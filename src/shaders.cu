#include <optix.h>
#include "optix_params.h"

extern "C" {
__constant__ Params params;
}

// --- Inline vector math helpers ---

static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __device__ float3 operator*(float3 a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(float s, float3 a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize(float3 v) {
  float inv_len = rsqrtf(fmaxf(dot(v, v), 1e-12f));
  return v * inv_len;
}

static __forceinline__ __device__ float3 reflect(float3 I, float3 N) {
  return I - 2.0f * dot(I, N) * N;
}

static __forceinline__ __device__ float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

static __forceinline__ __device__ uchar4 make_color(float3 c) {
  // Clamp and apply sRGB gamma
  float r = powf(clampf(c.x, 0.0f, 1.0f), 1.0f / 2.2f);
  float g = powf(clampf(c.y, 0.0f, 1.0f), 1.0f / 2.2f);
  float b = powf(clampf(c.z, 0.0f, 1.0f), 1.0f / 2.2f);
  return make_uchar4((unsigned char)(r * 255.0f + 0.5f), (unsigned char)(g * 255.0f + 0.5f),
                     (unsigned char)(b * 255.0f + 0.5f), 255u);
}

// --- Payload helpers (4 payload values: RGB color + depth) ---

static __forceinline__ __device__ void setPayload(float3 p) {
  optixSetPayload_0(__float_as_uint(p.x));
  optixSetPayload_1(__float_as_uint(p.y));
  optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload() {
  return make_float3(__uint_as_float(optixGetPayload_0()), __uint_as_float(optixGetPayload_1()),
                     __uint_as_float(optixGetPayload_2()));
}

// --- Shadow ray helper ---
// Traces a shadow ray toward a light. Returns 1.0 if lit, 0.0 if shadowed.
// Uses DISABLE_CLOSESTHIT + TERMINATE_ON_FIRST_HIT: if nothing is hit, miss sets p0=1.0.
// Before trace, p0 is set to 0.0 (assume shadowed).

static __forceinline__ __device__ float trace_shadow(float3 origin, float3 direction) {
  unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;  // must pass all 4 payloads
  optixTrace(params.handle, origin, direction,
             0.001f,   // tmin (offset to avoid self-intersection)
             1e16f,    // tmax
             0.0f,     // rayTime
             OptixVisibilityMask(255),
             OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT |
                 OPTIX_RAY_FLAG_DISABLE_ANYHIT,
             RAY_TYPE_SHADOW,   // SBT offset (shadow ray type)
             RAY_TYPE_COUNT,    // SBT stride
             RAY_TYPE_SHADOW,   // missSBTIndex (shadow miss)
             p0, p1, p2, p3);
  return __uint_as_float(p0);
}

// --- Lighting helper ---
// Computes Blinn-Phong shading from the global directional light.
// Each light casts a shadow ray to determine visibility.

static __forceinline__ __device__ float3 shade_lights(float3 hit_pos, float3 N, float3 V,
                                                       const HitGroupData* data) {
  float3 color = make_float3(0.0f, 0.0f, 0.0f);
  float shininess = fmaxf(2.0f / (data->roughness * data->roughness + 1e-4f) - 2.0f, 1.0f);
  float3 spec_tint = data->base_color * data->metallic + make_float3(1.0f, 1.0f, 1.0f) * (1.0f - data->metallic);

  // Helper lambda-like: shade one light
  // We'll inline it as a local function pattern
  #define SHADE_ONE_LIGHT(L_dir, L_color, L_strength) do { \
    float3 L = normalize((L_dir) * (-1.0f)); \
    float NdotL = fmaxf(dot(N, L), 0.0f); \
    if (NdotL > 0.0f) { \
      float shadow = params.shadows_enabled ? trace_shadow(hit_pos, L) : 1.0f; \
      float3 H = normalize(L + V); \
      float NdotH = fmaxf(dot(N, H), 0.0f); \
      float spec = powf(NdotH, shininess); \
      float3 diffuse = data->base_color * (L_color) * ((L_strength) * NdotL); \
      float3 specular = spec_tint * (L_color) * ((L_strength) * spec); \
      color = color + (diffuse + specular) * shadow; \
    } \
  } while(0)

  // Global directional light
  SHADE_ONE_LIGHT(data->light_dir, data->light_color, data->light_strength);

  #undef SHADE_ONE_LIGHT

  // Ambient
  float3 ambient = data->base_color * data->light_color * (data->light_strength * 0.15f);
  color = color + ambient;

  return color;
}

// --- Trace radiance ray helper ---

static __forceinline__ __device__ float3 trace_radiance(float3 origin, float3 direction,
                                                         unsigned int depth) {
  unsigned int p0, p1, p2, p3;
  p3 = depth;
  optixTrace(params.handle, origin, direction,
             0.001f,   // tmin
             1e16f,    // tmax
             0.0f,     // rayTime
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
             RAY_TYPE_RADIANCE,  // SBT offset
             RAY_TYPE_COUNT,     // SBT stride
             RAY_TYPE_RADIANCE,  // missSBTIndex
             p0, p1, p2, p3);
  return make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));
}

// --- Ray generation ---

// Simple hash-based pseudo-random for sub-pixel jitter
static __forceinline__ __device__ float halton(unsigned int index, unsigned int base) {
  float f = 1.0f;
  float r = 0.0f;
  unsigned int i = index;
  while (i > 0) {
    f /= static_cast<float>(base);
    r += f * static_cast<float>(i % base);
    i /= base;
  }
  return r;
}

extern "C" __global__ void __raygen__rg() {
  const uint3 idx = optixGetLaunchIndex();
  const uint3 dim = optixGetLaunchDimensions();
  const unsigned int spp = params.samples_per_pixel;

  float3 accum = make_float3(0.0f, 0.0f, 0.0f);

  for (unsigned int s = 0; s < spp; s++) {
    // Sub-pixel jitter using Halton sequence (centered around pixel)
    float jx = (spp > 1) ? halton(s + 1, 2) - 0.5f : 0.0f;
    float jy = (spp > 1) ? halton(s + 1, 3) - 0.5f : 0.0f;

    const float2 d = make_float2(
        2.0f * (static_cast<float>(idx.x) + jx) / static_cast<float>(dim.x) - 1.0f,
        2.0f * (static_cast<float>(idx.y) + jy) / static_cast<float>(dim.y) - 1.0f);

    float3 origin = params.cam_eye;
    float3 direction = normalize(d.x * params.cam_u + d.y * params.cam_v + params.cam_w);

    float3 color = trace_radiance(origin, direction, 0);
    accum = accum + color;
  }

  float3 result = accum * (1.0f / static_cast<float>(spp));
  params.image[idx.y * params.image_width + idx.x] = make_color(result);
}

// --- Miss programs ---

extern "C" __global__ void __miss__radiance() {
  MissData* data = reinterpret_cast<MissData*>(optixGetSbtDataPointer());
  // Background color is already in linear space, pass through
  setPayload(data->bg_color);
}

extern "C" __global__ void __miss__shadow() {
  // No blocker found — ray escaped. Set payload to 1.0 (lit).
  optixSetPayload_0(__float_as_uint(1.0f));
}

// --- Closest hit: mesh ---

extern "C" __global__ void __closesthit__ch() {
  HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
  if (!data || !data->indices || !data->normals || data->num_indices == 0 || data->num_normals == 0) {
    setPayload(make_float3(1.0f, 0.0f, 1.0f));
    return;
  }

  const float2 bary = optixGetTriangleBarycentrics();
  const unsigned int prim_idx = optixGetPrimitiveIndex();
  if (prim_idx >= data->num_indices) {
    setPayload(make_float3(1.0f, 0.0f, 1.0f));
    return;
  }

  // Interpolate vertex normal
  const uint3 tri = data->indices[prim_idx];
  if (tri.x >= data->num_normals || tri.y >= data->num_normals || tri.z >= data->num_normals) {
    setPayload(make_float3(1.0f, 0.0f, 1.0f));
    return;
  }
  float3 n0 = data->normals[tri.x];
  float3 n1 = data->normals[tri.y];
  float3 n2 = data->normals[tri.z];
  float3 N = normalize(n0 * (1.0f - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);

  // Ensure normal faces the ray (flip if backface)
  float3 ray_dir = optixGetWorldRayDirection();
  if (dot(N, ray_dir) > 0.0f) N = N * (-1.0f);

  // Compute hit position
  float3 hit_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

  // View direction
  float3 V = normalize(ray_dir * (-1.0f));

  // Shade with all lights + shadows
  float3 color = shade_lights(hit_pos, N, V, data) * data->opacity;

  setPayload(color);
}

// --- Closest hit: ground plane ---

extern "C" __global__ void __closesthit__ground() {
  HitGroupData* data = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());

  float3 N = make_float3(0.0f, 1.0f, 0.0f);
  float3 ray_dir = optixGetWorldRayDirection();

  // Flip normal if ray hits from below
  if (dot(N, ray_dir) > 0.0f) N = N * (-1.0f);

  float3 hit_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
  float3 V = normalize(ray_dir * (-1.0f));

  // Shade with all lights + shadows
  float3 color = shade_lights(hit_pos, N, V, data);

  // Reflection (only from primary rays to stay within maxTraceDepth=3)
  unsigned int depth = optixGetPayload_3();
  if (depth == 0 && data->metallic > 0.01f) {
    float3 refl_dir = reflect(ray_dir, N);
    float3 refl_color = trace_radiance(hit_pos, refl_dir, depth + 1);
    color = color * (1.0f - data->metallic) + refl_color * data->base_color * data->metallic;
  }

  color = color * data->opacity;

  setPayload(color);
}
