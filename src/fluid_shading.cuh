#pragma once

static __forceinline__ __device__ bool intersect_aabb(float3 ro, float3 rd, float3 bmin, float3 bmax, float& tmin,
                                                       float& tmax) {
  float3 inv_d = make_float3((fabsf(rd.x) > 1e-8f) ? (1.0f / rd.x) : 1e16f,
                             (fabsf(rd.y) > 1e-8f) ? (1.0f / rd.y) : 1e16f,
                             (fabsf(rd.z) > 1e-8f) ? (1.0f / rd.z) : 1e16f);

  float tx1 = (bmin.x - ro.x) * inv_d.x;
  float tx2 = (bmax.x - ro.x) * inv_d.x;
  if (tx1 > tx2) {
    float tmp = tx1;
    tx1 = tx2;
    tx2 = tmp;
  }

  float ty1 = (bmin.y - ro.y) * inv_d.y;
  float ty2 = (bmax.y - ro.y) * inv_d.y;
  if (ty1 > ty2) {
    float tmp = ty1;
    ty1 = ty2;
    ty2 = tmp;
  }

  float tz1 = (bmin.z - ro.z) * inv_d.z;
  float tz2 = (bmax.z - ro.z) * inv_d.z;
  if (tz1 > tz2) {
    float tmp = tz1;
    tz1 = tz2;
    tz2 = tmp;
  }

  tmin = fmaxf(tx1, fmaxf(ty1, tz1));
  tmax = fminf(tx2, fminf(ty2, tz2));
  return tmax > fmaxf(tmin, 0.0f);
}

static __forceinline__ __device__ float sample_fluid_density(float3 p) {
  if (!params.fluid_volume_enabled || params.fluid_density_tex == 0) return 0.0f;

  float3 lo = params.fluid_bounds_lo;
  float3 hi = params.fluid_bounds_hi;
  float3 extent = make_float3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z);
  if (extent.x <= 1e-8f || extent.y <= 1e-8f || extent.z <= 1e-8f) return 0.0f;

  float u = clampf((p.x - lo.x) / extent.x, 0.0f, 1.0f);
  float v = clampf((p.y - lo.y) / extent.y, 0.0f, 1.0f);
  float w = clampf((p.z - lo.z) / extent.z, 0.0f, 1.0f);
  return tex3D<float>(params.fluid_density_tex, u, v, w);
}

static __forceinline__ __device__ float march_light_transmittance(float3 p, float3 L, float step_base) {
  float t0 = 0.0f;
  float t1 = 0.0f;
  if (!intersect_aabb(p, L, params.fluid_bounds_lo, params.fluid_bounds_hi, t0, t1)) {
    return 1.0f;
  }

  float step = fmaxf(step_base * 2.0f, 1e-3f);
  float t = fmaxf(t0, 0.0f) + step * 0.5f;
  float optical = 0.0f;
  int max_steps = min(96, max(1, static_cast<int>((t1 - fmaxf(t0, 0.0f)) / step) + 1));

  for (int i = 0; i < max_steps && t < t1; ++i, t += step) {
    float dens = sample_fluid_density(p + L * t);
    if (dens > 1e-5f) {
      optical += dens * step;
      if (optical > 20.0f) {
        break;
      }
    }
  }

  float sigma = fmaxf(params.fluid_absorption_strength, 0.0f);
  return expf(-sigma * optical);
}

static __forceinline__ __device__ float3 apply_fluid_volume(float3 origin, float3 dir, const HitGroupData* data,
                                                             float3 trans_color) {
  if (!params.fluid_volume_enabled || params.fluid_density_tex == 0 || !data || data->is_ground) {
    return trans_color;
  }
  float volume_strength = clampf(data->opacity, 0.0f, 1.0f);
  if (volume_strength <= 1e-4f) {
    return trans_color;
  }
  if (params.fluid_absorption_strength <= 1e-5f && params.fluid_volume_scattering <= 1e-5f) {
    return trans_color;
  }

  float t0 = 0.0f;
  float t1 = 0.0f;
  if (!intersect_aabb(origin, dir, params.fluid_bounds_lo, params.fluid_bounds_hi, t0, t1)) {
    return trans_color;
  }

  const float step = fmaxf(params.fluid_step_size, 1e-4f);
  const float t_enter = fmaxf(t0, 0.0f);
  const float march_len = t1 - t_enter;
  if (march_len <= step * 0.5f) {
    return trans_color;
  }

  const float3 one = make_float3(1.0f, 1.0f, 1.0f);
  const float3 base = data->base_color;
  const float absorption = fmaxf(params.fluid_absorption_strength, 0.0f) * volume_strength;
  const float scattering = clampf(params.fluid_volume_scattering, 0.0f, 1.0f) * volume_strength;

  // Colored absorption (Beer-Lambert) + scattering coefficient.
  const float3 sigma_a = (one - base * 0.7f) * absorption;
  const float3 sigma_s = base * (0.25f + 0.75f * scattering);
  const float3 sigma_t = sigma_a + sigma_s;

  float3 throughput = one;
  float3 accum = make_float3(0.0f, 0.0f, 0.0f);
  const float3 L = normalize(data->light_dir * -1.0f);
  const float3 light_col = data->light_color * data->light_strength;
  const float3 ambient_col = data->light_color * (0.08f * data->light_strength);

  float t = t_enter + step * 0.5f;
  const int max_steps = min(1024, max(1, static_cast<int>(march_len / step) + 1));
  for (int i = 0; i < max_steps && t < t1; ++i, t += step) {
    const float3 p = origin + dir * t;
    float dens = sample_fluid_density(p);
    if (dens <= 1e-5f) {
      continue;
    }

    float3 ext = sigma_t * (dens * step);
    float3 Tr_step = make_float3(expf(-ext.x), expf(-ext.y), expf(-ext.z));

    float surface_shadow = params.shadows_enabled ? trace_shadow(p + L * 0.01f, L) : 1.0f;
    float volume_shadow = march_light_transmittance(p + L * 0.01f, L, step);
    float ao = expf(-dens * 0.8f);  // simple density-based ambient occlusion approximation

    float3 Li = light_col * (surface_shadow * volume_shadow);
    float3 single_scatter = sigma_s * (dens * step) * (Li + ambient_col * ao);

    accum = accum + throughput * single_scatter;
    throughput = throughput * Tr_step;

    float tr_max = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
    if (tr_max < 0.01f) {
      break;
    }
  }

  return accum + throughput * trans_color;
}

static __forceinline__ __device__ float3 shade_fluid_interface(float3 hit_pos, float3 N, float3 V,
                                                                const HitGroupData* data) {
  float roughness = clampf(data->roughness, 0.02f, 1.0f);
  float shininess = fmaxf(2.0f / (roughness * roughness + 1e-4f) - 2.0f, 1.0f);
  float3 L = normalize(data->light_dir * (-1.0f));
  float NdotL = fmaxf(dot(N, L), 0.0f);
  float shadow = params.shadows_enabled ? trace_shadow(hit_pos, L) : 1.0f;
  float3 H = normalize(L + V);
  float spec = powf(fmaxf(dot(N, H), 0.0f), shininess);
  float3 diffuse = data->base_color * data->light_color * (data->light_strength * 0.12f * NdotL);
  float3 specular = make_float3(1.0f, 1.0f, 1.0f) * (data->light_strength * spec);
  float3 ambient = data->base_color * data->light_color * (data->light_strength * 0.05f);
  return (diffuse + specular) * shadow + ambient;
}

extern "C" __global__ void __closesthit__fluid() {
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

  const uint3 tri = data->indices[prim_idx];
  if (tri.x >= data->num_normals || tri.y >= data->num_normals || tri.z >= data->num_normals) {
    setPayload(make_float3(1.0f, 0.0f, 1.0f));
    return;
  }

  float3 n0 = data->normals[tri.x];
  float3 n1 = data->normals[tri.y];
  float3 n2 = data->normals[tri.z];
  float3 N_geom = normalize(n0 * (1.0f - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);

  float3 ray_dir = optixGetWorldRayDirection();
  float3 N = N_geom;
  if (dot(N, ray_dir) > 0.0f) N = N * (-1.0f);

  float3 hit_pos = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
  float3 V = normalize(ray_dir * (-1.0f));
  unsigned int depth = optixGetPayload_3();

  if (!data->interface_visible) {
    float3 trans_color = make_float3(0.0f, 0.0f, 0.0f);
    if (depth < params.max_depth) {
      const bool entering = (dot(ray_dir, N_geom) < 0.0f);
      float3 march_origin = hit_pos + ray_dir * 0.01f;
      if (!entering) {
        // Back-face hit while volume-only: pass through and avoid applying
        // volume twice on the same segment.
        trans_color = trace_radiance(march_origin, ray_dir, depth + 1);
      } else {
        trans_color = trace_radiance(march_origin, ray_dir, depth + 1);
        trans_color = apply_fluid_volume(march_origin, ray_dir, data, trans_color);
      }
    }
    setPayload(trans_color);
    return;
  }

  float interface_opacity = clampf(data->opacity, 0.0f, 1.0f);
  float kIOR = clampf(data->ior, 1.0f, 2.5f);
  float cos_theta = fmaxf(dot(V, N), 0.0f);
  float fresnel = fresnel_schlick(cos_theta, kIOR);

  float3 refl_color = make_float3(0.0f, 0.0f, 0.0f);
  float3 trans_color = make_float3(0.0f, 0.0f, 0.0f);
  if (depth < params.max_depth) {
    float3 refl_dir = reflect(ray_dir, N);
    refl_color = trace_radiance(hit_pos + N * 0.01f, refl_dir, depth + 1);

    float3 refr_dir;
    if (refract_dir(ray_dir, N_geom, kIOR, refr_dir)) {
      trans_color = trace_radiance(hit_pos - N * 0.01f, refr_dir, depth + 1);
      trans_color = apply_fluid_volume(hit_pos - N * 0.01f, refr_dir, data, trans_color);
    } else {
      trans_color = refl_color;
    }
  }

  float3 direct = shade_fluid_interface(hit_pos, N, V, data);
  float3 dielectric = trans_color * (1.0f - fresnel) + refl_color * fresnel;
  float3 color = dielectric * (1.0f - interface_opacity) + direct * interface_opacity;
  setPayload(color);
}
