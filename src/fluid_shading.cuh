#pragma once

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
  float3 surface_color = shade_lights(hit_pos, N, V, data);
  float opacity = clampf(data->opacity, 0.0f, 1.0f);
  float3 color = surface_color * opacity;
  unsigned int depth = optixGetPayload_3();

  if (opacity < 0.999f && depth < params.max_depth) {
    float kIOR = clampf(data->ior, 1.0f, 2.5f);
    float cos_theta = fmaxf(dot(V, N), 0.0f);
    float fresnel = fresnel_schlick(cos_theta, kIOR);

    float3 refl_dir = reflect(ray_dir, N);
    float3 refl_color = trace_radiance(hit_pos + N * 0.01f, refl_dir, depth + 1);

    float3 refr_dir;
    float3 trans_color = refl_color;
    if (refract_dir(ray_dir, N_geom, kIOR, refr_dir)) {
      trans_color = trace_radiance(hit_pos - N * 0.01f, refr_dir, depth + 1);
    }

    float3 tint = make_float3(1.0f, 1.0f, 1.0f) * 0.75f + data->base_color * 0.25f;
    float3 glass_color = (trans_color * (1.0f - fresnel) + refl_color * fresnel) * tint;
    color = surface_color * opacity + glass_color * (1.0f - opacity);
  }

  setPayload(color);
}
