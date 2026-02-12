#pragma once

#include <cuda_runtime.h>
#include <optix.h>

enum RayType { RAY_TYPE_RADIANCE = 0, RAY_TYPE_SHADOW = 1, RAY_TYPE_COUNT };

struct Params {
  uchar4* image;
  unsigned int image_width;
  unsigned int image_height;
  unsigned int samples_per_pixel;
  unsigned int max_depth;
  float3 cam_eye;
  float3 cam_u, cam_v, cam_w;
  OptixTraversableHandle handle;
  int shadows_enabled;
  int fluid_volume_enabled;
  cudaTextureObject_t fluid_density_tex;
  float3 fluid_bounds_lo;
  float3 fluid_bounds_hi;
  float fluid_absorption_strength;
  float fluid_volume_mix;
  float fluid_step_size;
};

struct RayGenData {};

struct MissData {
  float3 bg_color;
};

struct HitGroupData {
  float3 base_color;
  float metallic;
  float roughness;
  float opacity;
  float ior;
  int interface_visible;
  float3 light_dir;
  float3 light_color;
  float light_strength;
  float3* normals;
  uint3* indices;
  unsigned int num_normals;
  unsigned int num_indices;
  int is_ground;
};
