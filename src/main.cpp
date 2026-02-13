#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <ImGuiFileDialog.h>
#include <viskores/io/VTKDataSetReader.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/Field.h>
#include <viskores/cont/CellSetStructured.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/CellShape.h>
#include <viskores/filter/contour/Contour.h>
#include <viskores/filter/field_conversion/PointAverage.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <sstream>
#include "config.hpp"
#include "optix_params.h"

// --- Error checking macros ---

#define CUDA_CHECK(call)                                                                                 \
  do {                                                                                                   \
    cudaError_t rc = call;                                                                               \
    if (rc != cudaSuccess) {                                                                             \
      std::ostringstream oss;                                                                            \
      oss << "CUDA error " << cudaGetErrorName(rc) << ": " << cudaGetErrorString(rc) << " at "         \
          << __FILE__ << ":" << __LINE__;                                                                \
      std::cerr << oss.str() << "\n";                                                                    \
      throw std::runtime_error(oss.str());                                                               \
    }                                                                                                    \
  } while (0)

#define OPTIX_CHECK(call)                                                            \
  do {                                                                               \
    OptixResult res = call;                                                          \
    if (res != OPTIX_SUCCESS) {                                                      \
      std::ostringstream oss;                                                        \
      oss << "OptiX error " << res << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
      std::cerr << oss.str() << "\n";                                                \
      throw std::runtime_error(oss.str());                                           \
    }                                                                                \
  } while (0)

static char optix_log[2048];
static size_t optix_log_size = sizeof(optix_log);

#define OPTIX_CHECK_LOG(call)                                    \
  do {                                                           \
    optix_log_size = sizeof(optix_log);                          \
    OptixResult res = call;                                      \
    if (res != OPTIX_SUCCESS) {                                  \
      std::ostringstream oss;                                    \
      oss << "OptiX error " << res << ": " << optix_log << "\n"; \
      throw std::runtime_error(oss.str());                       \
    }                                                            \
  } while (0)

// --- SBT record template ---

template <typename T>
struct SbtRecord {
  __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// --- Helper types ---

struct DataLayer {
  std::string name;
  bool show = true;
  bool prev_show = show;
  float threshold_min = 0.5f;
  float prev_threshold_min = threshold_min;
  float threshold_max = 1.0f;
  float prev_threshold_max = threshold_max;
  int fluid_flag = 0;
  int prev_fluid_flag = fluid_flag;
  ImVec4 color = ImVec4(0.8f, 0.9f, 1.0f, 1.0f);
  ImVec4 prev_color = color;
  float metallic = 0.0f;
  float prev_metallic = metallic;
  float roughness = 0.05f;
  float prev_roughness = roughness;
  float opacity = 1.0f;
  float prev_opacity = opacity;
  float glass_ior = 1.33f;
  float prev_glass_ior = glass_ior;
  int smooth_iterations = 8;
  int prev_smooth_iterations = smooth_iterations;
  float smooth_strength = 0.2f;
  float prev_smooth_strength = smooth_strength;
  bool prev_renderable = false;
  bool suppress_retry_after_error = false;
  std::string failed_source_file;
  std::string failed_mask_file;
  std::string failed_mask_field;
  std::string failed_field_name;
  int failed_fluid_flag = 0;
  float failed_threshold_min = 0.0f;
  float failed_threshold_max = 0.0f;
};

struct BBox {
  float3 lo = make_float3(1e30f, 1e30f, 1e30f);
  float3 hi = make_float3(-1e30f, -1e30f, -1e30f);
};

struct GpuMeshBuffers {
  CUdeviceptr d_vertices = 0;
  CUdeviceptr d_indices_buf = 0;
  CUdeviceptr d_normals = 0;
  CUdeviceptr d_indices = 0;
  unsigned int num_vertices = 0;
  unsigned int num_triangles = 0;

  bool IsValid() const { return d_vertices != 0 && d_indices_buf != 0 && d_normals != 0 && d_indices != 0 &&
                                num_vertices > 0 && num_triangles > 0; }
};

struct MeshCache {
  std::string source_file;
  std::string source_field;
  int source_solid_flag = 1;
  std::vector<float3> base_positions;
  std::vector<float3> base_normals;
  std::vector<uint3> indices;
  bool valid = false;

  void Clear() {
    source_file.clear();
    source_field.clear();
    source_solid_flag = 1;
    base_positions.clear();
    base_normals.clear();
    indices.clear();
    valid = false;
  }
};

struct LightingState {
  ImVec4 bg_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 prev_bg_color = bg_color;
  float strength = 1.0f;
  float prev_strength = strength;
  ImVec4 color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  ImVec4 prev_color = color;
  float3 dir = make_float3(0.0f, -1.0f, -0.5f);
  bool shadows_enabled = false;
  bool prev_shadows_enabled = true;
};

struct MaskState {
  std::string file;
  std::vector<std::string> field_names;
  int field_index = 0;
  int prev_field_index = 0;
  bool show = false;
  bool prev_show = false;
  int solid_flag = 1;
  int prev_solid_flag = 1;
  ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
  ImVec4 prev_color = color;
  float metallic = 0.0f;
  float prev_metallic = metallic;
  float roughness = 0.5f;
  float prev_roughness = roughness;
  float opacity = 1.0f;
  float prev_opacity = opacity;
  float glass_ior = 1.45f;
  float prev_glass_ior = glass_ior;
  int smooth_iterations = 0;
  int prev_smooth_iterations = 0;
  float smooth_strength = 0.5f;
  float prev_smooth_strength = smooth_strength;
};

struct GroundState {
  bool enabled = true;
  bool prev_enabled = true;
  float y_offset = 0.0f;
  float prev_y_offset = 0.0f;
  ImVec4 color = ImVec4(0.5f, 0.5f, 0.5f, 1.0f);
  ImVec4 prev_color = color;
  float metallic = 0.3f;
  float prev_metallic = metallic;
  float roughness = 0.5f;
  float prev_roughness = roughness;
  float opacity = 1.0f;
  float prev_opacity = opacity;
};

struct DatasetState {
  std::string vtk_dir;
  std::vector<std::string> vtk_files;
  int vtk_index = 0;
  std::vector<std::string> cell_names;
  std::vector<std::string> scalar_cell_names;
  std::vector<DataLayer> layers;
  std::vector<MeshCache> layer_mesh_caches;
  int loaded_field_vtk_index = -1;
  bool first_frame = true;
};

struct RenderMiscState {
  bool show_outlines = false;
  ImVec4 outline_color = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
  float outline_thickness = 2.0f;
  float rotate_x_deg = 0.0f;
  float prev_rotate_x_deg = rotate_x_deg;
  float rotate_y_deg = 0.0f;
  float prev_rotate_y_deg = rotate_y_deg;
  float rotate_z_deg = 0.0f;
  float prev_rotate_z_deg = rotate_z_deg;
  bool show_mask_error = false;
  std::string mask_error_msg;
};

struct RayTracingState {
  int bounces = 4;
  int prev_bounces = bounces;
  int samples = 1;
  int prev_samples = samples;
};

struct CameraState {
  float yaw = 0.0f;
  float pitch = 30.0f;
  float distance = 5.0f;
  float target[3] = {0.0f, 0.0f, 0.0f};
  float fov = 60.0f;
  bool viewport_needs_render = true;
  int prev_vp_w = 0;
  int prev_vp_h = 0;
};

struct OptixState {
  OptixDeviceContext context = nullptr;
  OptixModule module = nullptr;
  OptixPipeline pipeline = nullptr;
  // 6 program groups
  OptixProgramGroup raygen_pg = nullptr;
  OptixProgramGroup miss_radiance_pg = nullptr;
  OptixProgramGroup miss_shadow_pg = nullptr;
  OptixProgramGroup hitgroup_mesh_pg = nullptr;
  OptixProgramGroup hitgroup_fluid_pg = nullptr;
  OptixProgramGroup hitgroup_ground_pg = nullptr;
  OptixProgramGroup hitgroup_shadow_pg = nullptr;
  OptixShaderBindingTable sbt = {};
  CUdeviceptr d_raygen_record = 0;
  CUdeviceptr d_miss_records = 0;      // 2 miss records
  CUdeviceptr d_hitgroup_records = 0;
  int hitgroup_record_capacity = 0;
  int hitgroup_record_count = 0;
  int mask_sbt_index = -1;
  std::vector<int> fluid_sbt_indices;
  std::vector<int> fluid_layer_indices;
  int ground_sbt_index = -1;
  OptixTraversableHandle gas_handle = 0;
  CUdeviceptr d_gas_temp = 0;
  CUdeviceptr d_gas_output = 0;
  size_t gas_temp_capacity = 0;
  size_t gas_output_capacity = 0;
  GpuMeshBuffers mask_mesh;
  std::vector<GpuMeshBuffers> fluid_meshes;
  CUdeviceptr d_ground_vertices = 0;
  CUdeviceptr d_ground_indices = 0;
  CUdeviceptr d_image = 0;
  CUdeviceptr d_params = 0;
  int img_w = 0, img_h = 0;
};

// --- PTX file loader ---

static std::string load_ptx_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open PTX file: " + path);
  }
  return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// --- OptiX log callback ---

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
  std::cerr << "[" << level << "][" << tag << "]: " << message << "\n";
}

// --- Initialize OptiX pipeline (one-time) ---

static void init_optix(OptixState& state, const std::string& ptx_path, const ImVec4& bg_color) {
  // Init CUDA
  CUDA_CHECK(cudaFree(0));

  // Init OptiX
  OPTIX_CHECK(optixInit());

  // Create context
  OptixDeviceContextOptions ctx_options = {};
  ctx_options.logCallbackFunction = &context_log_cb;
  ctx_options.logCallbackLevel = 4;
  CUcontext cu_ctx = 0;
  OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &ctx_options, &state.context));

  // Load PTX
  std::string ptx = load_ptx_file(ptx_path);

  // Module
  OptixModuleCompileOptions module_options = {};
  OptixPipelineCompileOptions pipeline_options = {};
  pipeline_options.usesMotionBlur = false;
  pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipeline_options.numPayloadValues = 4;
  pipeline_options.numAttributeValues = 2;
  pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  pipeline_options.pipelineLaunchParamsVariableName = "params";
  pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  OPTIX_CHECK_LOG(optixModuleCreate(state.context, &module_options, &pipeline_options, ptx.c_str(), ptx.size(),
                                    optix_log, &optix_log_size, &state.module));

  // Program groups (7 total)
  OptixProgramGroupOptions pg_options = {};

  // 1. Raygen
  OptixProgramGroupDesc raygen_desc = {};
  raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  raygen_desc.raygen.module = state.module;
  raygen_desc.raygen.entryFunctionName = "__raygen__rg";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &raygen_desc, 1, &pg_options, optix_log, &optix_log_size,
                                          &state.raygen_pg));

  // 2. Miss radiance
  OptixProgramGroupDesc miss_radiance_desc = {};
  miss_radiance_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_radiance_desc.miss.module = state.module;
  miss_radiance_desc.miss.entryFunctionName = "__miss__radiance";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &miss_radiance_desc, 1, &pg_options, optix_log,
                                          &optix_log_size, &state.miss_radiance_pg));

  // 3. Miss shadow
  OptixProgramGroupDesc miss_shadow_desc = {};
  miss_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  miss_shadow_desc.miss.module = state.module;
  miss_shadow_desc.miss.entryFunctionName = "__miss__shadow";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &miss_shadow_desc, 1, &pg_options, optix_log, &optix_log_size,
                                          &state.miss_shadow_pg));

  // 4. Hitgroup mesh (closesthit__ch)
  OptixProgramGroupDesc hitgroup_mesh_desc = {};
  hitgroup_mesh_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_mesh_desc.hitgroup.moduleCH = state.module;
  hitgroup_mesh_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hitgroup_mesh_desc, 1, &pg_options, optix_log,
                                          &optix_log_size, &state.hitgroup_mesh_pg));

  // 5. Hitgroup fluid (closesthit__fluid)
  OptixProgramGroupDesc hitgroup_fluid_desc = {};
  hitgroup_fluid_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_fluid_desc.hitgroup.moduleCH = state.module;
  hitgroup_fluid_desc.hitgroup.entryFunctionNameCH = "__closesthit__fluid";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hitgroup_fluid_desc, 1, &pg_options, optix_log,
                                          &optix_log_size, &state.hitgroup_fluid_pg));

  // 6. Hitgroup ground (closesthit__ground)
  OptixProgramGroupDesc hitgroup_ground_desc = {};
  hitgroup_ground_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_ground_desc.hitgroup.moduleCH = state.module;
  hitgroup_ground_desc.hitgroup.entryFunctionNameCH = "__closesthit__ground";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hitgroup_ground_desc, 1, &pg_options, optix_log,
                                          &optix_log_size, &state.hitgroup_ground_pg));

  // 7. Hitgroup shadow (closesthit__shadow)
  OptixProgramGroupDesc hitgroup_shadow_desc = {};
  hitgroup_shadow_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  hitgroup_shadow_desc.hitgroup.moduleCH = state.module;
  hitgroup_shadow_desc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
  OPTIX_CHECK_LOG(optixProgramGroupCreate(state.context, &hitgroup_shadow_desc, 1, &pg_options, optix_log,
                                          &optix_log_size, &state.hitgroup_shadow_pg));

  // Pipeline
  const uint32_t max_trace_depth = 3;
  OptixProgramGroup program_groups[] = {state.raygen_pg, state.miss_radiance_pg, state.miss_shadow_pg,
                                        state.hitgroup_mesh_pg, state.hitgroup_fluid_pg, state.hitgroup_ground_pg,
                                        state.hitgroup_shadow_pg};

  OptixPipelineLinkOptions link_options = {};
  link_options.maxTraceDepth = max_trace_depth;
  OPTIX_CHECK_LOG(optixPipelineCreate(state.context, &pipeline_options, &link_options, program_groups, 7, optix_log,
                                      &optix_log_size, &state.pipeline));

  // Stack sizes
  OptixStackSizes stack_sizes = {};
  for (auto& pg : program_groups) {
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, state.pipeline));
  }

  uint32_t dcsft, dcsfst, css;
  OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &dcsft, &dcsfst, &css));
  css = std::max(css, 4096u);  // Ensure minimum for nested shadow/reflection traces
  OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, dcsft, dcsfst, css, 1));

  // SBT - raygen
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_raygen_record), sizeof(RayGenSbtRecord)));
  RayGenSbtRecord rg_sbt = {};
  OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_pg, &rg_sbt));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_raygen_record), &rg_sbt, sizeof(RayGenSbtRecord),
                        cudaMemcpyHostToDevice));

  // SBT - miss (2 records: radiance + shadow)
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_miss_records), 2 * sizeof(MissSbtRecord)));
  MissSbtRecord miss_records[2] = {};
  // Record 0: radiance miss
  miss_records[0].data.bg_color = make_float3(bg_color.x, bg_color.y, bg_color.z);
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_radiance_pg, &miss_records[0]));
  // Record 1: shadow miss (data unused)
  miss_records[1].data.bg_color = make_float3(0.0f, 0.0f, 0.0f);
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_shadow_pg, &miss_records[1]));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_miss_records), miss_records, 2 * sizeof(MissSbtRecord),
                        cudaMemcpyHostToDevice));

  // SBT - hitgroup (start with one geometry x 2 ray types; grows on demand)
  state.hitgroup_record_capacity = 2;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_hitgroup_records),
                        static_cast<size_t>(state.hitgroup_record_capacity) * sizeof(HitGroupSbtRecord)));
  HitGroupSbtRecord hg_defaults[2] = {};
  for (int i = 0; i < 2; i++) {
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_mesh_pg, &hg_defaults[i]));
  }
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_hitgroup_records), hg_defaults,
                        2 * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
  state.hitgroup_record_count = 2;  // default: one geometry x two ray types

  // Assemble SBT
  state.sbt.raygenRecord = state.d_raygen_record;
  state.sbt.missRecordBase = state.d_miss_records;
  state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
  state.sbt.missRecordCount = 2;
  state.sbt.hitgroupRecordBase = state.d_hitgroup_records;
  state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
  state.sbt.hitgroupRecordCount = state.hitgroup_record_count;

  // Allocate params buffer on device
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));
}

static void ensure_hitgroup_record_capacity(OptixState& state, int geometry_count) {
  int required_records = std::max(2, 2 * geometry_count);
  if (required_records <= state.hitgroup_record_capacity) {
    return;
  }
  if (state.d_hitgroup_records) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_hitgroup_records)));
    state.d_hitgroup_records = 0;
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_hitgroup_records),
                        static_cast<size_t>(required_records) * sizeof(HitGroupSbtRecord)));
  std::vector<HitGroupSbtRecord> defaults(static_cast<size_t>(required_records));
  for (int i = 0; i < required_records; i++) {
    defaults[static_cast<size_t>(i)] = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_mesh_pg, &defaults[static_cast<size_t>(i)]));
  }
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_hitgroup_records), defaults.data(),
                        static_cast<size_t>(required_records) * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
  state.hitgroup_record_capacity = required_records;
  state.sbt.hitgroupRecordBase = state.d_hitgroup_records;
}

// --- Update miss records (background color) ---

static void update_miss_sbt(OptixState& state, const ImVec4& bg_color) {
  MissSbtRecord miss_records[2] = {};
  miss_records[0].data.bg_color = make_float3(bg_color.x, bg_color.y, bg_color.z);
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_radiance_pg, &miss_records[0]));
  miss_records[1].data.bg_color = make_float3(0.0f, 0.0f, 0.0f);
  OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_shadow_pg, &miss_records[1]));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_miss_records), miss_records, 2 * sizeof(MissSbtRecord),
                        cudaMemcpyHostToDevice));
}

static void free_gpu_mesh_buffers(GpuMeshBuffers& mesh) {
  if (mesh.d_vertices) cudaFree(reinterpret_cast<void*>(mesh.d_vertices));
  if (mesh.d_indices_buf) cudaFree(reinterpret_cast<void*>(mesh.d_indices_buf));
  if (mesh.d_normals) cudaFree(reinterpret_cast<void*>(mesh.d_normals));
  if (mesh.d_indices) cudaFree(reinterpret_cast<void*>(mesh.d_indices));
  mesh = {};
}

static void upload_mesh_buffers_to_gpu(const std::vector<float3>& positions, const std::vector<float3>& normals,
                                       const std::vector<uint3>& indices, GpuMeshBuffers& mesh) {
  if (positions.empty() || normals.size() != positions.size() || indices.empty()) {
    throw std::runtime_error("Invalid mesh buffers for GPU upload.");
  }

  free_gpu_mesh_buffers(mesh);

  size_t vert_size = positions.size() * sizeof(float3);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mesh.d_vertices), vert_size));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mesh.d_vertices), positions.data(), vert_size, cudaMemcpyHostToDevice));

  size_t idx_size = indices.size() * sizeof(uint3);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mesh.d_indices_buf), idx_size));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mesh.d_indices_buf), indices.data(), idx_size, cudaMemcpyHostToDevice));

  size_t norm_size = normals.size() * sizeof(float3);
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mesh.d_normals), norm_size));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mesh.d_normals), normals.data(), norm_size, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&mesh.d_indices), idx_size));
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(mesh.d_indices), indices.data(), idx_size, cudaMemcpyHostToDevice));

  mesh.num_vertices = static_cast<unsigned int>(positions.size());
  mesh.num_triangles = static_cast<unsigned int>(indices.size());
}

static void write_shadow_hitgroup_record(OptixState& state, int sbt_index, const HitGroupData& data) {
  if (sbt_index < 0) return;
  HitGroupSbtRecord shadow_rec = {};
  shadow_rec.data = data;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_shadow_pg, &shadow_rec));
  size_t byte_offset = static_cast<size_t>(2 * sbt_index + RAY_TYPE_SHADOW) * sizeof(HitGroupSbtRecord);
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_hitgroup_records + byte_offset), &shadow_rec,
                        sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
}

static void update_surface_hitgroup_sbt(OptixState& state, int sbt_index, OptixProgramGroup program_group,
                                        const GpuMeshBuffers& mesh, float base_r, float base_g, float base_b,
                                        float metallic, float roughness, float opacity, float ior,
                                        bool interface_visible, const float3& light_dir, float light_r, float light_g,
                                        float light_b, float light_strength) {
  if (sbt_index < 0 || !mesh.IsValid()) return;

  HitGroupSbtRecord hg_sbt = {};
  hg_sbt.data.base_color = make_float3(base_r, base_g, base_b);
  hg_sbt.data.metallic = metallic;
  hg_sbt.data.roughness = roughness;
  hg_sbt.data.opacity = opacity;
  hg_sbt.data.ior = ior;
  hg_sbt.data.interface_visible = interface_visible ? 1 : 0;
  hg_sbt.data.light_dir = light_dir;
  hg_sbt.data.light_color = make_float3(light_r, light_g, light_b);
  hg_sbt.data.light_strength = light_strength;
  hg_sbt.data.normals = reinterpret_cast<float3*>(mesh.d_normals);
  hg_sbt.data.indices = reinterpret_cast<uint3*>(mesh.d_indices);
  hg_sbt.data.num_normals = mesh.num_vertices;
  hg_sbt.data.num_indices = mesh.num_triangles;
  hg_sbt.data.is_ground = 0;
  OPTIX_CHECK(optixSbtRecordPackHeader(program_group, &hg_sbt));
  size_t byte_offset = static_cast<size_t>(2 * sbt_index + RAY_TYPE_RADIANCE) * sizeof(HitGroupSbtRecord);
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_hitgroup_records + byte_offset), &hg_sbt,
                        sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
  write_shadow_hitgroup_record(state, sbt_index, hg_sbt.data);
}

static void update_ground_hitgroup_sbt(OptixState& state, int sbt_index, const ImVec4& ground_color,
                                       float ground_metallic, float ground_roughness, float ground_opacity,
                                       const float3& light_dir, float light_r, float light_g, float light_b,
                                       float light_strength) {
  if (sbt_index < 0) return;

  HitGroupSbtRecord hg_sbt = {};
  hg_sbt.data.base_color = make_float3(ground_color.x, ground_color.y, ground_color.z);
  hg_sbt.data.metallic = ground_metallic;
  hg_sbt.data.roughness = ground_roughness;
  hg_sbt.data.opacity = ground_opacity;
  hg_sbt.data.ior = 1.0f;
  hg_sbt.data.interface_visible = 1;
  hg_sbt.data.light_dir = light_dir;
  hg_sbt.data.light_color = make_float3(light_r, light_g, light_b);
  hg_sbt.data.light_strength = light_strength;
  hg_sbt.data.normals = nullptr;
  hg_sbt.data.indices = nullptr;
  hg_sbt.data.num_normals = 0;
  hg_sbt.data.num_indices = 0;
  hg_sbt.data.is_ground = 1;
  OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_ground_pg, &hg_sbt));
  size_t byte_offset = static_cast<size_t>(2 * sbt_index + RAY_TYPE_RADIANCE) * sizeof(HitGroupSbtRecord);
  CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_hitgroup_records + byte_offset), &hg_sbt,
                        sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));
  write_shadow_hitgroup_record(state, sbt_index, hg_sbt.data);
}

// --- Laplacian surface smoothing ---

static void laplacian_smooth(std::vector<float3>& positions, const std::vector<uint3>& indices,
                             std::vector<float3>& normals, int iterations, float lambda = 0.5f) {
  size_t num_verts = positions.size();
  if (num_verts == 0 || iterations <= 0) return;

  // Build adjacency list
  std::vector<std::vector<unsigned int>> adj(num_verts);
  for (const auto& tri : indices) {
    auto add_edge = [&](unsigned int a, unsigned int b) {
      // Avoid duplicates by checking (small lists, O(n) is fine)
      bool found = false;
      for (unsigned int n : adj[a]) {
        if (n == b) {
          found = true;
          break;
        }
      }
      if (!found) adj[a].push_back(b);
    };
    add_edge(tri.x, tri.y);
    add_edge(tri.x, tri.z);
    add_edge(tri.y, tri.x);
    add_edge(tri.y, tri.z);
    add_edge(tri.z, tri.x);
    add_edge(tri.z, tri.y);
  }

  // Iterative smoothing
  std::vector<float3> new_pos(num_verts);
  for (int iter = 0; iter < iterations; iter++) {
    for (size_t i = 0; i < num_verts; i++) {
      if (adj[i].empty()) {
        new_pos[i] = positions[i];
        continue;
      }
      float3 avg = make_float3(0.0f, 0.0f, 0.0f);
      for (unsigned int n : adj[i]) {
        avg.x += positions[n].x;
        avg.y += positions[n].y;
        avg.z += positions[n].z;
      }
      float inv_count = 1.0f / static_cast<float>(adj[i].size());
      avg.x *= inv_count;
      avg.y *= inv_count;
      avg.z *= inv_count;
      new_pos[i].x = (1.0f - lambda) * positions[i].x + lambda * avg.x;
      new_pos[i].y = (1.0f - lambda) * positions[i].y + lambda * avg.y;
      new_pos[i].z = (1.0f - lambda) * positions[i].z + lambda * avg.z;
    }
    positions.swap(new_pos);
  }

  // Recompute normals from smoothed positions
  for (size_t i = 0; i < num_verts; i++) {
    normals[i] = make_float3(0.0f, 0.0f, 0.0f);
  }
  for (const auto& tri : indices) {
    float3 p0 = positions[tri.x];
    float3 p1 = positions[tri.y];
    float3 p2 = positions[tri.z];
    // Cross product (p1-p0) x (p2-p0)
    float3 e1 = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    float3 e2 = make_float3(p2.x - p0.x, p2.y - p0.y, p2.z - p0.z);
    float3 fn = make_float3(e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x);
    normals[tri.x].x += fn.x;
    normals[tri.x].y += fn.y;
    normals[tri.x].z += fn.z;
    normals[tri.y].x += fn.x;
    normals[tri.y].y += fn.y;
    normals[tri.y].z += fn.z;
    normals[tri.z].x += fn.x;
    normals[tri.z].y += fn.y;
    normals[tri.z].z += fn.z;
  }
  for (size_t i = 0; i < num_verts; i++) {
    float len = std::sqrt(normals[i].x * normals[i].x + normals[i].y * normals[i].y + normals[i].z * normals[i].z);
    if (len > 1e-12f) {
      normals[i].x /= len;
      normals[i].y /= len;
      normals[i].z /= len;
    }
  }
}

static float3 rotate_euler_xyz(const float3& v, float sx, float cx, float sy, float cy, float sz, float cz) {
  float3 r = v;

  // Rotate around X axis.
  float y = cx * r.y - sx * r.z;
  float z = sx * r.y + cx * r.z;
  r.y = y;
  r.z = z;

  // Rotate around Y axis.
  float x = cy * r.x + sy * r.z;
  z = -sy * r.x + cy * r.z;
  r.x = x;
  r.z = z;

  // Rotate around Z axis.
  x = cz * r.x - sz * r.y;
  y = sz * r.x + cz * r.y;
  r.x = x;
  r.y = y;

  return r;
}

static void apply_geometry_rotation(std::vector<float3>& positions, std::vector<float3>& normals, float rotate_x_deg,
                                    float rotate_y_deg, float rotate_z_deg) {
  const float deg_to_rad = 3.14159265f / 180.0f;
  float rx = rotate_x_deg * deg_to_rad;
  float ry = rotate_y_deg * deg_to_rad;
  float rz = rotate_z_deg * deg_to_rad;

  if (std::fabs(rx) <= 1e-8f && std::fabs(ry) <= 1e-8f && std::fabs(rz) <= 1e-8f) {
    return;
  }

  float sx = std::sin(rx), cx = std::cos(rx);
  float sy = std::sin(ry), cy = std::cos(ry);
  float sz = std::sin(rz), cz = std::cos(rz);

  for (auto& p : positions) {
    p = rotate_euler_xyz(p, sx, cx, sy, cy, sz, cz);
  }
  for (auto& n : normals) {
    n = rotate_euler_xyz(n, sx, cx, sy, cy, sz, cz);
    float len = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
    if (len > 1e-12f) {
      n.x /= len;
      n.y /= len;
      n.z /= len;
    }
  }
}

static void compute_bbox(const std::vector<float3>& positions, BBox& bbox) {
  bbox.lo = make_float3(1e30f, 1e30f, 1e30f);
  bbox.hi = make_float3(-1e30f, -1e30f, -1e30f);
  for (const auto& p : positions) {
    bbox.lo.x = std::min(bbox.lo.x, p.x);
    bbox.lo.y = std::min(bbox.lo.y, p.y);
    bbox.lo.z = std::min(bbox.lo.z, p.z);
    bbox.hi.x = std::max(bbox.hi.x, p.x);
    bbox.hi.y = std::max(bbox.hi.y, p.y);
    bbox.hi.z = std::max(bbox.hi.z, p.z);
  }
}

static int find_field_index(const std::vector<std::string>& field_names, const std::string& field_name) {
  for (int i = 0; i < static_cast<int>(field_names.size()); i++) {
    if (field_names[i] == field_name) {
      return i;
    }
  }
  return -1;
}

static bool load_dataset_field_lists(const std::string& vtk_file, std::vector<std::string>& all_field_names,
                                     std::vector<std::string>& scalar_cell_field_names) {
  all_field_names.clear();
  scalar_cell_field_names.clear();
  if (vtk_file.empty()) {
    return false;
  }

  viskores::io::VTKDataSetReader reader(vtk_file);
  viskores::cont::DataSet ds = reader.ReadDataSet();

  for (viskores::IdComponent i = 0; i < ds.GetNumberOfFields(); i++) {
    const auto& field = ds.GetField(i);
    if (field.IsPointField() || field.IsCellField()) {
      all_field_names.push_back(field.GetName());
    }
    if (!field.IsCellField()) {
      continue;
    }
    if (field.GetData().GetNumberOfComponentsFlat() != 1) {
      continue;
    }
    scalar_cell_field_names.push_back(field.GetName());
  }
  return true;
}

static viskores::cont::ArrayHandle<int> cast_scalar_field_to_int32(const viskores::cont::Field& field,
                                                                    viskores::Id expected_num_values,
                                                                    const char* field_context) {
  if (field.GetData().GetNumberOfComponentsFlat() != 1) {
    throw std::runtime_error(std::string(field_context) + " must be scalar (1 component).");
  }

  viskores::cont::UnknownArrayHandle scalar_data = field.GetDataAsDefaultFloat();
  if (scalar_data.GetNumberOfComponentsFlat() != 1 || scalar_data.GetNumberOfValues() != expected_num_values) {
    throw std::runtime_error(std::string(field_context) + " size does not match point count.");
  }

  auto scalar_array = scalar_data.AsArrayHandle<viskores::cont::ArrayHandle<float>>();
  auto scalar_portal = scalar_array.ReadPortal();

  viskores::cont::ArrayHandle<int> out_int32;
  out_int32.Allocate(expected_num_values);
  auto out_portal = out_int32.WritePortal();
  for (viskores::Id i = 0; i < expected_num_values; i++) {
    out_portal.Set(i, static_cast<int>(std::llround(scalar_portal.Get(i))));
  }
  return out_int32;
}

static void sync_data_layer_caches(std::vector<DataLayer>& layers, std::vector<MeshCache>& layer_mesh_caches) {
  if (layer_mesh_caches.size() < layers.size()) {
    layer_mesh_caches.resize(layers.size());
  } else if (layer_mesh_caches.size() > layers.size()) {
    layer_mesh_caches.resize(layers.size());
  }
}

static void add_data_layer(std::vector<DataLayer>& layers, std::vector<MeshCache>& layer_mesh_caches,
                           const std::string& field_name) {
  DataLayer layer;
  layer.name = field_name;
  layers.push_back(layer);
  layer_mesh_caches.emplace_back();
}

static BBox merge_bbox(const BBox& a, const BBox& b) {
  BBox out;
  out.lo = make_float3(std::min(a.lo.x, b.lo.x), std::min(a.lo.y, b.lo.y), std::min(a.lo.z, b.lo.z));
  out.hi = make_float3(std::max(a.hi.x, b.hi.x), std::max(a.hi.y, b.hi.y), std::max(a.hi.z, b.hi.z));
  return out;
}

static void build_render_mesh_from_cache(const MeshCache& mesh_cache, int smooth_iters, float smooth_lambda,
                                         float rotate_x_deg, float rotate_y_deg, float rotate_z_deg,
                                         std::vector<float3>& out_positions, std::vector<float3>& out_normals,
                                         BBox& out_bbox) {
  if (!mesh_cache.valid || mesh_cache.base_positions.empty() || mesh_cache.indices.empty()) {
    throw std::runtime_error("Cached mesh is not available.");
  }

  out_positions = mesh_cache.base_positions;
  out_normals = mesh_cache.base_normals;
  if (out_normals.size() != out_positions.size()) {
    out_normals.assign(out_positions.size(), make_float3(0.0f, 1.0f, 0.0f));
  }

  if (smooth_iters > 0) {
    laplacian_smooth(out_positions, mesh_cache.indices, out_normals, smooth_iters, smooth_lambda);
  }
  apply_geometry_rotation(out_positions, out_normals, rotate_x_deg, rotate_y_deg, rotate_z_deg);
  compute_bbox(out_positions, out_bbox);
}

// --- Rebuild GAS (geometry acceleration structure) ---

static void rebuild_gas(OptixState& state, const BBox& bbox, bool include_mask, int fluid_mesh_count, bool ground_enabled,
                        float ground_y_offset) {
  if (!include_mask && fluid_mesh_count <= 0) {
    throw std::runtime_error("Scene rebuild requires at least one surface geometry.");
  }

  // Prepare ground plane geometry
  float3 ground_verts[4];
  uint3 ground_tris[2];
  if (ground_enabled) {
    float3 extent = make_float3(bbox.hi.x - bbox.lo.x, bbox.hi.y - bbox.lo.y, bbox.hi.z - bbox.lo.z);
    float diag = std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
    float half_ext = 5.0f * diag;
    float y = bbox.lo.y + ground_y_offset;
    float cx = (bbox.lo.x + bbox.hi.x) * 0.5f;
    float cz = (bbox.lo.z + bbox.hi.z) * 0.5f;
    ground_verts[0] = make_float3(cx - half_ext, y, cz - half_ext);
    ground_verts[1] = make_float3(cx + half_ext, y, cz - half_ext);
    ground_verts[2] = make_float3(cx + half_ext, y, cz + half_ext);
    ground_verts[3] = make_float3(cx - half_ext, y, cz + half_ext);
    ground_tris[0] = make_uint3(0, 1, 2);
    ground_tris[1] = make_uint3(0, 2, 3);

    if (!state.d_ground_vertices) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_ground_vertices), 4 * sizeof(float3)));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_ground_vertices), ground_verts, 4 * sizeof(float3),
                          cudaMemcpyHostToDevice));
    if (!state.d_ground_indices) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_ground_indices), 2 * sizeof(uint3)));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(state.d_ground_indices), ground_tris, 2 * sizeof(uint3),
                          cudaMemcpyHostToDevice));
  }

  // Build GAS
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  int num_inputs_expected = (include_mask ? 1 : 0) + std::max(0, fluid_mesh_count) + (ground_enabled ? 1 : 0);
  std::vector<uint32_t> flags(static_cast<size_t>(num_inputs_expected), OPTIX_GEOMETRY_FLAG_NONE);
  std::vector<CUdeviceptr> vertex_buffers(static_cast<size_t>(num_inputs_expected), 0);
  std::vector<OptixBuildInput> build_inputs(static_cast<size_t>(num_inputs_expected));
  int num_inputs = 0;

  state.mask_sbt_index = -1;
  state.fluid_sbt_indices.assign(static_cast<size_t>(std::max(0, fluid_mesh_count)), -1);
  state.ground_sbt_index = -1;

  if (include_mask) {
    state.mask_sbt_index = num_inputs;
    vertex_buffers[num_inputs] = state.mask_mesh.d_vertices;
    build_inputs[num_inputs].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_inputs[num_inputs].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_inputs[num_inputs].triangleArray.numVertices = state.mask_mesh.num_vertices;
    build_inputs[num_inputs].triangleArray.vertexBuffers = &vertex_buffers[num_inputs];
    build_inputs[num_inputs].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_inputs[num_inputs].triangleArray.numIndexTriplets = state.mask_mesh.num_triangles;
    build_inputs[num_inputs].triangleArray.indexBuffer = state.mask_mesh.d_indices_buf;
    build_inputs[num_inputs].triangleArray.flags = &flags[static_cast<size_t>(num_inputs)];
    build_inputs[num_inputs].triangleArray.numSbtRecords = 1;
    num_inputs++;
  }

  for (int i = 0; i < fluid_mesh_count; i++) {
    if (i >= static_cast<int>(state.fluid_meshes.size()) || !state.fluid_meshes[static_cast<size_t>(i)].IsValid()) {
      continue;
    }
    const GpuMeshBuffers& fluid_mesh = state.fluid_meshes[static_cast<size_t>(i)];
    state.fluid_sbt_indices[static_cast<size_t>(i)] = num_inputs;
    vertex_buffers[num_inputs] = fluid_mesh.d_vertices;
    build_inputs[num_inputs].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_inputs[num_inputs].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_inputs[num_inputs].triangleArray.numVertices = fluid_mesh.num_vertices;
    build_inputs[num_inputs].triangleArray.vertexBuffers = &vertex_buffers[num_inputs];
    build_inputs[num_inputs].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_inputs[num_inputs].triangleArray.numIndexTriplets = fluid_mesh.num_triangles;
    build_inputs[num_inputs].triangleArray.indexBuffer = fluid_mesh.d_indices_buf;
    build_inputs[num_inputs].triangleArray.flags = &flags[static_cast<size_t>(num_inputs)];
    build_inputs[num_inputs].triangleArray.numSbtRecords = 1;
    num_inputs++;
  }

  if (ground_enabled) {
    state.ground_sbt_index = num_inputs;
    vertex_buffers[num_inputs] = state.d_ground_vertices;
    build_inputs[num_inputs].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_inputs[num_inputs].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_inputs[num_inputs].triangleArray.numVertices = 4;
    build_inputs[num_inputs].triangleArray.vertexBuffers = &vertex_buffers[num_inputs];
    build_inputs[num_inputs].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_inputs[num_inputs].triangleArray.numIndexTriplets = 2;
    build_inputs[num_inputs].triangleArray.indexBuffer = state.d_ground_indices;
    build_inputs[num_inputs].triangleArray.flags = &flags[static_cast<size_t>(num_inputs)];
    build_inputs[num_inputs].triangleArray.numSbtRecords = 1;
    num_inputs++;
  }

  if (num_inputs <= 0) {
    throw std::runtime_error("No valid geometry inputs available for GAS rebuild.");
  }
  ensure_hitgroup_record_capacity(state, num_inputs);

  OptixAccelBufferSizes gas_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, build_inputs.data(), num_inputs, &gas_sizes));

  if (state.gas_temp_capacity < gas_sizes.tempSizeInBytes) {
    if (state.d_gas_temp) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_temp)));
      state.d_gas_temp = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_temp), gas_sizes.tempSizeInBytes));
    state.gas_temp_capacity = gas_sizes.tempSizeInBytes;
  }
  if (state.gas_output_capacity < gas_sizes.outputSizeInBytes) {
    if (state.d_gas_output) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output)));
      state.d_gas_output = 0;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output), gas_sizes.outputSizeInBytes));
    state.gas_output_capacity = gas_sizes.outputSizeInBytes;
  }

  OPTIX_CHECK(optixAccelBuild(state.context, 0, &accel_options, build_inputs.data(), num_inputs, state.d_gas_temp,
                              state.gas_temp_capacity, state.d_gas_output, state.gas_output_capacity, &state.gas_handle,
                              nullptr, 0));

  // Update SBT counts: 2 ray types per geometry SBT index.
  state.hitgroup_record_count = 2 * num_inputs;
  state.sbt.hitgroupRecordCount = state.hitgroup_record_count;
}

static void rebuild_scene_from_caches(const MeshCache* mask_cache, const std::vector<int>& fluid_layer_indices,
                                      const std::vector<DataLayer>& data_layers,
                                      const std::vector<MeshCache>& fluid_layer_caches, OptixState& state,
                                      const float3& light_dir, float light_r, float light_g, float light_b,
                                      float light_strength, bool ground_enabled, const ImVec4& ground_color,
                                      float ground_metallic, float ground_roughness, float ground_opacity,
                                      float ground_y_offset, float rotate_x_deg, float rotate_y_deg, float rotate_z_deg,
                                      int mask_smooth_iters, float mask_smooth_lambda, const ImVec4& mask_color,
                                      float mask_metallic, float mask_roughness, float mask_opacity, float mask_ior,
                                      BBox& bbox) {
  bool include_mask = (mask_cache != nullptr && mask_cache->valid);
  int fluid_count = static_cast<int>(fluid_layer_indices.size());
  if (!include_mask && fluid_count <= 0) {
    throw std::runtime_error("No renderable mask/fluid mesh is available.");
  }

  BBox scene_bbox;
  bool has_bbox = false;

  if (include_mask) {
    std::vector<float3> positions;
    std::vector<float3> normals;
    BBox mask_bbox;
    build_render_mesh_from_cache(*mask_cache, mask_smooth_iters, mask_smooth_lambda, rotate_x_deg, rotate_y_deg,
                                 rotate_z_deg, positions, normals, mask_bbox);
    upload_mesh_buffers_to_gpu(positions, normals, mask_cache->indices, state.mask_mesh);
    scene_bbox = mask_bbox;
    has_bbox = true;
  } else {
    free_gpu_mesh_buffers(state.mask_mesh);
  }

  for (auto& mesh : state.fluid_meshes) {
    free_gpu_mesh_buffers(mesh);
  }
  state.fluid_meshes.clear();
  state.fluid_layer_indices.clear();
  state.fluid_meshes.reserve(static_cast<size_t>(fluid_count));
  state.fluid_layer_indices.reserve(static_cast<size_t>(fluid_count));

  for (int layer_idx : fluid_layer_indices) {
    if (layer_idx < 0 || layer_idx >= static_cast<int>(data_layers.size()) ||
        layer_idx >= static_cast<int>(fluid_layer_caches.size())) {
      continue;
    }
    const DataLayer& layer = data_layers[static_cast<size_t>(layer_idx)];
    const MeshCache& cache = fluid_layer_caches[static_cast<size_t>(layer_idx)];
    if (!cache.valid) {
      continue;
    }
    std::vector<float3> positions;
    std::vector<float3> normals;
    BBox fluid_bbox;
    build_render_mesh_from_cache(cache, layer.smooth_iterations, layer.smooth_strength, rotate_x_deg, rotate_y_deg,
                                 rotate_z_deg, positions, normals, fluid_bbox);
    state.fluid_meshes.emplace_back();
    upload_mesh_buffers_to_gpu(positions, normals, cache.indices, state.fluid_meshes.back());
    state.fluid_layer_indices.push_back(layer_idx);
    scene_bbox = has_bbox ? merge_bbox(scene_bbox, fluid_bbox) : fluid_bbox;
    has_bbox = true;
  }

  fluid_count = static_cast<int>(state.fluid_meshes.size());
  if (!include_mask && fluid_count <= 0) {
    throw std::runtime_error("No renderable mask/fluid mesh is available.");
  }

  bbox = scene_bbox;
  rebuild_gas(state, bbox, include_mask, fluid_count, ground_enabled, ground_y_offset);

  if (state.mask_sbt_index >= 0 && include_mask) {
    update_surface_hitgroup_sbt(state, state.mask_sbt_index, state.hitgroup_mesh_pg, state.mask_mesh, mask_color.x,
                                mask_color.y, mask_color.z, mask_metallic, mask_roughness, mask_opacity, mask_ior, true,
                                light_dir, light_r, light_g, light_b, light_strength);
  }
  for (size_t i = 0; i < state.fluid_meshes.size(); i++) {
    if (i >= state.fluid_sbt_indices.size() || i >= state.fluid_layer_indices.size()) {
      continue;
    }
    int sbt_index = state.fluid_sbt_indices[i];
    int layer_idx = state.fluid_layer_indices[i];
    if (sbt_index < 0 || layer_idx < 0 || layer_idx >= static_cast<int>(data_layers.size())) {
      continue;
    }
    const DataLayer& layer = data_layers[static_cast<size_t>(layer_idx)];
    update_surface_hitgroup_sbt(state, sbt_index, state.hitgroup_fluid_pg, state.fluid_meshes[i], layer.color.x,
                                layer.color.y, layer.color.z, layer.metallic, layer.roughness, layer.opacity,
                                layer.glass_ior, true, light_dir, light_r, light_g, light_b, light_strength);
  }
  if (state.ground_sbt_index >= 0 && ground_enabled) {
    update_ground_hitgroup_sbt(state, state.ground_sbt_index, ground_color, ground_metallic, ground_roughness,
                               ground_opacity, light_dir, light_r, light_g, light_b, light_strength);
  }
}

static void update_scene_material_sbt(OptixState& state, const float3& light_dir, float light_r, float light_g,
                                      float light_b, float light_strength, bool ground_enabled,
                                      const ImVec4& ground_color, float ground_metallic, float ground_roughness,
                                      float ground_opacity, const ImVec4& mask_color, float mask_metallic,
                                      float mask_roughness, float mask_opacity, float mask_ior,
                                      const std::vector<DataLayer>& data_layers) {
  if (state.mask_sbt_index >= 0 && state.mask_mesh.IsValid()) {
    update_surface_hitgroup_sbt(state, state.mask_sbt_index, state.hitgroup_mesh_pg, state.mask_mesh, mask_color.x,
                                mask_color.y, mask_color.z, mask_metallic, mask_roughness, mask_opacity, mask_ior, true,
                                light_dir, light_r, light_g, light_b, light_strength);
  }
  for (size_t i = 0; i < state.fluid_meshes.size(); i++) {
    if (i >= state.fluid_sbt_indices.size() || i >= state.fluid_layer_indices.size()) {
      continue;
    }
    int sbt_index = state.fluid_sbt_indices[i];
    int layer_idx = state.fluid_layer_indices[i];
    if (sbt_index < 0 || layer_idx < 0 || layer_idx >= static_cast<int>(data_layers.size()) ||
        !state.fluid_meshes[i].IsValid()) {
      continue;
    }
    const DataLayer& layer = data_layers[static_cast<size_t>(layer_idx)];
    update_surface_hitgroup_sbt(state, sbt_index, state.hitgroup_fluid_pg, state.fluid_meshes[i], layer.color.x,
                                layer.color.y, layer.color.z, layer.metallic, layer.roughness, layer.opacity,
                                layer.glass_ior, true, light_dir, light_r, light_g, light_b, light_strength);
  }
  if (ground_enabled && state.ground_sbt_index >= 0) {
    update_ground_hitgroup_sbt(state, state.ground_sbt_index, ground_color, ground_metallic, ground_roughness,
                               ground_opacity, light_dir, light_r, light_g, light_b, light_strength);
  }
}

static void extract_mesh(const std::string& mask_filepath, const std::string& field_name, int solid_val,
                         MeshCache& mesh_cache) {
  viskores::io::VTKDataSetReader reader(mask_filepath);
  viskores::cont::DataSet ds = reader.ReadDataSet();

  auto selectedField = ds.GetField(field_name);
  if (selectedField.GetData().GetNumberOfComponentsFlat() != 1) {
    throw std::runtime_error("Selected mask field must be scalar (1 component).");
  }

  // Contour requires point-associated scalar data.
  viskores::cont::DataSet pointDS = ds;
  if (selectedField.IsCellField()) {
    viskores::filter::field_conversion::PointAverage cellToPoint;
    cellToPoint.SetActiveField(field_name);
    cellToPoint.SetOutputFieldName(field_name);
    pointDS = cellToPoint.Execute(ds);
  } else if (!selectedField.IsPointField()) {
    throw std::runtime_error("Selected mask field must be a point or cell field.");
  }

  auto& pointField = pointDS.GetPointField(field_name);
  viskores::cont::ArrayHandle<int> mask_int32 =
      cast_scalar_field_to_int32(pointField, pointDS.GetNumberOfPoints(),
                                 "Contour input mask field");
  pointField.SetData(mask_int32);

  // Marching Cubes: extract smooth isosurface at the boundary of solid_val
  viskores::filter::contour::Contour contour;
  contour.SetActiveField(field_name);
  contour.SetIsoValue(static_cast<double>(solid_val) - 0.5);
  contour.SetGenerateNormals(true);
  contour.SetNormalArrayName("Normals");
  viskores::cont::DataSet result = contour.Execute(pointDS);

  // Extract positions
  auto coords = result.GetCoordinateSystem();
  auto coordData = coords.GetData();
  viskores::Id numPoints = result.GetNumberOfPoints();
  if (numPoints <= 0) {
    throw std::runtime_error("Contour produced no points.");
  }

  std::vector<float3> positions(numPoints);
  auto coordArray = coordData.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
  {
    auto portal = coordArray.ReadPortal();
    for (viskores::Id i = 0; i < numPoints; i++) {
      auto p = portal.Get(i);
      positions[i] = make_float3(p[0], p[1], p[2]);
    }
  }

  // Extract normals
  std::vector<float3> normal_data;
  if (result.HasPointField("Normals")) {
    auto normalField = result.GetPointField("Normals");
    auto normalArray = normalField.GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
    auto portal = normalArray.ReadPortal();
    normal_data.resize(numPoints);
    for (viskores::Id i = 0; i < numPoints; i++) {
      auto n = portal.Get(i);
      normal_data[i] = make_float3(n[0], n[1], n[2]);
    }
  } else {
    // Fallback: default normals
    normal_data.resize(numPoints, make_float3(0.0f, 1.0f, 0.0f));
  }

  // Extract connectivity
  viskores::Id numCells = result.GetNumberOfCells();
  std::vector<uint3> indices;
  indices.reserve(static_cast<size_t>(numCells));
  auto cellSet = result.GetCellSet();
  const viskores::Id max_u32 = static_cast<viskores::Id>(std::numeric_limits<unsigned int>::max());

  for (viskores::Id c = 0; c < numCells; c++) {
    viskores::IdComponent npts = cellSet.GetNumberOfPointsInCell(c);
    if (npts < 3) continue;

    std::vector<viskores::Id> ids(static_cast<size_t>(npts));
    cellSet.GetCellPointIds(c, ids.data());

    auto append_tri_if_valid = [&](viskores::Id a, viskores::Id b, viskores::Id d) {
      if (a < 0 || b < 0 || d < 0) return;
      if (a >= numPoints || b >= numPoints || d >= numPoints) return;
      if (a > max_u32 || b > max_u32 || d > max_u32) return;
      indices.push_back(
          make_uint3(static_cast<unsigned int>(a), static_cast<unsigned int>(b), static_cast<unsigned int>(d)));
    };

    if (cellSet.GetCellShape(c) == viskores::CELL_SHAPE_TRIANGLE && npts == 3) {
      append_tri_if_valid(ids[0], ids[1], ids[2]);
      continue;
    }

    // Fallback: triangulate polygon-like cells with a simple fan.
    for (viskores::IdComponent i = 1; i + 1 < npts; i++) {
      append_tri_if_valid(ids[0], ids[i], ids[i + 1]);
    }
  }

  if (indices.empty()) {
    throw std::runtime_error("Contour produced no valid triangles.");
  }

  mesh_cache.source_file = mask_filepath;
  mesh_cache.source_field = field_name;
  mesh_cache.source_solid_flag = solid_val;
  mesh_cache.base_positions = std::move(positions);
  mesh_cache.base_normals = std::move(normal_data);
  mesh_cache.indices = std::move(indices);
  mesh_cache.valid = true;
}

static void extract_fluid_mesh(const std::string& density_filepath, const std::string& density_field_name,
                               float density_threshold_min, float density_threshold_max,
                               const std::string& mask_filepath,
                               const std::string& mask_field_name, int fluid_flag, MeshCache& mesh_cache) {
  viskores::io::VTKDataSetReader density_reader(density_filepath);
  viskores::cont::DataSet density_ds = density_reader.ReadDataSet();
  viskores::cont::DataSet mask_ds = density_ds;
  if (mask_filepath != density_filepath) {
    viskores::io::VTKDataSetReader mask_reader(mask_filepath);
    mask_ds = mask_reader.ReadDataSet();
  }

  viskores::cont::Field density_field;
  try {
    density_field = density_ds.GetField(density_field_name, viskores::cont::Field::Association::Cells);
  } catch (...) {
    throw std::runtime_error("Selected density field must be a scalar cell field.");
  }
  if (density_field.GetData().GetNumberOfComponentsFlat() != 1) {
    throw std::runtime_error("Selected density field must be scalar (1 component).");
  }
  auto mask_field = mask_ds.GetField(mask_field_name);
  if (mask_field.GetData().GetNumberOfComponentsFlat() != 1) {
    throw std::runtime_error("Selected mask field must be scalar (1 component).");
  }

  viskores::cont::DataSet density_point_ds = density_ds;
  viskores::filter::field_conversion::PointAverage cell_to_point;
  cell_to_point.SetActiveField(density_field_name);
  cell_to_point.SetOutputFieldName(density_field_name);
  density_point_ds = cell_to_point.Execute(density_ds);

  viskores::cont::DataSet mask_point_ds = mask_ds;
  if (mask_field.IsCellField()) {
    viskores::filter::field_conversion::PointAverage cell_to_point;
    cell_to_point.SetActiveField(mask_field_name);
    cell_to_point.SetOutputFieldName(mask_field_name);
    mask_point_ds = cell_to_point.Execute(mask_ds);
  } else if (!mask_field.IsPointField()) {
    throw std::runtime_error("Selected mask field must be a point or cell field.");
  }

  auto& density_point_field = density_point_ds.GetPointField(density_field_name);
  viskores::cont::UnknownArrayHandle density_scalar = density_point_field.GetDataAsDefaultFloat();
  if (density_scalar.GetNumberOfComponentsFlat() != 1 ||
      density_scalar.GetNumberOfValues() != density_point_ds.GetNumberOfPoints()) {
    throw std::runtime_error("Density field size does not match point count.");
  }
  density_point_field.SetData(density_scalar);

  auto& mask_point_field = mask_point_ds.GetPointField(mask_field_name);
  viskores::cont::ArrayHandle<int> mask_int32 =
      cast_scalar_field_to_int32(mask_point_field, mask_point_ds.GetNumberOfPoints(),
                                 "Mask field");
  mask_point_field.SetData(mask_int32);

  if (density_point_ds.GetNumberOfPoints() != mask_point_ds.GetNumberOfPoints()) {
    throw std::runtime_error("Density and mask datasets must share the same point count.");
  }

  viskores::Id num_points = density_point_ds.GetNumberOfPoints();
  auto density_array = density_point_field.GetData().AsArrayHandle<viskores::cont::ArrayHandle<float>>();
  auto mask_array = mask_point_field.GetData().AsArrayHandle<viskores::cont::ArrayHandle<int>>();
  auto density_portal = density_array.ReadPortal();
  auto mask_portal = mask_array.ReadPortal();

  float threshold_min = std::min(density_threshold_min, density_threshold_max);
  float threshold_max = std::max(density_threshold_min, density_threshold_max);

  bool found_fluid_flag = false;
  bool found_in_range = false;
  for (viskores::Id i = 0; i < num_points; i++) {
    int mask_value = mask_portal.Get(i);
    if (mask_value == fluid_flag) {
      found_fluid_flag = true;
      float d = density_portal.Get(i);
      if (d >= threshold_min && d <= threshold_max) {
        found_in_range = true;
      }
    }
  }
  if (!found_fluid_flag) {
    throw std::runtime_error("No points match the selected fluid flag in the mask field.");
  }
  if (!found_in_range) {
    throw std::runtime_error("No fluid points fall inside the selected threshold range.");
  }
  std::vector<float> filtered_density(static_cast<size_t>(num_points), 0.0f);

  for (viskores::Id i = 0; i < num_points; i++) {
    int mask_value = mask_portal.Get(i);
    if (mask_value == fluid_flag) {
      float d = density_portal.Get(i);
      if (d >= threshold_min && d <= threshold_max) {
        filtered_density[static_cast<size_t>(i)] = 1.0f;
      }
    }
  }

  viskores::cont::ArrayHandle<float> filtered_density_array;
  filtered_density_array.Allocate(num_points);
  auto filtered_portal = filtered_density_array.WritePortal();
  for (viskores::Id i = 0; i < num_points; i++) {
    filtered_portal.Set(i, filtered_density[static_cast<size_t>(i)]);
  }
  density_point_field.SetData(filtered_density_array);

  viskores::cont::DataSet contour_input = density_point_ds;
  auto density_cell_set = density_point_ds.GetCellSet();
  if (density_cell_set.CanConvert<viskores::cont::CellSetStructured<3>>()) {
    auto structured = density_cell_set.AsCellSet<viskores::cont::CellSetStructured<3>>();
    viskores::Id3 dims = structured.GetPointDimensions();
    viskores::Id nx = dims[0];
    viskores::Id ny = dims[1];
    viskores::Id nz = dims[2];
    if (nx > 1 && ny > 1 && nz > 1 && nx * ny * nz == num_points) {
      viskores::Bounds bounds = density_point_ds.GetCoordinateSystem().GetBounds();
      if (bounds.X.IsNonEmpty() && bounds.Y.IsNonEmpty() && bounds.Z.IsNonEmpty()) {
        float sx = static_cast<float>((bounds.X.Max - bounds.X.Min) / static_cast<double>(nx - 1));
        float sy = static_cast<float>((bounds.Y.Max - bounds.Y.Min) / static_cast<double>(ny - 1));
        float sz = static_cast<float>((bounds.Z.Max - bounds.Z.Min) / static_cast<double>(nz - 1));
        viskores::Id px = nx + 2;
        viskores::Id py = ny + 2;
        viskores::Id pz = nz + 2;
        std::vector<float> padded(static_cast<size_t>(px * py * pz), 0.0f);
        auto pad_index = [px, py](viskores::Id x, viskores::Id y, viskores::Id z) -> size_t {
          return static_cast<size_t>(x + px * (y + py * z));
        };
        auto src_index = [nx, ny](viskores::Id x, viskores::Id y, viskores::Id z) -> size_t {
          return static_cast<size_t>(x + nx * (y + ny * z));
        };

        for (viskores::Id z = 0; z < nz; z++) {
          for (viskores::Id y = 0; y < ny; y++) {
            for (viskores::Id x = 0; x < nx; x++) {
              padded[pad_index(x + 1, y + 1, z + 1)] = filtered_density[src_index(x, y, z)];
            }
          }
        }

        viskores::cont::ArrayHandle<float> padded_array;
        padded_array.Allocate(px * py * pz);
        auto padded_portal = padded_array.WritePortal();
        for (viskores::Id i = 0; i < px * py * pz; i++) {
          padded_portal.Set(i, padded[static_cast<size_t>(i)]);
        }

        viskores::Vec3f origin(static_cast<float>(bounds.X.Min) - sx, static_cast<float>(bounds.Y.Min) - sy,
                               static_cast<float>(bounds.Z.Min) - sz);
        viskores::Vec3f spacing(sx, sy, sz);
        contour_input = viskores::cont::DataSetBuilderUniform::Create(viskores::Id3(px, py, pz), origin, spacing, "coords");
        contour_input.AddField(viskores::cont::Field(density_field_name, viskores::cont::Field::Association::Points,
                                                     padded_array));
      }
    }
  }

  viskores::filter::contour::Contour contour;
  contour.SetActiveField(density_field_name);
  contour.SetIsoValue(0.5);
  contour.SetGenerateNormals(true);
  contour.SetNormalArrayName("Normals");
  viskores::cont::DataSet result = contour.Execute(contour_input);

  auto coords = result.GetCoordinateSystem();
  auto coord_data = coords.GetData();
  viskores::Id contour_points = result.GetNumberOfPoints();
  if (contour_points <= 0) {
    throw std::runtime_error("Fluid contour produced no points.");
  }

  std::vector<float3> positions(static_cast<size_t>(contour_points));
  auto coord_array = coord_data.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
  {
    auto portal = coord_array.ReadPortal();
    for (viskores::Id i = 0; i < contour_points; i++) {
      auto p = portal.Get(i);
      positions[static_cast<size_t>(i)] = make_float3(p[0], p[1], p[2]);
    }
  }

  std::vector<float3> normal_data;
  if (result.HasPointField("Normals")) {
    auto normal_field = result.GetPointField("Normals");
    auto normal_array = normal_field.GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
    auto portal = normal_array.ReadPortal();
    normal_data.resize(static_cast<size_t>(contour_points));
    for (viskores::Id i = 0; i < contour_points; i++) {
      auto n = portal.Get(i);
      normal_data[static_cast<size_t>(i)] = make_float3(n[0], n[1], n[2]);
    }
  } else {
    normal_data.resize(static_cast<size_t>(contour_points), make_float3(0.0f, 1.0f, 0.0f));
  }

  viskores::Id num_cells = result.GetNumberOfCells();
  std::vector<uint3> indices;
  indices.reserve(static_cast<size_t>(num_cells));
  auto contour_cell_set = result.GetCellSet();
  const viskores::Id max_u32 = static_cast<viskores::Id>(std::numeric_limits<unsigned int>::max());

  for (viskores::Id c = 0; c < num_cells; c++) {
    viskores::IdComponent npts = contour_cell_set.GetNumberOfPointsInCell(c);
    if (npts < 3) continue;

    std::vector<viskores::Id> ids(static_cast<size_t>(npts));
    contour_cell_set.GetCellPointIds(c, ids.data());

    auto append_tri_if_valid = [&](viskores::Id a, viskores::Id b, viskores::Id d) {
      if (a < 0 || b < 0 || d < 0) return;
      if (a >= contour_points || b >= contour_points || d >= contour_points) return;
      if (a > max_u32 || b > max_u32 || d > max_u32) return;
      indices.push_back(
          make_uint3(static_cast<unsigned int>(a), static_cast<unsigned int>(b), static_cast<unsigned int>(d)));
    };

    if (contour_cell_set.GetCellShape(c) == viskores::CELL_SHAPE_TRIANGLE && npts == 3) {
      append_tri_if_valid(ids[0], ids[1], ids[2]);
      continue;
    }

    for (viskores::IdComponent i = 1; i + 1 < npts; i++) {
      append_tri_if_valid(ids[0], ids[i], ids[i + 1]);
    }
  }

  if (indices.empty()) {
    throw std::runtime_error("Fluid contour produced no valid triangles.");
  }

  mesh_cache.source_file = density_filepath;
  mesh_cache.source_field = density_field_name;
  mesh_cache.source_solid_flag = fluid_flag;
  mesh_cache.base_positions = std::move(positions);
  mesh_cache.base_normals = std::move(normal_data);
  mesh_cache.indices = std::move(indices);
  mesh_cache.valid = true;
}

// --- Cleanup OptiX ---

static void cleanup_optix(OptixState& state) {
  if (state.d_image) cudaFree(reinterpret_cast<void*>(state.d_image));
  if (state.d_params) cudaFree(reinterpret_cast<void*>(state.d_params));
  if (state.d_gas_temp) cudaFree(reinterpret_cast<void*>(state.d_gas_temp));
  if (state.d_gas_output) cudaFree(reinterpret_cast<void*>(state.d_gas_output));
  free_gpu_mesh_buffers(state.mask_mesh);
  for (auto& mesh : state.fluid_meshes) {
    free_gpu_mesh_buffers(mesh);
  }
  state.fluid_meshes.clear();
  state.fluid_sbt_indices.clear();
  state.fluid_layer_indices.clear();
  if (state.d_ground_vertices) cudaFree(reinterpret_cast<void*>(state.d_ground_vertices));
  if (state.d_ground_indices) cudaFree(reinterpret_cast<void*>(state.d_ground_indices));
  if (state.d_raygen_record) cudaFree(reinterpret_cast<void*>(state.d_raygen_record));
  if (state.d_miss_records) cudaFree(reinterpret_cast<void*>(state.d_miss_records));
  if (state.d_hitgroup_records) cudaFree(reinterpret_cast<void*>(state.d_hitgroup_records));
  if (state.pipeline) optixPipelineDestroy(state.pipeline);
  if (state.hitgroup_shadow_pg) optixProgramGroupDestroy(state.hitgroup_shadow_pg);
  if (state.hitgroup_ground_pg) optixProgramGroupDestroy(state.hitgroup_ground_pg);
  if (state.hitgroup_fluid_pg) optixProgramGroupDestroy(state.hitgroup_fluid_pg);
  if (state.hitgroup_mesh_pg) optixProgramGroupDestroy(state.hitgroup_mesh_pg);
  if (state.miss_shadow_pg) optixProgramGroupDestroy(state.miss_shadow_pg);
  if (state.miss_radiance_pg) optixProgramGroupDestroy(state.miss_radiance_pg);
  if (state.raygen_pg) optixProgramGroupDestroy(state.raygen_pg);
  if (state.module) optixModuleDestroy(state.module);
  if (state.context) optixDeviceContextDestroy(state.context);
}

int main(int argc, char* argv[]) {
  viskores::cont::Initialize(argc, argv);

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    SDL_Log("SDL failed to initialize: %s", SDL_GetError());
    return 1;
  }

  SDL_Window* window = SDL_CreateWindow("CaustiX", width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!window) {
    SDL_Log("Failed to create window: %s", SDL_GetError());
    SDL_Quit();
    return 1;
  }

  SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
  if (!renderer) {
    SDL_Log("Failed to create renderer: %s", SDL_GetError());
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }
  SDL_SetRenderVSync(renderer, 1);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

  ImGui::StyleColorsDark();

  ImGui_ImplSDL3_InitForSDLRenderer(window, renderer);
  ImGui_ImplSDLRenderer3_Init(renderer);

  io.Fonts->AddFontFromFileTTF(font_regular, ui_font_size);
  ImFont* bold_font = io.Fonts->AddFontFromFileTTF(font_bold, ui_heading_font_size);

  LightingState lighting;
  MaskState mask;
  GroundState ground;
  DatasetState dataset;
  RenderMiscState misc;
  RayTracingState rt;
  CameraState camera;

  ImVec4& bg_color = lighting.bg_color;
  ImVec4& prev_bg_color = lighting.prev_bg_color;
  float& light_strength = lighting.strength;
  float& prev_light_strength = lighting.prev_strength;
  ImVec4& light_color = lighting.color;
  ImVec4& prev_light_color = lighting.prev_color;
  float3& light_dir = lighting.dir;
  bool& shadows_enabled = lighting.shadows_enabled;
  bool& prev_shadows_enabled = lighting.prev_shadows_enabled;

  std::string& vtk_dir = dataset.vtk_dir;
  std::vector<std::string>& vtk_files = dataset.vtk_files;
  int& vtk_index = dataset.vtk_index;
  std::vector<std::string>& dataset_cell_names = dataset.cell_names;
  std::vector<std::string>& dataset_scalar_cell_names = dataset.scalar_cell_names;
  std::vector<DataLayer>& data_layers = dataset.layers;
  std::vector<MeshCache>& data_layer_mesh_caches = dataset.layer_mesh_caches;
  int& dataset_loaded_field_vtk_index = dataset.loaded_field_vtk_index;
  bool& first_frame = dataset.first_frame;

  std::string& mask_file = mask.file;
  std::vector<std::string>& mask_field_names = mask.field_names;
  int& mask_field_index = mask.field_index;
  int& prev_mask_field_index = mask.prev_field_index;
  bool& show_mask = mask.show;
  bool& prev_show_mask = mask.prev_show;
  int& solid_flag = mask.solid_flag;
  int& prev_solid_flag = mask.prev_solid_flag;
  ImVec4& mask_color = mask.color;
  ImVec4& prev_mask_color = mask.prev_color;
  float& mask_metallic = mask.metallic;
  float& prev_mask_metallic = mask.prev_metallic;
  float& mask_roughness = mask.roughness;
  float& prev_mask_roughness = mask.prev_roughness;
  float& mask_opacity = mask.opacity;
  float& prev_mask_opacity = mask.prev_opacity;
  float& mask_glass_ior = mask.glass_ior;
  float& prev_mask_glass_ior = mask.prev_glass_ior;
  int& smooth_iterations = mask.smooth_iterations;
  int& prev_smooth_iterations = mask.prev_smooth_iterations;
  float& smooth_strength = mask.smooth_strength;
  float& prev_smooth_strength = mask.prev_smooth_strength;

  bool& ground_enabled = ground.enabled;
  bool& prev_ground_enabled = ground.prev_enabled;
  float& ground_y_offset = ground.y_offset;
  float& prev_ground_y_offset = ground.prev_y_offset;
  ImVec4& ground_color = ground.color;
  ImVec4& prev_ground_color = ground.prev_color;
  float& ground_metallic = ground.metallic;
  float& prev_ground_metallic = ground.prev_metallic;
  float& ground_roughness = ground.roughness;
  float& prev_ground_roughness = ground.prev_roughness;
  float& ground_opacity = ground.opacity;
  float& prev_ground_opacity = ground.prev_opacity;

  bool& show_outlines = misc.show_outlines;
  ImVec4& outline_color = misc.outline_color;
  float& outline_thickness = misc.outline_thickness;
  float& rotate_x_deg = misc.rotate_x_deg;
  float& prev_rotate_x_deg = misc.prev_rotate_x_deg;
  float& rotate_y_deg = misc.rotate_y_deg;
  float& prev_rotate_y_deg = misc.prev_rotate_y_deg;
  float& rotate_z_deg = misc.rotate_z_deg;
  float& prev_rotate_z_deg = misc.prev_rotate_z_deg;
  bool& show_mask_error = misc.show_mask_error;
  std::string& mask_error_msg = misc.mask_error_msg;

  int& rt_bounces = rt.bounces;
  int& prev_rt_bounces = rt.prev_bounces;
  int& rt_samples = rt.samples;
  int& prev_rt_samples = rt.prev_samples;

  float& cam_yaw = camera.yaw;
  float& cam_pitch = camera.pitch;
  float& cam_distance = camera.distance;
  float (&cam_target)[3] = camera.target;
  float& cam_fov = camera.fov;
  bool& viewport_needs_render = camera.viewport_needs_render;
  int& prev_vp_w = camera.prev_vp_w;
  int& prev_vp_h = camera.prev_vp_h;

  // Viewport texture
  SDL_Texture* viewport_tex = nullptr;
  int viewport_tex_w = 0, viewport_tex_h = 0;

  // OptiX state
  OptixState optix_state;
  bool mesh_loaded = false;

  // Find PTX file relative to executable
  std::string exe_path = argv[0];
  std::string exe_dir = std::filesystem::path(exe_path).parent_path().string();
  std::string ptx_path = exe_dir + "/shaders.ptx";

  try {
    init_optix(optix_state, ptx_path, bg_color);
  } catch (const std::exception& e) {
    SDL_Log("OptiX initialization failed: %s", e.what());
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  // Persistent mesh bounding box for outline drawing
  BBox mesh_bbox;
  MeshCache mask_mesh_cache;

  // Host pixel buffer for copying from GPU
  std::vector<uchar4> host_pixels;

  bool running = true;
  while (running) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL3_ProcessEvent(&event);
      if (event.type == SDL_EVENT_QUIT) running = false;
      if (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED && event.window.windowID == SDL_GetWindowID(window))
        running = false;
    }

    if (SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) {
      SDL_Delay(10);
      continue;
    }

    ImGui_ImplSDLRenderer3_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();

    ImGuiID dockspace_id =
        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode);

    if (first_frame) {
      first_frame = false;
      ImGui::DockBuilderRemoveNode(dockspace_id);
      ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
      ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->Size);

      ImGuiID dock_left;
      ImGuiID dock_main;
      ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.125f, &dock_left, &dock_main);
      ImGui::DockBuilderDockWindow("Config", dock_left);
      ImGui::DockBuilderDockWindow("Render", dock_left);
      ImGui::DockBuilderDockWindow("###ViewportPanel", dock_main);
      if (ImGuiDockNode* main_node = ImGui::DockBuilderGetNode(dock_main)) {
        main_node->LocalFlags |= ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoWindowMenuButton;
      }
      ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::Begin("Config");
    ImGui::PushFont(bold_font);
    ImGui::Text("Background Color");
    ImGui::PopFont();
    ImGui::ColorEdit3("##bg", (float*)&bg_color);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushFont(bold_font);
    ImGui::Text("Global Illumination");
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::Text("Strength");
    ImGui::SameLine();
    ImGui::InputFloat("##strength", &light_strength);
    ImGui::Text("Color");
    ImGui::ColorEdit3("##light_color", (float*)&light_color);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushFont(bold_font);
    ImGui::Text("Ray Tracing");
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::Text("Bounces");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##rt_bounces", &rt_bounces);
    rt_bounces = std::max(1, std::min(16, rt_bounces));
    ImGui::Text("Samples");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##rt_samples", &rt_samples);
    rt_samples = std::max(1, std::min(64, rt_samples));

    // Ground Plane section
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::PushFont(bold_font);
    ImGui::Text("Ground Plane");
    ImGui::PopFont();
    ImGui::Spacing();
    ImGui::Checkbox("Enable##ground", &ground_enabled);
    if (ground_enabled) {
      ImGui::Text("Y Offset");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##ground_y_offset", &ground_y_offset, 0.01f);
      ImGui::Text("Color");
      ImGui::ColorEdit3("##ground_color", (float*)&ground_color);
      ImGui::Text("Metallic");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##ground_metallic", &ground_metallic, 0.0f, 1.0f);
      ImGui::Text("Roughness");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##ground_roughness", &ground_roughness, 0.0f, 1.0f);
      ImGui::Text("Opacity");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##ground_opacity", &ground_opacity, 0.0f, 1.0f);
    }

    ImGui::End();

    ImGui::Begin("Render");
    ImGui::PushFont(bold_font);
    ImGui::Text("Dataset");
    ImGui::PopFont();
    ImGui::SameLine();
    if (ImGui::Button("Open")) {
      IGFD::FileDialogConfig config;
      config.path = getenv("HOME");
      ImGuiFileDialog::Instance()->OpenDialog("OpenFileDlg", "Open File", nullptr, config);
    }
    ImGui::SameLine();
    if (ImGui::Button("Play")) {
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
      vtk_dir.clear();
      vtk_files.clear();
      vtk_index = 0;
      dataset_cell_names.clear();
      dataset_scalar_cell_names.clear();
      dataset_loaded_field_vtk_index = -1;
      data_layers.clear();
      data_layer_mesh_caches.clear();
      mesh_loaded = false;
    }
    if (!vtk_files.empty()) {
      ImGui::Spacing();
      if (ImGui::Button("Begin")) vtk_index = 0;
      ImGui::SameLine();
      if (ImGui::Button("Prev") && vtk_index > 0) vtk_index--;
      ImGui::SameLine();
      if (ImGui::Button("Next") && vtk_index < (int)vtk_files.size() - 1) vtk_index++;
      ImGui::SameLine();
      if (ImGui::Button("End")) vtk_index = (int)vtk_files.size() - 1;
      if (dataset_loaded_field_vtk_index != vtk_index) {
        try {
          if (load_dataset_field_lists(vtk_files[vtk_index], dataset_cell_names, dataset_scalar_cell_names)) {
            for (auto& layer_cache : data_layer_mesh_caches) {
              layer_cache.Clear();
            }
            dataset_loaded_field_vtk_index = vtk_index;
          } else {
            dataset_loaded_field_vtk_index = -1;
            for (auto& layer_cache : data_layer_mesh_caches) {
              layer_cache.Clear();
            }
          }
        } catch (...) {
          dataset_cell_names.clear();
          dataset_scalar_cell_names.clear();
          dataset_loaded_field_vtk_index = -1;
          for (auto& layer_cache : data_layer_mesh_caches) {
            layer_cache.Clear();
          }
        }
      }
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushFont(bold_font);
      ImGui::Text("Render:Misc");
      ImGui::PopFont();
      ImGui::Checkbox("Enable Shadows", &shadows_enabled);
      ImGui::Checkbox("Show Outlines", &show_outlines);
      if (show_outlines) {
        ImGui::Text("Color");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::ColorEdit3("##outline_color", (float*)&outline_color);
        ImGui::Text("Thickness");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        ImGui::SliderFloat("##outline_thickness", &outline_thickness, 1.0f, 10.0f);
      }
      ImGui::Text("Rotate X");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##rotate_x", &rotate_x_deg, 0.5f);
      ImGui::Text("Rotate Y");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##rotate_y", &rotate_y_deg, 0.5f);
      ImGui::Text("Rotate Z");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::DragFloat("##rotate_z", &rotate_z_deg, 0.5f);
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushFont(bold_font);
      ImGui::Text("Render:Mask");
      ImGui::PopFont();
      ImGui::Spacing();
      ImGui::Text("File");
      ImGui::SameLine();
      if (ImGui::Button("Open##mask")) {
        IGFD::FileDialogConfig mask_config;
        mask_config.path = vtk_dir;
        ImGuiFileDialog::Instance()->OpenDialog("OpenMaskDlg", "Open Mask", ".vtk", mask_config);
      }
      ImGui::SameLine();
      if (ImGui::Button("Clear##mask")) {
        mask_file.clear();
        mask_field_names.clear();
        mask_field_index = 0;
        mask_mesh_cache.Clear();
        for (auto& layer_cache : data_layer_mesh_caches) {
          layer_cache.Clear();
        }
        mesh_loaded = false;
      }

      if (!mask_file.empty()) {
        ImGui::TextWrapped("%s", std::filesystem::path(mask_file).filename().string().c_str());
      }
      if (!mask_field_names.empty()) {
        ImGui::Spacing();
        ImGui::Text("Field");
        ImGui::SameLine();
        const char* preview = mask_field_names[mask_field_index].c_str();
        if (ImGui::BeginCombo("##mask_field", preview)) {
          for (int i = 0; i < (int)mask_field_names.size(); i++) {
            bool selected = (i == mask_field_index);
            if (ImGui::Selectable(mask_field_names[i].c_str(), selected)) mask_field_index = i;
            if (selected) ImGui::SetItemDefaultFocus();
          }
          ImGui::EndCombo();
        }
        ImGui::Spacing();
        ImGui::Text("Solid Flag");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        ImGui::InputInt("##solid_flag", &solid_flag);
        ImGui::Spacing();
        ImGui::Checkbox("Show##mask", &show_mask);

        if (show_mask) {
          ImGui::Spacing();
          ImGui::Text("Color");
          ImGui::ColorEdit3("##mask_color", (float*)&mask_color);
          ImGui::Text("Metallic");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##mask_metallic", &mask_metallic, 0.0f, 1.0f);
          ImGui::Text("Roughness");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##mask_roughness", &mask_roughness, 0.0f, 1.0f);
          ImGui::Text("Opacity");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##mask_opacity", &mask_opacity, 0.0f, 1.0f);
          ImGui::Text("Glass IOR");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##mask_ior", &mask_glass_ior, 1.0f, 2.5f);
          ImGui::Text("Smoothing");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderInt("##smooth_iters", &smooth_iterations, 0, 50);
          ImGui::Text("Smooth Strength");
          ImGui::SameLine();
          ImGui::SetNextItemWidth(-1);
          ImGui::SliderFloat("##smooth_strength", &smooth_strength, 0.0f, 1.0f);
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PushFont(bold_font);
        ImGui::Text("Render:Data");
        ImGui::PopFont();
        ImGui::Spacing();
        if (ImGui::Button("Add##data")) {
          ImGui::OpenPopup("AddDataLayer");
        }
        if (ImGui::BeginPopup("AddDataLayer")) {
          if (dataset_scalar_cell_names.empty()) {
            ImGui::TextDisabled("No scalar cell arrays in current frame.");
          } else {
            for (const auto& field_name : dataset_scalar_cell_names) {
              if (ImGui::Selectable(field_name.c_str())) {
                add_data_layer(data_layers, data_layer_mesh_caches, field_name);
              }
            }
          }
          ImGui::EndPopup();
        }
        sync_data_layer_caches(data_layers, data_layer_mesh_caches);
        if (data_layers.empty()) {
          ImGui::TextDisabled("Add a scalar cell array to enable fluid surface rendering.");
        } else {
          for (size_t i = 0; i < data_layers.size();) {
            DataLayer& layer = data_layers[i];
            MeshCache& layer_cache = data_layer_mesh_caches[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::Spacing();
            if (i > 0) {
              ImGui::Separator();
              ImGui::Spacing();
            }

            ImGui::Text("Entry %d", static_cast<int>(i + 1));
            ImGui::SameLine();
            if (ImGui::Button("Clear")) {
              data_layers.erase(data_layers.begin() + static_cast<long>(i));
              data_layer_mesh_caches.erase(data_layer_mesh_caches.begin() + static_cast<long>(i));
              mesh_loaded = false;
              ImGui::PopID();
              continue;
            }

            ImGui::Checkbox("Show", &layer.show);

            ImGui::Text("Field");
            ImGui::SameLine();
            const char* field_preview = layer.name.empty() ? "<none>" : layer.name.c_str();
            if (ImGui::BeginCombo("##field", field_preview)) {
              for (const auto& field_name : dataset_scalar_cell_names) {
                bool selected = (layer.name == field_name);
                if (ImGui::Selectable(field_name.c_str(), selected)) {
                  layer.name = field_name;
                  layer_cache.Clear();
                  layer.suppress_retry_after_error = false;
                }
                if (selected) ImGui::SetItemDefaultFocus();
              }
              ImGui::EndCombo();
            }

            bool selected_available = (find_field_index(dataset_scalar_cell_names, layer.name) >= 0);
            if (!selected_available) {
              ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "Selected field is missing in current frame.");
            }

            ImGui::Text("Threshold Min");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat("##threshold_min", &layer.threshold_min, 0.01f);
            ImGui::Text("Threshold Max");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::DragFloat("##threshold_max", &layer.threshold_max, 0.01f);

            ImGui::Text("Fluid Flag");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(100.0f);
            ImGui::InputInt("##fluid_flag", &layer.fluid_flag);

            ImGui::Text("Color");
            ImGui::ColorEdit3("##color", (float*)&layer.color);

            ImGui::Text("Metallic");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##metallic", &layer.metallic, 0.0f, 1.0f);

            ImGui::Text("Roughness");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##roughness", &layer.roughness, 0.0f, 1.0f);

            ImGui::Text("Opacity");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##opacity", &layer.opacity, 0.0f, 1.0f);

            ImGui::Text("Glass IOR");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##ior", &layer.glass_ior, 1.0f, 2.5f);

            ImGui::Text("Boundary Smoothing");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderInt("##smooth_iters", &layer.smooth_iterations, 0, 50);

            ImGui::Text("Boundary Smooth Strength");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##smooth_strength", &layer.smooth_strength, 0.0f, 1.0f);
            ImGui::PopID();
            ++i;
          }
        }
      }
    }
    ImGui::End();

    if (ImGuiFileDialog::Instance()->Display("OpenFileDlg")) {
      if (ImGuiFileDialog::Instance()->IsOk()) {
        vtk_dir = ImGuiFileDialog::Instance()->GetCurrentPath();
        vtk_files.clear();
        for (auto& entry : std::filesystem::directory_iterator(vtk_dir)) {
          if (entry.is_regular_file() && entry.path().extension() == ".vtk") {
            vtk_files.push_back(entry.path().string());
          }
        }
        std::sort(vtk_files.begin(), vtk_files.end());
        vtk_index = 0;
        dataset_cell_names.clear();
        dataset_scalar_cell_names.clear();
        dataset_loaded_field_vtk_index = -1;
        data_layers.clear();
        data_layer_mesh_caches.clear();
        if (!vtk_files.empty()) {
          try {
            if (load_dataset_field_lists(vtk_files[vtk_index], dataset_cell_names, dataset_scalar_cell_names)) {
              dataset_loaded_field_vtk_index = vtk_index;
            }
          } catch (...) {
            dataset_cell_names.clear();
            dataset_scalar_cell_names.clear();
            data_layers.clear();
            data_layer_mesh_caches.clear();
            dataset_loaded_field_vtk_index = -1;
          }
        }
      }
      ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("OpenMaskDlg")) {
      if (ImGuiFileDialog::Instance()->IsOk()) {
        std::string selected_file = ImGuiFileDialog::Instance()->GetFilePathName();

        try {
          viskores::io::VTKDataSetReader reader(selected_file);
          viskores::cont::DataSet mask_dataset = reader.ReadDataSet();

          bool is_3d = true;
          auto cellSet = mask_dataset.GetCellSet();
          if (cellSet.CanConvert<viskores::cont::CellSetStructured<1>>() ||
              cellSet.CanConvert<viskores::cont::CellSetStructured<2>>()) {
            is_3d = false;
          }

          if (!is_3d) {
            show_mask_error = true;
            mask_error_msg = "Only 3D files are supported.";
          } else {
            mask_file = selected_file;
            mask_field_names.clear();
            mask_field_index = 0;
            mask_mesh_cache.Clear();
            for (auto& layer_cache : data_layer_mesh_caches) {
              layer_cache.Clear();
            }
            mesh_loaded = false;
            for (viskores::IdComponent i = 0; i < mask_dataset.GetNumberOfFields(); i++) {
              const auto& field = mask_dataset.GetField(i);
              if (field.IsPointField() || field.IsCellField()) {
                mask_field_names.push_back(field.GetName());
              }
            }
            // Auto-select field containing "mask" (case-insensitive)
            for (int i = 0; i < (int)mask_field_names.size(); i++) {
              std::string lower_name = mask_field_names[i];
              std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(),
                             [](unsigned char c) { return std::tolower(c); });
              if (lower_name.find("mask") != std::string::npos) {
                mask_field_index = i;
                break;
              }
            }
          }
        } catch (const viskores::io::ErrorIO& e) {
          show_mask_error = true;
          mask_error_msg = std::string("Failed to read file: ") + e.what();
        } catch (...) {
          show_mask_error = true;
          mask_error_msg = "Failed to read file as a VTK dataset.";
        }
      }
      ImGuiFileDialog::Instance()->Close();
    }

    if (show_mask_error) {
      ImGui::OpenPopup("Mask Error");
      show_mask_error = false;
    }
    if (ImGui::BeginPopupModal("Mask Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("%s", mask_error_msg.c_str());
      if (ImGui::Button("OK")) {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }

    // Detect mesh extraction triggers (full rebuild) vs smoothing-only cache rebuild.
    bool mask_selection_changed = show_mask && (mask_field_index != prev_mask_field_index || solid_flag != prev_solid_flag);
    bool mask_smooth_changed =
        (smooth_iterations != prev_smooth_iterations) || (std::fabs(smooth_strength - prev_smooth_strength) > 1e-4f);
    bool transform_changed = (std::fabs(rotate_x_deg - prev_rotate_x_deg) > 1e-4f) ||
                             (std::fabs(rotate_y_deg - prev_rotate_y_deg) > 1e-4f) ||
                             (std::fabs(rotate_z_deg - prev_rotate_z_deg) > 1e-4f);
    bool ground_toggled = (ground_enabled != prev_ground_enabled);
    bool ground_offset_changed = (ground_y_offset != prev_ground_y_offset);

    bool has_mask_selection = !mask_file.empty() && !mask_field_names.empty() && mask_field_index >= 0 &&
                              mask_field_index < static_cast<int>(mask_field_names.size());
    std::string selected_mask_field = has_mask_selection ? mask_field_names[mask_field_index] : std::string();
    std::string fluid_source_file = (!vtk_files.empty() && vtk_index >= 0 && vtk_index < static_cast<int>(vtk_files.size()))
                                        ? vtk_files[vtk_index]
                                        : std::string();

    bool wants_mask = show_mask && has_mask_selection;
    bool mask_cache_matches_selection = wants_mask && mask_mesh_cache.valid && mask_mesh_cache.source_file == mask_file &&
                                        mask_mesh_cache.source_field == selected_mask_field &&
                                        mask_mesh_cache.source_solid_flag == solid_flag;
    bool needs_mask_extract = wants_mask && (mask_selection_changed || !mask_cache_matches_selection);

    struct LayerFrameState {
      bool renderable = false;
      bool same_failed_request = false;
      bool cache_matches = false;
      bool threshold_changed = false;
      bool smoothing_changed = false;
      bool visibility_changed = false;
      bool material_changed = false;
      bool needs_extract = false;
    };

    sync_data_layer_caches(data_layers, data_layer_mesh_caches);
    std::vector<LayerFrameState> layer_states(data_layers.size());
    bool any_layer_extract = false;
    bool any_layer_smooth_changed = false;
    bool any_layer_visibility_changed = false;

    for (size_t i = 0; i < data_layers.size(); i++) {
      DataLayer& layer = data_layers[i];
      MeshCache& layer_cache = data_layer_mesh_caches[i];
      LayerFrameState& state = layer_states[i];

      if (layer.threshold_min > layer.threshold_max) {
        std::swap(layer.threshold_min, layer.threshold_max);
      }

      bool has_density_field = (find_field_index(dataset_scalar_cell_names, layer.name) >= 0);
      state.renderable = layer.show && has_mask_selection && has_density_field && !fluid_source_file.empty();
      state.visibility_changed = (state.renderable != layer.prev_renderable);
      state.threshold_changed = (std::fabs(layer.threshold_min - layer.prev_threshold_min) > 1e-5f) ||
                                (std::fabs(layer.threshold_max - layer.prev_threshold_max) > 1e-5f);
      state.smoothing_changed =
          state.renderable && ((layer.smooth_iterations != layer.prev_smooth_iterations) ||
                               (std::fabs(layer.smooth_strength - layer.prev_smooth_strength) > 1e-4f));
      state.material_changed = state.renderable && (layer.color.x != layer.prev_color.x || layer.color.y != layer.prev_color.y ||
                                                    layer.color.z != layer.prev_color.z ||
                                                    std::fabs(layer.metallic - layer.prev_metallic) > 1e-5f ||
                                                    std::fabs(layer.roughness - layer.prev_roughness) > 1e-5f ||
                                                    std::fabs(layer.opacity - layer.prev_opacity) > 1e-5f ||
                                                    std::fabs(layer.glass_ior - layer.prev_glass_ior) > 1e-5f);

      state.same_failed_request =
          state.renderable && layer.suppress_retry_after_error && layer.failed_source_file == fluid_source_file &&
          layer.failed_mask_file == mask_file && layer.failed_mask_field == selected_mask_field &&
          layer.failed_field_name == layer.name && layer.failed_fluid_flag == layer.fluid_flag &&
          std::fabs(layer.failed_threshold_min - layer.threshold_min) <= 1e-5f &&
          std::fabs(layer.failed_threshold_max - layer.threshold_max) <= 1e-5f;
      if (state.renderable && layer.suppress_retry_after_error && !state.same_failed_request) {
        layer.suppress_retry_after_error = false;
        state.same_failed_request = false;
      }

      state.cache_matches = state.renderable && layer_cache.valid && layer_cache.source_file == fluid_source_file &&
                            layer_cache.source_field == layer.name && layer_cache.source_solid_flag == layer.fluid_flag;
      state.needs_extract = state.renderable && !state.same_failed_request && (!state.cache_matches || state.threshold_changed);

      any_layer_extract = any_layer_extract || state.needs_extract;
      any_layer_smooth_changed = any_layer_smooth_changed || state.smoothing_changed;
      any_layer_visibility_changed = any_layer_visibility_changed || state.visibility_changed;
    }

    bool needs_extract = needs_mask_extract || any_layer_extract;
    bool extraction_attempted = false;

    if (needs_mask_extract) {
      extraction_attempted = true;
      try {
        extract_mesh(mask_file, selected_mask_field, solid_flag, mask_mesh_cache);
      } catch (const std::exception& e) {
        show_mask_error = true;
        mask_error_msg = std::string("Mesh extraction failed: ") + e.what();
        mask_mesh_cache.Clear();
      } catch (...) {
        show_mask_error = true;
        mask_error_msg = "Mesh extraction failed.";
        mask_mesh_cache.Clear();
      }
    }

    for (size_t i = 0; i < data_layers.size(); i++) {
      if (!layer_states[i].needs_extract) {
        continue;
      }
      extraction_attempted = true;
      DataLayer& layer = data_layers[i];
      MeshCache& layer_cache = data_layer_mesh_caches[i];
      try {
        extract_fluid_mesh(fluid_source_file, layer.name, layer.threshold_min, layer.threshold_max, mask_file,
                           selected_mask_field, layer.fluid_flag, layer_cache);
        layer.suppress_retry_after_error = false;
      } catch (const std::exception& e) {
        show_mask_error = true;
        mask_error_msg = std::string("Mesh extraction failed: ") + e.what();
        layer_cache.Clear();
        layer.suppress_retry_after_error = true;
        layer.failed_source_file = fluid_source_file;
        layer.failed_mask_file = mask_file;
        layer.failed_mask_field = selected_mask_field;
        layer.failed_field_name = layer.name;
        layer.failed_fluid_flag = layer.fluid_flag;
        layer.failed_threshold_min = layer.threshold_min;
        layer.failed_threshold_max = layer.threshold_max;
      } catch (...) {
        show_mask_error = true;
        mask_error_msg = "Mesh extraction failed.";
        layer_cache.Clear();
        layer.suppress_retry_after_error = true;
        layer.failed_source_file = fluid_source_file;
        layer.failed_mask_file = mask_file;
        layer.failed_mask_field = selected_mask_field;
        layer.failed_field_name = layer.name;
        layer.failed_fluid_flag = layer.fluid_flag;
        layer.failed_threshold_min = layer.threshold_min;
        layer.failed_threshold_max = layer.threshold_max;
      }
    }

    bool render_mask = wants_mask && mask_mesh_cache.valid;
    std::vector<int> active_fluid_layer_indices;
    active_fluid_layer_indices.reserve(data_layers.size());
    for (size_t i = 0; i < data_layers.size(); i++) {
      const LayerFrameState& layer_state = layer_states[i];
      const MeshCache& layer_cache = data_layer_mesh_caches[i];
      if (layer_state.renderable && !layer_state.same_failed_request && layer_cache.valid) {
        active_fluid_layer_indices.push_back(static_cast<int>(i));
      }
    }
    bool render_fluid = !active_fluid_layer_indices.empty();
    bool render_any = render_mask || render_fluid;

    bool visibility_changed = (show_mask != prev_show_mask) || any_layer_visibility_changed;
    bool needs_mesh_rebuild_from_cache =
        render_any && !needs_extract &&
        ((render_mask && (mask_smooth_changed || transform_changed)) ||
         (render_fluid && (any_layer_smooth_changed || transform_changed)) || visibility_changed || !mesh_loaded);
    bool needs_full_rebuild = render_any && (extraction_attempted || needs_mesh_rebuild_from_cache);
    bool needs_gas_rebuild = render_any && mesh_loaded && !needs_full_rebuild && (ground_toggled || ground_offset_changed);

    if (needs_full_rebuild) {
      try {
        bool mesh_was_loaded = mesh_loaded;
        BBox bbox;
        rebuild_scene_from_caches(render_mask ? &mask_mesh_cache : nullptr, active_fluid_layer_indices, data_layers,
                                  data_layer_mesh_caches, optix_state, light_dir, light_color.x, light_color.y,
                                  light_color.z, light_strength, ground_enabled, ground_color, ground_metallic,
                                  ground_roughness, ground_opacity, ground_y_offset, rotate_x_deg, rotate_y_deg,
                                  rotate_z_deg, smooth_iterations, smooth_strength, mask_color, mask_metallic,
                                  mask_roughness, mask_opacity, mask_glass_ior, bbox);
        mesh_loaded = true;
        mesh_bbox = bbox;
        viewport_needs_render = true;

        if (extraction_attempted || !mesh_was_loaded) {
          float3 center =
              make_float3((bbox.lo.x + bbox.hi.x) * 0.5f, (bbox.lo.y + bbox.hi.y) * 0.5f, (bbox.lo.z + bbox.hi.z) * 0.5f);
          cam_target[0] = center.x;
          cam_target[1] = center.y;
          cam_target[2] = center.z;
          float3 extent = make_float3(bbox.hi.x - bbox.lo.x, bbox.hi.y - bbox.lo.y, bbox.hi.z - bbox.lo.z);
          float diag = std::sqrt(extent.x * extent.x + extent.y * extent.y + extent.z * extent.z);
          cam_distance = diag * 1.5f;
        }
      } catch (const std::exception& e) {
        show_mask_error = true;
        mask_error_msg = std::string("Mesh rebuild failed: ") + e.what();
        mesh_loaded = false;
      } catch (...) {
        show_mask_error = true;
        mask_error_msg = "Mesh rebuild failed.";
        mesh_loaded = false;
      }
    } else if (!render_any) {
      mesh_loaded = false;
    }

    if (needs_gas_rebuild && !needs_full_rebuild) {
      rebuild_gas(optix_state, mesh_bbox, render_mask, static_cast<int>(active_fluid_layer_indices.size()),
                  ground_enabled, ground_y_offset);
      update_scene_material_sbt(optix_state, light_dir, light_color.x, light_color.y, light_color.z, light_strength,
                                ground_enabled, ground_color, ground_metallic, ground_roughness, ground_opacity,
                                mask_color, mask_metallic, mask_roughness, mask_opacity, mask_glass_ior, data_layers);
      prev_ground_color = ground_color;
      prev_ground_metallic = ground_metallic;
      prev_ground_roughness = ground_roughness;
      prev_ground_opacity = ground_opacity;
      viewport_needs_render = true;
    }

    // Detect background color / render settings change
    if (bg_color.x != prev_bg_color.x || bg_color.y != prev_bg_color.y || bg_color.z != prev_bg_color.z ||
        rt_samples != prev_rt_samples || rt_bounces != prev_rt_bounces) {
      if (render_any && mesh_loaded) {
        update_miss_sbt(optix_state, bg_color);
      }
      viewport_needs_render = true;
      prev_bg_color = bg_color;
      prev_rt_samples = rt_samples;
      prev_rt_bounces = rt_bounces;
    }

    // Detect material/light changes (cheap SBT update, no GAS rebuild)
    if (render_any && mesh_loaded) {
      bool light_changed = (light_strength != prev_light_strength || light_color.x != prev_light_color.x ||
                            light_color.y != prev_light_color.y || light_color.z != prev_light_color.z);
      bool mask_mat_changed =
          render_mask && (mask_color.x != prev_mask_color.x || mask_color.y != prev_mask_color.y ||
                          mask_color.z != prev_mask_color.z || std::fabs(mask_metallic - prev_mask_metallic) > 1e-5f ||
                          std::fabs(mask_roughness - prev_mask_roughness) > 1e-5f ||
                          std::fabs(mask_opacity - prev_mask_opacity) > 1e-5f ||
                          std::fabs(mask_glass_ior - prev_mask_glass_ior) > 1e-4f);
      bool fluid_mat_changed = false;
      for (int layer_idx : active_fluid_layer_indices) {
        if (layer_idx < 0 || layer_idx >= static_cast<int>(layer_states.size()) ||
            layer_idx >= static_cast<int>(data_layers.size())) {
          continue;
        }
        fluid_mat_changed = fluid_mat_changed || layer_states[static_cast<size_t>(layer_idx)].material_changed;
      }
      bool ground_mat_changed = ground_enabled &&
                                (ground_color.x != prev_ground_color.x || ground_color.y != prev_ground_color.y ||
                                 ground_color.z != prev_ground_color.z ||
                                 std::fabs(ground_metallic - prev_ground_metallic) > 1e-5f ||
                                 std::fabs(ground_roughness - prev_ground_roughness) > 1e-5f ||
                                 std::fabs(ground_opacity - prev_ground_opacity) > 1e-5f);

      if (light_changed || mask_mat_changed || fluid_mat_changed || ground_mat_changed) {
        update_scene_material_sbt(optix_state, light_dir, light_color.x, light_color.y, light_color.z, light_strength,
                                  ground_enabled, ground_color, ground_metallic, ground_roughness, ground_opacity,
                                  mask_color, mask_metallic, mask_roughness, mask_opacity, mask_glass_ior, data_layers);
        viewport_needs_render = true;

        prev_mask_color = mask_color;
        prev_mask_metallic = mask_metallic;
        prev_mask_roughness = mask_roughness;
        prev_mask_opacity = mask_opacity;
        prev_mask_glass_ior = mask_glass_ior;
        prev_light_strength = light_strength;
        prev_light_color = light_color;
        prev_ground_color = ground_color;
        prev_ground_metallic = ground_metallic;
        prev_ground_roughness = ground_roughness;
        prev_ground_opacity = ground_opacity;
      }
    }

    // Detect shadow toggle changes (params only, no SBT update needed)
    if (render_any && mesh_loaded) {
      if (shadows_enabled != prev_shadows_enabled) {
        viewport_needs_render = true;
        prev_shadows_enabled = shadows_enabled;
      }
    }

    prev_show_mask = show_mask;
    prev_mask_field_index = mask_field_index;
    prev_solid_flag = solid_flag;
    prev_smooth_iterations = smooth_iterations;
    prev_smooth_strength = smooth_strength;
    prev_rotate_x_deg = rotate_x_deg;
    prev_rotate_y_deg = rotate_y_deg;
    prev_rotate_z_deg = rotate_z_deg;
    prev_ground_enabled = ground_enabled;
    prev_ground_y_offset = ground_y_offset;
    for (size_t i = 0; i < data_layers.size(); i++) {
      DataLayer& layer = data_layers[i];
      layer.prev_show = layer.show;
      layer.prev_threshold_min = layer.threshold_min;
      layer.prev_threshold_max = layer.threshold_max;
      layer.prev_fluid_flag = layer.fluid_flag;
      layer.prev_color = layer.color;
      layer.prev_metallic = layer.metallic;
      layer.prev_roughness = layer.roughness;
      layer.prev_opacity = layer.opacity;
      layer.prev_glass_ior = layer.glass_ior;
      layer.prev_smooth_iterations = layer.smooth_iterations;
      layer.prev_smooth_strength = layer.smooth_strength;
      layer.prev_renderable = (i < layer_states.size()) ? layer_states[i].renderable : false;
    }
    // Viewport window
    ImGuiWindowClass viewport_window_class;
    viewport_window_class.DockNodeFlagsOverrideSet =
        ImGuiDockNodeFlags_NoTabBar | ImGuiDockNodeFlags_NoWindowMenuButton;
    ImGui::SetNextWindowClass(&viewport_window_class);
    ImGui::Begin("###ViewportPanel", nullptr, ImGuiWindowFlags_NoTitleBar);

    if (ImGui::IsWindowHovered()) {
      // Scroll wheel: zoom
      float scroll = io.MouseWheel;
      if (scroll != 0.0f) {
        cam_distance *= (1.0f - scroll * 0.1f);
        cam_distance = std::max(0.1f, cam_distance);
        viewport_needs_render = true;
      }

      // Left mouse drag: orbit
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        ImVec2 delta = io.MouseDelta;
        cam_yaw -= delta.x * 0.3f;
        cam_pitch += delta.y * 0.3f;
        cam_pitch = std::max(-89.0f, std::min(89.0f, cam_pitch));
        viewport_needs_render = true;
      }

      // Middle mouse drag: pan
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
        ImVec2 delta = io.MouseDelta;
        float yaw_rad = cam_yaw * 3.14159265f / 180.0f;
        float pitch_rad = cam_pitch * 3.14159265f / 180.0f;

        // Camera right and up vectors
        float3 forward = make_float3(std::cos(pitch_rad) * std::sin(yaw_rad), std::sin(pitch_rad),
                                     std::cos(pitch_rad) * std::cos(yaw_rad));
        float3 world_up = make_float3(0, 1, 0);
        // cross(forward, world_up)
        float3 right = make_float3(forward.y * world_up.z - forward.z * world_up.y,
                                   forward.z * world_up.x - forward.x * world_up.z,
                                   forward.x * world_up.y - forward.y * world_up.x);
        float right_len = std::sqrt(right.x * right.x + right.y * right.y + right.z * right.z);
        if (right_len > 1e-6f) {
          right.x /= right_len;
          right.y /= right_len;
          right.z /= right_len;
        }
        // cross(right, forward)
        float3 up = make_float3(right.y * forward.z - right.z * forward.y, right.z * forward.x - right.x * forward.z,
                                right.x * forward.y - right.y * forward.x);
        float up_len = std::sqrt(up.x * up.x + up.y * up.y + up.z * up.z);
        if (up_len > 1e-6f) {
          up.x /= up_len;
          up.y /= up_len;
          up.z /= up_len;
        }

        float pan_speed = cam_distance * 0.002f;
        cam_target[0] -= (right.x * delta.x + up.x * delta.y) * pan_speed;
        cam_target[1] -= (right.y * delta.x + up.y * delta.y) * pan_speed;
        cam_target[2] -= (right.z * delta.x + up.z * delta.y) * pan_speed;
        viewport_needs_render = true;
      }
    }

    ImVec2 avail = ImGui::GetContentRegionAvail();
    int vp_w = std::max(1, (int)avail.x);
    int vp_h = std::max(1, (int)avail.y);

    if (vp_w != prev_vp_w || vp_h != prev_vp_h) {
      viewport_needs_render = true;
      prev_vp_w = vp_w;
      prev_vp_h = vp_h;
    }

    // Ensure viewport texture exists at the right size
    if (!viewport_tex || viewport_tex_w != vp_w || viewport_tex_h != vp_h) {
      if (viewport_tex) SDL_DestroyTexture(viewport_tex);
      viewport_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, vp_w, vp_h);
      viewport_tex_w = vp_w;
      viewport_tex_h = vp_h;
    }

    if (render_any && mesh_loaded && viewport_needs_render) {
      viewport_needs_render = false;
      try {
        // Compute camera eye from spherical coordinates
        float yaw_rad = cam_yaw * 3.14159265f / 180.0f;
        float pitch_rad = cam_pitch * 3.14159265f / 180.0f;
        float3 eye = make_float3(cam_target[0] + cam_distance * std::cos(pitch_rad) * std::sin(yaw_rad),
                                 cam_target[1] + cam_distance * std::sin(pitch_rad),
                                 cam_target[2] + cam_distance * std::cos(pitch_rad) * std::cos(yaw_rad));
        float3 target = make_float3(cam_target[0], cam_target[1], cam_target[2]);

        // Compute camera frame vectors
        float3 W = make_float3(target.x - eye.x, target.y - eye.y, target.z - eye.z);
        float w_len = std::sqrt(W.x * W.x + W.y * W.y + W.z * W.z);
        if (w_len > 1e-6f) {
          W.x /= w_len;
          W.y /= w_len;
          W.z /= w_len;
        }

        float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
        // U = cross(W, world_up)
        float3 U = make_float3(W.y * world_up.z - W.z * world_up.y, W.z * world_up.x - W.x * world_up.z,
                               W.x * world_up.y - W.y * world_up.x);
        float u_len = std::sqrt(U.x * U.x + U.y * U.y + U.z * U.z);
        if (u_len > 1e-6f) {
          U.x /= u_len;
          U.y /= u_len;
          U.z /= u_len;
        }

        // V = cross(U, W)
        float3 V = make_float3(U.y * W.z - U.z * W.y, U.z * W.x - U.x * W.z, U.x * W.y - U.y * W.x);

        // Scale U and V by FOV
        float half_h = std::tan(cam_fov * 3.14159265f / 360.0f);
        float aspect = (float)vp_w / (float)vp_h;
        float half_w = half_h * aspect;

        float3 cam_u = make_float3(U.x * half_w, U.y * half_w, U.z * half_w);
        // Negate V so that y=0 (d.y=-1) maps to camera UP, matching screen convention
        float3 cam_v = make_float3(-V.x * half_h, -V.y * half_h, -V.z * half_h);

        // Resize device image buffer if needed
        if (optix_state.img_w != vp_w || optix_state.img_h != vp_h) {
          if (optix_state.d_image) {
            cudaFree(reinterpret_cast<void*>(optix_state.d_image));
          }
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&optix_state.d_image), vp_w * vp_h * sizeof(uchar4)));
          optix_state.img_w = vp_w;
          optix_state.img_h = vp_h;
          host_pixels.resize(vp_w * vp_h);
        }

        // Fill launch params
        Params launch_params = {};
        launch_params.image = reinterpret_cast<uchar4*>(optix_state.d_image);
        launch_params.image_width = vp_w;
        launch_params.image_height = vp_h;
        launch_params.samples_per_pixel = static_cast<unsigned int>(rt_samples);
        launch_params.max_depth = static_cast<unsigned int>(rt_bounces);
        launch_params.cam_eye = eye;
        launch_params.cam_u = cam_u;
        launch_params.cam_v = cam_v;
        launch_params.cam_w = W;
        launch_params.handle = optix_state.gas_handle;

        // Fill shadow toggle
        launch_params.shadows_enabled = shadows_enabled ? 1 : 0;

        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(optix_state.d_params), &launch_params, sizeof(Params),
                              cudaMemcpyHostToDevice));

        // Launch ray tracing
        OPTIX_CHECK(optixLaunch(optix_state.pipeline, 0, optix_state.d_params, sizeof(Params), &optix_state.sbt, vp_w,
                                vp_h, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result to host
        CUDA_CHECK(cudaMemcpy(host_pixels.data(), reinterpret_cast<void*>(optix_state.d_image),
                              vp_w * vp_h * sizeof(uchar4), cudaMemcpyDeviceToHost));

        // Upload to SDL texture
        void* tex_pixels = nullptr;
        int tex_pitch = 0;
        if (SDL_LockTexture(viewport_tex, nullptr, &tex_pixels, &tex_pitch)) {
          for (int y = 0; y < vp_h; y++) {
            memcpy((uint8_t*)tex_pixels + y * tex_pitch, host_pixels.data() + y * vp_w, vp_w * sizeof(uchar4));
          }
          SDL_UnlockTexture(viewport_tex);
        }
      } catch (const std::exception& e) {
        show_mask_error = true;
        mask_error_msg = std::string("Rendering failed: ") + e.what();
        mesh_loaded = false;
        viewport_needs_render = true;
      }
    } else if ((!render_any || !mesh_loaded) && viewport_needs_render) {
      viewport_needs_render = false;

      // Fill viewport with background color (no OptiX)
      host_pixels.resize(vp_w * vp_h);
      uchar4 bg = make_uchar4((unsigned char)(bg_color.x * 255.0f + 0.5f), (unsigned char)(bg_color.y * 255.0f + 0.5f),
                              (unsigned char)(bg_color.z * 255.0f + 0.5f), 255);
      std::fill(host_pixels.begin(), host_pixels.end(), bg);

      void* tex_pixels = nullptr;
      int tex_pitch = 0;
      if (SDL_LockTexture(viewport_tex, nullptr, &tex_pixels, &tex_pitch)) {
        for (int y = 0; y < vp_h; y++) {
          memcpy((uint8_t*)tex_pixels + y * tex_pitch, host_pixels.data() + y * vp_w, vp_w * sizeof(uchar4));
        }
        SDL_UnlockTexture(viewport_tex);
      }
    }

    if (viewport_tex) {
      ImVec2 img_cursor = ImGui::GetCursorScreenPos();
      ImGui::Image((ImTextureID)(intptr_t)viewport_tex, avail);

      // Draw bounding box outline overlay
      if (render_any && mesh_loaded && show_outlines) {
        float yaw_rad = cam_yaw * 3.14159265f / 180.0f;
        float pitch_rad = cam_pitch * 3.14159265f / 180.0f;
        float3 eye = make_float3(cam_target[0] + cam_distance * std::cos(pitch_rad) * std::sin(yaw_rad),
                                 cam_target[1] + cam_distance * std::sin(pitch_rad),
                                 cam_target[2] + cam_distance * std::cos(pitch_rad) * std::cos(yaw_rad));
        float3 tgt = make_float3(cam_target[0], cam_target[1], cam_target[2]);
        float3 fwd = make_float3(tgt.x - eye.x, tgt.y - eye.y, tgt.z - eye.z);
        float fwd_len = std::sqrt(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z);
        if (fwd_len > 1e-6f) {
          fwd.x /= fwd_len;
          fwd.y /= fwd_len;
          fwd.z /= fwd_len;
        }
        float3 world_up = make_float3(0.0f, 1.0f, 0.0f);
        float3 cam_right = make_float3(fwd.y * world_up.z - fwd.z * world_up.y, fwd.z * world_up.x - fwd.x * world_up.z,
                                       fwd.x * world_up.y - fwd.y * world_up.x);
        float cr_len = std::sqrt(cam_right.x * cam_right.x + cam_right.y * cam_right.y + cam_right.z * cam_right.z);
        if (cr_len > 1e-6f) {
          cam_right.x /= cr_len;
          cam_right.y /= cr_len;
          cam_right.z /= cr_len;
        }
        float3 cam_up =
            make_float3(cam_right.y * fwd.z - cam_right.z * fwd.y, cam_right.z * fwd.x - cam_right.x * fwd.z,
                        cam_right.x * fwd.y - cam_right.y * fwd.x);
        float cu_len = std::sqrt(cam_up.x * cam_up.x + cam_up.y * cam_up.y + cam_up.z * cam_up.z);
        if (cu_len > 1e-6f) {
          cam_up.x /= cu_len;
          cam_up.y /= cu_len;
          cam_up.z /= cu_len;
        }

        float half_h = std::tan(cam_fov * 3.14159265f / 360.0f);
        float aspect = (float)vp_w / (float)vp_h;
        float half_w_val = half_h * aspect;

        // Project 3D point to screen
        auto project = [&](const float3& p, bool& behind) -> ImVec2 {
          float3 v = make_float3(p.x - eye.x, p.y - eye.y, p.z - eye.z);
          float cz = v.x * fwd.x + v.y * fwd.y + v.z * fwd.z;
          if (cz <= 0.001f) {
            behind = true;
            return {0, 0};
          }
          float cx = v.x * cam_right.x + v.y * cam_right.y + v.z * cam_right.z;
          float cy = v.x * cam_up.x + v.y * cam_up.y + v.z * cam_up.z;
          float ndc_x = cx / (cz * half_w_val);
          float ndc_y = cy / (cz * half_h);
          behind = false;
          return ImVec2(img_cursor.x + (ndc_x * 0.5f + 0.5f) * avail.x, img_cursor.y + (0.5f - ndc_y * 0.5f) * avail.y);
        };

        float3 lo = mesh_bbox.lo, hi = mesh_bbox.hi;
        float3 corners[8] = {{lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {lo.x, hi.y, lo.z},
                             {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z}, {lo.x, hi.y, hi.z}};
        int edges[12][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
                            {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImU32 ol_col = ImGui::ColorConvertFloat4ToU32(outline_color);
        for (int e = 0; e < 12; e++) {
          bool b0, b1;
          ImVec2 p0 = project(corners[edges[e][0]], b0);
          ImVec2 p1 = project(corners[edges[e][1]], b1);
          if (!b0 && !b1) {
            draw_list->AddLine(p0, p1, ol_col, outline_thickness);
          }
        }
      }
    }

    ImGui::End();

    ImGui::Render();
    SDL_SetRenderScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
    SDL_SetRenderDrawColorFloat(renderer, bg_color.x, bg_color.y, bg_color.z, bg_color.w);
    SDL_RenderClear(renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
    SDL_RenderPresent(renderer);
  }

  if (viewport_tex) SDL_DestroyTexture(viewport_tex);

  ImGui_ImplSDLRenderer3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  cleanup_optix(optix_state);

  return 0;
}
