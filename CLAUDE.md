# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## Build

```bash
# From repo root
cmake -S . -B build
make -C build -j6
```

Outputs:
- App binary: `bin/CaustiX`
- OptiX PTX: `bin/shaders.ptx` (built by CMake custom target)

## Runtime Dependencies

- CUDA Toolkit 13.1+ (`/opt/cuda` on this machine)
- OptiX SDK 9.1.0 (`/opt/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64`)
- NVIDIA GPU (project currently targets `compute_75` in CMake)

No test infrastructure exists in this repo.

## Architecture

CaustiX is a 3D scientific visualization app that ray-traces isosurfaces from VTK datasets.

Stack:
- SDL3 window + renderer
- ImGui docking UI
- Viskores data processing (VTK read + contour extraction)
- OptiX 9.1 GPU ray tracing

### Key Source Files

- `src/main.cpp`
  - App/UI loop
  - VTK/mask loading
  - Mesh extraction + smoothing + rotation
  - Hybrid fluid extraction (masked density isosurface + volume texture)
  - OptiX host setup, SBT/GAS updates, launch
- `src/shaders.cu`
  - `__raygen__rg`, `__miss__radiance`, `__miss__shadow`
  - `__closesthit__ch` (mesh), `__closesthit__ground`
- `src/optix_params.h`
  - Shared launch and SBT structs (`Params`, `MissData`, `HitGroupData`)
- `src/config.hpp`
  - Window/font constants

### Main Runtime State Organization

`main.cpp` groups UI/runtime variables into structs:
- `LightingState`
- `MaskState`
- `GroundState`
- `FluidState`
- `DatasetState`
- `RenderMiscState`
- `RayTracingState`
- `CameraState`
- `OptixState`

## Data + Mesh Pipeline

### Mask Surface (`extract_mesh`)
1. Read mask VTK via `viskores::io::VTKDataSetReader`.
2. Validate selected field is scalar.
3. If field is cell-associated, convert to point field using `PointAverage`.
4. Normalize field storage to default float using `GetDataAsDefaultFloat()`.
5. Run contour extraction at isovalue `solid_flag - 0.5`, generate normals.
6. Extract positions/normals/connectivity.
7. Triangulate non-triangle cells with a fan fallback.
8. Optional CPU Laplacian smoothing:
   - Iterations from UI slider
   - Strength (`lambda`) from UI slider
   - Recompute normals afterward
9. Optional geometry rotation:
   - Rotate X / Y / Z (degrees) from `Render:Misc`
   - Applied to positions and normals before upload
10. Upload mesh buffers to GPU.
11. Rebuild GAS and update SBT hitgroups.

### Hybrid Fluid (`extract_fluid_mesh`)
1. Read density dataset (from current `Render:Dataset` frame) and mask dataset (from `Render:Mask`).
   - The active density file follows `Begin/Prev/Next/End` frame navigation (`vtk_files[vtk_index]`).
   - UI density candidates are scalar cell arrays only; non-scalar cell arrays are excluded.
   - Default field name is chosen from the first dataset file in the sequence: first scalar cell field whose name contains `rho` (case-insensitive), fallback first scalar cell field.
2. Convert density and mask fields to point scalar arrays.
3. Build masked density:
   - keep density where `mask == liquid_flag`
   - set non-liquid points below threshold so contour ignores them
4. Contour masked density at `Density Threshold` to get liquid surface.
5. Build normalized masked volume density and upload as a 3D CUDA texture.
6. Rebuild mesh/GAS with fluid-specific surface material controls.

## OptiX Pipeline + SBT

One-time init:
1. `cudaFree(0)` then `optixInit()`
2. PTX module creation from `bin/shaders.ptx`
3. Program groups: raygen, 2 miss, mesh hitgroup, ground hitgroup
4. Pipeline creation (`maxTraceDepth=3`)
5. SBT allocation:
   - 1 raygen record
   - 2 miss records (radiance + shadow)
   - Up to 4 hitgroup records (mesh/ground x radiance/shadow)

Ray types:
- `RAY_TYPE_RADIANCE`
- `RAY_TYPE_SHADOW`

SBT hitgroup indexing:
- `index = ray_type + RAY_TYPE_COUNT * geometry_index`

## Error Handling Macros

`src/main.cpp` uses macro-based error checks:
- `CUDA_CHECK(call)`
- `OPTIX_CHECK(call)`
- `OPTIX_CHECK_LOG(call)`

They build detailed `std::ostringstream` messages and throw `std::runtime_error`.

Important:
- These are multi-line macros and require trailing `\` line continuations.
- If line continuations are removed/altered, parsing fails early (before normal compilation), often near the top of
  `main.cpp`.

## Shading and Materials

### Lighting

- Single global directional light (from `HitGroupData`)
- Optional hard shadows via dedicated shadow rays (`Params.shadows_enabled`)

### Glass-like Transparency

Mesh shading in `__closesthit__ch` uses:
- Local direct lighting (Blinn-Phong style)
- Reflection + refraction blend when `opacity < 1`
- Schlick Fresnel
- Runtime IOR from mask material (`HitGroupData.ior`, controlled by UI slider)
- Applies on secondary hits too (`depth < params.max_depth`) to avoid black interior artifacts

### Hybrid Fluid Volume

When `Render:Data > Show Fluid` is enabled:
- Refracted rays sample a masked 3D density texture (`cudaTextureObject_t`) inside fluid bounds.
- Beer-Lambert attenuation uses `Volume Absorption`.
- A soft in-scatter tint uses `Volume Mix`.
- Marching step length is controlled by `Volume Step`.
- `Show Volume` toggles volumetric contribution on/off at launch-param level.
- `Show Interface` toggles surface/interface shading visibility.

### Ground Plane

- Optional large quad under mesh (`bbox.lo.y + ground_y_offset`)
- Ground has separate material controls (color/metallic/roughness/opacity)
- Reflection path on ground for primary hits when metallic is non-trivial

## UI Layout

- Docked `Config` + `Render` sidebar and central viewport.
- Viewport panel intentionally hides title/tab bar for a clean look (`NoTitleBar`, dock node no-tab flags).

### Config Panel

- Background color
- Global illumination (strength/color)
- Ray tracing (`Bounces`, `Samples`)
- Ground plane controls

### Render Panel

- Dataset open/navigation
- `Render:Misc`:
  - Enable Shadows
  - Show Outlines
  - Rotate X / Rotate Y / Rotate Z (degrees)
- `Render:Mask`:
  - Mask file + field
  - `Solid Flag`
  - Show
  - Material: color/metallic/roughness/opacity
  - `Glass IOR`
  - Smoothing iterations + smooth strength
- `Render:Data`:
  - `Show Fluid`
  - `Show Interface`
  - `Show Volume`
  - `Density Field` (scalar cell fields from current dataset frame, defaulting to first `rho*` match)
  - `Density Threshold`
  - `Liquid Flag` (default `0`)
  - Fluid material controls (separate from mask): color/metallic/roughness/opacity/`Glass IOR`
  - `Volume Absorption`, `Volume Mix`, `Volume Step`

## Rebuild/Update Strategy

- Full mesh rebuild (extract + upload + GAS + SBT):
  - mask show/field/solid flag changes
  - smoothing iteration/strength changes
  - geometry rotation changes (X/Y/Z)
  - fluid interface/volume mode, field/threshold/liquid-flag/source-file changes
- GAS-only rebuild:
  - ground enabled toggle
  - ground y offset changes
- Cheap SBT updates only:
  - mask material (including glass IOR)
  - fluid surface material (including glass IOR/interface visibility)
  - ground material
  - global light params
  - background miss color
- Launch params only:
  - camera
  - samples/bounces
  - shadow toggle
  - fluid volume shading params (`Volume Absorption`, `Volume Mix`, `Volume Step`)

## Viskores Notes

- `viskores::cont::Initialize(argc, argv)` is called at startup.
- Viskores may warn for VTK file versions > 4.2 (common with 5.1 files); this is a reader warning and not necessarily fatal.

## Submodules

Under `lib/`:
- SDL (3.4.0)
- imgui (docking branch, 1.92.5)
- ImGuiFileDialog (0.6.8)
- viskores (1.1.0)

Viskores is intentionally trimmed in `CMakeLists.txt`:
- tests/rendering/examples/tutorials/benchmarks/docs disabled
- MPI/CUDA/Kokkos/TBB/OpenMP disabled

## Code Style

- Google C++ style
- 120 column limit
- Includes intentionally not sorted (`SortIncludes: false`)
