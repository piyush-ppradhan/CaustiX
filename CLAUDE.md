# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Build

```bash
cmake -S . -B build
make -C build -j6
```

Outputs:
- `bin/CaustiX`
- `bin/shaders.ptx`

## Runtime Dependencies

- CUDA Toolkit 13.1+
- OptiX SDK 9.1.0
- NVIDIA GPU

No automated test suite is currently present.

## Architecture

CaustiX is a 3D scientific visualization app that ray-traces extracted surfaces from VTK data.

Stack:
- SDL3 window + renderer
- ImGui docking UI
- Viskores for VTK IO/filtering
- OptiX for GPU ray tracing

## Key Source Files

- `src/main.cpp`
  - UI and application loop
  - Dataset/mask loading
  - Mask/fluid surface extraction
  - Mesh smoothing + rotation
  - OptiX host setup, SBT/GAS rebuilds, launch
- `src/shaders.cu`
  - raygen/miss programs
  - mesh + ground closest-hit shading
- `src/fluid_shading.cuh`
  - fluid closest-hit shading (surface-based, no volumetric marching)
- `src/optix_params.h`
  - shared host/device launch and hitgroup structs

## Runtime State Organization (`main.cpp`)

Runtime/UI state is grouped into structs:
- `LightingState`
- `MaskState`
- `GroundState`
- `FluidState`
- `DatasetState`
- `RenderMiscState`
- `RayTracingState`
- `CameraState`
- `OptixState`

## Data + Extraction Pipeline

### Mask surface (`extract_mesh`)

1. Read mask VTK.
2. Validate selected mask field is scalar.
3. Convert cell field to point field when needed.
4. Convert to default float storage.
5. Contour at `solid_flag - 0.5`.
6. Extract points/normals/connectivity.
7. Optional Laplacian smoothing.
8. Optional rotation (`Rotate X/Y/Z`).
9. Upload to GPU and rebuild GAS/SBT.

### Fluid surface (`extract_fluid_mesh`)

1. Read density dataset from current `Render:Dataset` frame.
2. Read mask dataset from `Render:Mask` selection (same file or separate file).
3. Validate selected data field is scalar cell field.
4. Convert density and mask to point scalar fields.
5. Keep points where:
   - `mask == fluid_flag`
   - `threshold_min <= data_value <= threshold_max`
6. Build binary occupancy field from the filtered points.
7. Contour occupancy at `0.5` to extract fluid surface.
8. Triangulate polygons when needed.
9. Cache mesh for rebuild/material updates.

No volumetric texture upload or ray marching is used in the current design.

## Material + Shading

### Mask and fluid surfaces

- Surface shading with directional light + optional shadows.
- Supports reflection/refraction blend when opacity is below 1.
- Uses Fresnel + IOR for glass-like appearance.
- Fluid has independent material controls from mask:
  - color
  - metallic
  - roughness
  - opacity
  - glass IOR

### Ground

- Optional ground plane with independent material controls.

## UI Mapping

### `Render:Mask`

- File/field selection
- `Solid Flag`
- `Show`
- Material controls
- Smoothing controls

### `Render:Data`

- `Add`/`Clear` scalar cell fields
- `Show`
- `Field`
- `Threshold Min`, `Threshold Max`
- `Fluid Flag`
- Fluid material controls (`Color`, `Metallic`, `Roughness`, `Opacity`, `Glass IOR`)
- Fluid smoothing controls (`Boundary Smoothing`, `Boundary Smooth Strength`)

## Rebuild Strategy

- Full extract + rebuild:
  - file/field/flag/threshold changes
  - smoothing changes
  - rotation changes
  - visibility changes requiring mesh inclusion
- GAS-only rebuild:
  - ground enable/disable or ground offset changes
- SBT-only update:
  - material and lighting parameter changes

## Error Handling

`main.cpp` uses:
- `CUDA_CHECK`
- `OPTIX_CHECK`
- `OPTIX_CHECK_LOG`

These macros throw `std::runtime_error` with location/context.

## Viskores Notes

- `viskores::cont::Initialize(argc, argv)` is called during app startup.
- Reader warnings about VTK versions > 4.2 can appear for VTK 5.1 files.

## Submodules (`lib/`)

- SDL
- ImGui
- ImGuiFileDialog
- Viskores
