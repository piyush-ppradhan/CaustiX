# CaustiX

CaustiX is a GPU ray-traced visualization app for volumetric VTK data.
It extracts and ray-traces isosurfaces from scalar fields, with an optional hybrid fluid mode that combines a masked density surface with a lightweight volume pass.

## Features

- VTK dataset loading and frame navigation
- Mask/isosurface extraction from scalar fields (`solid_flag - 0.5`)
- Interactive ray-traced rendering (OptiX)
- Glass-like transparency with adjustable `Glass IOR`
- Hybrid fluid rendering:
  - Surface: masked density isosurface
  - Volume: masked density ray-marched on refracted rays
  - Independent toggles for surface interface and volumetric pass
  - Dedicated fluid material controls (separate from mask material)
  - Dedicated fluid interface smoothing controls
  - Physically-inspired volume controls (absorption + scattering + step size)
  - Density field list is scalar cell fields from the active dataset frame, with default selected from first sequence file (`rho*`, case-insensitive)
- Shadow toggle and global directional light controls
- Ground plane with material and offset controls
- CPU Laplacian surface smoothing:
  - Iterations
  - Smooth strength
- Geometry transform controls:
  - Rotate X / Rotate Y / Rotate Z
- Docked ImGui UI with clean viewport panel

## Requirements

- Linux
- CMake 3.15+
- CUDA Toolkit 13.1+
- NVIDIA OptiX SDK 9.1.0
- NVIDIA GPU (project currently compiles shaders for `compute_75`)

Expected local paths in current `CMakeLists.txt`:
- CUDA: discovered via `find_package(CUDAToolkit REQUIRED)`
- OptiX SDK: `/opt/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64`

## Build

```bash
cmake -S . -B build
make -C build -j6
```

Build outputs:
- `bin/CaustiX`
- `bin/shaders.ptx`

## Run

```bash
./bin/CaustiX
```

## Quick Start

1. Open a dataset folder in `Render > Dataset > Open`.
2. In `Render:Mask`, open/select a mask VTK file.
3. Choose mask `Field` and enable `Show` for regular mask-surface rendering.
4. For fluid rendering, go to `Render:Data`:
   - enable `Show Fluid`
   - select `Density Field` (default comes from first dataset file: first scalar cell field containing `rho`, case-insensitive)
   - set `Density Threshold`
   - set `Fluid Flag` (default `0`)
   - choose `Show Interface` and/or `Show Volume`
   - tune fluid material (`Color`, `Metallic`, `Roughness`, `Opacity`, `Glass IOR`)
5. Use `Begin/Prev/Next/End` in `Render > Dataset` to pick the frame used by fluid rendering.
6. Tune fluid interface smoothing (`Interface Smoothing`, `Interface Smooth Strength`) and volume controls (`Volume Absorption`, `Volume Scattering`, `Volume Step`).

## Controls

### Camera

- Mouse wheel: zoom
- Left drag: orbit
- Middle drag: pan

### Config Panel

- Background color
- Global illumination color/strength
- Ray tracing:
  - `Bounces`
  - `Samples`
- Ground plane material and offset

### Render Panel

- `Render:Misc`
  - `Enable Shadows`
  - `Show Outlines`
  - `Rotate X`, `Rotate Y`, `Rotate Z` (degrees, default `0`)
- `Render:Mask`
  - Mask file/field selection
  - `Solid Flag`
  - `Show`
  - Material controls
  - `Glass IOR` (`1.0` to `2.5`)
  - Smoothing iterations (`0` to `50`)
  - Smooth strength (`0.0` to `1.0`)
- `Render:Data`
  - `Show Fluid`
  - `Show Interface`
  - `Show Volume`
  - `Density Field` (scalar cell fields from current dataset frame; default picked from first dataset file by `rho*` match)
  - `Density Threshold`
  - `Fluid Flag` (default `0`)
  - Fluid material:
    - `Color`
    - `Metallic`
    - `Roughness`
    - `Opacity`
    - `Glass IOR`
  - `Interface Smoothing`
  - `Interface Smooth Strength`
  - `Volume Absorption`
  - `Volume Scattering`
  - `Volume Step`

## Notes on Glass Rendering

- `Opacity` near `0` makes the mesh mostly transmissive/reflection-driven.
- `Glass IOR` controls refraction bending:
  - Lower: weaker bending
  - Higher: stronger bending
- For cleaner transparent objects, use higher `Bounces` (e.g. `3-5`).

## Troubleshooting

- Viskores warning about VTK version > 4.2:
  - This warning can appear with VTK 5.1 files and is not always fatal.
- If mask extraction fails:
  - Ensure selected mask field is scalar.
  - Ensure the dataset is 3D.
- If fluid rendering fails:
  - Ensure density and mask datasets have matching point topology.
  - Ensure the selected density field is a scalar cell field.
  - Ensure the active dataset frame has scalar cell fields (fluid density picker ignores non-scalar fields).
  - Ensure the dataset is a 3D structured grid (required for the hybrid volume texture).
- If fluid is enabled but nothing shows:
  - Ensure at least one of `Show Interface` or `Show Volume` is enabled.
- If transparent objects look too dark:
  - Increase `Bounces`.
  - Reduce extreme `Metallic`.
- If performance is slow:
  - Reduce `Samples`.
  - Reduce viewport size.
  - Lower smoothing iterations.

## Repository Layout

- `src/main.cpp`: application loop, UI, OptiX host code, mesh extraction
- `src/shaders.cu`: OptiX device programs
- `src/optix_params.h`: shared host/device structs
- `assets/`: UI fonts
- `lib/`: submodules (SDL, ImGui, ImGuiFileDialog, Viskores)
