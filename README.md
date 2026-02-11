# CaustiX

CaustiX is a GPU ray-traced visualization app for volumetric VTK data.
It extracts an isosurface from a scalar field and renders it with OptiX using interactive material, glass, smoothing, and ground-plane controls.

## Features

- VTK dataset loading and frame navigation
- Mask/isosurface extraction from scalar fields (`solid_flag - 0.5`)
- Interactive ray-traced rendering (OptiX)
- Glass-like transparency with adjustable `Glass IOR`
- Shadow toggle and global directional light controls
- Ground plane with material and offset controls
- CPU Laplacian surface smoothing:
  - Iterations
  - Smooth strength
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
3. Choose mask `Field` and `Solid Flag`.
4. Enable `Show`.
5. Tune material (`Opacity`, `Metallic`, `Roughness`, `Glass IOR`) and smoothing controls.

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
- `Render:Mask`
  - Mask file/field selection
  - `Solid Flag`
  - `Show`
  - Material controls
  - `Glass IOR` (`1.0` to `2.5`)
  - Smoothing iterations (`0` to `50`)
  - Smooth strength (`0.0` to `1.0`)

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

## Development Notes

- No automated test suite is currently included.
- Main project guidance for coding agents lives in `CLAUDE.md`.
