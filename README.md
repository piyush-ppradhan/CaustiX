# CaustiX

CaustiX is a GPU ray-traced visualization app for VTK datasets.
It extracts and renders surfaces from scalar fields for both solid masks and fluid regions.

<p align="center">
  <img src="assets/3D_evaporation.gif" alt="" width="600">
</p>

## Features

- VTK dataset loading and frame navigation
- Mask surface extraction from scalar fields (`Solid Flag - 0.5` isovalue)
- Fluid surface extraction from scalar cell fields using:
  - dataset frame selected in `Render > Dataset`
  - mask-based fluid gating via `Fluid Flag`
  - lower/upper scalar threshold (`Threshold Min`, `Threshold Max`)
- Geometry import via Assimp (`OBJ/STL/PLY/FBX/glTF`) with per-entry transform/material
- Independent smoothing controls for mask and fluid surfaces
- Surface material controls for mask/fluid/ground:
  - color, metallic, roughness, opacity
  - glass IOR (for transparent/refraction look)
- Optional shadows and ground plane
- Geometry rotation controls (`Rotate X/Y/Z`)

## Requirements

- Linux
- CMake 3.15+
- CUDA Toolkit 13.1+
- NVIDIA OptiX SDK 9.1.0
- NVIDIA GPU

## Build

```bash
cmake -S . -B build
make -C build -j6
```

Outputs:
- `bin/CaustiX`
- `bin/shaders.ptx`

## Run

```bash
./bin/CaustiX
```

## Quick Start

1. Open a dataset folder from `Render > Dataset > Open`.
2. In `Render:Mask`, choose a mask `.vtk` and select the mask field.
3. Enable `Show` in `Render:Mask` to render solids.
4. In `Render:Data`:
   - click `Add` and choose a scalar **cell** array
   - enable `Show`
   - choose `Field`
   - set `Threshold Min/Max`
   - set `Fluid Flag` (default `0`)
5. Tune fluid material (`Color`, `Metallic`, `Roughness`, `Opacity`, `Glass IOR`) and fluid boundary smoothing.
6. Use `Begin/Prev/Next/End` under `Render > Dataset` to change the active frame.
7. In `Render:Geometry`, click `Add Geometry`, then use each entry's `Config` button to tune
   scale/rotation/material.

## UI Summary

### Render:Misc

- `Enable Shadows`
- `Show Outlines`
- `Rotate X`, `Rotate Y`, `Rotate Z`
- Ray tracing `Bounces` supports `1..20`

### Config

- `Load Config from File`
- `Save Config to File`
- Background color
- Global illumination
- Camera axis buttons `X/Y/Z`
- `Save Camera Config` to store the current camera
- Saved camera selector + `Load` to restore a saved view

Config files (`.cfg`) store:
- background color
- global illumination strength and color
- ground plane properties
- all saved camera positions

### Render:Mask

- `File`, `Field`, `Solid Flag`, `Show`
- Material: `Color`, `Metallic`, `Roughness`, `Opacity`, `Glass IOR`
- Smoothing: `Smoothing`, `Smooth Strength`
- `Render:Geometry` appears below this section in the Render panel

### Render:Data

- `Add`, `Clear`
- `Show`
- `Field` (from added scalar cell arrays)
- `Threshold Min`, `Threshold Max`
- `Fluid Flag`
- Material: `Color`, `Metallic`, `Roughness`, `Opacity`, `Glass IOR`
- Smoothing: `Boundary Smoothing`, `Boundary Smooth Strength`

### Render:Geometry

- `Add Geometry`
- Per-entry:
  - `Show`
  - `Config` (scale, local rotation, color, metallic, roughness, opacity, glass IOR)
  - `Clear`

## Notes

- Fluid extraction uses the mask field selected in `Render:Mask`.
- Points with `mask == Fluid Flag` are treated as fluid candidates.
- Thresholding is then applied to the selected data field to build the fluid surface.

## Troubleshooting

- If mask extraction fails:
  - ensure the selected mask field is scalar
  - ensure the dataset is 3D
- If fluid extraction fails:
  - ensure `Render:Mask` has a valid mask file/field
  - ensure selected `Render:Data` field is scalar cell data
  - ensure `Threshold Min/Max` includes values present in fluid regions
  - ensure `Fluid Flag` matches the fluid label in the mask
- If rendering is slow:
  - reduce `Samples`
  - reduce smoothing iterations
  - reduce viewport size

## Performance Notes

- Assimp is built with only required importers enabled:
  - `OBJ`, `STL`, `PLY`, `FBX`, `GLTF`
- Exporters, tools, docs, tests, samples, and Draco are disabled in CMake.

## Repository Layout

- `src/main.cpp`: app loop, UI, extraction, OptiX host setup
- `src/shaders.cu`: raygen/miss/mesh/ground shaders
- `src/fluid_shading.cuh`: fluid closest-hit shader (surface-based)
- `src/optix_params.h`: shared launch/SBT structs
- `assets/`: UI fonts
- `lib/`: dependencies
