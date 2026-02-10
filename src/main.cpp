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
#include <viskores/filter/entity_extraction/Threshold.h>
#include <viskores/filter/entity_extraction/ExternalFaces.h>
#include <viskores/filter/geometry_refinement/Triangulate.h>
#include <viskores/filter/vector_analysis/SurfaceNormals.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include "config.hpp"
#include "pathtracer.hpp"

struct DataLayer {
  std::string name;
  ImVec4 color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
  float opacity = 1.0f;
  bool visible = true;
  float line_width = 1.0f;
  bool show_config = false;
};

static void extract_mesh(const std::string& mask_filepath, const std::string& field_name,
                         int solid_val, pt::PathTracer& tracer, const pt::Material& mat) {
  viskores::io::VTKDataSetReader reader(mask_filepath);
  viskores::cont::DataSet ds = reader.ReadDataSet();

  // Threshold: keep cells where field == solid_val
  viskores::filter::entity_extraction::Threshold threshold;
  threshold.SetActiveField(field_name);
  threshold.SetThresholdBetween((double)solid_val, (double)solid_val);
  viskores::cont::DataSet thresholded = threshold.Execute(ds);

  // External faces
  viskores::filter::entity_extraction::ExternalFaces externalFaces;
  externalFaces.SetCompactPoints(true);
  viskores::cont::DataSet surface = externalFaces.Execute(thresholded);

  // Triangulate
  viskores::filter::geometry_refinement::Triangulate triangulate;
  viskores::cont::DataSet triangulated = triangulate.Execute(surface);

  // Surface normals (point normals)
  viskores::filter::vector_analysis::SurfaceNormals normals;
  normals.SetGeneratePointNormals(true);
  normals.SetGenerateCellNormals(false);
  normals.SetAutoOrientNormals(true);
  viskores::cont::DataSet result = normals.Execute(triangulated);

  // Extract positions
  auto coords = result.GetCoordinateSystem();
  auto coordData = coords.GetData();
  viskores::Id numPoints = result.GetNumberOfPoints();

  std::vector<float> positions(numPoints * 3);
  auto coordArray = coordData.AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
  {
    auto portal = coordArray.ReadPortal();
    for (viskores::Id i = 0; i < numPoints; i++) {
      auto p = portal.Get(i);
      positions[i * 3 + 0] = p[0];
      positions[i * 3 + 1] = p[1];
      positions[i * 3 + 2] = p[2];
    }
  }

  // Extract normals
  std::vector<float> normal_data;
  if (result.HasPointField("Normals")) {
    auto normalField = result.GetPointField("Normals");
    auto normalArray = normalField.GetData().AsArrayHandle<viskores::cont::ArrayHandle<viskores::Vec3f_32>>();
    auto portal = normalArray.ReadPortal();
    normal_data.resize(numPoints * 3);
    for (viskores::Id i = 0; i < numPoints; i++) {
      auto n = portal.Get(i);
      normal_data[i * 3 + 0] = n[0];
      normal_data[i * 3 + 1] = n[1];
      normal_data[i * 3 + 2] = n[2];
    }
  }

  // Extract connectivity
  viskores::Id numCells = result.GetNumberOfCells();
  std::vector<int> indices(numCells * 3);
  auto cellSet = result.GetCellSet();

  for (viskores::Id c = 0; c < numCells; c++) {
    viskores::Id ids[3];
    cellSet.GetCellPointIds(c, ids);
    indices[c * 3 + 0] = (int)ids[0];
    indices[c * 3 + 1] = (int)ids[1];
    indices[c * 3 + 2] = (int)ids[2];
  }

  tracer.set_mesh(positions, normal_data, indices, mat);
}

int main(int /*argc*/, char* /*argv*/[]) {
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

  ImVec4 bg_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  float light_strength = 1.0f;
  ImVec4 light_color = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
  std::string vtk_dir;
  std::string mask_file;
  bool show_outlines = false;
  bool show_mask = false;
  bool prev_show_mask = false;
  int solid_flag = 1;
  int prev_solid_flag = 1;
  std::vector<std::string> vtk_files;
  std::vector<std::string> mask_field_names;
  int mask_field_index = 0;
  int prev_mask_field_index = 0;
  bool show_mask_error = false;
  std::string mask_error_msg;
  int vtk_index = 0;
  std::vector<std::string> dataset_cell_names;
  std::vector<DataLayer> data_layers;
  bool first_frame = true;

  // Mask material
  ImVec4 mask_color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);
  float mask_metallic = 0.0f;
  float mask_roughness = 0.5f;
  float mask_opacity = 1.0f;

  // Ray tracing config
  int rt_bounces = 4;
  int rt_samples = 1;
  int rt_width = 512;
  int rt_height = 512;

  // Orbit camera
  float cam_yaw = 0.0f;
  float cam_pitch = 30.0f;
  float cam_distance = 5.0f;
  float cam_target[3] = {0, 0, 0};
  float cam_fov = 60.0f;
  bool viewport_needs_render = true;

  // Viewport texture
  SDL_Texture* viewport_tex = nullptr;
  int viewport_tex_w = 0, viewport_tex_h = 0;
  std::vector<uint8_t> pixel_buffer;

  // Path tracer
  pt::PathTracer path_tracer;
  bool mesh_loaded = false;

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
      ImGui::DockBuilderDockWindow("Viewport", dock_main);
      ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::Begin("Config");
    ImGui::PushFont(bold_font);
    ImGui::Text("Background Color");
    ImGui::PopFont();
    ImGui::ColorPicker3("##bg", (float*)&bg_color, ImGuiColorEditFlags_PickerHueWheel);
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
    ImGui::ColorPicker3("##light_color", (float*)&light_color, ImGuiColorEditFlags_PickerHueWheel);
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
    ImGui::Text("Width");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##rt_width", &rt_width);
    rt_width = std::max(64, std::min(2048, rt_width));
    ImGui::Text("Height");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::InputInt("##rt_height", &rt_height);
    rt_height = std::max(64, std::min(2048, rt_height));
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
      data_layers.clear();
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
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::PushFont(bold_font);
      ImGui::Text("Render:Misc");
      ImGui::PopFont();
      ImGui::Checkbox("Show Outlines", &show_outlines);
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
        ImGui::InputInt(" ", &solid_flag);
        ImGui::Spacing();
        ImGui::Checkbox("Show", &show_mask);

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
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::PushFont(bold_font);
        ImGui::Text("Render:Data");
        ImGui::PopFont();
        ImGui::Spacing();
        if (!dataset_cell_names.empty()) {
          if (ImGui::Button("Add##data")) {
            ImGui::OpenPopup("AddDataLayer");
          }
          if (ImGui::BeginPopup("AddDataLayer")) {
            for (int i = 0; i < (int)dataset_cell_names.size(); i++) {
              if (ImGui::Selectable(dataset_cell_names[i].c_str())) {
                data_layers.push_back(DataLayer{dataset_cell_names[i]});
              }
            }
            ImGui::EndPopup();
          }
          ImGui::SameLine();
        }
        if (ImGui::Button("Clear##data")) {
          data_layers.clear();
        }
        for (int i = 0; i < (int)data_layers.size(); i++) {
          ImGui::Text("%s", data_layers[i].name.c_str());
          ImGui::SameLine();
          std::string config_id = "Config##data_config_" + std::to_string(i);
          if (ImGui::SmallButton(config_id.c_str())) {
            data_layers[i].show_config = true;
          }
          ImGui::SameLine();
          std::string remove_id = "X##data_remove_" + std::to_string(i);
          if (ImGui::SmallButton(remove_id.c_str())) {
            data_layers.erase(data_layers.begin() + i);
            i--;
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
        data_layers.clear();
        if (!vtk_files.empty()) {
          try {
            viskores::io::VTKDataSetReader data_reader(vtk_files[0]);
            viskores::cont::DataSet ds = data_reader.ReadDataSet();
            for (viskores::IdComponent i = 0; i < ds.GetNumberOfFields(); i++) {
              const auto& field = ds.GetField(i);
              if (field.IsPointField() || field.IsCellField()) {
                dataset_cell_names.push_back(field.GetName());
              }
            }
          } catch (...) {
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
          viskores::cont::DataSet dataset = reader.ReadDataSet();

          bool is_3d = true;
          auto cellSet = dataset.GetCellSet();
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
            for (viskores::IdComponent i = 0; i < dataset.GetNumberOfFields(); i++) {
              const auto& field = dataset.GetField(i);
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

    for (int i = 0; i < (int)data_layers.size(); i++) {
      std::string popup_id = "Data Config##" + std::to_string(i);
      if (data_layers[i].show_config) {
        ImGui::OpenPopup(popup_id.c_str());
        data_layers[i].show_config = false;
      }
      if (ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Layer: %s", data_layers[i].name.c_str());
        ImGui::Spacing();
        ImGui::Text("Color");
        std::string color_id = "##data_color_" + std::to_string(i);
        ImGui::ColorPicker3(color_id.c_str(), (float*)&data_layers[i].color, ImGuiColorEditFlags_PickerHueWheel);
        ImGui::Spacing();
        ImGui::Text("Opacity");
        std::string opacity_id = "##data_opacity_" + std::to_string(i);
        ImGui::SliderFloat(opacity_id.c_str(), &data_layers[i].opacity, 0.0f, 1.0f);
        ImGui::Spacing();
        std::string visible_id = "Visible##data_visible_" + std::to_string(i);
        ImGui::Checkbox(visible_id.c_str(), &data_layers[i].visible);
        ImGui::Spacing();
        ImGui::Text("Line Width");
        std::string lw_id = "##data_lw_" + std::to_string(i);
        ImGui::SliderFloat(lw_id.c_str(), &data_layers[i].line_width, 0.5f, 5.0f);
        ImGui::Spacing();
        if (ImGui::Button("OK")) {
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
    }

    // Detect mesh extraction triggers
    bool mask_params_changed = (show_mask != prev_show_mask) ||
                               (show_mask && (mask_field_index != prev_mask_field_index ||
                                              solid_flag != prev_solid_flag));
    prev_show_mask = show_mask;
    prev_mask_field_index = mask_field_index;
    prev_solid_flag = solid_flag;

    if (show_mask && mask_params_changed && !mask_file.empty() && !mask_field_names.empty()) {
      pt::Material mat;
      mat.albedo = {mask_color.x, mask_color.y, mask_color.z};
      mat.metallic = mask_metallic;
      mat.roughness = mask_roughness;
      mat.opacity = mask_opacity;
      try {
        extract_mesh(mask_file, mask_field_names[mask_field_index], solid_flag, path_tracer, mat);
        mesh_loaded = true;
        viewport_needs_render = true;

        // Auto-fit camera to mesh bounding box
        auto& tris = path_tracer.bvh().triangles;
        if (!tris.empty()) {
          pt::Vec3f lo(1e30f), hi(-1e30f);
          for (auto& tri : tris) {
            lo = pt::vmin(lo, pt::vmin(tri.v0, pt::vmin(tri.v1, tri.v2)));
            hi = pt::vmax(hi, pt::vmax(tri.v0, pt::vmax(tri.v1, tri.v2)));
          }
          pt::Vec3f center = (lo + hi) * 0.5f;
          cam_target[0] = center.x;
          cam_target[1] = center.y;
          cam_target[2] = center.z;
          float extent = (hi - lo).length();
          cam_distance = extent * 1.5f;
        }
      } catch (const std::exception& e) {
        show_mask_error = true;
        mask_error_msg = std::string("Mesh extraction failed: ") + e.what();
        mesh_loaded = false;
      } catch (...) {
        show_mask_error = true;
        mask_error_msg = "Mesh extraction failed.";
        mesh_loaded = false;
      }
    }

    // Viewport window
    ImGui::Begin("Viewport");

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
        pt::Vec3f forward = {
            std::cos(pitch_rad) * std::sin(yaw_rad),
            std::sin(pitch_rad),
            std::cos(pitch_rad) * std::cos(yaw_rad)};
        pt::Vec3f world_up = {0, 1, 0};
        pt::Vec3f right = pt::cross(forward, world_up).normalized();
        pt::Vec3f up = pt::cross(right, forward).normalized();

        float pan_speed = cam_distance * 0.002f;
        cam_target[0] -= (right.x * delta.x + up.x * delta.y) * pan_speed;
        cam_target[1] -= (right.y * delta.x + up.y * delta.y) * pan_speed;
        cam_target[2] -= (right.z * delta.x + up.z * delta.y) * pan_speed;
        viewport_needs_render = true;
      }
    }

    if (show_mask && mesh_loaded && viewport_needs_render) {
      viewport_needs_render = false;

      // Compute camera eye from spherical coordinates
      float yaw_rad = cam_yaw * 3.14159265f / 180.0f;
      float pitch_rad = cam_pitch * 3.14159265f / 180.0f;
      pt::Vec3f eye = {
          cam_target[0] + cam_distance * std::cos(pitch_rad) * std::sin(yaw_rad),
          cam_target[1] + cam_distance * std::sin(pitch_rad),
          cam_target[2] + cam_distance * std::cos(pitch_rad) * std::cos(yaw_rad)};
      pt::Vec3f target = {cam_target[0], cam_target[1], cam_target[2]};

      // Update material
      pt::Material mat;
      mat.albedo = {mask_color.x, mask_color.y, mask_color.z};
      mat.metallic = mask_metallic;
      mat.roughness = mask_roughness;
      mat.opacity = mask_opacity;

      path_tracer.set_camera(eye, target, {0, 1, 0}, cam_fov);
      path_tracer.set_config(rt_width, rt_height, rt_bounces, rt_samples);
      path_tracer.set_background(bg_color.x * 0.2f, bg_color.y * 0.2f, bg_color.z * 0.2f);
      path_tracer.render(pixel_buffer);

      // Create or resize SDL texture
      if (!viewport_tex || viewport_tex_w != rt_width || viewport_tex_h != rt_height) {
        if (viewport_tex) SDL_DestroyTexture(viewport_tex);
        viewport_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32,
                                         SDL_TEXTUREACCESS_STREAMING, rt_width, rt_height);
        viewport_tex_w = rt_width;
        viewport_tex_h = rt_height;
      }

      // Upload pixels to texture
      void* tex_pixels = nullptr;
      int tex_pitch = 0;
      if (SDL_LockTexture(viewport_tex, nullptr, &tex_pixels, &tex_pitch)) {
        for (int y = 0; y < rt_height; y++) {
          memcpy((uint8_t*)tex_pixels + y * tex_pitch,
                 pixel_buffer.data() + y * rt_width * 4, rt_width * 4);
        }
        SDL_UnlockTexture(viewport_tex);
      }
    }

    if (viewport_tex && show_mask && mesh_loaded) {
      ImVec2 avail = ImGui::GetContentRegionAvail();
      float aspect = (float)viewport_tex_w / (float)viewport_tex_h;
      float disp_w = avail.x;
      float disp_h = avail.x / aspect;
      if (disp_h > avail.y) {
        disp_h = avail.y;
        disp_w = avail.y * aspect;
      }
      ImGui::Image((ImTextureID)(intptr_t)viewport_tex, ImVec2(disp_w, disp_h));
    } else {
      ImGui::TextDisabled("Enable mask rendering to see viewport");
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

  return 0;
}
