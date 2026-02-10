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
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include <iostream>
#include "config.hpp"

struct DataLayer {
  std::string name;
  ImVec4 color = ImVec4(1.0f, 0.0f, 0.0f, 1.0f);
  float opacity = 1.0f;
  bool visible = true;
  float line_width = 1.0f;
  bool show_config = false;
};

int main(int argc, char* argv[]) {
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
  int solid_flag = 1;
  std::vector<std::string> vtk_files;
  std::vector<std::string> mask_field_names;
  int mask_field_index = 0;
  bool show_mask_error = false;
  std::string mask_error_msg;
  int vtk_index = 0;
  std::vector<std::string> dataset_cell_names;
  std::vector<DataLayer> data_layers;
  bool first_frame = true;

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
    // ImGui::Separator();
    // ImGui::PushFont(bold_font);
    // ImGui::Text("Lights");
    // ImGui::PopFont();
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
        // ImGui::InputInt(" ", &solid_flag, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::InputInt(" ", &solid_flag);
        ImGui::Spacing();
        ImGui::Checkbox("Show", &show_mask);
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

          std::cout << "DEBUG: Number of points: " << dataset.GetNumberOfPoints() << std::endl;
          std::cout << "DEBUG: Number of cells: " << dataset.GetNumberOfCells() << std::endl;
          std::cout << "DEBUG: Number of fields: " << dataset.GetNumberOfFields() << std::endl;

          bool is_3d = true;
          auto cellSet = dataset.GetCellSet();
          if (cellSet.CanConvert<viskores::cont::CellSetStructured<1>>() ||
              cellSet.CanConvert<viskores::cont::CellSetStructured<2>>()) {
            is_3d = false;
            std::cout << "DEBUG: Structured 1D or 2D dataset" << std::endl;
          } else {
            std::cout << "DEBUG: 3D or unstructured dataset" << std::endl;
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

    ImGui::Render();
    SDL_SetRenderScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
    SDL_SetRenderDrawColorFloat(renderer, bg_color.x, bg_color.y, bg_color.z, bg_color.w);
    SDL_RenderClear(renderer);
    ImGui_ImplSDLRenderer3_RenderDrawData(ImGui::GetDrawData(), renderer);
    SDL_RenderPresent(renderer);
  }

  ImGui_ImplSDLRenderer3_Shutdown();
  ImGui_ImplSDL3_Shutdown();
  ImGui::DestroyContext();

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
