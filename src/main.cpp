#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <imgui.h>
#include <imgui_internal.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_sdlrenderer3.h>
#include <ImGuiFileDialog.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkNew.h>
#include <vtkDataSet.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkImageData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredGrid.h>
#include <vtkRectilinearGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkPolyData.h>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>
#include "config.hpp"
#include <iostream>

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
      }
      ImGuiFileDialog::Instance()->Close();
    }

    if (ImGuiFileDialog::Instance()->Display("OpenMaskDlg")) {
      if (ImGuiFileDialog::Instance()->IsOk()) {
        std::string selected_file = ImGuiFileDialog::Instance()->GetFilePathName();

        vtkNew<vtkGenericDataObjectReader> reader;
        reader->SetFileName(selected_file.c_str());
        reader->ReadAllScalarsOn();
        reader->ReadAllVectorsOn();
        reader->ReadAllTensorsOn();
        reader->ReadAllFieldsOn();
        reader->ReadAllNormalsOn();
        reader->ReadAllTCoordsOn();
        reader->ReadAllColorScalarsOn();
        reader->Update();

        vtkDataSet* dataset = nullptr;
        if (reader->IsFileStructuredPoints()) {
          std::cout << "DEBUG: File is StructuredPoints" << std::endl;
          dataset = reader->GetStructuredPointsOutput();
        } else if (reader->IsFileStructuredGrid()) {
          std::cout << "DEBUG: File is StructuredGrid" << std::endl;
          dataset = reader->GetStructuredGridOutput();
        } else if (reader->IsFileRectilinearGrid()) {
          std::cout << "DEBUG: File is RectilinearGrid" << std::endl;
          dataset = reader->GetRectilinearGridOutput();
        } else if (reader->IsFileUnstructuredGrid()) {
          std::cout << "DEBUG: File is UnstructuredGrid" << std::endl;
          dataset = reader->GetUnstructuredGridOutput();
        } else if (reader->IsFilePolyData()) {
          std::cout << "DEBUG: File is PolyData" << std::endl;
          dataset = reader->GetPolyDataOutput();
        } else {
          std::cout << "DEBUG: File type not recognized" << std::endl;
        }

        if (!dataset) {
          show_mask_error = true;
          mask_error_msg = "Failed to read file as a VTK dataset.";
        } else {
          std::cout << "DEBUG: Number of points: " << dataset->GetNumberOfPoints() << std::endl;
          std::cout << "DEBUG: Point data arrays: " << dataset->GetPointData()->GetNumberOfArrays() << std::endl;
          std::cout << "DEBUG: Cell data arrays: " << dataset->GetCellData()->GetNumberOfArrays() << std::endl;

          bool is_3d = true;
          int dims[3];
          if (auto* img = vtkImageData::SafeDownCast(dataset)) {
            img->GetDimensions(dims);
            std::cout << "DEBUG: ImageData dims: " << dims[0] << "x" << dims[1] << "x" << dims[2] << std::endl;
            if (dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) is_3d = false;
          } else if (auto* sg = vtkStructuredGrid::SafeDownCast(dataset)) {
            sg->GetDimensions(dims);
            std::cout << "DEBUG: StructuredGrid dims: " << dims[0] << "x" << dims[1] << "x" << dims[2] << std::endl;
            if (dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) is_3d = false;
          } else if (auto* rg = vtkRectilinearGrid::SafeDownCast(dataset)) {
            rg->GetDimensions(dims);
            std::cout << "DEBUG: RectilinearGrid dims: " << dims[0] << "x" << dims[1] << "x" << dims[2] << std::endl;
            if (dims[0] <= 1 || dims[1] <= 1 || dims[2] <= 1) is_3d = false;
          } else {
            std::cout << "DEBUG: No dimension check (unstructured/polydata)" << std::endl;
          }

          if (!is_3d) {
            show_mask_error = true;
            mask_error_msg = "Only 3D files are supported.";
          } else {
            mask_file = selected_file;
            mask_field_names.clear();
            mask_field_index = 0;
            vtkPointData* pd = dataset->GetPointData();
            for (int i = 0; i < pd->GetNumberOfArrays(); i++) {
              mask_field_names.push_back(pd->GetArrayName(i));
            }
            vtkCellData* cd = dataset->GetCellData();
            for (int i = 0; i < cd->GetNumberOfArrays(); i++) {
              mask_field_names.push_back(cd->GetArrayName(i));
            }
          }
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
