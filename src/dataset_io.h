#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <viskores/cont/DataSet.h>

struct DatasetFileInfo {
  bool is_3d = false;
  std::vector<std::string> all_field_names;
  std::vector<std::string> scalar_cell_field_names;
};

bool is_supported_dataset_file(const std::filesystem::path& path);
bool is_supported_dataset_sequence_entry(const std::filesystem::path& path);

DatasetFileInfo inspect_dataset_file(const std::string& path);
viskores::cont::DataSet read_dataset_with_compat(const std::string& path);
