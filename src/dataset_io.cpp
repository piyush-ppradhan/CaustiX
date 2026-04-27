#include "dataset_io.h"

#include <hdf5.h>
#include <tinyxml2.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CellSetStructured.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/io/VTKDataSetReader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class H5Handle {
 public:
  H5Handle() = default;
  H5Handle(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
  H5Handle(const H5Handle&) = delete;
  H5Handle& operator=(const H5Handle&) = delete;
  H5Handle(H5Handle&& other) noexcept : id_(other.id_), closer_(other.closer_) { other.id_ = -1; }
  H5Handle& operator=(H5Handle&& other) noexcept {
    if (this != &other) {
      reset();
      id_ = other.id_;
      closer_ = other.closer_;
      other.id_ = -1;
    }
    return *this;
  }
  ~H5Handle() { reset(); }

  hid_t get() const { return id_; }
  explicit operator bool() const { return id_ >= 0; }

 private:
  void reset() {
    if (id_ >= 0 && closer_ != nullptr) {
      closer_(id_);
    }
    id_ = -1;
  }

  hid_t id_ = -1;
  herr_t (*closer_)(hid_t) = nullptr;
};

struct Hdf5SourceRef {
  std::filesystem::path file_path;
  std::string dataset_path;
  std::vector<hsize_t> start;
  std::vector<hsize_t> stride;
  std::vector<hsize_t> count;
};

struct XdmfAttributeRef {
  std::string name;
  Hdf5SourceRef source;
};

struct XdmfGridInfo {
  bool is_3d = false;
  viskores::Id3 cell_dims = viskores::Id3(1, 1, 1);
  viskores::Vec3f origin = viskores::Vec3f(0.0f, 0.0f, 0.0f);
  viskores::Vec3f spacing = viskores::Vec3f(1.0f, 1.0f, 1.0f);
  std::vector<XdmfAttributeRef> attributes;
};

static std::string trim_copy(const std::string& text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
    begin++;
  }
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    end--;
  }
  return text.substr(begin, end - begin);
}

static std::string lowercase_copy(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

static bool ends_with_case_insensitive(const std::string& text, const std::string& suffix) {
  if (suffix.size() > text.size()) {
    return false;
  }
  return lowercase_copy(text.substr(text.size() - suffix.size())) == lowercase_copy(suffix);
}

static std::vector<long long> parse_integer_list(const std::string& text) {
  std::vector<long long> values;
  std::istringstream iss(text);
  long long value = 0;
  while (iss >> value) {
    values.push_back(value);
  }
  return values;
}

static std::vector<double> parse_double_list(const std::string& text) {
  std::vector<double> values;
  std::istringstream iss(text);
  double value = 0.0;
  while (iss >> value) {
    values.push_back(value);
  }
  return values;
}

static bool replace_all_padded(std::string& text, const std::string& token, const std::string& replacement) {
  if (replacement.size() > token.size()) {
    return false;
  }
  std::string padded = replacement + std::string(token.size() - replacement.size(), ' ');
  size_t pos = 0;
  bool changed = false;
  while ((pos = text.find(token, pos)) != std::string::npos) {
    text.replace(pos, token.size(), padded);
    pos += padded.size();
    changed = true;
  }
  return changed;
}

static bool normalize_legacy_vtk_binary(std::string& bytes) {
  bool changed = false;
  changed = replace_all_padded(bytes, "vtk DataFile Version 5.1", "vtk DataFile Version 4.2") || changed;
  changed = replace_all_padded(bytes, "signed_char", "char") || changed;
  changed = replace_all_padded(bytes, "vtktypeint64", "long") || changed;
  changed = replace_all_padded(bytes, "vtktypeint32", "int") || changed;
  changed = replace_all_padded(bytes, "vtktypeuint8", "unsigned_char") || changed;
  return changed;
}

static viskores::cont::DataSet read_vtk_dataset_with_compat_impl(const std::string& vtk_file) {
  viskores::io::VTKDataSetReader reader(vtk_file);
  try {
    return reader.ReadDataSet();
  } catch (const viskores::io::ErrorIO&) {
    std::ifstream in(vtk_file, std::ios::binary);
    if (!in.is_open()) {
      throw;
    }
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
      throw;
    }
    if (!normalize_legacy_vtk_binary(bytes)) {
      throw;
    }

    std::filesystem::path temp_path =
        std::filesystem::path("/tmp") / ("caustix_compat_" + std::to_string(std::hash<std::string>{}(vtk_file)) + ".vtk");
    {
      std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
      out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      if (!out.good()) {
        throw;
      }
    }

    try {
      viskores::io::VTKDataSetReader compat_reader(temp_path.string());
      viskores::cont::DataSet converted = compat_reader.ReadDataSet();
      std::error_code remove_ec;
      std::filesystem::remove(temp_path, remove_ec);
      return converted;
    } catch (...) {
      std::error_code remove_ec;
      std::filesystem::remove(temp_path, remove_ec);
      throw;
    }
  }
}

static Hdf5SourceRef parse_hdf5_source_ref(const std::filesystem::path& xdmf_path, const std::string& ref_text) {
  std::string trimmed = trim_copy(ref_text);
  size_t split = trimmed.find(":/");
  if (split == std::string::npos) {
    throw std::runtime_error("Unsupported XDMF HDF reference: " + trimmed);
  }

  std::filesystem::path h5_path = trimmed.substr(0, split);
  if (h5_path.is_relative()) {
    h5_path = xdmf_path.parent_path() / h5_path;
  }

  Hdf5SourceRef ref;
  ref.file_path = std::filesystem::weakly_canonical(h5_path);
  ref.dataset_path = trimmed.substr(split + 1);
  return ref;
}

static Hdf5SourceRef parse_xdmf_data_source(const std::filesystem::path& xdmf_path,
                                            const tinyxml2::XMLElement* data_item) {
  const char* format_attr = data_item != nullptr ? data_item->Attribute("Format") : nullptr;
  const char* item_type_attr = data_item != nullptr ? data_item->Attribute("ItemType") : nullptr;

  if (data_item == nullptr) {
    throw std::runtime_error("Malformed XDMF attribute: missing DataItem.");
  }

  if (format_attr != nullptr && std::string(format_attr) == "HDF") {
    const char* text = data_item->GetText();
    if (text == nullptr) {
      throw std::runtime_error("Malformed XDMF attribute: empty HDF DataItem.");
    }
    return parse_hdf5_source_ref(xdmf_path, text);
  }

  if (item_type_attr != nullptr && std::string(item_type_attr) == "HyperSlab") {
    const tinyxml2::XMLElement* slab_spec = data_item->FirstChildElement("DataItem");
    const tinyxml2::XMLElement* slab_source = slab_spec != nullptr ? slab_spec->NextSiblingElement("DataItem") : nullptr;
    if (slab_spec == nullptr || slab_source == nullptr) {
      throw std::runtime_error("Malformed XDMF HyperSlab attribute.");
    }
    const char* slab_text = slab_spec->GetText();
    if (slab_text == nullptr) {
      throw std::runtime_error("Malformed XDMF HyperSlab spec.");
    }

    auto slab_dims = parse_integer_list(slab_spec->Attribute("Dimensions") != nullptr ? slab_spec->Attribute("Dimensions")
                                                                                      : "");
    if (slab_dims.size() != 2 || slab_dims[0] < 3 || slab_dims[1] <= 0) {
      throw std::runtime_error("Unsupported XDMF HyperSlab dimensions.");
    }
    auto slab_values = parse_integer_list(slab_text);
    size_t rows = static_cast<size_t>(slab_dims[0]);
    size_t cols = static_cast<size_t>(slab_dims[1]);
    if (slab_values.size() < rows * cols) {
      throw std::runtime_error("Malformed XDMF HyperSlab values.");
    }

    Hdf5SourceRef ref = parse_xdmf_data_source(xdmf_path, slab_source);
    ref.start.resize(cols);
    ref.stride.resize(cols);
    ref.count.resize(cols);
    for (size_t c = 0; c < cols; c++) {
      ref.start[c] = static_cast<hsize_t>(slab_values[c]);
      ref.stride[c] = static_cast<hsize_t>(slab_values[cols + c]);
      ref.count[c] = static_cast<hsize_t>(slab_values[2 * cols + c]);
    }
    return ref;
  }

  throw std::runtime_error("Unsupported XDMF DataItem format.");
}

static XdmfGridInfo parse_xdmf_grid_info(const std::string& xdmf_file) {
  tinyxml2::XMLDocument doc;
  if (doc.LoadFile(xdmf_file.c_str()) != tinyxml2::XML_SUCCESS) {
    throw std::runtime_error("Failed to parse XDMF file.");
  }

  const tinyxml2::XMLElement* root = doc.FirstChildElement("Xdmf");
  const tinyxml2::XMLElement* domain = root != nullptr ? root->FirstChildElement("Domain") : nullptr;
  const tinyxml2::XMLElement* grid = domain != nullptr ? domain->FirstChildElement("Grid") : nullptr;
  if (grid == nullptr) {
    throw std::runtime_error("Malformed XDMF file: missing Grid.");
  }

  const char* grid_type = grid->Attribute("GridType");
  if (grid_type != nullptr && std::string(grid_type) == "Collection") {
    grid = grid->FirstChildElement("Grid");
  }
  if (grid == nullptr) {
    throw std::runtime_error("Malformed XDMF file: empty Grid collection.");
  }

  const tinyxml2::XMLElement* topology = grid->FirstChildElement("Topology");
  const tinyxml2::XMLElement* geometry = grid->FirstChildElement("Geometry");
  if (topology == nullptr || geometry == nullptr) {
    throw std::runtime_error("Malformed XDMF file: missing topology or geometry.");
  }

  const char* topology_type = topology->Attribute("TopologyType");
  const char* dimensions_attr = topology->Attribute("Dimensions");
  if (topology_type == nullptr || dimensions_attr == nullptr) {
    throw std::runtime_error("Malformed XDMF topology.");
  }

  XdmfGridInfo info;
  auto topology_dims = parse_integer_list(dimensions_attr);
  std::string topo = topology_type;
  if (topo == "3DCoRectMesh") {
    if (topology_dims.size() != 3) {
      throw std::runtime_error("Malformed 3D XDMF topology dimensions.");
    }
    info.is_3d = true;
    info.cell_dims = viskores::Id3(static_cast<viskores::Id>(topology_dims[2] - 1),
                                   static_cast<viskores::Id>(topology_dims[1] - 1),
                                   static_cast<viskores::Id>(topology_dims[0] - 1));
  } else {
    throw std::runtime_error("Only 3D XDMF rectilinear cell data is supported.");
  }

  const char* geometry_type = geometry->Attribute("GeometryType");
  if (geometry_type == nullptr || std::string(geometry_type) != "ORIGIN_DXDYDZ") {
    throw std::runtime_error("Only ORIGIN_DXDYDZ XDMF geometry is supported.");
  }

  const tinyxml2::XMLElement* origin_item = geometry->FirstChildElement("DataItem");
  const tinyxml2::XMLElement* spacing_item =
      origin_item != nullptr ? origin_item->NextSiblingElement("DataItem") : nullptr;
  if (origin_item == nullptr || spacing_item == nullptr || origin_item->GetText() == nullptr ||
      spacing_item->GetText() == nullptr) {
    throw std::runtime_error("Malformed XDMF geometry.");
  }

  auto origin_vals = parse_double_list(origin_item->GetText());
  auto spacing_vals = parse_double_list(spacing_item->GetText());
  if (origin_vals.size() != 3 || spacing_vals.size() != 3) {
    throw std::runtime_error("Malformed XDMF geometry values.");
  }
  info.origin = viskores::Vec3f(static_cast<float>(origin_vals[2]), static_cast<float>(origin_vals[1]),
                                static_cast<float>(origin_vals[0]));
  info.spacing = viskores::Vec3f(static_cast<float>(spacing_vals[2]), static_cast<float>(spacing_vals[1]),
                                 static_cast<float>(spacing_vals[0]));

  std::filesystem::path xdmf_path = std::filesystem::absolute(xdmf_file);
  for (const tinyxml2::XMLElement* attr = grid->FirstChildElement("Attribute"); attr != nullptr;
       attr = attr->NextSiblingElement("Attribute")) {
    const char* center = attr->Attribute("Center");
    const char* name = attr->Attribute("Name");
    if (center == nullptr || name == nullptr || std::string(center) != "Cell") {
      continue;
    }
    const tinyxml2::XMLElement* data_item = attr->FirstChildElement("DataItem");
    XdmfAttributeRef field;
    field.name = name;
    field.source = parse_xdmf_data_source(xdmf_path, data_item);
    info.attributes.push_back(std::move(field));
  }

  return info;
}

static std::vector<float> read_hdf5_field_as_float(const Hdf5SourceRef& source) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

  H5Handle file(H5Fopen(source.file_path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (!file) {
    throw std::runtime_error("Failed to open HDF5 file: " + source.file_path.string());
  }
  H5Handle dataset(H5Dopen2(file.get(), source.dataset_path.c_str(), H5P_DEFAULT), H5Dclose);
  if (!dataset) {
    throw std::runtime_error("Failed to open HDF5 dataset: " + source.dataset_path);
  }
  H5Handle datatype(H5Dget_type(dataset.get()), H5Tclose);
  if (!datatype) {
    throw std::runtime_error("Failed to inspect HDF5 dataset type.");
  }
  H5T_class_t type_class = H5Tget_class(datatype.get());
  if (type_class != H5T_INTEGER && type_class != H5T_FLOAT) {
    throw std::runtime_error("Only numeric HDF5 datasets are supported.");
  }

  H5Handle file_space(H5Dget_space(dataset.get()), H5Sclose);
  if (!file_space) {
    throw std::runtime_error("Failed to inspect HDF5 dataset dataspace.");
  }
  int rank = H5Sget_simple_extent_ndims(file_space.get());
  if (rank <= 0) {
    throw std::runtime_error("Invalid HDF5 dataset rank.");
  }
  std::vector<hsize_t> dims(static_cast<size_t>(rank), 0);
  if (H5Sget_simple_extent_dims(file_space.get(), dims.data(), nullptr) < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset dimensions.");
  }

  std::vector<hsize_t> mem_dims = dims;
  if (!source.count.empty()) {
    if (source.start.size() != static_cast<size_t>(rank) || source.stride.size() != static_cast<size_t>(rank) ||
        source.count.size() != static_cast<size_t>(rank)) {
      throw std::runtime_error("XDMF HyperSlab rank does not match HDF5 dataset rank.");
    }
    if (H5Sselect_hyperslab(file_space.get(), H5S_SELECT_SET, source.start.data(), source.stride.data(),
                            source.count.data(), nullptr) < 0) {
      throw std::runtime_error("Failed to select HDF5 hyperslab.");
    }
    mem_dims = source.count;
  }

  size_t num_values = 1;
  for (hsize_t dim : mem_dims) {
    num_values *= static_cast<size_t>(dim);
  }
  if (num_values == 0) {
    return {};
  }

  H5Handle mem_space(H5Screate_simple(static_cast<int>(mem_dims.size()), mem_dims.data(), nullptr), H5Sclose);
  if (!mem_space) {
    throw std::runtime_error("Failed to create HDF5 memory dataspace.");
  }

  std::vector<float> values(num_values, 0.0f);
  if (H5Dread(dataset.get(), H5T_NATIVE_FLOAT, mem_space.get(), file_space.get(), H5P_DEFAULT, values.data()) < 0) {
    throw std::runtime_error("Failed to read HDF5 dataset.");
  }
  return values;
}

static viskores::cont::DataSet read_xdmf_dataset_impl(const std::string& xdmf_file) {
  XdmfGridInfo info = parse_xdmf_grid_info(xdmf_file);
  if (!info.is_3d) {
    throw std::runtime_error("Only 3D XDMF datasets are supported.");
  }

  viskores::Id nx = info.cell_dims[0];
  viskores::Id ny = info.cell_dims[1];
  viskores::Id nz = info.cell_dims[2];
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::runtime_error("Invalid XDMF cell dimensions.");
  }

  viskores::cont::DataSet dataset =
      viskores::cont::DataSetBuilderUniform::Create(viskores::Id3(nx + 1, ny + 1, nz + 1), info.origin, info.spacing,
                                                    "coords");
  viskores::Id expected_num_cells = nx * ny * nz;
  for (const XdmfAttributeRef& field : info.attributes) {
    std::vector<float> values = read_hdf5_field_as_float(field.source);
    if (static_cast<viskores::Id>(values.size()) != expected_num_cells) {
      std::ostringstream oss;
      oss << "Field '" << field.name << "' size mismatch. Expected " << expected_num_cells << " values, got "
          << values.size() << ".";
      throw std::runtime_error(oss.str());
    }

    viskores::cont::ArrayHandle<float> array;
    array.Allocate(expected_num_cells);
    auto portal = array.WritePortal();
    for (viskores::Id i = 0; i < expected_num_cells; i++) {
      portal.Set(i, values[static_cast<size_t>(i)]);
    }
    dataset.AddField(viskores::cont::Field(field.name, viskores::cont::Field::Association::Cells, array));
  }

  return dataset;
}

static DatasetFileInfo inspect_xdmf_file(const std::string& xdmf_file) {
  XdmfGridInfo info = parse_xdmf_grid_info(xdmf_file);
  DatasetFileInfo out;
  out.is_3d = info.is_3d;
  out.all_field_names.reserve(info.attributes.size());
  out.scalar_cell_field_names.reserve(info.attributes.size());
  for (const XdmfAttributeRef& field : info.attributes) {
    out.all_field_names.push_back(field.name);
    out.scalar_cell_field_names.push_back(field.name);
  }
  return out;
}

static DatasetFileInfo inspect_vtk_file(const std::string& vtk_file) {
  DatasetFileInfo out;
  viskores::cont::DataSet ds = read_vtk_dataset_with_compat_impl(vtk_file);
  auto cell_set = ds.GetCellSet();
  out.is_3d = !(cell_set.CanConvert<viskores::cont::CellSetStructured<1>>() ||
                cell_set.CanConvert<viskores::cont::CellSetStructured<2>>());
  for (viskores::IdComponent i = 0; i < ds.GetNumberOfFields(); i++) {
    const auto& field = ds.GetField(i);
    if (field.IsPointField() || field.IsCellField()) {
      out.all_field_names.push_back(field.GetName());
    }
    if (field.IsCellField() && field.GetData().GetNumberOfComponentsFlat() == 1) {
      out.scalar_cell_field_names.push_back(field.GetName());
    }
  }
  return out;
}

}  // namespace

bool is_supported_dataset_file(const std::filesystem::path& path) {
  if (!path.has_extension()) {
    return false;
  }
  std::string ext = lowercase_copy(path.extension().string());
  return ext == ".vtk" || ext == ".xdmf";
}

bool is_supported_dataset_sequence_entry(const std::filesystem::path& path) {
  if (!is_supported_dataset_file(path)) {
    return false;
  }
  if (lowercase_copy(path.extension().string()) == ".xdmf" &&
      ends_with_case_insensitive(path.filename().string(), "_series.xdmf")) {
    return false;
  }
  return true;
}

DatasetFileInfo inspect_dataset_file(const std::string& path) {
  std::filesystem::path fs_path(path);
  std::string ext = lowercase_copy(fs_path.extension().string());
  if (ext == ".vtk") {
    return inspect_vtk_file(path);
  }
  if (ext == ".xdmf") {
    return inspect_xdmf_file(path);
  }
  throw std::runtime_error("Unsupported dataset format.");
}

viskores::cont::DataSet read_dataset_with_compat(const std::string& path) {
  std::filesystem::path fs_path(path);
  std::string ext = lowercase_copy(fs_path.extension().string());
  if (ext == ".vtk") {
    return read_vtk_dataset_with_compat_impl(path);
  }
  if (ext == ".xdmf") {
    return read_xdmf_dataset_impl(path);
  }
  throw std::runtime_error("Unsupported dataset format.");
}
