cmake_minimum_required(VERSION 3.15)
project(CaustiX LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep IPO/LTO off by default; global IPO causes unresolved Viskores symbols in this tree.
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE OFF)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO OFF)
option(CAUSTIX_ENABLE_IPO "Enable IPO/LTO for the CaustiX executable" OFF)

if (UNIX)
	find_package(PkgConfig QUIET)
	set(CAUSTIX_HAS_WAYLAND OFF)
	set(CAUSTIX_HAS_X11 OFF)

	if (PkgConfig_FOUND)
		pkg_check_modules(CAUSTIX_WAYLAND QUIET wayland-client)
		pkg_check_modules(CAUSTIX_X11 QUIET x11)
		if (CAUSTIX_WAYLAND_FOUND)
			set(CAUSTIX_HAS_WAYLAND ON)
		endif()
		if (CAUSTIX_X11_FOUND)
			set(CAUSTIX_HAS_X11 ON)
		endif()
	endif()

	if (CAUSTIX_HAS_WAYLAND)
		set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
		set(SDL_X11 OFF CACHE BOOL "" FORCE)
		if (CAUSTIX_HAS_X11)
			message(STATUS "Both Wayland and X11 detected; preferring Wayland.")
		else()
			message(STATUS "Wayland detected; enabling Wayland backend.")
		endif()
	elseif (CAUSTIX_HAS_X11)
		set(SDL_WAYLAND OFF CACHE BOOL "" FORCE)
		set(SDL_X11 ON CACHE BOOL "" FORCE)
		message(STATUS "Wayland not detected; falling back to X11 backend.")
	else()
		# Keep Wayland as the preferred default when desktop stack detection is unavailable.
		set(SDL_WAYLAND ON CACHE BOOL "" FORCE)
		set(SDL_X11 OFF CACHE BOOL "" FORCE)
		message(STATUS "Could not detect Wayland/X11 via pkg-config; defaulting to Wayland.")
	endif()

	set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
	set(SDL_X11_XSCRNSAVER OFF CACHE BOOL "" FORCE)
	set(SDL_X11_SCRNSAVER OFF CACHE BOOL "" FORCE)
	set(SDL_VULKAN ON CACHE BOOL "" FORCE)
endif()

set(SDL_AUDIO OFF)
set(SDL_HAPTIC OFF)
set(SDL_HIDAPI OFF)
set(SDL_SENSOR OFF)
set(SDL_CAMERA OFF)
set(SDL_DIALOG OFF)
set(SDL_JOYSTICK OFF)

set(CAUSTIX_CUDA_ARCH "75" CACHE STRING "CUDA compute capability for OptiX PTX (e.g. 75, 86, 89)")

find_package(OpenMP QUIET)
set(CAUSTIX_VISKORES_OPENMP_DEFAULT OFF)
if (OpenMP_CXX_FOUND)
	set(CAUSTIX_VISKORES_OPENMP_DEFAULT ON)
endif()
option(CAUSTIX_ENABLE_VISKORES_OPENMP "Enable OpenMP backend for Viskores filters" ${CAUSTIX_VISKORES_OPENMP_DEFAULT})

set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/lib/imgui)

add_executable(${CMAKE_PROJECT_NAME}
	src/main.cpp
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMGUI_DIR}/imgui_demo.cpp
	${IMGUI_DIR}/backends/imgui_impl_sdl3.cpp
	${IMGUI_DIR}/backends/imgui_impl_sdlrenderer3.cpp
	${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog/ImGuiFileDialog.cpp
)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall)
if (CAUSTIX_ENABLE_IPO)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT CAUSTIX_IPO_SUPPORTED OUTPUT CAUSTIX_IPO_ERROR)
	if (CAUSTIX_IPO_SUPPORTED)
		set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
		set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
	else()
		message(WARNING "CAUSTIX_ENABLE_IPO=ON but IPO is not supported: ${CAUSTIX_IPO_ERROR}")
	endif()
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/SDL)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
	lib/SDL/include
	${IMGUI_DIR}
	${IMGUI_DIR}/backends
	${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog
)
set(Viskores_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_RENDERING OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_TUTORIALS OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_DOCUMENTATION OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_MPI OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_KOKKOS OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_TBB OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_OPENMP ${CAUSTIX_ENABLE_VISKORES_OPENMP} CACHE BOOL "" FORCE)
set(Viskores_ENABLE_LOGGING OFF CACHE BOOL "" FORCE)
set(Viskores_ENABLE_DEVELOPER_FLAGS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/viskores)

find_package(CUDAToolkit REQUIRED)
set(OptiX_INSTALL_DIR /opt/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64)

# Compile shaders.cu -> shaders.ptx via nvcc
set(SHADER_SRC ${CMAKE_SOURCE_DIR}/src/shaders.cu)
set(SHADER_PTX ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders.ptx)
add_custom_command(
	OUTPUT ${SHADER_PTX}
	COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -ptx
		-I${OptiX_INSTALL_DIR}/include
		-I${CMAKE_SOURCE_DIR}/src
		--gpu-architecture=compute_${CAUSTIX_CUDA_ARCH}
		-o ${SHADER_PTX}
		${SHADER_SRC}
	DEPENDS ${SHADER_SRC} ${CMAKE_SOURCE_DIR}/src/optix_params.h
	COMMENT "Compiling OptiX shaders to PTX"
)
add_custom_target(shaders ALL DEPENDS ${SHADER_PTX})
add_dependencies(${CMAKE_PROJECT_NAME} shaders)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
	${OptiX_INSTALL_DIR}/include
)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
	SDL3::SDL3
	viskores_io viskores_cont
	viskores_filter_contour
	viskores_filter_field_conversion
	CUDA::cudart
)
