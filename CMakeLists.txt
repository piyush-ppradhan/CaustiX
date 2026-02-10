cmake_minimum_required(VERSION 3.12)
project(CaustiX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_CXX_FLAGS "-O3 -Wall")
else ()
	message(WARNING "CMAKE_BUILD_TYPE not set, using debug.")
	set(CMAKE_BUILD_TYPE "Debug")
	set(CMAKE_CXX_FLAGS "-O2 -Wall")
endif()

if (UNIX)
	set(SDL_WAYLAND ON)
	set(SDL_X11 OFF)
	set(SDL_VULKAN ON)
endif()

set(SDL_AUDIO OFF)
set(SDL_HAPTIC OFF)
set(SDL_HIDAPI OFF)
set(SDL_SENSOR OFF)
set(SDL_CAMERA OFF)
set(SDL_DIALOG OFF)
set(SDL_JOYSTICK OFF)

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

add_subdirectory(${CMAKE_SOURCE_DIR}/lib/SDL)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
	lib/SDL/include
	${IMGUI_DIR}
	${IMGUI_DIR}/backends
	${CMAKE_SOURCE_DIR}/lib/ImGuiFileDialog
)
set(VTK_BUILD_ALL_MODULES OFF)
set(VTK_BUILD_TESTING OFF)
set(VTK_ENABLE_WRAPPING OFF)
set(VTK_GROUP_ENABLE_Rendering "NO")
set(VTK_GROUP_ENABLE_StandAlone "DEFAULT")
set(VTK_GROUP_ENABLE_Tk "NO")
set(VTK_GROUP_ENABLE_MPI "NO")
set(VTK_GROUP_ENABLE_Web "NO")
set(VTK_GROUP_ENABLE_Qt "NO")
set(VTK_BUILD_EXAMPLES OFF)
set(VTK_BUILD_DOCUMENTATION OFF)
set(VTK_MODULE_ENABLE_VTK_IOLegacy "YES")
set(VTK_MODULE_ENABLE_VTK_CommonDataModel "YES")
add_subdirectory(${CMAKE_SOURCE_DIR}/lib/vtk)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE SDL3::SDL3 VTK::IOLegacy VTK::CommonDataModel)
