cmake_minimum_required (VERSION 3.9) # CMP0069 NEW

project (PybindGrid)

add_definitions(
    -Wall
    -Wextra
    # Other flags
)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

# Look for an installation of pagmo in the system.
find_package(Pagmo REQUIRED)

find_package(Eigen3 3.2.92 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(SYSTEM ${Python_INCLUDE_DIRS})

add_subdirectory(${PROJECT_SOURCE_DIR}/pybind11)
include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)

add_library(grid SHARED ./main.cpp)

target_compile_definitions(grid PUBLIC ${PYTHON_DEFINES})
target_link_libraries(grid ${Python_LIBRARIES} ${PYTHON_LINKS})
target_link_libraries(grid Pagmo::pagmo)
set_target_properties(grid PROPERTIES PREFIX "")
