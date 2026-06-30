cmake_minimum_required(VERSION 3.15)
project(structstats LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /arch:AVX2 /fp:fast)
else()
  add_compile_options(-O3 -march=native -ffast-math)
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(structstats_core STABLE_ABI src/structstats/core.cpp)
install(TARGETS structstats_core LIBRARY DESTINATION structstats)
