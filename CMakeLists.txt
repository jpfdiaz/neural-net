cmake_minimum_required(VERSION 3.20)
project(neural-net VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

option(ENABLE_COVERAGE "Build with coverage instrumentation" OFF)
include(cmake/Coverage.cmake)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(demos)