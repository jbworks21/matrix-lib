cmake_minimum_required(VERSION 3.16)
project(MatrixLib LANGUAGES CXX VERSION 0.3.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -Wpedantic)

# Header-only library target
add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tests
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests
    tests/test_matrix.cpp
    tests/test_core.cpp
    tests/test_decompose.cpp
)
target_link_libraries(tests PRIVATE matrix Catch2::Catch2WithMain)

# Runnable examples
add_executable(tier2_demo examples/tier2_demo.cpp)
target_link_libraries(tier2_demo PRIVATE matrix)
