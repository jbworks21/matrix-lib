cmake_minimum_required(VERSION 3.16)
project(MatrixLib LANGUAGES CXX VERSION 0.1.0)

# Use C++17 for a balance of modern features and Termux compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enforce strict warnings to catch bugs early
add_compile_options(-Wall -Wextra -Wpedantic)

# --- Header-only library target ---
add_library(matrix INTERFACE)
target_include_directories(matrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Testing (Catch2) ---
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.2 # Pin a specific version for reproducibility
)
FetchContent_MakeAvailable(Catch2)

add_executable(tests tests/test_matrix.cpp)
target_link_libraries(tests PRIVATE matrix Catch2::Catch2WithMain)
