cmake_minimum_required(VERSION 3.16)
project(MatrixLib LANGUAGES CXX VERSION 0.4.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

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

# Examples
add_executable(tier2_demo examples/tier2_demo.cpp)
target_link_libraries(tier2_demo PRIVATE matrix)

# Benchmarks — need Threads::Threads for std::thread
find_package(Threads REQUIRED)
add_executable(bench benchmarks/bench.cpp)
target_link_libraries(bench PRIVATE matrix Threads::Threads)
