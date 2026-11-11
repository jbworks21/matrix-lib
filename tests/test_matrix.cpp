#include <catch2/catch_test_macros.hpp>
#include "matrix/matrix.hpp"

TEST_CASE("Matrix multiplication works", "[matmul]") {
    Matrix<double> A = {{1, 2}, {3, 4}};
    Matrix<double> B = {{5, 6}, {7, 8}};
    auto C = A * B;

    CHECK(C(0, 0) == 19);
    CHECK(C(0, 1) == 22);
    CHECK(C(1, 0) == 43);
    CHECK(C(1, 1) == 50);
}

TEST_CASE("Dimension mismatch throws", "[matmul]") {
    Matrix<double> A(2, 3);
    Matrix<double> B(2, 2);
    // Use CHECK_THROWS to verify the exception logic
    CHECK_THROWS_AS(A * B, std::invalid_argument);
}
