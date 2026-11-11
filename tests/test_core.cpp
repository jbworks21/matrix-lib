#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "matrix/matrix.hpp"

using Catch::Matchers::WithinAbs;

TEST_CASE("transpose", "[core]") {
    Matrix<double> A = {{1, 2, 3}, {4, 5, 6}};  // 2x3
    auto T = A.transpose();
    REQUIRE(T.rows() == 3);
    REQUIRE(T.cols() == 2);
    CHECK(T(0, 0) == 1);
    CHECK(T(2, 1) == 6);
    REQUIRE(T.transpose().isApprox(A));         // double transpose == identity
}

TEST_CASE("element-wise arithmetic", "[core]") {
    Matrix<double> A = {{1, 2}, {3, 4}};
    Matrix<double> B = {{10, 20}, {30, 40}};

    REQUIRE((A + B).isApprox(Matrix<double>{{11, 22}, {33, 44}}));
    REQUIRE((B - A).isApprox(Matrix<double>{{9, 18}, {27, 36}}));
    REQUIRE((-A).isApprox(Matrix<double>{{-1, -2}, {-3, -4}}));
    REQUIRE((2.0 * A).isApprox(Matrix<double>{{2, 4}, {6, 8}}));
    REQUIRE((A * 2.0).isApprox(Matrix<double>{{2, 4}, {6, 8}}));
    REQUIRE((B / 10.0).isApprox(A));
}

TEST_CASE("trace and norms", "[core]") {
    Matrix<double> A = {{1, 2}, {3, 4}};
    CHECK_THAT(A.trace(), WithinAbs(5.0, 1e-12));
    CHECK_THAT(A.norm(),  WithinAbs(std::sqrt(30.0), 1e-12));  // sqrt(1+4+9+16)
    CHECK_THAT(A.norm1(), WithinAbs(6.0, 1e-12));              // max column sum = 4+2? no: col0=1+3=4, col1=2+4=6
    CHECK_THAT(A.normInf(), WithinAbs(7.0, 1e-12));            // row0=1+2=3, row1=3+4=7
}

TEST_CASE("isApprox respects tolerance and shape", "[core]") {
    Matrix<double> A = {{1, 2}, {3, 4}};
    Matrix<double> B = {{1 + 1e-10, 2}, {3, 4}};
    Matrix<double> C = {{1, 2, 3}, {4, 5, 6}};

    CHECK(A.isApprox(B));                  // tiny difference, well within default tol
    CHECK_FALSE(A.isApprox(B, 1e-15));     // strict tol rejects it
    CHECK_FALSE(A.isApprox(C));            // shape mismatch
}

TEST_CASE("arithmetic rejects dimension mismatches", "[core]") {
    Matrix<double> A(2, 3);
    Matrix<double> B(3, 2);
    CHECK_THROWS_AS(A + B, std::invalid_argument);
    CHECK_THROWS_AS(A * A, std::invalid_argument);
    CHECK_THROWS_AS(A.trace(), std::invalid_argument);  // non-square
}

TEST_CASE("blocked matmul matches operator*", "[matmul][blocked]") {
    Matrix<double> A = {{1, 2, 3}, {4, 5, 6}};
    Matrix<double> B = {{7, 8}, {9, 10}, {11, 12}};
    REQUIRE(matmul_blocked(A, B, 2).isApprox(A * B));
    REQUIRE(matmul_blocked(A, B, 64).isApprox(A * B));
}
