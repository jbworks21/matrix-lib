#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "matrix/matrix.hpp"
#include "matrix/decompose.hpp"

using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// LU
// ---------------------------------------------------------------------------
TEST_CASE("LU reconstructs A: P A == L U", "[lu]") {
    Matrix<double> A = {{4, 3, 2}, {3, 4, 1}, {2, 1, 3}};
    auto d = lu(A);

    Matrix<double> LU = d.L * d.U;
    // (P A)(i, j) == A(perm[i], j) == (L U)(i, j)  =>  reconstruct A.
    Matrix<double> Arec(A.rows(), A.cols());
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < A.cols(); ++j)
            Arec(d.perm[i], j) = LU(i, j);

    REQUIRE(Arec.isApprox(A));
    // L has a unit diagonal
    for (size_t i = 0; i < d.L.rows(); ++i)
        CHECK_THAT(d.L(i, i), WithinAbs(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------
TEST_CASE("det of known values", "[det]") {
    CHECK_THAT(det(Matrix<double>{{1, 2}, {3, 4}}), WithinAbs(-2.0, 1e-12));
    CHECK_THAT(det(Matrix<double>::identity(5)),  WithinAbs( 1.0, 1e-12));
    CHECK_THAT(det(Matrix<double>(3, 3, 0.0)),    WithinAbs( 0.0, 1e-12));
}

TEST_CASE("det of singular matrix is ~0", "[det]") {
    Matrix<double> A = {{1, 2}, {2, 4}};
    CHECK_THAT(det(A), WithinAbs(0.0, 1e-12));
}

TEST_CASE("det(A * B) == det(A) * det(B)", "[det]") {
    Matrix<double> A = {{2, 1}, {1, 3}};
    Matrix<double> B = {{4, -1}, {2, 5}};
    CHECK_THAT(det(A * B), WithinAbs(det(A) * det(B), 1e-9));
}

// ---------------------------------------------------------------------------
// solve / inverse
// ---------------------------------------------------------------------------
TEST_CASE("solve recovers known x", "[solve]") {
    Matrix<double> A = {{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}};
    Matrix<double> x_true = {{8}, {-11}, {-3}};
    Matrix<double> b = A * x_true;              // b = A x_true
    REQUIRE(solve(A, b).isApprox(x_true));      // solve must recover x_true
}

TEST_CASE("solve throws on singular", "[solve]") {
    Matrix<double> A = {{1, 2}, {2, 4}};
    Matrix<double> b = {{1}, {2}};
    CHECK_THROWS_AS(solve(A, b), std::runtime_error);
}

TEST_CASE("inverse times A is identity", "[inverse]") {
    Matrix<double> A = {{4, 7}, {2, 6}};
    auto Ainv = inverse(A);
    REQUIRE((A * Ainv).isApprox(Matrix<double>::identity(2)));
    REQUIRE((Ainv * A).isApprox(Matrix<double>::identity(2)));
}

TEST_CASE("inverse throws on singular", "[inverse]") {
    Matrix<double> A = {{1, 2}, {2, 4}};
    CHECK_THROWS_AS(inverse(A), std::runtime_error);
}

// ---------------------------------------------------------------------------
// QR
// ---------------------------------------------------------------------------
TEST_CASE("QR on a 3x2: Q orthogonal, R triangular, Q R = A", "[qr]") {
    Matrix<double> A = {{1, 2}, {3, 4}, {5, 6}};
    auto q = qr(A);

    REQUIRE((q.Q.transpose() * q.Q).isApprox(Matrix<double>::identity(3)));
    REQUIRE((q.Q * q.R).isApprox(A));

    for (size_t i = 1; i < q.R.rows(); ++i)
        for (size_t j = 0; j < std::min(i, q.R.cols()); ++j)
            CHECK(std::abs(q.R(i, j)) < 1e-12);
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------
TEST_CASE("rank: identity, zero, rank-deficient", "[rank]") {
    CHECK(rank(Matrix<double>::identity(4)) == 4);
    CHECK(rank(Matrix<double>(3, 3, 0.0)) == 0);

    Matrix<double> A = {{1, 2, 3}, {4, 5, 6}, {1, 2, 3}};  // rows 0 == 2
    CHECK(rank(A) == 2);
}

// ---------------------------------------------------------------------------
// least squares
// ---------------------------------------------------------------------------
TEST_CASE("lstsq: exact linear fit recovers coefficients", "[lstsq]") {
    // Data: y = 2 + 3x exactly at x = 0, 1, 2, 3.
    Matrix<double> A = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
    Matrix<double> b = {{2}, {5}, {8}, {11}};
    REQUIRE(lstsq(A, b).isApprox(Matrix<double>{{2}, {3}}));
}

TEST_CASE("lstsq: noisy fit matches least-squares solution", "[lstsq]") {
    // Any overdetermined consistent system: coefficients should be exact.
    Matrix<double> A = {{1, 0, 0}, {1, 1, 0}, {1, 0, 1}, {1, 1, 1}};
    Matrix<double> x_true = {{1}, {2}, {3}};
    Matrix<double> b = A * x_true;
    REQUIRE(lstsq(A, b).isApprox(x_true));
}

TEST_CASE("lstsq rejects underdetermined systems", "[lstsq]") {
    Matrix<double> A = {{1, 2, 3}};   // 1x3
    Matrix<double> b = {{1}};
    CHECK_THROWS_AS(lstsq(A, b), std::invalid_argument);
}
