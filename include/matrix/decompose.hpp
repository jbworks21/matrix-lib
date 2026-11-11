#pragma once
#include "matrix.hpp"          // lives in the same directory
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

// ============================================================================
// LU decomposition with partial pivoting:  P A = L U
// ============================================================================
// P is a permutation (stored as a row-index vector perm with
//   (P A)(i, j) == A(perm[i], j)).
// L is unit lower triangular (1s on the diagonal).
// U is upper triangular.
// `sign` is (-1)^(number of row swaps), which is det(P).
// `singular` is true if any pivot was numerically zero.
template <typename T>
struct LUDecomp {
    Matrix<T> L;
    Matrix<T> U;
    std::vector<size_t> perm;
    int  sign     = 1;
    bool singular = false;
};

template <typename T>
LUDecomp<T> lu(const Matrix<T>& A) {
    const size_t n = A.rows();
    if (A.cols() != n) throw std::invalid_argument("lu: requires square matrix");

    Matrix<T> U = A;
    Matrix<T> L = Matrix<T>::identity(n);
    std::vector<size_t> perm(n);
    for (size_t i = 0; i < n; ++i) perm[i] = i;

    int  sign     = 1;
    bool singular = false;
    // Type-adaptive pivot threshold. Not scale-invariant — if you ever feed
    // in matrices with wildly different magnitudes, pass an explicit tol.
    const T tol = std::sqrt(std::numeric_limits<T>::epsilon());

    for (size_t k = 0; k < n; ++k) {
        // --- Partial pivoting: find row with largest |A(i, k)| for i >= k ---
        size_t pivot   = k;
        T      max_abs = std::abs(U(k, k));
        for (size_t i = k + 1; i < n; ++i) {
            const T v = std::abs(U(i, k));
            if (v > max_abs) { max_abs = v; pivot = i; }
        }
        if (max_abs <= tol) { singular = true; continue; }

        if (pivot != k) {
            for (size_t j = 0; j < n; ++j) std::swap(U(k, j), U(pivot, j));
            for (size_t j = 0; j < k; ++j) std::swap(L(k, j), L(pivot, j));  // keep L consistent with P
            std::swap(perm[k], perm[pivot]);
            sign = -sign;
        }

        // --- Eliminate below the diagonal ---
        const T inv_pivot = T{1} / U(k, k);
        for (size_t i = k + 1; i < n; ++i) {
            const T factor = U(i, k) * inv_pivot;
            L(i, k) = factor;
            U(i, k) = T{};
            for (size_t j = k + 1; j < n; ++j) U(i, j) -= factor * U(k, j);
        }
    }

    return {std::move(L), std::move(U), std::move(perm), sign, singular};
}

// Determinant via LU: det(A) = det(P^T) * det(L) * det(U) = sign * prod(diag(U)).
// det(L) = 1 because L is unit triangular.
template <typename T>
T det(const Matrix<T>& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("det: requires square matrix");
    auto d = lu(A);
    T result = static_cast<T>(d.sign);
    for (size_t i = 0; i < A.rows(); ++i) result *= d.U(i, i);
    return result;
}

// Solve A x = b for a single right-hand side column vector.
// Costs one LU + two triangular solves; do NOT compute inverse()*b.
template <typename T>
Matrix<T> solve(const Matrix<T>& A, const Matrix<T>& b) {
    const size_t n = A.rows();
    if (A.cols() != n)                 throw std::invalid_argument("solve: A must be square");
    if (b.rows() != n || b.cols() != 1) throw std::invalid_argument("solve: b must be n x 1");

    auto d = lu(A);
    if (d.singular) throw std::runtime_error("solve: matrix is singular");

    // y = P b
    std::vector<T> y(n);
    for (size_t i = 0; i < n; ++i) y[i] = b(d.perm[i], 0);

    // Forward substitution: L z = y  (in place; y[j] for j < i is already z[j])
    for (size_t i = 0; i < n; ++i) {
        T s = y[i];
        for (size_t j = 0; j < i; ++j) s -= d.L(i, j) * y[j];
        y[i] = s;
    }

    // Back substitution: U x = z
    std::vector<T> x(n);
    for (size_t i = n; i-- > 0; ) {
        T s = y[i];
        for (size_t j = i + 1; j < n; ++j) s -= d.U(i, j) * x[j];
        if (std::abs(d.U(i, i)) <= std::numeric_limits<T>::epsilon())
            throw std::runtime_error("solve: zero pivot in back substitution");
        x[i] = s / d.U(i, i);
    }

    Matrix<T> result(n, 1);
    for (size_t i = 0; i < n; ++i) result(i, 0) = x[i];
    return result;
}

// Inverse via LU: factor once, then solve A x = e_col for each column of I.
template <typename T>
Matrix<T> inverse(const Matrix<T>& A) {
    const size_t n = A.rows();
    if (A.cols() != n) throw std::invalid_argument("inverse: requires square matrix");

    auto d = lu(A);
    if (d.singular) throw std::runtime_error("inverse: matrix is singular");

    Matrix<T> inv(n, n);
    std::vector<T> y(n), x(n);

    for (size_t col = 0; col < n; ++col) {
        // y = P e_col, i.e. y[i] = 1 iff the original row `col` landed at row i.
        for (size_t i = 0; i < n; ++i) y[i] = (d.perm[i] == col) ? T{1} : T{};

        for (size_t i = 0; i < n; ++i) {
            T s = y[i];
            for (size_t j = 0; j < i; ++j) s -= d.L(i, j) * y[j];
            y[i] = s;
        }
        for (size_t i = n; i-- > 0; ) {
            T s = y[i];
            for (size_t j = i + 1; j < n; ++j) s -= d.U(i, j) * x[j];
            x[i] = s / d.U(i, i);
        }
        for (size_t i = 0; i < n; ++i) inv(i, col) = x[i];
    }
    return inv;
}

// ============================================================================
// QR decomposition via Householder reflections:  A = Q R
// ============================================================================
// Q is m x m orthogonal (Q^T Q = I), R is m x n upper triangular.
// (This is the "full" QR; the bottom m-n rows of R are ~0 when m > n.)
template <typename T>
struct QRDecomp {
    Matrix<T> Q;
    Matrix<T> R;
};

template <typename T>
QRDecomp<T> qr(const Matrix<T>& A) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    Matrix<T> R = A;
    Matrix<T> Q = Matrix<T>::identity(m);
    const size_t steps = std::min(m, n);

    std::vector<T> v(m);   // workspace: Householder vector for the current step

    for (size_t k = 0; k < steps; ++k) {
        const size_t len = m - k;

        // x = R[k:m, k]
        T x_norm_sq = T{};
        for (size_t i = 0; i < len; ++i) {
            v[i] = R(k + i, k);
            x_norm_sq += v[i] * v[i];
        }
        const T x_norm = std::sqrt(x_norm_sq);
        if (x_norm <= std::numeric_limits<T>::epsilon()) continue;  // column already zero

        // Reflect x onto the -sign(x0)*||x|| axis. Choosing the sign this way
        // prevents catastrophic cancellation when x[0] is close to +||x||.
        const T alpha = (v[0] >= T{}) ? -x_norm : x_norm;
        v[0] -= alpha;

        // Normalise the Householder vector so H = I - 2 v v^T is orthogonal.
        T v_norm_sq = T{};
        for (size_t i = 0; i < len; ++i) v_norm_sq += v[i] * v[i];
        if (v_norm_sq <= std::numeric_limits<T>::epsilon()) continue;
        const T inv_norm = T{1} / std::sqrt(v_norm_sq);
        for (size_t i = 0; i < len; ++i) v[i] *= inv_norm;

        // Apply H from the LEFT to R: R[k:m, :] -= 2 v (v^T R[k:m, :])
        for (size_t j = 0; j < n; ++j) {
            T dot = T{};
            for (size_t i = 0; i < len; ++i) dot += v[i] * R(k + i, j);
            const T factor = T{2} * dot;
            for (size_t i = 0; i < len; ++i) R(k + i, j) -= factor * v[i];
        }

        // Apply H from the RIGHT to Q: Q[:, k:m] -= 2 (Q[:, k:m] v) v^T
        // (Right-multiplication accumulates Q = H_0 H_1 ... H_{k}.)
        for (size_t i = 0; i < m; ++i) {
            T dot = T{};
            for (size_t j = 0; j < len; ++j) dot += Q(i, k + j) * v[j];
            const T factor = T{2} * dot;
            for (size_t j = 0; j < len; ++j) Q(i, k + j) -= factor * v[j];
        }
    }

    // Numerical dust below the diagonal — zero it for a clean upper triangular R.
    for (size_t i = 1; i < m; ++i)
        for (size_t j = 0; j < std::min(i, n); ++j)
            R(i, j) = T{};

    return {std::move(Q), std::move(R)};
}

// Numerical rank via QR diagonal: count diag entries of R that are
// "big enough" relative to the largest one.
template <typename T>
size_t rank(const Matrix<T>& A,
            T tol = std::sqrt(std::numeric_limits<T>::epsilon())) {
    if (A.empty()) return 0;
    const auto q = qr(A);
    const size_t steps = std::min(A.rows(), A.cols());

    T max_diag = T{};
    for (size_t i = 0; i < steps; ++i)
        max_diag = std::max(max_diag, std::abs(q.R(i, i)));

    const T threshold = tol * std::max<T>(T{1}, max_diag);
    size_t r = 0;
    for (size_t i = 0; i < steps; ++i)
        if (std::abs(q.R(i, i)) > threshold) ++r;
    return r;
}

// Least-squares: minimise ||A x - b||_2 for an m x n matrix with m >= n.
// Reduction: A = QR  =>  ||A x - b|| = ||R x - Q^T b||  =>  solve Rx = Q^T b
// using only the top n rows of R (square, triangular).
template <typename T>
Matrix<T> lstsq(const Matrix<T>& A, const Matrix<T>& b) {
    const size_t m = A.rows();
    const size_t n = A.cols();
    if (m < n)                         throw std::invalid_argument("lstsq: requires m >= n");
    if (b.rows() != m || b.cols() != 1) throw std::invalid_argument("lstsq: b must be m x 1");

    const auto q = qr(A);

    // y = Q^T b  (first n entries are enough)
    std::vector<T> y(n);
    for (size_t i = 0; i < n; ++i) {
        T s = T{};
        for (size_t k = 0; k < m; ++k) s += q.Q(k, i) * b(k, 0);
        y[i] = s;
    }

    // Back-substitute R[0:n, 0:n] x = y
    std::vector<T> x(n);
    for (size_t i = n; i-- > 0; ) {
        T s = y[i];
        for (size_t j = i + 1; j < n; ++j) s -= q.R(i, j) * x[j];
        if (std::abs(q.R(i, i)) <= std::numeric_limits<T>::epsilon())
            throw std::runtime_error("lstsq: A is rank deficient");
        x[i] = s / q.R(i, i);
    }

    Matrix<T> result(n, 1);
    for (size_t i = 0; i < n; ++i) result(i, 0) = x[i];
    return result;
}
