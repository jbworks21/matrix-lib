#pragma once
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

template <typename T>
class Matrix {
    static_assert(std::is_arithmetic_v<T>, "Matrix<T>: T must be arithmetic");

private:
    size_t rows_, cols_;
    // Row-major: element (i, j) lives at index i * cols_ + j.
    std::vector<T> data_;

public:
    // --- Constructors --------------------------------------------------------
    Matrix() : rows_(0), cols_(0) {}

    Matrix(size_t rows, size_t cols, const T& value = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        rows_ = list.size();
        cols_ = (rows_ > 0) ? list.begin()->size() : 0;
        data_.reserve(rows_ * cols_);
        for (const auto& row : list) {
            if (row.size() != cols_)
                throw std::invalid_argument("Matrix: inconsistent row lengths");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // --- Static factories ----------------------------------------------------
    static Matrix identity(size_t n) {
        Matrix I(n, n, T{});
        for (size_t i = 0; i < n; ++i) I(i, i) = T{1};
        return I;
    }
    static Matrix zeros(size_t r, size_t c) { return Matrix(r, c, T{}); }
    static Matrix ones (size_t r, size_t c) { return Matrix(r, c, T{1}); }

    // --- Shape ---------------------------------------------------------------
    size_t rows()  const { return rows_; }
    size_t cols()  const { return cols_; }
    size_t size()  const { return rows_ * cols_; }
    bool   empty() const { return size() == 0; }

    // --- Element access ------------------------------------------------------
    // operator() instead of [][] avoids proxy objects.
    T&       operator()(size_t r, size_t c)       { return data_[r * cols_ + c]; }
    const T& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    // --- Element-wise arithmetic --------------------------------------------
    Matrix operator+(const Matrix& o) const {
        checkSameShape(o, "+");
        Matrix r(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Matrix operator-(const Matrix& o) const {
        checkSameShape(o, "-");
        Matrix r(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    Matrix operator-() const {
        Matrix r(rows_, cols_);
        for (size_t i = 0; i < data_.size(); ++i) r.data_[i] = -data_[i];
        return r;
    }
    Matrix& operator+=(const Matrix& o) { checkSameShape(o, "+="); for (size_t i = 0; i < data_.size(); ++i) data_[i] += o.data_[i]; return *this; }
    Matrix& operator-=(const Matrix& o) { checkSameShape(o, "-="); for (size_t i = 0; i < data_.size(); ++i) data_[i] -= o.data_[i]; return *this; }
    Matrix& operator*=(const T& s)      { for (auto& x : data_) x *= s; return *this; }
    Matrix& operator/=(const T& s)      { for (auto& x : data_) x /= s; return *this; }

    // --- Matrix multiply -----------------------------------------------------
    Matrix operator*(const Matrix& o) const {
        if (cols_ != o.rows_) throw std::invalid_argument("Matrix*: dimension mismatch");
        Matrix result(rows_, o.cols_, T{});
        // i-k-j ordering: inner loop walks a row of `o` (cache-friendly-ish).
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t k = 0; k < cols_; ++k) {
                const T a = (*this)(i, k);
                if (a == T{}) continue;          // skip zero, minor win
                for (size_t j = 0; j < o.cols_; ++j)
                    result(i, j) += a * o(k, j);
            }
        }
        return result;
    }

    // --- Structural ----------------------------------------------------------
    Matrix transpose() const {
        Matrix t(cols_, rows_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                t(j, i) = (*this)(i, j);
        return t;
    }

    // --- Reductions ----------------------------------------------------------
    T trace() const {
        if (rows_ != cols_) throw std::invalid_argument("trace: requires square matrix");
        T s = T{};
        for (size_t i = 0; i < rows_; ++i) s += (*this)(i, i);
        return s;
    }
    // Frobenius norm: sqrt(sum of squared entries).
    T norm() const {
        T s = T{};
        for (const auto& x : data_) s += x * x;
        return std::sqrt(s);
    }
    // Induced 1-norm: max column sum of |entries|.
    T norm1() const {
        T best = T{};
        for (size_t j = 0; j < cols_; ++j) {
            T s = T{};
            for (size_t i = 0; i < rows_; ++i) s += std::abs((*this)(i, j));
            best = std::max(best, s);
        }
        return best;
    }
    // Induced infinity-norm: max row sum of |entries|.
    T normInf() const {
        T best = T{};
        for (size_t i = 0; i < rows_; ++i) {
            T s = T{};
            for (size_t j = 0; j < cols_; ++j) s += std::abs((*this)(i, j));
            best = std::max(best, s);
        }
        return best;
    }

    // --- Comparison ----------------------------------------------------------
    // Relative tolerance, scaled by the magnitudes being compared. The default
    // sqrt(machine-epsilon) is a standard "reasonable" threshold for double.
    bool isApprox(const Matrix& o,
                  T tol = std::sqrt(std::numeric_limits<T>::epsilon())) const {
        if (rows_ != o.rows_ || cols_ != o.cols_) return false;
        for (size_t i = 0; i < data_.size(); ++i) {
            const T a = data_[i], b = o.data_[i];
            const T diff  = std::abs(a - b);
            const T scale = std::max(std::abs(a), std::abs(b));
            if (diff > tol * std::max<T>(T{1}, scale)) return false;
        }
        return true;
    }

    // --- I/O -----------------------------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m) {
        for (size_t i = 0; i < m.rows_; ++i) {
            for (size_t j = 0; j < m.cols_; ++j) {
                os << m(i, j);
                if (j + 1 < m.cols_) os << ' ';
            }
            os << '\n';
        }
        return os;
    }

private:
    void checkSameShape(const Matrix& o, const char* op) const {
        if (rows_ != o.rows_ || cols_ != o.cols_)
            throw std::invalid_argument(std::string("Matrix") + op + ": dimension mismatch");
    }
};

// Scalar on the left / right. SFINAE keeps these from matching `M * M`.
template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Matrix<T> operator*(const S& s, const Matrix<T>& m) { Matrix<T> r = m; r *= static_cast<T>(s); return r; }

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Matrix<T> operator*(const Matrix<T>& m, const S& s) { Matrix<T> r = m; r *= static_cast<T>(s); return r; }

template <typename T, typename S, typename = std::enable_if_t<std::is_arithmetic_v<S>>>
Matrix<T> operator/(const Matrix<T>& m, const S& s) { Matrix<T> r = m; r /= static_cast<T>(s); return r; }

// Blocked matmul: cache-tiled i-k-j. ~5-15x faster than operator* on large
// matrices. Block size 64 (three 64x64 slabs of double = ~96KB, fits in L2).
template <typename T>
Matrix<T> matmul_blocked(const Matrix<T>& A, const Matrix<T>& B, size_t BS = 64) {
    if (A.cols() != B.rows()) throw std::invalid_argument("matmul_blocked: dimension mismatch");
    const size_t m = A.rows(), k = A.cols(), n = B.cols();
    Matrix<T> C(m, n, T{});
    for (size_t i0 = 0; i0 < m; i0 += BS)
        for (size_t k0 = 0; k0 < k; k0 += BS)
            for (size_t j0 = 0; j0 < n; j0 += BS) {
                const size_t i_hi = std::min(i0 + BS, m);
                const size_t k_hi = std::min(k0 + BS, k);
                const size_t j_hi = std::min(j0 + BS, n);
                for (size_t i = i0; i < i_hi; ++i)
                    for (size_t kk = k0; kk < k_hi; ++kk) {
                        const T a = A(i, kk);
                        for (size_t j = j0; j < j_hi; ++j)
                            C(i, j) += a * B(kk, j);
                    }
            }
    return C;
}
