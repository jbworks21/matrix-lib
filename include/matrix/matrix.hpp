#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

template <typename T>
class Matrix {
private:
    size_t rows_, cols_;
    std::vector<T> data_; // Row-major storage

public:
    // Constructors
    Matrix(size_t rows, size_t cols, const T& value = T{})
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    Matrix(std::initializer_list<std::initializer_list<T>> list) {
        rows_ = list.size();
        cols_ = (rows_ > 0) ? list.begin()->size() : 0;
        data_.reserve(rows_ * cols_);
        for (const auto& row : list) {
            if (row.size() != cols_) throw std::invalid_argument("Row size mismatch");
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    // Accessors
    // Using operator() instead of [][] is standard for 2D math objects
    // because it avoids the need for a proxy object.
    T& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    const T& operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Matrix multiplication (naive, to be optimized later)
    Matrix operator*(const Matrix& other) const {
        if (cols_ != other.rows_) throw std::invalid_argument("Dimension mismatch");
        Matrix result(rows_, other.cols_, T{});
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                for (size_t k = 0; k < cols_; ++k) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    // ... (Implement transpose, identity, etc. here)
};
