#include "matrix/matrix.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Matmul variants — all return an identical n x n result up to floating error.
// ============================================================================

// (1) Baseline: textbook i-j-k dot-product loop. Column of B strides badly.
template <typename T>
Matrix<T> matmul_ijk(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t j = 0; j < B.cols(); ++j)
            for (size_t k = 0; k < A.cols(); ++k)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

// (2) i-k-j: each inner loop walks a row of B and a row of C — both contiguous.
// This is what the library's operator* already does.
template <typename T>
Matrix<T> matmul_ikj(const Matrix<T>& A, const Matrix<T>& B) {
    Matrix<T> C(A.rows(), B.cols(), T{});
    for (size_t i = 0; i < A.rows(); ++i)
        for (size_t k = 0; k < A.cols(); ++k) {
            const T a = A(i, k);
            for (size_t j = 0; j < B.cols(); ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

// (4) Parallel i-k-j: partition the rows of C across threads.
template <typename T>
Matrix<T> matmul_parallel(const Matrix<T>& A, const Matrix<T>& B, unsigned nthreads) {
    const size_t m = A.rows(), k = A.cols(), n = B.cols();
    Matrix<T> C(m, n, T{});
    auto worker = [&](size_t i_begin, size_t i_end) {
        for (size_t i = i_begin; i < i_end; ++i)
            for (size_t kk = 0; kk < k; ++kk) {
                const T a = A(i, kk);
                for (size_t j = 0; j < n; ++j)
                    C(i, j) += a * B(kk, j);
            }
    };
    const size_t chunk = (m + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(lo + chunk, m);
        if (lo < hi) threads.emplace_back(worker, lo, hi);
    }
    for (auto& th : threads) th.join();
    return C;
}

// (5) Blocked + parallel — the one that usually wins.
template <typename T>
Matrix<T> matmul_blocked_parallel(const Matrix<T>& A, const Matrix<T>& B,
                                  size_t BS, unsigned nthreads) {
    const size_t m = A.rows(), k = A.cols(), n = B.cols();
    Matrix<T> C(m, n, T{});
    auto worker = [&](size_t i_begin, size_t i_end) {
        for (size_t i0 = i_begin; i0 < i_end; i0 += BS)
            for (size_t k0 = 0; k0 < k; k0 += BS)
                for (size_t j0 = 0; j0 < n; j0 += BS) {
                    const size_t i_hi = std::min(i0 + BS, i_end);
                    const size_t k_hi = std::min(k0 + BS, k);
                    const size_t j_hi = std::min(j0 + BS, n);
                    for (size_t i = i0; i < i_hi; ++i)
                        for (size_t kk = k0; kk < k_hi; ++kk) {
                            const T a = A(i, kk);
                            for (size_t j = j0; j < j_hi; ++j)
                                C(i, j) += a * B(kk, j);
                        }
                }
    };
    const size_t chunk = (m + nthreads - 1) / nthreads;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        const size_t lo = t * chunk;
        const size_t hi = std::min(lo + chunk, m);
        if (lo < hi) threads.emplace_back(worker, lo, hi);
    }
    for (auto& th : threads) th.join();
    return C;
}

// ============================================================================
// Harness
// ============================================================================

struct Result { std::string name; double seconds; double gflops; };

template <typename Fn>
Result bench(const std::string& name, Fn&& fn, size_t n, int runs = 3) {
    fn();  // warm up: page in buffers, spin up caches
    double best = std::numeric_limits<double>::max();
    for (int i = 0; i < runs; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        const double s = std::chrono::duration<double>(t1 - t0).count();
        best = std::min(best, s);
    }
    const double flops = 2.0 * n * n * n;      // n^3 mul + n^3 add
    return {name, best, flops / best / 1e9};
}

void print(const std::vector<Result>& rows, const Result* baseline) {
    std::cout << std::left  << std::setw(28) << "variant"
              << std::right << std::setw(12) << "sec"
              << std::setw(12) << "GFLOPS"
              << std::setw(10) << "speedup" << "\n"
              << std::string(62, '-') << "\n";
    for (const auto& r : rows) {
        const double sp = baseline ? baseline->seconds / r.seconds : 1.0;
        std::cout << std::left  << std::setw(28) << r.name
                  << std::right << std::setw(12) << std::fixed << std::setprecision(4) << r.seconds
                  << std::setw(12) << std::setprecision(2) << r.gflops
                  << std::setw(9)  << std::setprecision(2) << sp << "x\n";
    }
    std::cout.unsetf(std::ios::fixed);
}

// Sanity check: every variant must agree with the baseline at small n.
// If a "faster" variant is numerically wrong, we want to know before trusting
// the GFLOPS numbers.
void check_all_variants() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    const size_t n = 64;

    Matrix<double> A(n, n), B(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }

    const auto ref = matmul_ijk(A, B);
    const auto hw  = std::max(1u, std::thread::hardware_concurrency());

    const std::vector<std::pair<std::string, Matrix<double>>> variants = {
        {"ikj",              matmul_ikj(A, B)},
        {"blocked",          matmul_blocked(A, B, 32)},
        {"parallel",         matmul_parallel(A, B, hw)},
        {"blocked+parallel", matmul_blocked_parallel(A, B, 32, hw)},
    };
    for (const auto& [name, M] : variants) {
        if (!M.isApprox(ref, 1e-10))
            throw std::runtime_error("variant " + name + " disagrees with baseline");
    }
    std::cout << "correctness check: OK (n=64, tol=1e-10)\n\n";
}

int main(int argc, char** argv) {
    const size_t n = (argc > 1) ? std::stoul(argv[1]) : 400;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    std::cout << "matrix size: " << n << " x " << n
              << "   hardware_concurrency: " << hw << "\n\n";

    check_all_variants();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    Matrix<double> A(n, n), B(n, n);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }

    std::vector<Result> results;
    results.push_back(bench("ijk (baseline)",   [&]{ matmul_ijk(A, B); },                    n));
    results.push_back(bench("ikj",              [&]{ matmul_ikj(A, B); },                    n));
    results.push_back(bench("blocked BS=32",    [&]{ matmul_blocked(A, B, 32); },            n));
    results.push_back(bench("blocked BS=64",    [&]{ matmul_blocked(A, B, 64); },            n));
    results.push_back(bench("blocked BS=128",   [&]{ matmul_blocked(A, B, 128); },           n));
    results.push_back(bench("parallel T=" + std::to_string(hw),
                            [&]{ matmul_parallel(A, B, hw); },                               n));
    results.push_back(bench("blocked+parallel BS=64",
                            [&]{ matmul_blocked_parallel(A, B, 64, hw); },                   n));

    print(results, &results[0]);
}
