#include "matrix/matrix.hpp"
#include "matrix/decompose.hpp"
#include <iostream>

int main() {
    Matrix<double> A = {{4, 7}, {2, 6}};
    std::cout << "A =\n" << A;
    std::cout << "det(A)   = " << det(A)   << "\n";
    std::cout << "trace(A) = " << A.trace() << "\n";
    std::cout << "A^-1 =\n" << inverse(A);
    std::cout << "solve(A x = e0): x =\n" << solve(A, Matrix<double>{{1}, {0}});

    Matrix<double> C = {{1, 2}, {3, 4}, {5, 6}};  // 3x2
    auto q = qr(C);
    std::cout << "\nC =\n" << C;
    std::cout << "Q =\n" << q.Q;
    std::cout << "R =\n" << q.R;
    std::cout << "Q*R =\n" << q.Q * q.R;
    std::cout << "rank(C) = " << rank(C) << "\n";

    Matrix<double> X = {{1, 0}, {1, 1}, {1, 2}, {1, 3}};
    Matrix<double> y = {{1.1}, {2.9}, {5.2}, {6.8}};
    Matrix<double> coef = lstsq(X, y);
    std::cout << "\nLeast-squares fit y = a + b*x:\n";
    std::cout << "  a = " << coef(0, 0) << "\n";
    std::cout << "  b = " << coef(1, 0) << "\n";
}
