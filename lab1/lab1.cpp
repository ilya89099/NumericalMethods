#include<vector>
#include<cmath>
#include<iomanip>
#include "Matrix.h"

using namespace std;

using LMatrix = Matrix<long double>;
using LColumn = Column<long double>;

int main() {
    cout << fixed << setprecision(3);
    LMatrix m({{7, 8, 4, -6},
               {5, 9, 1, 1},
               {-1,6,-2, -6},
               {2, 9,6, -4}});
    LColumn b{-126,-42,-115,-67};
    auto decomposition = Math::lu_decomposition(m);
    auto x = Math::solve_linear_system(decomposition, b);
    cout << decomposition.L * decomposition.U << "\n\n";
    cout << "Matrix:\n" << m << "\nSolution:" << x << "\n\n";
    auto inverse_matrix = Math::solve_matrix_equation(decomposition, LMatrix::ones(m.size_rows(), m.size_columns()));
    cout << "Inverse matrix:\n" << inverse_matrix << "\n";
    cout << "M * Inverse\n" << m * inverse_matrix << "\n";
    cout << "Determinant = " << Math::determinant(decomposition) << "\n";

}