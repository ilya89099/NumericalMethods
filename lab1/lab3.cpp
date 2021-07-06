#include<vector>
#include<cmath>
#include<iomanip>
#include "Matrix.h"

using namespace std;


using MatrixType = long double;
using LMatrix = Matrix<MatrixType>;
using LColumn = Column<MatrixType>;

int main() {
    LMatrix m({{10, -1, -2, 5},
               {4, 28, 7, 9},
               {6, 5, -23, 4},
               {1,4,5,-15}});

    LColumn b{-99,0,67,58};
    long double epsilon = 0.0001;
    auto simple_solution = Math::simple_iterations(m, b, epsilon);
    auto zeidel_solution = Math::zeidel(m, b, epsilon);
    cout << "Matrix:\n" << m << "\nValues:\n" << b << "\n\n";
    cout << "Simple iteration method\n";
    for (int i = 0; i < simple_solution.size(); ++i) {
        cout << "iteration: " << i << ", solution" << simple_solution[i] << "\n";
    }
    cout << "\n";

    cout << "Zeidel method\n";
    for (int i = 0; i < zeidel_solution.size(); ++i) {
        cout << "iteration: " << i << ", solution " << zeidel_solution[i] << "\n";
    }
    cout << "\n";

    return 0;
}