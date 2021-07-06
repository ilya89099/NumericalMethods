#include<vector>
#include<cmath>
#include<iomanip>
#include "Matrix.h"

using namespace std;

using MatrixType = long double;
using LMatrix = Matrix<MatrixType>;
using LColumn = Column<MatrixType>;

int main() {
    cout << fixed << setprecision(1);
    LMatrix m({{-6, 6, 0, 0, 0},
               {2, 10, -7, 0, 0},
               {0, -8, 18, 9, 0},
               {0,0,6,-17,-6},
               {0,0,0,9,14}});
    LColumn b{30,-31,108, -114, 124};

    cout << "Matrix:\n " << m << "\nValues:\n" << b << "\n\nSolution:\n" << Math::sweep_method(m,b) << "\n";

}