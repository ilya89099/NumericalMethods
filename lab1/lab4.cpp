#include<vector>
#include<cmath>
#include<iomanip>
#include "Matrix.h"

using namespace std;


using MatrixType = long double;
using LMatrix = Matrix<MatrixType>;
using LColumn = Column<MatrixType>;

int main() {
    LMatrix m({{-7,-9,1},
               {-9,7,2},
               {1,2,9}});


    long double epsilon = 0.0001;
    auto [eigenvalues, eigenvectors, epsilons] = Math::rotation_method(m, epsilon);
    cout << "Eigenvalues\n";
    cout << eigenvalues << "\n\n";
    cout << "Eigenvectors\n";
    for (const auto& v : eigenvectors) {
        cout << v << "\n";
    }
    cout << "\n";
    for (int i = 0; i < eigenvalues.size(); ++i) {
        cout << "i: " << i + 1 << ", Ax " << LColumn(m * eigenvectors[i]) << ", lambda * x" << eigenvectors[i] * eigenvalues[i] << "\n";
    }
    cout << "\n";
    for (int i = 0; i < epsilons.size(); ++i) {
        cout << "epsilon on iteration " << i << ": " << epsilons[i] << "\n";
    }
    return 0;
}