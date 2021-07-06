#include<vector>
#include<cmath>
#include<iomanip>
#include "Matrix.h"

using namespace std;


using MatrixType = long double;
using LMatrix = Matrix<MatrixType>;
using LColumn = Column<MatrixType>;

int main() {
    LMatrix A({{6,5,-6},{4,-6,9},{-6,6,1}});
    cout << fixed << setprecision(3);
    auto [Q, R] = Math::qr_decomposition(A);
    cout << A << "\n";
    cout << Q << "\n" << R << "\n";
    cout << Q * R << "\n";
    //cout << Math::find_roots(-2.01, -2.58, 0.98, -1.33).first;
    //cout << "qr\n" << Q * R << "\n";
    vector<complex<long double>> values = Math::qr_algorithm(A, 0.01);
    cout << values << "\n";
    cout << "Test: " << Math::qr_algorithm(A, 0.01) << "\n";
    return 0;
}