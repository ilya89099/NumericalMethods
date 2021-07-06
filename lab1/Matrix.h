#ifndef LAB1_MATRIX_H
#define LAB1_MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <tuple>
#include <complex>
#include <functional>

template <typename T>
using Column = std::vector<T>;

template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {
    for (const auto& i : v) {
        os << i << " ";
    }
    return os;
}

template <typename T>
std::ostream& operator << (std::ostream& os, std::complex<T> cp) {
    return os << "(re: " << cp.real() << ", im:" << cp.imag() << ")";
}

template <typename T>
Column<T> operator * (Column<T> v, T value) {
    for (T& i : v) {
        i *= value;
    }
    return v;
}

template <typename T>
Column<T> operator * (T value, Column<T> v) {
    return std::move(v) * value;
}

template <typename T>
Column<T> operator / (Column<T> v, T value) {
    for (T& i : v) {
        i /= value;
    }
    return v;
}

template <typename T>
Column<T> operator / (T value, Column<T> v) {
    return std::move(v) / value;
}

template <typename T>
class Matrix {
public:
    Matrix(const Matrix<T>&);
    Matrix(Matrix<T>&&);

    Matrix& operator = (const Matrix<T>&other);
    Matrix& operator = (Matrix<T>&&other);

    Matrix(std::vector<std::vector<T>>);

    static Matrix<T> ones(int n, int m);
    static Matrix<T> zeros(int n, int m);

    static Matrix<T> construct_row_matrix(std::vector<T>);
    static Matrix<T> construct_row_matrix(size_t);

    static Matrix<T> construct_column_matrix(const std::vector<T>&);
    static Matrix<T> construct_column_matrix(size_t);

    static Matrix<T> concatenate_columns(const std::vector<Column<T>>&);

    auto begin();
    auto begin() const;
    auto end();
    auto end() const;


    Matrix<T> operator * (T other) const;
    Matrix<T> operator * (const Matrix<T>& other) const;
    Matrix<T> operator * (const Column<T>& other) const;

    Matrix<T> operator + (const Matrix<T>& other) const;
    Matrix<T> operator - (const Matrix<T>& other) const;



    Matrix<T> transpose() const;
    const std::vector<std::vector<T>>& get_data() const;

    Column<T> get_column(size_t index) const;
    explicit operator Column<T>() const;
    explicit operator T() const;
    std::vector<T>& operator[](size_t index);
    const std::vector<T>& operator[](size_t index) const;

    size_t size_rows() const;
    size_t size_columns() const;

    void swap_rows(int f, int s);
    void swap_columns(int f, int s);

    std::vector<std::vector<T>> data;
};


template <typename T>
struct LUDecomposition {
    using value_type = T;
    Matrix<T> L;
    Matrix<T> U;
    Matrix<T> P;
    bool odd;
};

template <typename T>
std::ostream& operator << (std::ostream& os, const Matrix<T>& m) {
    for (int i = 0; i < m.size_rows(); ++i) {
        for (int j = 0; j < m.size_columns(); ++j) {
            os << m[i][j] << " ";
        }
        os << "\n";
    }
    return os;
}

template <typename T>
Matrix<T>::operator T() const {
    if (size_rows() == 1 && size_columns() == 1) {
        return (*this)[0][0];
    }
    throw std::logic_error("cant cast matrix to number");
}

template<typename T>
Matrix<T> operator * (Matrix<T> mt, T value) {
    for (int i = 0; mt.size_rows(); ++i) {
        for (int j = 0; mt.size_columns(); ++i) {
            mt[i][j] *= value;
        }
    }
    return mt;
}

template<typename T>
Matrix<T> operator * (T value, Matrix<T> mt) {
    return mt * value;
}


template <typename T>
Matrix<T>::Matrix(const Matrix<T>& other)
: data(other.data) {}

template <typename T>
Matrix<T>::Matrix(Matrix<T>&& other)
: data(std::move(other.data)) {}

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T>> v)
: data(std::move(v)) {}

template <typename T>
const std::vector<std::vector<T>>& Matrix<T>::get_data() const {
    return data;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(int n, int m) {
    return std::vector<std::vector<T>>(n, std::vector<T>(m, T(0)));
}

template <typename T>
Matrix<T> Matrix<T>::ones(int n, int m) {
    auto result = Matrix<T>::zeros(n,m);
    for (int i = 0, j = 0; i < n && j < m; ++i, ++j) {
        result[i][j] = T(1);
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> result = Matrix<T>::zeros(size_columns(), size_rows());
    for (int i = 0; i < size_rows(); ++i) {
        for (int j = 0; j < size_columns(); ++j) {
            result[j][i] = data[i][j];
        }
    }
    return result;
}

template <typename T>
size_t Matrix<T>::size_rows() const {
    return data.size();
}
template <typename T>
auto Matrix<T>::begin() {
    return data.begin();
}

template <typename T>
auto Matrix<T>::begin() const {
    return data.begin();
}

template <typename T>
auto Matrix<T>::end() {
    return data.end();
}

template <typename T>
auto Matrix<T>::end() const {
    return data.end();
}

template<typename T>
Matrix<T>::operator Column<T>() const {
    if (size_columns() != 1) {
        throw std::logic_error("Wrong cast to column");
    }
    Column<T> result;
    for (int i = 0; i < size_rows(); ++i) {
        result.push_back(data[i][0]);
    }
    return result;
}

template <typename T>
size_t Matrix<T>::size_columns() const {
    return (data.size() ? data[0].size() : 0);
}

template <typename T>
std::vector<T>& Matrix<T>::operator[](size_t index) {
    return data[index];
}

template <typename T>
const std::vector<T>& Matrix<T>::operator[](size_t index) const {
    return data[index];
}

template <typename T>
Matrix<T> Matrix<T>::operator * (T other) const {
    auto res = *this;
    for (int i = 0; i < res.size_rows(); ++i) {
        for (int j = 0; j < res.size_columns(); ++j) {
            res[i][j] *= other;
        }
    }
    return res;
}

template <typename T>
Matrix<T> Matrix<T>::operator * (const Matrix<T>& other) const {
    if (size_columns() != other.size_rows()) {
        throw std::logic_error("cant multiply matrices");
    }
    auto result = zeros(size_rows(), other.size_columns());
    for (int i = 0; i < size_rows(); ++i) {
        for (int j = 0; j < other.size_columns(); ++j) {
            T val = 0;
            for (int k = 0; k < size_columns(); ++k) {
                val += data[i][k] * other[k][j];
            }
            result.data[i][j] = val;
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator + (const Matrix<T>& other) const {
    if (size_columns() != other.size_columns() || size_rows() != other.size_rows()) {
        throw std::logic_error("cant add matrices");
    }
    auto result = zeros(size_rows(), size_columns());
    for (int i = 0; i < size_rows(); ++i) {
        for (int j = 0; j < other.size_columns(); ++j) {
            result[i][j] = *this[i][j] + other[i][j];
        }
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator - (const Matrix<T>& other) const {
    if (size_columns() != other.size_columns() || size_rows() != other.size_rows()) {
        throw std::logic_error("cant subtract matrices");
    }
    auto result = zeros(size_rows(), size_columns());
    for (int i = 0; i < size_rows(); ++i) {
        for (int j = 0; j < other.size_columns(); ++j) {
            result[i][j] = (*this)[i][j] - other[i][j];
        }
    }
    return result;
}

template <typename T>
void Matrix<T>::swap_rows(int f, int s) {
    data[f].swap(data[s]);
}

template <typename T>
void Matrix<T>::swap_columns(int f, int s) {
    for (int i = 0; i < size_rows(); ++i) {
        std::swap(data[i][f], data[i][s]);
    }
}

template<typename T>
Matrix<T> to_column_matrix(const Column<T>& c) {
    Matrix<T> result = Matrix<T>::zeros(c.size(), 1);
    for (int i = 0; i < c.size(); ++i) {
        result[i][0] = c[i];
    }
    return result;
}

template<typename T>
Matrix<T> to_row_matrix(const Column<T>& c) {
    return Matrix<T>({c});
}


template <typename T>
Matrix<T>& Matrix<T>::operator = (const Matrix<T>&other) {
    data = other.data;
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator = (Matrix<T>&& other) {
    data = std::move(other.data);
    return *this;
}

template <typename T>
Matrix<T> Matrix<T>::concatenate_columns(const std::vector<Column<T>>& cols) {
    Matrix<T> result = zeros((cols.size() ? cols[0].size() : 0), cols.size());
    for (int j = 0; j < cols.size(); ++j) {
        for (int i = 0; i < cols[j].size(); ++i) {
            result[i][j] = cols[j][i];
        }
    }
    return result;
}

template<typename T>
Column<T> operator + (const Column<T>& lhs, const Column<T>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::logic_error("error while summing vectors : sizes are not equal");
    }
    Column<T> result(lhs.size());
    for (int i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

template<typename T>
Column<T> operator - (const Column<T>& lhs, const Column<T>& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::logic_error("error while summing vectors : sizes are not equal");
    }
    Column<T> result(lhs.size());
    for (int i = 0; i < lhs.size(); ++i) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}


template <typename T>
Matrix<T> Matrix<T>::construct_row_matrix(std::vector<T> v) {
    return Matrix<T>{{std::move(v)}};
}

template <typename T>
Matrix<T> Matrix<T>::construct_column_matrix(const std::vector<T>& v) {
    Matrix<T> result = zeros( v.size(), 1);
    for (int i = 0; i < v.size(); ++i) {
        result[i][0] = v[i];
    }
    return result;
}

template <typename T>
Matrix<T> Matrix<T>::operator * (const Column<T>& other) const {
    return *this * construct_column_matrix(other);
}

template <typename T>
Matrix<T> Matrix<T>::construct_column_matrix(size_t size) {
    return zeros(size,1);
}

template <typename T>
Matrix<T> Matrix<T>::construct_row_matrix(size_t size) {
    return zeros(1,size);
}

template <typename T>
Column<T> Matrix<T>::get_column(size_t index) const {
    Column<T> result;
    for (int i = 0; i < size_rows(); ++i) {
        result.push_back(data[i][index]);
    }
    return result;
}

namespace Math{
    template <typename T>
    LUDecomposition<T> lu_decomposition(const Matrix<T>& other) {
        Matrix<T> L = Matrix<T>::zeros(other.size_rows(), other.size_columns());
        Matrix<T> U = other;
        Matrix<T> P = Matrix<T>::ones(other.size_rows(), other.size_columns());
        bool odd = false;
        int n = other.size_rows();
        for (int i = 0; i < n; ++i) {
            int index = i;
            for (int j = i; j < n; ++j) {
                if (std::abs(U[j][i]) > std::abs(U[index][i])) {
                    index = j;
                }
            }
            if (i != index) {
                L.swap_rows(i, index);
                U.swap_rows(i, index);
                P.swap_rows(i, index);
                odd = !odd;
            }


            L[i][i] = 1;
            for (int j = i + 1; j < n; ++j) {
                L[j][i] = U[j][i] / U[i][i];
                U[j][i] = 0;
            }
            for (int j = i + 1; j < n; ++j) {
                for (int k = i + 1; k < n; ++k) {
                    U[j][k] = U[j][k] - U[i][k] * L[j][i];
                }
            }
        }
        return {L, U, P, odd};
    }



    template <typename T>
    Column<T> solve_linear_system(const LUDecomposition<T>& dec, Column<T> b) {
        const auto& [L,U,P,odd] = dec;
        int n = L.size_rows();
        b = Column<T>(P * b);
        // PA = LU
        //Solve Lz = b
        Column<T> z(n);
        z[0] = b[0];
        for (int i = 1; i < n; ++i) {
            T sum = 0;
            for (int j = 0; j < i; ++j) {
                sum += L[i][j] * z[j];
            }
            z[i] = b[i] - sum;

        }

        //Solve Ux = z
        Column<T> x(n);
        x[n - 1] = z[n - 1] / U[n - 1][n - 1];
        for (int i = n - 2; i >= 0; --i) {
            T sum = 0;
            for (int j = i + 1; j < n; ++j) {
                sum += U[i][j] * x[j];
            }
            x[i] = (z[i] - sum) / U[i][i];
        }

        //Return x
        return x;
    }

    template <typename T>
    Matrix<T> solve_matrix_equation(const LUDecomposition<T>& dec, const Matrix<T>& b) {
        std::vector<Column<T>> result;
        for (int i = 0; i < dec.L.size_columns(); ++i) {
            result.push_back(solve_linear_system(dec, b.get_column(i)));
        }
        return Matrix<T>::concatenate_columns(result);

    }

    template <typename T>
    Matrix<T> solve_matrix_equation(const Matrix<T>& a, const Matrix<T>& b) {
        return solve_matrix_equation(lu_decomposition(a), b);

    }

    template <typename T>
    T determinant(const LUDecomposition<T>& dec) {
        const auto& [L, U, P, odd] = dec;
        double Ldet = 1, Udet = 1;
        int n = L.size_rows();
        for (int i = 0; i < n; ++i) {
            Ldet *= L[i][i];
            Udet *= U[i][i];
        }
        return Ldet * Udet * (odd ? -1 : 1);
    }

    template <typename T>
    long double norm(const Column<T>& d, const std::function<bool(int)>& filter = [] (int) {return true;}) {
        long double sum = 0;
        for (int i = 0; i < d.size(); ++i) {
            if (filter(i)) {
                sum += d[i] * d[i];
            }
        }
        return std::sqrt(sum);
    }

    template <typename T>
    long double norm(const Matrix<T>& d, const std::function<bool(int,int)>& filter = [] (int,int) {return true;}) {
        long double sum = 0;
        for (int i = 0; i < d.size_rows(); ++i) {
            for (int j = 0; j < d.size_columns(); ++j) {
                if (filter(i, j)) {
                    sum += d[i][j] * d[i][j];
                }
            }
        }
        return std::sqrt(sum);
    }



    template <typename T>
    Column<T> sweep_method(const Matrix<T>& matrix, const Column<T>& d) {
        struct index_pair {int i; int j;};
        std::vector<double> P(d.size()), Q(d.size());
        int n = d.size();
        auto M = [&matrix] (index_pair elem, int inc) -> T {
            return matrix[elem.i + inc][elem.j + inc];
        };
        index_pair a = {0,-1}, b = {0,0}, c = {0,1};
        P[0] = -M(c,0) / M(b,0);
        Q[0] = d[0] / M(b,0);
        for (int i = 1; i < n - 1; ++i) {
            P[i] = -M(c,i) / (M(b,i) + M(a,i) * P[i - 1]);
            Q[i] = (d[i] - M(a,i) * Q[i - 1]) / (M(b,i) + M(a,i) * P[i - 1]);
        }
        P[n - 1] = 0;
        Q[n - 1] = (d[n - 1] - M(a, n - 1) * Q[n - 2]) / (M(b, n - 1) + M(a, n - 1) * P[n - 2]);
        Column<T> x(d.size());
        x[n - 1] = Q.back();
        for (int i = x.size() - 2; i >= 0; --i) {
            x[i] = P[i] * x[i + 1] + Q[i];
        }
        return x;
    }

    template <typename T>
    std::vector<Column<T>> simple_iterations(const Matrix<T>& a, const Column<T>& b, long double epsilon, int max_iteration_count = 1'000'000) {
        int n = b.size();
        Column<T> beta(n);
        Matrix<T> alpha = Matrix<T>::zeros(n,n);
        for (int i = 0; i < n; ++i) {
            beta[i] = b[i] / a[i][i];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                alpha[i][j] = -a[i][j] / a[i][i];
                if (i == j) {
                    alpha[i][j] = 0;
                }
            }
        }
        std::vector<Column<T>> result = {beta};
        Column<T> x = beta;
        for (int i = 0; i < max_iteration_count; ++i) {
            Column<T> new_x = beta + Column<T>(alpha * x);
            double iter_epsilon = norm(new_x - x) * norm(alpha) / (1 - norm(alpha));
            result.push_back(new_x);//std::cout << "Simple iteration " << i + 1 << " eps " << iter_epsilon << "\n";
            x = new_x;
            if (iter_epsilon <= epsilon) {
                break;
            }
        }
        return result;
    }

    template <typename T>
    std::vector<Column<T>> zeidel(const Matrix<T>& a, const Column<T>& b, long double epsilon, int max_iteration_count = 1'000'000) {
        int n = b.size();
        Column<T> beta(n);
        Matrix<T> alpha = Matrix<T>::zeros(n,n);
        for (int i = 0; i < n; ++i) {
            beta[i] = b[i] / a[i][i];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                alpha[i][j] = -a[i][j] / a[i][i];
                if (i == j) {
                    alpha[i][j] = 0;
                }
            }
        }
        std::vector<Column<T>> result = {beta};
        Column<T> x = beta;
        for (int i = 0; i < max_iteration_count; ++i) {
            Column<T> new_x(n);
            for (int j = 0; j < n; ++j) {
                new_x[j] = beta[j];
                for (int k = 0; k < n; ++k) {
                    new_x[j] += alpha[j][k] * (k < j ? new_x[k] : x[k]);
                }
            }
            double iter_epsilon = (norm(alpha, [] (int i, int j) {return j >= i;}))/(1 - norm(alpha)) * norm(new_x - x);
            result.push_back(new_x);//std::cout << "Zeidel iteration " << i + 1 << " eps " << iter_epsilon << "\n";
            x = new_x;
            if (iter_epsilon <= epsilon) {
                break;
            }
        }
        return result;
    }

    template<typename T>
    struct Eigen {
        Column<T> eigenvalues;
        std::vector<Column<T>> eigenvectors;
        std::vector<long double> epsilons;
    };

    template <typename T>
    Eigen<T> rotation_method(const Matrix<T>& m, long double epsilon, int max_iteration_count = 1'000'000) {
        int n = m.size_columns();
        Matrix<T> A = m;
        Matrix<T> U = Matrix<T>::ones(n,n);
        std::vector<long double> epsilons;
        for (int iter = 0; iter < max_iteration_count; ++iter) {
            T max = std::abs(A[0][1]);
            int max_i = 0;
            int max_j = 1;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i != j && std::abs(A[i][j]) > max) {
                        max = std::abs(A[i][j]);
                        max_i = i, max_j = j;
                    }
                }
            }
            Matrix<T> rotate_matrix = Matrix<T>::ones(n,n);
            long double phi = (A[max_i][max_i] == A[max_j][max_j] ?
                               M_PI_4 :
                               atan2(2 * A[max_i][max_j], (A[max_i][max_i] - A[max_j][max_j])) / 2.);
            rotate_matrix[max_i][max_i] = rotate_matrix[max_j][max_j] = std::cos(phi);
            rotate_matrix[max_i][max_j] = -std::sin(phi);
            rotate_matrix[max_j][max_i] = std::sin(phi);
            U = U * rotate_matrix;
            A = rotate_matrix.transpose() * A * rotate_matrix;
            epsilons.push_back(norm(A, [] (int i, int j) {return i != j;}));
            if (epsilons.back() < epsilon) {
                break;
            }
        }
        Column<T> lambdas(n);
        for (int i = 0; i < n; ++i) {
            lambdas[i] = A[i][i];
        }
        return {lambdas, U.transpose().get_data(), epsilons};
    }

    template <typename T>
    struct QRDecomposition {
        Matrix<T> Q;
        Matrix<T> R;
    };

    template <typename T>
    int sign(T v) {
        return (v < 0 ? 1 : -1);
    }

    template <typename T>
    std::pair<std::complex<T>,std::complex<T>> find_roots(T a0, T a1, T a2, T a3) {
        T a = 1;
        T b = - (a0 + a3);
        T c = a0 * a3 - a1 * a2;
        T discriminant = std::pow(b, 2) - 4 * a * c;
        if (discriminant < 0) {
            return {{-b / 2., std::sqrt(std::abs(discriminant)) / 2.}, {-b / 2., - std::sqrt(std::abs(discriminant)) / 2.}};
        }
        return {{(-b + std::sqrt(discriminant)) / 2.,0},{(-b - std::sqrt(discriminant)) / 2.,0}};
    }

    template <typename T>
    QRDecomposition<T> qr_decomposition(Matrix<T> A) {
        if (A.size_columns() != A.size_rows()) {
            throw std::logic_error("Not square matrix");
        }
        int n = A.size_columns();
        Matrix<T> H = Matrix<T>::ones(n,n);


        for (int i = 0; i < n - 1; ++i) {
            Column<T> a = A.get_column(i);
            Column<T> v = a;
            for (int j = 0; j < i; ++j) {
                v[j] = 0;
            }
            v[i] = a[i] + sign(a[i]) * norm(a, [i] (int index) {return index >= i;});
            Matrix<T> v_col = Matrix<T>::construct_column_matrix(v);
            Matrix<T> v_row = Matrix<T>::construct_row_matrix(v);
            Matrix<T> H_cur = Matrix<T>::ones(n,n) - ((v_col * v_row) * (2 / T(v_row * v_col)));


            H = H * H_cur;
            A = H_cur * A;
        }
        return {H, A};
    }

    template <typename T>
    std::vector<T> reverse_iter_process(const Matrix<T>& A, T value, long double epsilon, int max_iteration_count = 1'000'000) {
        Column<T> result(A.size_columns(), 1);
        Matrix<T> ones = Matrix<T>::ones(A.size_rows(), A.size_columns());
        for (int i = 0; i < max_iteration_count; ++i) {
            Column<T> current_result = Column<T>(solve_linear_system(Math::lu_decomposition<T>(A - (value * ones)), result));
            if (norm(result - (current_result) / norm(current_result)) < epsilon) {
                result = current_result;
                break;
            }
            result = current_result / norm(current_result);
        }
        return result;
    }

    template<typename T>
    Eigen<std::complex<T>> reverse_iterations(const Matrix<T>& A, std::vector<std::complex<T>> eigenvalues, long double epsilon, int max_iteration_count = 1'000'000) {
        int n = A.size_rows();
        std::vector<std::vector<T>> real_vectors, imag_vectors;
        for (auto num : eigenvalues) {
            if (num.real() != 0) {
                real_vectors.push_back(reverse_iter_process(A, num.real(), epsilon, max_iteration_count));
            } else {
                real_vectors.emplace_back(A.size_columns());
            }

            if (num.imag() != 0) {
                imag_vectors.push_back(reverse_iter_process(A, num.imag(), epsilon, max_iteration_count));
            } else {
                imag_vectors.emplace_back(A.size_columns());
            }
        }
        Eigen<std::complex<T>> result;
        result.eigenvalues = eigenvalues;
        result.eigenvectors.resize(eigenvalues.size());
        for (int i = 0; i < real_vectors.size(); ++i) {
            for (int j = 0; j < n; ++j) {
                result.eigenvectors[i].push_back(std::complex(real_vectors[i][j], imag_vectors[i][j]));
            }

        }
        return result;
    }


    template <typename T>
    std::vector<std::complex<T>> qr_algorithm(Matrix<T> A, long double epsilon, int max_iteration_count = 1'000'000) {
        if (A.size_columns() != A.size_rows()) {
            throw std::logic_error("not square matrix in qr algo");
        }
        int n = A.size_columns();
        std::vector<bool> complex_convergence(n - 1, false);
        std::vector<bool> real_convergence(n - 1, false);
        std::vector<std::complex<T>> complex_roots(n - 1);
        std::vector<std::complex<T>> result(n);
        for (int i = 0; i < n - 1; ++i) {
            auto [root1, root2] = find_roots(A[i][i], A[i][i + 1], A[i + 1][i], A[i + 1][i + 1]);
            complex_roots[i] = root1;
        }
        for (int iter = 0; iter < max_iteration_count; ++iter) {
            auto [Q,R] = qr_decomposition(A);
            A = R * Q;

            for (int j = 0; j < n - 1; ++j) {
                real_convergence[j] = (norm(A.get_column(j), [j] (int index) {return index > j;}) < epsilon);
                auto [root1, root2] = find_roots(A[j][j], A[j][j + 1], A[j + 1][j], A[j + 1][j + 1]);
                if (root1.imag() != 0 && abs(root1 - complex_roots[j]) < epsilon) {
                    complex_convergence[j] = true;
                }
                complex_roots[j] = root1;
            }
            bool end = true;
            for (int j = 0; j < n - 1; ++j) {
                if (real_convergence[j]) {
                    result[j] = A[j][j];
                } else if (complex_convergence[j]) {
                    std::complex<T> root = complex_roots[j];
                    result[j] = root;
                    result[j + 1] = {root.real(), - root.imag()};
                    ++j;
                } else {
                    end = false;
                    break;
                }
            }
            std::cout << "iter\n" << iter << "\n" << real_convergence << "\n" << complex_convergence << "\n" << A << "\n";
            if (!complex_convergence[n - 2]) {
                result[n - 1] = A[n - 1][n - 1];
            }
            if (end) {
                break;
            }
        }
        return result;
    }

};

#endif //LAB1_MATRIX_H
