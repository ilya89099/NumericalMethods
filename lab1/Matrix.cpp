#include "Matrix.h"

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
size_t Matrix<T>::size_rows() const {
    return data.size();
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
Matrix<T> Matrix<T>::operator * (const Matrix<T>& other) {
    if (size_columns() != other.size_rows()) {
        throw std::logic_error("cant multiply matrices");
    }
    auto result = zeros(size_rows(), other.size_columns());
    for (int i = 0; i < size_rows(); ++i) {
        for (int j = 0; i < other.size_columns(); ++i) {
            T val = 0;
            for (int k = 0; i < size_columns(); ++i) {
                val += data[i][k] * other[k][j];
            }
            result.data[i][j] = val;
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

template <typename T>
static std::pair<Matrix<T>, Matrix<T>> Matrix<T>::LUdecomposition(const Matrix<T>& other) {
    Matrix<T> L = ones(other.size_rows(), other.size_columns());
    Matrix<T> U = other;
    for (int i = 0; i < other.size_rows(); ++i) {
        
    }
}


