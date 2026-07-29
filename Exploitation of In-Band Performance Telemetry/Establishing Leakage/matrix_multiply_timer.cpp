#include <iostream>
#include <vector>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int N = 100; // Matrix size (N x N)

void initialize_matrix_fixed(vector<vector<double>>& mat, double value) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            mat[i][j] = value;
}

void multiply_matrices(const vector<vector<double>>& A,
                       const vector<vector<double>>& B,
                       vector<vector<double>>& C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            C[i][j] = 0.0;
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int main() {
    vector<vector<double>> A(N, vector<double>(N));
    vector<vector<double>> B(N, vector<double>(N));
    vector<vector<double>> C(N, vector<double>(N));

    // Initialize A and B with fixed values
    initialize_matrix_fixed(A, 1.0);  // All elements of A are 1.0
    initialize_matrix_fixed(B, 1.0);  // All elements of B are 1.0

    auto start_time = high_resolution_clock::now();
    auto end_time = start_time + minutes(5);
    int iterations = 0;

    while (high_resolution_clock::now() < end_time) {
        multiply_matrices(A, B, C);
        ++iterations;
    }

    //cout << "Matrix multiplication completed " << iterations << " iterations in 10 minutes.\n";
    return 0;
}

