#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <lambda_lanczos.hpp>

using std::cout;
using std::endl;
using std::setprecision;
using lambda_lanczos::LambdaLanczos;

template<typename T>
using vector = std::vector<T>;

int main1() {
    //const int n = 1000;
    //float* matrix = new float[n * n];
    //std::random_device e;
    //std::uniform_real_distribution<float> u(0, 1);
    //for (int i = 0; i < n; i++)
    //{
    //    for (int j = 0; j < n; j++)
    //    {
    //        matrix[i*n+j] = u(e);
    //    }
    //}
    float matrix[] = { 5., 4., 3., 6., 4., 3., 5., 7., 3., 5., 9., 6., 6., 7., 6., 8. };
    /* Its eigenvalues are {4, 1, 1} */

    // the matrix-vector multiplication routine
    auto mv_mul = [&](const vector<float>& in, vector<float>& out) {
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                out[i] += matrix[i*4+j] * in[j];
            }
        }
    };

    LambdaLanczos<float> engine(mv_mul, 4, false); // true means to calculate the largest eigenvalue.
    float eigenvalue;
    vector<float> eigenvector(4);
    int itern = engine.run(eigenvalue, eigenvector);

    cout << "Iteration count: " << itern << endl;
    cout << "Eigen value: " << setprecision(16) << eigenvalue << endl;
    cout << "Eigen vector: ";
    for (int i = 0; i < 4; ++i) {
        cout << eigenvector[i] << " ";
    }
    cout << endl;

    return EXIT_SUCCESS;
}