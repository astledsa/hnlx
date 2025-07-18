#include <random>
#include "hnlx.h"
#include <iostream>

int main() {

    const int D = 128;
    const int N = 100;

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<Vector> all_vectors;
    all_vectors.reserve(N);

    for (int i = 0; i < N; ++i) {
        std::vector<float> data_vec(D);
        for (int j = 0; j < D; ++j) {
            data_vec[j] = dist(gen);
        }

        all_vectors.push_back(mx::array(data_vec.begin(), {D}, mx::float16));
    }

    // (M, max level, threshold, efconstruction)
    int batch = 50;
    HNSW hnsw({16, 4, batch, 200});
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        hnsw.Insert(all_vectors[i]);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    std::cout << "\n";
    std::cout << "Insertion complete.\n";
    std::cout << "Total insertion time for " << N << " vectors with batching " << batch << ": " << static_cast<double>(duration.count())/1000000 << " seconds.\n";
    std::cout << "\n" << std::endl;
    hnsw.Report();

    return 0;
}
