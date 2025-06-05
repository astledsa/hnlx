#include <random>
#include "hnlx.h"
#include <iostream>

int main() {
    const int dim = 128;
    std::vector<float> data1(dim);
    std::vector<float> data2(dim);
    std::vector<float> data3(dim);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < dim; ++i) {
        data1[i] = dist(gen);
        data2[i] = dist(gen);
        data3[i] = dist(gen);
    }

    Vector vec1 = mx::array(data1.begin(), {dim}, mx::float16);
    Vector vec2 = mx::array(data2.begin(), {dim}, mx::float16);
    Vector vec3 = mx::array(data3.begin(), {dim}, mx::float16);

    HNSW hnsw(16, 200);
    hnsw.Insert(vec1);
    hnsw.Insert(vec2);
    hnsw.Insert(vec3);

    return 0;
}
