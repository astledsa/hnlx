#ifndef HNLX_H
#define HNLX_H

#include <map>
#include <list>
#include <string>
#include <cstdint>
#include <iostream>
#include "mlx/mlx.h"
#include <unordered_set>
#include <algorithm>

namespace mx = mlx::core;
using Vector = mx::array;
enum class Dist { nearest, farthest };

template <typename T>
T get_index (const Vector& array, int i, int j) {
    if (m.ndim() != 2) {
        throw std::invalid_argument(
            "[get_index] Input array must be 2-dimensional (a matrix).");
    }

    if (i < 0 || i >= m.shape(0) || j < 0 || j >= m.shape(1)) {
        throw std::out_of_range(
            "[get_index] Index (" + std::to_string(i) + ", " + 
            std::to_string(j) + ") is out of bounds for array with shape (" +
            std::to_string(m.shape(0)) + ", " + std::to_string(m.shape(1)) + ").");
    }

    auto start_indices = {i, j};
    auto end_indices = {i + 1, j + 1};
    auto element_slice = mx::slice(m, start_indices, end_indices);

    return element_slice.item<T>();
}

template <typename T>
class UniqueVector {
    private:
        std::unordered_set<T> unique;
    public:
        std::vector<T> data;
        UniqueVector() {};
        
        explicit UniqueVector (const T& val) {
            insert(val);
        }
        bool contains (const T& val) {
            return unique.find(val) != unique.end();
        }
        void insert(const T& val) {
            if (unique.insert(val).second) {
                data.push_back(val);
            }
        }
        void remove(const T& val) {
            if (unique.erase(val)) {
                auto it = std::find(data.begin(), data.end(), val);
                if (it != data.end()) {
                    data.erase(it);
                }
            }
        }
        const T& operator[](size_t idx) const { return data[idx]; }
        size_t size() const { return data.size(); }
};

class Cache {
    public:
        std::vector<std::tuple<int, size_t, Vector>> Data;

        Cache() {};
        
        void Insert (size_t id, Vector q, int lvl) { 
            Data.push_back(std::make_tuple(lvl, id, q)); 
        }

        std::vector<size_t> get_ids (int lvl) {
            std::vector<size_t> result_ids;
            for (const auto& item: Data) {
                if (std::get<0>(item) > lvl) {
                    result_ids.push_back(std::get<1>(item));
                }
            }
            return result_ids;
        }

        std::vector<Vector> get_vectors (int lvl) {
            std::vector<Vector> result_vectors;
            for (const auto& item: Data) {
                if (std::get<0>(item) > lvl) {
                    result_vectors.push_back(std::get<2>(item));
                }
            }
            return result_vectors;
        }
   
        int get_max_level () {
            if (Data.empty()) { return 0; }
            std::vector<std::tuple<int, size_t, Vector>>::iterator max_ID = std::max_element(Data.begin(), Data.end(),
            [](const auto& a, const auto& b) {
                return std::get<0>(a) < std::get<0>(b);
            });
            return std::get<0>(*max_ID);
        }

        size_t get_entry_id () {
            int max_lvl = get_max_level();
            for (const std::tuple<int, size_t, Vector>& tup: Data) {
                if (std::get<0>(tup) == max_lvl) {
                    return std::get<1>(tup);
                }
            }
            throw std::logic_error("No max ID found.");
        }
};

class Node {
    public:
        int max_level;
        Vector vector;
        std::map<int, std::vector<size_t>> Neighbours;
        Node (Vector q, int max_level);
};

class HNSW {
    public:
        float M;
        bool pruning;
        int threshold;
        int max_level;
        int total_nodes;
        int ef_construction;
        std::optional<size_t> Ep;
        void Insert(const Vector& vector);
        HNSW (int M, int ef_construction, int threshold);
        // std::vector<Vector> Search (const Vector& q, int K, int efsearch);
    
    private:
        Cache cache;
        std::vector<Node> NodeMap;
        int generate_level(float ml, int max_level);
        float cosine_similarity(size_t id, const Vector& q);
        bool search_layer_break_condition (size_t c, size_t f, std::vector<Vector> q);
        std::vector<size_t> select_neighbours(const Vector& q, std::vector<size_t> ids, int M);
        size_t get_node_by_distance(const std::vector<size_t>& nodes, std::vector<Vector> q, Dist dist);
        UniqueVector<size_t> search_layer (std::vector<Vector> q, size_t ep, int ef, int layer);
        void bidirectional_connection (std::vector<size_t> ids, size_t id, int M, int layer);
        std::vector<UniqueVector<size_t>> select_neighbours_batch(
            std::vector<Vector> points, 
            std::vector<std::vector<size_t>> ids, 
            int M
        );
};

#endif
