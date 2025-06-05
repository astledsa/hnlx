#ifndef HNLX_H
#define HNLX_H

#include <map>
#include <list>
#include <string>
#include <cstdint>
#include <iostream>
#include "mlx/mlx.h"
#include <unordered_set>

namespace mx = mlx::core;
using Vector = mx::array;
enum class Dist { nearest, farthest };

template <typename T>
class UniqueVector {
    private:
        std::unordered_set<T> unique;
    public:
        std::vector<T> data;
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
        int max_level;
        int total_nodes;
        int ef_construction;
        std::optional<size_t> Ep;
        void Insert(const Vector& vector);
        HNSW (int M, int ef_construction);
        // std::vector<Vector> Search (const Vector& q, int K, int efsearch);
    
    private:
        std::vector<Node> NodeMap;
        int generate_level(float ml, int max_level);
        float cosine_similarity(size_t id, const Vector& q);
        std::vector<size_t> select_neighbours(const Vector& q, std::vector<size_t> ids, int M);
        size_t get_node_by_distance(const std::vector<size_t>& nodes, const Vector& q, Dist dist);
        UniqueVector<size_t> search_layer (const Vector& q, size_t ep, int ef, int layer);
        void bidirectional_connection (std::vector<size_t> ids, size_t id, int M, int layer);
};

#endif
