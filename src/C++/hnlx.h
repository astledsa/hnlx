#ifndef HNLX_H
#define HNLX_H

#include <map>
#include <list>
#include <vector>
#include <string>
#include "mlx/mlx.h"

namespace mx = mlx::core;
using Vector = mx::array;
enum class Dist { nearest, farthest };

class Node {
    public:
        int8_t max_level;
        Vector vector;
        std::map<int8_t, std::list<Node*>> Neighbours;
};

class HNSW {
    public:
        Node* Ep;
        int8_t M;
        int8_t ef_construction;
        int8_t total_nodes;
        void Insert(Vector vector);
    
    private:
        std::vector<Node> NodeMap;
        float cosine_similarity(size_t id, Vector q);
        int8_t generate_level(float ml, int8_t max_level);
        std::list<size_t> select_neighbours(Vector q, std::list<size_t> ids);
        size_t get_node_by_distance(std::list<size_t> nodes, Vector q, Dist dist);
        std::list<size_t> select_layer (Vector q, size_t ep, int8_t ef, int8_t layer);
        void bidirectional_connection (std::list<size_t> ids, size_t id, int8_t M, int8_t layer);
};

#endif
