#include <cmath>
#include "hnlx.h" 
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include "mlx/mlx.h"
#include <algorithm>
#include <random>

namespace mx = mlx::core;

void PrintShape (Vector v) {
    if (v.ndim() < 1) {
        std::cout << "no dim" << std::endl;
        return;
    }
    if (v.ndim() < 2) {
        std::cout << "(" << v.shape(0) << ")" << std::endl;
        return;
    }
    std::cout << "(" << v.shape(0) << "," << v.shape(1) << ")" << std::endl;
}

mx::array cosine_sim (const Vector& v1, const Vector& v2) {
    mx::array q = v1; 
    mx::array v = mx::transpose(v2);

    mx::array q_norm = mx::linalg::norm(q); 
    mx::array v_norm = mx::linalg::norm(v);
    mx::array dot_product = mx::matmul(q, v);
    mx::array denominator = mx::maximum(q_norm * v_norm, mx::array(0.000001f));

    return  dot_product / denominator;;
}

HNSW::HNSW (int M, int ef_construction) :
    M(M), ef_construction(ef_construction), total_nodes(0), Ep(std::nullopt) {}

Node::Node (Vector v, int max_level) :
    vector(v), max_level(max_level) {
    for (int i = 0; i <= max_level; ++i) {
        Neighbours.emplace(i, std::vector<size_t>{});
    }
}

float HNSW::cosine_similarity(size_t id, const Vector& q) {
    const Node& node = this->NodeMap[id];
    mx::array c =  cosine_sim (node.vector, q);
    return c.item<float>();
}

int HNSW::generate_level(float ml, int max_level) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    r = std::max(r, 1e-10f);
    float val = -std::log(r) * ml;
    int level = static_cast<int>(val);
    return std::min(level, max_level);
}

std::vector<size_t> HNSW::select_neighbours(const Vector& q, std::vector<size_t> ids, int M) {
    std::unordered_map<int, float> id_dist_map;
    for (const size_t& id: ids) {
        id_dist_map[id] = this->cosine_similarity(id, q);
    }
    std::vector<size_t> sorted;
    for (const auto& [id, _]: id_dist_map) { sorted.push_back(id); };
    std::sort(sorted.begin(), sorted.end(), [&id_dist_map] (const int& a, const int& b) {
        return id_dist_map[a] > id_dist_map[b];
    });
    if (sorted.size() > M) { sorted.resize(M); }
    return sorted;
}

size_t HNSW::get_node_by_distance(const std::vector<size_t>& nodes, const Vector& q, Dist dist) {

    assert(!nodes.empty());
    std::vector<Vector> vecs;
    vecs.reserve(nodes.size());

    std::transform(nodes.begin(), nodes.end(), std::back_inserter(vecs), [this](size_t id) {
        return this->NodeMap[id].vector;
    });
    assert(vecs.size() == nodes.size());

    mx::array matrix = mx::squeeze(mx::stack(vecs));
    if (matrix.ndim() < 2) {
        matrix = mx::transpose(mx::expand_dims(matrix, 1));
    }
    assert(matrix.ndim() == 2);

    size_t idx;
    Vector dists = cosine_sim(matrix, mx::transpose(mx::expand_dims(q, 1)));
    switch (dist) {
        case Dist::nearest:
            idx = mx::argmax(dists).item<size_t>();
            break;   
        case Dist::farthest:
            idx = mx::argmin(dists).item<size_t>();
            break;
    };

    assert(idx < nodes.size());
    return nodes[idx];
}

UniqueVector<size_t> HNSW::search_layer(const Vector& q, size_t ep, int ef, int layer) {

    UniqueVector<size_t> visited(ep);
    UniqueVector<size_t> candidates(ep);
    UniqueVector<size_t> neighbours(ep);

    while (candidates.size() > 0) {
        size_t c = this->get_node_by_distance(candidates.data, q, Dist::nearest);
        size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);

        if (this->cosine_similarity(c, q) < this->cosine_similarity(f, q)) {
            break;
        }

        for (const size_t& e_id: this->NodeMap[c].Neighbours[layer]) {
            if (!visited.contains(e_id)) {
                visited.insert(e_id);
                size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);
                if (this->cosine_similarity(e_id, q) < this->cosine_similarity(f, q) || neighbours.size() < ef) {
                    candidates.insert(e_id);
                    neighbours.insert(e_id);
                    if (neighbours.size() > ef) {
                        size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);
                        neighbours.remove(f);
                    }
                }
            }
        }
        candidates.remove(c);
    }

    return neighbours;
}

void HNSW::bidirectional_connection(std::vector<size_t> ids, size_t id, int M, int layer) {
    for (const size_t& n: ids) {
        this->NodeMap[n].Neighbours[layer].push_back(id);
        this->NodeMap[id].Neighbours[layer].push_back(n);
        this->NodeMap[n].Neighbours[layer] = this->select_neighbours(
            this->NodeMap[n].vector,
            this->NodeMap[n].Neighbours[layer],
            M
        );
    };
    this->NodeMap[id].Neighbours[layer] = this->select_neighbours(
        this->NodeMap[id].vector,
        this->NodeMap[id].Neighbours[layer],
        M
    );
}

void HNSW::Insert(const Vector& q) {
    
    float ml = 1 / (-std::log(1 - (1 / this->M)));
    assert (ml != -INFINITY);

    if (!this->Ep) {
        int node_level = this->generate_level(ml, this->max_level);
        this->NodeMap.push_back(Node (q, node_level));
        this->Ep = this->NodeMap.size() - 1;
        this->total_nodes += 1;
        return;
    }

    size_t ep = this->Ep.value();
    int L = this->max_level;
    int l = this->generate_level(ml, this->max_level);
    this->NodeMap.push_back(Node(q, l));
    size_t new_node_id = this->NodeMap.size() - 1;

    for (int l_c = L; l_c >= l; --l_c) {
        UniqueVector<size_t> W = this->search_layer(q, ep, 1, l_c);
        ep = this->get_node_by_distance(W.data, q, Dist::nearest);
    }

    for (int l_c = std::min(L, l); l_c >= 0; --l_c) {
        UniqueVector<size_t> W = this->search_layer(q, ep, this->ef_construction, l_c);
        std::vector<size_t> neighbours = this->select_neighbours(q, W.data, this->M);
        this->bidirectional_connection(neighbours, new_node_id, this->M, l_c);
        ep = this->get_node_by_distance(W.data, q, Dist::nearest);
    }

    if (l > L) { this->Ep = new_node_id; }
    this->total_nodes += 1;
};

// std::vector<Vector> HNSW::Search (const Vector& q, int8_t K, int8_t efsearch) {};