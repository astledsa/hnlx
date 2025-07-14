#include <cmath>
#include <random>
#include "hnlx.h" 
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include "mlx/mlx.h"
#include <algorithm>

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

Vector cosine_sim (const Vector& v1, const Vector& v2) {
    mx::array q = v1; 
    mx::array v = mx::transpose(v2);

    mx::array q_norm = mx::linalg::norm(q); 
    mx::array v_norm = mx::linalg::norm(v);
    mx::array dot_product = mx::matmul(q, v);
    mx::array denominator = mx::maximum(q_norm * v_norm, mx::array(0.000001f));

    return  dot_product / denominator;;
}

Vector get_slice (int index, const Vector& v) {
    auto start_indices = {index, 0};
    auto end_indices = {index + 1, v.shape(1)};
    auto inner_array_slice = mx::slice(v, start_indices, end_indices);

    return mx::squeeze(inner_array_slice, 0);
}

HNSW::HNSW (int M, int ef_construction, int threshold) :
    M(M), pruning(false), ef_construction(ef_construction), total_nodes(0), Ep(std::nullopt), threshold(threshold) {}

Node::Node (Vector v, int max_level) :
    vector(v), max_level(max_level) {
    for (int i = 0; i <= max_level; ++i) {
        Neighbours.emplace(i, std::vector<size_t>{});
    }
}

float HNSW::cosine_similarity(size_t id, const Vector& q) {
    const Node& node = this->NodeMap[id];
    Vector c =  cosine_sim (node.vector, q);
    return c.item<float>();
}

bool HNSW::search_layer_break_condition (size_t c, size_t f, std::vector<Vector> q) {
    Vector cv = this->NodeMap[c].vector;
    Vector fv = this->NodeMap[f].vector;
    Vector qv = mx::stack(q);
    Vector cv_qv = cosine_sim(mx::transpose(mx::expand_dims(cv, 1)), qv);
    Vector fv_qv = cosine_sim(mx::transpose(mx::expand_dims(fv, 1)), qv);
    return mx::all(cv_qv < fv_qv).item<bool>();
}

int HNSW::generate_level(float ml, int max_level) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    r = std::max(r, 1e-10f);
    float val = -std::log(r) * ml;
    int level = static_cast<int>(val);
    return std::min(level, max_level);
}

std::vector<UniqueVector<size_t>> HNSW::select_neighbours_batch(
    std::vector<Vector> points, 
    std::vector<std::vector<size_t>> ids, 
    int M
) {
    if (ids.empty()) { return {}; }
    std::vector<std::vector<size_t>> ks = ids;
    int B = static_cast<int>(ks.size());
    int d = static_cast<int>(this->NodeMap[ks[0][0]].vector.shape(0));
    int L = 0;
    for (const auto& inner_list : ks) {
        L = std::max(L, static_cast<int>(inner_list.size()));
    }
    Vector X = mx::zeros(mx::Shape{B, L, d});
    Vector mask = mx::zeros(mx::Shape{B, L});
    for (int i = 0; i < ks.size(); i++) {
        std::vector<Vector> e_list;
        e_list.reserve(ks[i].size());
        std::transform(ks[i].begin(), ks[i].end(), std::back_inserter(e_list), [this](size_t id) {
            return this->NodeMap[id].vector;
        });
        Vector e = mx::stack(e_list);
        int n = static_cast<int>(e.shape(0));
        X = mx::slice_update(
            X,
            e,
            {i, 0},
            {i+1, n}
        );
        mask = mx::slice_update(
            mask, 
            mx::ones(mx::Shape{1, n}), 
            {i, 0},
            {i+1, n}
        );
    }
    Vector V = mx::stack(points);
    X = X / mx::linalg::norm(X, -1, true);
    V = V / mx::linalg::norm(V, -1, true);
    Vector S = mx::sum(X * mx::expand_dims(V, 1), -1) + (mask - 1) * 1e9;
    Vector top = mx::argsort(-S, 1);

    assert(M < top.shape(1));
    assert(ks.size() == static_cast<size_t>(top.shape(0)));
    std::vector<UniqueVector<size_t>> neighbours = {};

    for (int i = 0; i < ks.size(); i++) {
        UniqueVector<size_t> current = {};
        for (int j = 0; j < M; j++) {
            size_t top_index_i_j = get_index<size_t>(top, i, j);
            if (top_index_i_j < ks[i].size()) {
                current.insert(top_index_i_j);
            }
        }
        neighbours.push_back(current);
    }
    return neighbours;
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

size_t HNSW::get_node_by_distance(const std::vector<size_t>& nodes, std::vector<Vector> q, Dist dist) {

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
    Vector dists = cosine_sim(matrix, mx::transpose(mx::stack(q)));
    switch (dist) {
        case Dist::nearest:
            idx = mx::argmax(mx::argmax(dists, 1)).item<size_t>();
            break;   
        case Dist::farthest:
            idx = mx::argmin(mx::argmin(dists, 1)).item<size_t>();
            break;
    };

    assert(idx < nodes.size());
    return nodes[idx];
}

UniqueVector<size_t> HNSW::search_layer(std::vector<Vector> q, size_t ep, int ef, int layer) {

    UniqueVector<size_t> visited(ep);
    UniqueVector<size_t> candidates(ep);
    UniqueVector<size_t> neighbours(ep);

    while (candidates.size() > 0) {
        size_t c = this->get_node_by_distance(candidates.data, q, Dist::nearest);
        size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);

        if (this->search_layer_break_condition(c, f, q)) {
            break;
        }

        for (const size_t& e_id: this->NodeMap[c].Neighbours[layer]) {
            if (!visited.contains(e_id)) {
                visited.insert(e_id);
                size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);
                if (this->search_layer_break_condition(e_id, f, q) || neighbours.size() < ef) {
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
    };
    
    if (this->pruning) { 
        // Implement Pruning logic here
    };
}

void HNSW::Insert(const Vector& q) {
    
    float ml = 1 / (-std::log(1 - (1 / this->M)));
    assert (ml != -INFINITY);

    if (!this->Ep) {
        int node_level = this->generate_level(ml, this->max_level);
        this->NodeMap.push_back(Node (q, node_level));
        this->Ep = this->NodeMap.size() - 1;
        this->cache.Insert(this->Ep.value(), q, node_level);
        this->total_nodes += 1;
        return;
    }

    size_t ep = this->Ep.value();
    int l = this->generate_level(ml, this->max_level);
    this->NodeMap.push_back(Node(q, l));
    size_t new_node_id = this->NodeMap.size() - 1;
    this->cache.Insert(new_node_id, q, l);

    if (this->cache.Data.size() == this->threshold) {
        int max_lvl = this->cache.get_max_level();
        for (int l_c = max_lvl; l_c >= 0; --l_c) {
            std::vector<Vector> batch_vector = this->cache.get_vectors(l_c);
            UniqueVector<size_t> W = this->search_layer(batch_vector, ep, 1, l_c);
            ep = this->get_node_by_distance(W.data, batch_vector, Dist::nearest);
        }

        for (int l_c = max_lvl; l_c >= 0; --l_c) {
            std::vector<size_t> batch_ids = this->cache.get_ids(l_c);
            std::vector<Vector> batch_vectors = this->cache.get_vectors(l_c);
            UniqueVector<size_t> W = this->search_layer(batch_vectors, ep, this->ef_construction, l_c);
            std::vector<std::vector<size_t>> copied_W (batch_vectors.size(), W.data);
            std::vector<UniqueVector<size_t>> neighbours = this->select_neighbours_batch(
                batch_vectors, 
                copied_W, 
                this->M
            );

            assert(neighbours.size() == batch_ids.size());
            for (int i = 0; i < neighbours.size(); i++) {
                this->bidirectional_connection(
                    neighbours[i].data,
                    batch_ids[i],
                    this->M,
                    l_c
                );
            };

            ep = this->get_node_by_distance(W.data, batch_vectors, Dist::nearest);
        }

        this->Ep = this->cache.get_entry_id();
        this->total_nodes += 1;
        this->cache.Data.clear();
    }
};

// std::vector<Vector> HNSW::Search (const Vector& q, int K, int efsearch) {
//     assert(K <= efsearch);
//     assert(this->Ep != std::nullopt);
//     assert (std::find(this->NodeMap.begin(), this->NodeMap.end(), this->Ep) != this->NodeMap.end());

//     size_t ep = this->Ep.value();
//     int L = this->NodeMap[ep].max_level;

//     for (int l_c = L; l_c >= 1; --l_c) {
//         UniqueVector<size_t> W = this->search_layer(q, ep, efsearch, l_c);
//         ep = this->get_node_by_distance(W.data, q, Dist::nearest);
//     }

//     UniqueVector<size_t> nearest_ids = this->search_layer(q, ep, efsearch, 0);
//     std::vector<std::tuple<float, Vector>> vec_dist_tuple;

// };