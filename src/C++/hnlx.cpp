#include <cmath>
#include <random>
#include "hnlx.h"
#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include "mlx/mlx.h"
#include <algorithm>

static constexpr int THREADGROUP_SIZE = 256;

namespace mx = mlx::core;

namespace
{
        static const auto nearest_index_fn = mx::compile([](const std::vector<mx::array> &inputs)
                                                         {
        const auto &dists = inputs[0];
        return std::vector<mx::array>{mx::argmax(mx::argmax(dists, 1))}; });

        static const auto farthest_index_fn = mx::compile([](const std::vector<mx::array> &inputs)
                                                          {
        const auto &dists = inputs[0];
        return std::vector<mx::array>{mx::argmin(mx::argmin(dists, 1))}; });
}

void PrintShape(Vector v)
{
        if (v.ndim() < 1)
        {
                std::cout << "no dim" << std::endl;
                return;
        }
        if (v.ndim() < 2)
        {
                std::cout << "(" << v.shape(0) << ")" << std::endl;
                return;
        }
        std::cout << "(" << v.shape(0) << "," << v.shape(1) << ")" << std::endl;
}

Vector cosine_sim(const Vector &v1, const Vector &v2)
{
        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();
        mx::array q = v1;
        mx::array v = mx::transpose(v2);

        mx::array q_norm = mx::linalg::norm(q);
        mx::array v_norm = mx::linalg::norm(v);
        mx::array dot_product = mx::matmul(q, v);
        mx::array denominator = mx::maximum(q_norm * v_norm, mx::array(0.000001f));

        return dot_product / denominator;
}

Vector get_slice(int index, const Vector &v)
{
        auto start_indices = {index, 0};
        auto end_indices = {index + 1, v.shape(1)};
        auto inner_array_slice = mx::slice(v, start_indices, end_indices);

        return mx::squeeze(inner_array_slice, 0);
}

HNSW::HNSW(HNSWParams params) : M(params.M),
                                pruning(false),
                                ef_construction(params.ef_construction),
                                total_nodes(0),
                                Ep(std::nullopt),
                                threshold(params.threshold),
                                max_level(params.max_level),
                                Time({{HNSWFunctions::main::insert, {}},
                                      {HNSWFunctions::main::search, {}},
                                      {HNSWFunctions::sub::searchLayer, {}},
                                      {HNSWFunctions::sub::selectNeightbour, {}},
                                      {HNSWFunctions::sub::gnbd, {}},
                                      {HNSWFunctions::sub::bidirectionalConn, {}},
                                      {HNSWFunctions::subsub::searchBreakCond, {}},
                                      {HNSWFunctions::subsub::cosSim, {}}})
{
}

Node::Node(Vector v, int max_level) : vector(v), max_level(max_level)
{
        for (int i = 0; i <= max_level; ++i)
        {
                Neighbours.emplace(i, std::vector<size_t>{});
        }
}

float HNSW::cosine_similarity(size_t id, const Vector &q)
{
        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        const Node &node = this->NodeMap[id];
        Vector c = cosine_sim(node.vector, q);

        auto end_initial_search_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_initial_search_loop - start_initial_search_loop;
        this->Time[HNSWFunctions::subsub::cosSim].push_back(duration.count());

        return c.item<float>();
}

bool HNSW::search_layer_break_condition(size_t c, size_t f, std::vector<Vector> q)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        Vector cv = this->NodeMap[c].vector;
        Vector fv = this->NodeMap[f].vector;
        Vector qv = mx::stack(q);
        Vector cv_qv = cosine_sim(mx::transpose(mx::expand_dims(cv, 1)), qv);
        Vector fv_qv = cosine_sim(mx::transpose(mx::expand_dims(fv, 1)), qv);

        auto end_initial_search_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_initial_search_loop - start_initial_search_loop;
        this->Time[HNSWFunctions::subsub::searchBreakCond].push_back(duration.count());

        float c_float = static_cast<float>(mx::mean(cv_qv).item<float>());
        float f_float = static_cast<float>(mx::mean(fv_qv).item<float>());

        return c_float < f_float;
}

int HNSW::generate_level(float ml, int max_level)
{
        static thread_local std::mt19937 generator(std::random_device{}());
        static thread_local std::uniform_real_distribution<float> distribution(1e-10f, 1.0f);

        float r = distribution(generator);
        float val = -std::log(r) * ml;
        int level = static_cast<int>(val);
        return std::min(level, max_level);
}

std::vector<UniqueVector<size_t>> HNSW::select_neighbours_batch(std::vector<Vector> points, std::vector<std::vector<size_t>> ids, int M)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        if (ids.empty())
        {
                return {};
        }
        std::vector<std::vector<size_t>> ks = ids;
        int B = static_cast<int>(ks.size());
        int d = static_cast<int>(this->NodeMap[ks[0][0]].vector.shape(0));
        int L = 0;
        for (const auto &inner_list : ks)
        {
                L = std::max(L, static_cast<int>(inner_list.size()));
        }

        // Vector X = mx::zeros(mx::Shape{B, L, d});
        // Vector mask = mx::zeros(mx::Shape{B, L});

        std::vector<Vector> to_be_stacked_X(B, mx::zeros(mx::Shape{L, d}));
        std::vector<Vector> to_be_stacked_mask_mlx;
        to_be_stacked_mask_mlx.reserve(B);

        for (int i = 0; i < ks.size(); i++)
        {
                std::vector<Vector> e_list;
                e_list.reserve(ks[i].size());
                std::transform(ks[i].begin(), ks[i].end(), std::back_inserter(e_list), [this](size_t id)
                               { return this->NodeMap[id].vector; });
                Vector e = mx::stack(e_list);
                int n = static_cast<int>(e.shape(0));

                to_be_stacked_X[i] = e;
                std::vector<float> current_mask_data(n, 1.0f);
                to_be_stacked_mask_mlx.push_back(mx::array(current_mask_data.data(), {n}, mx::float32));

                // Old code
                // X = mx::slice_update(
                //     X,
                //     e,
                //     {i, 0, 0},
                //     {i + 1, static_cast<int>(e.shape(0)), d});
                // mask = mx::slice_update(
                //     mask,
                //     mx::ones(mx::Shape{1, n}),
                //     {i, 0},
                //     {i + 1, n});
                // End of old code
        }

        Vector X = mx::stack(to_be_stacked_X);
        Vector mask = mx::stack(to_be_stacked_mask_mlx);

        Vector V = mx::stack(points);
        X = X / mx::linalg::norm(X, -1, true);
        V = V / mx::linalg::norm(V, -1, true);
        Vector S = mx::sum(X * mx::expand_dims(V, 1), -1) + (mask - 1) * 1e9;
        Vector top = mx::argsort(-S, 1);

        assert(ks.size() == static_cast<size_t>(top.shape(0)));
        std::vector<UniqueVector<size_t>> neighbours = {};

        for (int i = 0; i < ks.size(); i++)
        {
                UniqueVector<size_t> current = {};
                for (int j = 0; j < std::min(M, top.shape(1)); j++)
                {
                        int top_index_i_j = get_index<int>(top, i, j);
                        if (top_index_i_j < ks[i].size())
                        {
                                current.insert(ks[i][top_index_i_j]);
                        }
                }
                neighbours.push_back(current);
        }

        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start_initial_search_loop;
        this->Time[HNSWFunctions::sub::selectNeightbour].push_back(duration.count());

        return neighbours;
}

std::vector<size_t> HNSW::select_neighbours(const Vector &q, std::vector<size_t> ids, int M)
{
        std::unordered_map<int, float> id_dist_map;
        for (const size_t &id : ids)
        {
                id_dist_map[id] = this->cosine_similarity(id, q);
        }
        std::vector<size_t> sorted;
        for (const auto &[id, _] : id_dist_map)
        {
                sorted.push_back(id);
        };
        std::sort(sorted.begin(), sorted.end(), [&id_dist_map](const int &a, const int &b)
                  { return id_dist_map[a] > id_dist_map[b]; });
        if (sorted.size() > M)
        {
                sorted.resize(M);
        }
        return sorted;
}

size_t HNSW::get_node_by_distance(const std::vector<size_t> &nodes, std::vector<Vector> q, Dist dist)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        std::vector<Vector> vecs;
        vecs.reserve(nodes.size());
        std::transform(nodes.begin(), nodes.end(), std::back_inserter(vecs), [this](size_t id)
                       { return this->NodeMap[id].vector; });
        mx::array matrix = mx::squeeze(mx::stack(vecs));
        if (matrix.ndim() < 2)
        {
                matrix = mx::transpose(mx::expand_dims(matrix, 1));
        }

        Vector dists = cosine_sim(matrix, mx::stack(q));
        Vector res = (dist == Dist::nearest)
                         ? nearest_index_fn({dists})[0]
                         : farthest_index_fn({dists})[0];

        size_t idx = res.item<size_t>();

        this->Time[HNSWFunctions::sub::gnbd].push_back(
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - start_initial_search_loop)
                .count());

        return nodes[idx];
}

UniqueVector<size_t> HNSW::search_layer(std::vector<Vector> q, size_t ep, int ef, int layer)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        UniqueVector<size_t> visited(ep);
        UniqueVector<size_t> candidates(ep);
        UniqueVector<size_t> neighbours(ep);

        while (candidates.size() > 0)
        {
                size_t c = this->get_node_by_distance(candidates.data, q, Dist::nearest);
                size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);

                if (this->search_layer_break_condition(c, f, q))
                {
                        break;
                }

                for (const size_t &e_id : this->NodeMap[c].Neighbours[layer])
                {
                        if (!visited.contains(e_id))
                        {
                                visited.insert(e_id);
                                size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);
                                if (this->search_layer_break_condition(e_id, f, q) || neighbours.data.size() < ef)
                                {
                                        candidates.insert(e_id);
                                        neighbours.insert(e_id);
                                        if (neighbours.size() > ef)
                                        {
                                                size_t f = this->get_node_by_distance(neighbours.data, q, Dist::farthest);
                                                neighbours.remove(f);
                                        }
                                }
                        }
                }

                candidates.remove(c);
        }

        auto end_initial_search_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_initial_search_loop - start_initial_search_loop;
        this->Time[HNSWFunctions::sub::searchLayer].push_back(duration.count());

        return neighbours;
}

void HNSW::bidirectional_connection(std::vector<size_t> ids, size_t id, int M, int layer)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        for (const size_t &n : ids)
        {
                this->NodeMap[n].Neighbours[layer].push_back(id);
                this->NodeMap[id].Neighbours[layer].push_back(n);
        };

        auto end_initial_search_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_initial_search_loop - start_initial_search_loop;
        this->Time[HNSWFunctions::sub::bidirectionalConn].push_back(duration.count());

        if (this->pruning)
        {
                // Implement Pruning logic here
        };
}

void HNSW::Insert(const Vector &q)
{

        auto start_initial_search_loop = std::chrono::high_resolution_clock::now();

        float ml = 1 / (-std::log(1 - (1 / this->M)));
        assert(ml != -INFINITY);

        if (!this->Ep)
        {
                int node_level = this->generate_level(ml, this->max_level);
                this->NodeMap.push_back(Node(q, node_level));
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

        if (this->cache.Data.size() == this->threshold)
        {
                int max_lvl = this->cache.get_max_level();

                for (int l_c = max_lvl; l_c >= 0; --l_c)
                {
                        std::vector<Vector> batch_vector = this->cache.get_vectors(l_c);
                        UniqueVector<size_t> W = this->search_layer(batch_vector, ep, 1, l_c);
                        ep = this->get_node_by_distance(W.data, batch_vector, Dist::nearest);
                }

                for (int l_c = max_lvl; l_c >= 0; --l_c)
                {
                        std::vector<size_t> batch_ids = this->cache.get_ids(l_c);
                        std::vector<Vector> batch_vectors = this->cache.get_vectors(l_c);
                        UniqueVector<size_t> W = this->search_layer(batch_vectors, ep, this->ef_construction, l_c);
                        std::vector<std::vector<size_t>> copied_W(batch_vectors.size(), W.data);
                        std::vector<UniqueVector<size_t>> neighbours = this->select_neighbours_batch(
                            batch_vectors,
                            copied_W,
                            this->M
                        );
                        assert(neighbours.size() == batch_ids.size());

                        for (int i = 0; i < neighbours.size(); i++)
                        {
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
                this->total_nodes += this->cache.Data.size();
                this->cache.Data.clear();
        }

        auto end_initial_search_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_initial_search_loop - start_initial_search_loop;
        this->Time[HNSWFunctions::main::insert].push_back(duration.count());
};
