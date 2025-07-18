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
#include <iomanip>

namespace mx = mlx::core;
using Vector = mx::array;

enum class Dist
{
    nearest,
    farthest
};

namespace HNSWFunctions
{
    enum class main
    {
        insert,
        search
    };
    enum class sub
    {
        searchLayer,
        selectNeightbour,
        gnbd,
        bidirectionalConn
    };
    enum class subsub
    {
        searchBreakCond,
        cosSim
    };
};

using HNSWKeyType = std::variant<
    HNSWFunctions::main,
    HNSWFunctions::sub,
    HNSWFunctions::subsub>;

struct HNSWKeyHasher
{
    std::size_t operator()(const HNSWKeyType &k) const
    {
        return std::visit([](const auto &val)
                          {
                              using T = std::decay_t<decltype(val)>;

                              if constexpr (std::is_same_v<T, HNSWFunctions::main>)
                              {
                                  return std::hash<int>{}(static_cast<int>(val));
                              }
                              else if constexpr (std::is_same_v<T, HNSWFunctions::sub>)
                              {
                                  return std::hash<int>{}(static_cast<int>(val));
                              }
                              else if constexpr (std::is_same_v<T, HNSWFunctions::subsub>)
                              {
                                  return std::hash<int>{}(static_cast<int>(val));
                              } },
                          k);
    }
};

struct HNSWParams
{
    int M;
    int max_level;
    int threshold;
    int ef_construction;
};

inline std::string hnswKeyTypeToString(const HNSWKeyType &key)
{
    return std::visit([](const auto &val) -> std::string
                      {
                          using T = std::decay_t<decltype(val)>;

                          if constexpr (std::is_same_v<T, HNSWFunctions::main>)
                          {
                              switch (val)
                              {
                              case HNSWFunctions::main::insert:
                                  return "HNSWFunctions::main::Insert";
                              case HNSWFunctions::main::search:
                                  return "HNSWFunctions::main::Search";
                              }
                          }
                          else if constexpr (std::is_same_v<T, HNSWFunctions::sub>)
                          {
                              switch (val)
                              {
                              case HNSWFunctions::sub::searchLayer:
                                  return "HNSWFunctions::sub::SearchLayer";
                              case HNSWFunctions::sub::selectNeightbour:
                                  return "HNSWFunctions::sub::SelectNeighbour";
                              case HNSWFunctions::sub::gnbd:
                                  return "HNSWFunctions::sub::GNBD";
                              case HNSWFunctions::sub::bidirectionalConn:
                                  return "HNSWFunctions::sub::BidirectionalConn";
                              }
                          }
                          else if constexpr (std::is_same_v<T, HNSWFunctions::subsub>)
                          {
                              switch (val)
                              {
                              case HNSWFunctions::subsub::searchBreakCond:
                                  return "HNSWFunctions::subsub::SearchBreakCond";
                              case HNSWFunctions::subsub::cosSim:
                                  return "HNSWFunctions::subsub::CosineSim";
                              }
                          }
                          return "Unknown_HNSWKeyType"; },
                      key);
}

template <typename T>
T get_index(const Vector &array, int i, int j)
{
    if (array.ndim() != 2)
    {
        throw std::invalid_argument(
            "[get_index] Input array must be 2-dimensional (a matrix).");
    }

    if (i < 0 || i >= array.shape(0) || j < 0 || j >= array.shape(1))
    {
        throw std::out_of_range(
            "[get_index] Index (" + std::to_string(i) + ", " +
            std::to_string(j) + ") is out of bounds for array with shape (" +
            std::to_string(array.shape(0)) + ", " + std::to_string(array.shape(1)) + ").");
    }

    auto start_indices = {i, j};
    auto end_indices = {i + 1, j + 1};
    auto element_slice = mx::slice(array, start_indices, end_indices);

    return element_slice.item<T>();
}

template <typename T>
class UniqueVector
{
private:
    std::unordered_set<T> unique;

public:
    std::vector<T> data;
    UniqueVector() {};

    explicit UniqueVector(const T &val)
    {
        insert(val);
    }
    bool contains(const T &val)
    {
        return unique.find(val) != unique.end();
    }
    void insert(const T &val)
    {
        if (unique.insert(val).second)
        {
            data.push_back(val);
        }
    }
    void remove(const T &val)
    {
        if (unique.erase(val))
        {
            auto it = std::find(data.begin(), data.end(), val);
            if (it != data.end())
            {
                data.erase(it);
            }
        }
    }
    const T &operator[](size_t idx) const { return data[idx]; }
    size_t size() const { return data.size(); }
};

class Cache
{
public:
    std::vector<std::tuple<int, size_t, Vector>> Data;

    Cache() {};

    void Insert(size_t id, Vector q, int lvl)
    {
        Data.push_back(std::make_tuple(lvl, id, q));
    }

    std::vector<size_t> get_ids(int lvl)
    {
        std::vector<size_t> result_ids;
        for (const auto &item : Data)
        {
            if (std::get<0>(item) >= lvl)
            {
                result_ids.push_back(std::get<1>(item));
            }
        }
        if (result_ids.empty())
        {
            throw std::logic_error("empty return: IDs");
        }
        return result_ids;
    }

    std::vector<Vector> get_vectors(int lvl)
    {
        std::vector<Vector> result_vectors;
        for (const auto &item : Data)
        {
            if (std::get<0>(item) >= lvl)
            {
                result_vectors.push_back(std::get<2>(item));
            }
        }
        if (result_vectors.empty())
        {
            throw std::logic_error("empty return: Vectors");
        }
        return result_vectors;
    }

    int get_max_level()
    {
        if (Data.empty())
        {
            return 0;
        }
        std::vector<std::tuple<int, size_t, Vector>>::iterator max_ID = std::max_element(Data.begin(), Data.end(),
                                                                                         [](const auto &a, const auto &b)
                                                                                         {
                                                                                             return std::get<0>(a) < std::get<0>(b);
                                                                                         });
        return std::get<0>(*max_ID);
    }

    size_t get_entry_id()
    {
        int max_lvl = get_max_level();
        for (const std::tuple<int, size_t, Vector> &tup : Data)
        {
            if (std::get<0>(tup) == max_lvl)
            {
                return std::get<1>(tup);
            }
        }
        throw std::logic_error("No max ID found.");
    }
};

class Node
{
public:
    int max_level;
    Vector vector;
    std::map<int, std::vector<size_t>> Neighbours;
    Node(Vector q, int max_level);
};

class HNSW
{
public:
    float M;
    bool pruning;
    int threshold;
    int max_level;
    int total_nodes;
    int ef_construction;
    std::optional<size_t> Ep;
    HNSW(HNSWParams params);
    void Insert(const Vector &vector);
    std::unordered_map<HNSWKeyType, std::vector<float>, HNSWKeyHasher> Time;
    void Report()
    {
        std::cout << std::fixed << std::setprecision(6);

        std::cout << "--- HNSW Performance Report ---" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << std::setw(30) << std::left << "Function"
                  << std::setw(18) << std::right << "Total Time (s)"
                  << std::setw(25) << std::right << "Avg Time (s/call)"
                  << std::setw(15) << std::right << "Call Count" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        for (const auto &pair : Time)
        {
            const HNSWKeyType &function_key = pair.first;
            const std::vector<float> &times = pair.second;

            std::string function_name = hnswKeyTypeToString(function_key);

            double total_time = 0.0;
            size_t call_count = 0;
            double average_time = 0.0;

            if (!times.empty())
            {
                total_time = std::accumulate(times.begin(), times.end(), 0.0);
                call_count = times.size();
                average_time = total_time / call_count;
            }

            std::cout << std::setw(30) << std::left << function_name;
            std::cout << std::setw(18) << std::right << total_time;
            if (call_count > 0)
            {
                std::cout << std::setw(25) << std::right << average_time;
            }
            else
            {
                std::cout << std::setw(25) << std::right << "N/A";
            }
            std::cout << std::setw(15) << std::right << call_count;
            std::cout << std::endl;
        }
        std::cout << "---------------------------------------------------------" << std::endl;
        std::cout << std::endl;
    }
    // std::vector<Vector> Search (const Vector& q, int K, int efsearch);

private:
    Cache cache;
    std::vector<Node> NodeMap;
    int generate_level(float ml, int max_level);
    float cosine_similarity(size_t id, const Vector &q);
    bool search_layer_break_condition(size_t c, size_t f, std::vector<Vector> q);
    std::vector<size_t> select_neighbours(const Vector &q, std::vector<size_t> ids, int M);
    size_t get_node_by_distance(const std::vector<size_t> &nodes, std::vector<Vector> q, Dist dist);
    UniqueVector<size_t> search_layer(std::vector<Vector> q, size_t ep, int ef, int layer);
    void bidirectional_connection(std::vector<size_t> ids, size_t id, int M, int layer);
    std::vector<UniqueVector<size_t>> select_neighbours_batch(
        std::vector<Vector> points,
        std::vector<std::vector<size_t>> ids,
        int M);
};

#endif
