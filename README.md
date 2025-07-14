# HNLX

This is a python library for the implementation of the HNSW graph, utilising the MLX array framework (for Apple silicon). This library is currently unfinished. A list of future features and to-dos is given below for further reference. The main goal is to keep improving this index in accordance with latest research (and adding more vector indexes) with focus on making them performant on apple silicon using mlx.

## Deliverable (v1.0)

- [x] Multiple distance functions (Cosine, Inner, Manhattan, Hamming, Jaccard, Euclidean)
- [x] C++ Implementation with MLX C++
- [ ] ARM NEON Vector instructions (optional)
- [ ] Improve the `search_neighbour` algorithm 
- [ ] Python, Golang and Typescript bindings 

## Future Features (>v1.0)

- [ ] Add visualization (3D)
- Explore current research and experiment 
   - [ ] [d-HNSW](https://arxiv.org/abs/2505.11783)
   - [ ] [Accelerating Graph Indexing for ANNS](https://arxiv.org/abs/2502.18113)
   - [ ] [In-Place Updates of a Graph Index](https://arxiv.org/abs/2502.13826)
   - [ ] [Dual-Branch HNSW](https://arxiv.org/abs/2501.13992)
   - [ ] [Hubs NSW](https://arxiv.org/abs/2412.01940)
   - [ ] [Self-healing HNSW](https://arxiv.org/html/2407.07871v1)


## Testing/Benchmarks

- [x] Basic tests
- [x] Comparison against popular implementations
