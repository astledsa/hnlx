# HNLX 

This is a python library for the implementation of the HNSW graph, utilising the MLX array framework (for Apple silicon). This library is currently unfinished. A list of future features and to-dos is given below for further reference. The main goal is to keep improving this index in accordance with latest research (and adding more vector indexes) with focus on making them performant on apple silicon using mlx.

## Features

- [x] Multiple distance functions (Cosine, Inner, Manhattan, Hamming, Jaccard, Euclidean)
- [ ] Improve the `search_neighbour` algorithm 
- [ ] Add visualization (3D)
- Explore current research and experiment 
   - [ ] [d-HNSW](https://arxiv.org/abs/2505.11783)
   - [ ] [Accelerating Graph Indexing for ANNS](https://arxiv.org/abs/2502.18113)
   - [ ] [In-Place Updates of a Graph Index](https://arxiv.org/abs/2502.13826)
   - [ ] [Dual-Branch HNSW](https://arxiv.org/abs/2501.13992)
   - [ ] [Hubs NSW](https://arxiv.org/abs/2412.01940)

## Testing/Benchmarks

- [x] Basic tests
- [ ] Comparison against popular implementations
