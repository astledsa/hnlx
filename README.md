# HNLX

A Python library for Hierarchical Navigable Small World (HNSW) graphs, built using Apple's MLX framework for performant execution on Apple silicon.

The primary goal of this project is to provide an efficient and up-to-date implementation of vector search indexes, starting with HNSW. The library will continuously evolve by incorporating findings from the latest research to enhance performance and capabilities on Apple silicon.

## Key Features

*   **MLX-Powered Performance:** Leverages Apple's MLX framework to deliver highly optimized HNSW graph operations, taking full advantage of Apple Silicon's unified memory architecture and neural engine.
*   **Efficient Vector Search:** Provides fast and accurate approximate nearest neighbor (ANN) search capabilities for high-dimensional vector data.
*   **Research-Driven Development:** Continuously integrates insights and techniques from cutting-edge research papers to improve performance, scalability, and functionality.
*   **Hybrid Implementation:** Combines Python for ease of use and rapid prototyping with C++ for performance-critical components, seamlessly integrated via MLX.

## Technical Details

### Hierarchical Navigable Small World (HNSW) Algorithm

HNLX implements the HNSW algorithm, a state-of-the-art approach for approximate nearest neighbor search. HNSW constructs a multi-layer graph structure where each layer represents a different level of connectivity. This hierarchical organization allows for efficient navigation and search, significantly reducing the number of distance calculations required to find nearest neighbors compared to brute-force methods. The algorithm is particularly well-suited for large-scale vector databases due to its logarithmic search complexity and high recall rates.

### MLX Framework Integration

The core of HNLX's performance lies in its deep integration with Apple's MLX framework. MLX is a numerical computing library designed for efficient machine learning on Apple Silicon. By utilizing MLX, HNLX benefits from:

*   **Unified Memory:** Seamless data sharing between CPU and GPU, eliminating costly data transfers.
*   **Optimized Primitives:** Access to highly optimized low-level operations for vector and matrix computations.
*   **Automatic Differentiation:** While not directly used in the HNSW graph construction itself, MLX's capabilities provide a robust foundation for potential future extensions involving differentiable graph operations or integration with ML models.
*   **Hardware Acceleration:** Direct utilization of the Neural Engine and GPU cores on Apple Silicon for accelerated graph traversal and distance calculations.

The C++ components of HNLX are designed to interface directly with MLX's C++ API, ensuring minimal overhead and maximum performance. Python bindings are then provided to expose these high-performance operations to the user-friendly Python environment.

## Installation

(Installation instructions will be provided here)

## Usage

(Usage examples will be provided here)

## Future Directions

Future development will focus on exploring and implementing techniques from the following research papers:

- [d-HNSW](https://arxiv.org/abs/2505.11783)
- [Accelerating Graph Indexing for ANNS](https://arxiv.org/abs/2502.18113)
- [In-Place Updates of a Graph Index](https://arxiv.org/abs/2502.13826)
- [Dual-Branch HNSW](https://arxiv.org/abs/2501.13992)
- [Hubs NSW](https://arxiv.org/abs/2412.01940)
- [Self-healing HNSW](https://arxiv.org/html/2407.07871v1)
