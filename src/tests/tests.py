import hnswlib
import numpy as np
import datetime
import mlx.nn as nn
import mlx.core as mx
from Python.hnsw import HNSW
from Python.models import Vector, TestConfig, Report

def BruteForceTest (values: TestConfig) -> Report:
    
    def brute_force_search (vectors: list[Vector], Q: Vector, K: int) -> list[tuple[float, Vector]] :
        q: Vector = mx.expand_dims(Q, axis=1)
        vecs: list[Vector] = [mx.expand_dims(vec, axis=1) for vec in vectors]
        vec_dist_tuple = [(float(nn.losses.cosine_similarity_loss(vec.T, q.T)), vec) for vec in vecs]
        sorted_vecs = sorted(vec_dist_tuple, key=lambda x: x[0], reverse=True)
        return sorted_vecs[:K]
    
    def Precision (b: list[tuple[float, Vector]], k: list[tuple[float, Vector]]) -> float:
        bv: list[Vector] = [mx.round(x[1], 3) for x in b]
        kv: list[Vector] = [mx.round(x[1], 3) for x in k]
        
        n = 0
        for i in range(len(bv)):
            if mx.array_equal(bv[i].squeeze(), kv[i]):
                n += 1     
        return n / len(bv)
        
    
    num_vectors: int = values.num_vectors
    dimensions: int = values.vec_dimensions
    
    index: HNSW = HNSW(values.M, values.efconstruction, 4, pruning=values.Prune, threshold=values.Batch_Threshold)
    random_vectors: list[Vector] = [
        mx.random.normal(shape=(dimensions,)) for _ in range(num_vectors)
    ]  
    
    start = datetime.datetime.now()
    for i in range(num_vectors):
        index.Insert(random_vectors[i])
    end = datetime.datetime.now()
    
    construction_duration = end - start
    query: Vector = mx.random.normal(shape=(dimensions,))
    bruteResults: list[tuple[float, Vector]] = brute_force_search(random_vectors, query, values.K)
    
    start = datetime.datetime.now()
    knnResults: list[tuple[float, Vector]] = index.Search(query, values.K, values.efsearch)
    end = datetime.datetime.now()
    
    search_duration = end - start
    precision: float = Precision(bruteResults, knnResults)
    index.TimeReport()

    return Report(
                precision=precision, 
                search_time=search_duration.total_seconds(), 
                construction_time=construction_duration.total_seconds()
            )

def ComparisonTest(values: TestConfig) -> Report:

    def l2_normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def brute_force_search(vectors: np.ndarray, Q: np.ndarray, K: int) -> list[tuple[float, np.ndarray]]:
        q = Q.reshape(1, -1)
        similarities = np.dot(vectors, q.T).squeeze()
        top_k_indices = np.argsort(similarities)[-K:][::-1]
        return [(similarities[i], vectors[i]) for i in top_k_indices]

    def Precision(b: list[tuple[float, np.ndarray]], k: list[tuple[float, np.ndarray]]) -> float:
        bv = [np.round(x[1], 3) for x in b]
        kv = [np.round(x[1], 3) for x in k]
        n = 0
        for i in range(len(bv)):
            if np.array_equal(bv[i], kv[i]):
                n += 1
        return n / len(bv)

    xb = l2_normalize(np.random.normal(0, 2, size=(values.num_vectors, values.vec_dimensions)).astype(np.float32))
    xq = l2_normalize(np.random.normal(0, 2, size=(1, values.vec_dimensions)).astype(np.float32))

    index = hnswlib.Index(space="cosine", dim=values.vec_dimensions)
    index.init_index(max_elements=values.num_vectors, ef_construction=values.efconstruction, M=values.M)

    start = datetime.datetime.now()
    index.add_items(xb, ids=np.arange(values.num_vectors))
    end = datetime.datetime.now()
    construction_duration = end - start

    index.set_ef(values.efsearch)

    start = datetime.datetime.now()
    labels, distances = index.knn_query(xq, k=values.K)
    end = datetime.datetime.now()
    search_duration = end - start

    knnResults = [(1 - float(distances[0][i]), xb[labels[0][i]]) for i in range(values.K)]
    bruteResults = brute_force_search(xb, xq[0], values.K)
    precision = Precision(bruteResults, knnResults)

    return Report(
        precision=precision,
        search_time=search_duration.total_seconds(),
        construction_time=construction_duration.total_seconds()
    )

config1 = TestConfig(
    M=16,             
    K=5,             
    efsearch=1000,     
    num_vectors=100, 
    efconstruction=200, 
    vec_dimensions=128,
    Prune=False,
    Batch_Threshold=100
)

print(BruteForceTest(config1).model_dump_json(indent=2))
