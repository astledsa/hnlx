import datetime
import mlx.nn as nn
import mlx.core as mx
from hnlx.hnsw import HNSW
from hnlx.models import Vector, TestConfig, Report

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
    
    index: HNSW = HNSW(values.M, values.efconstruction, 4)
    random_vectors: list[Vector] = [
        mx.random.bernoulli(shape=(dimensions,)) for _ in range(num_vectors)
    ]
    
    start = datetime.datetime.now()
    for i in range(num_vectors):
        index.Insert(random_vectors[i])
    end = datetime.datetime.now()
    
    construction_duration = end - start
    query: Vector = mx.random.bernoulli(shape=(dimensions,))
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

config1 = TestConfig(
    M=16,             
    K=10,             
    efsearch=100,     
    num_vectors=50, 
    efconstruction=200, 
    vec_dimensions=128
)
