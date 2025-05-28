import mlx.core as mx
from typing import Literal
from pydantic import BaseModel

type Vector = mx.array
type distance = Literal['Cosine', 'Hamming', 'Jaccard', 'Eucilidean', 'Inner', 'Manhattan']

class TestConfig (BaseModel):
    M: int
    K: int
    efsearch: int
    num_vectors: int
    efconstruction: int
    vec_dimensions: int

class Report (BaseModel):
    precision: float
    search_time: float
    construction_time: float

class NodeNotFoundError(Exception):
    """Raised when a node ID is not found in the node map."""
    def __init__(self, message: str):
        super().__init__(f"Error in finding the node: {message}")

class CosineSimilarityError(Exception):
    """Raised during cosine similarity calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating cosine similarity: {message}")

class EuclideanDistanceError(Exception):
    """Raised during euclidean distance calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating euclidean distance: {message}")

class ManhattanDistanceError(Exception):
    """Raised during manhattan distance calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating manhattan distance: {message}")

class InnerProductError(Exception):
    """Raised during inner product calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating inner product: {message}")

class HammingDistanceError(Exception):
    """Raised during hamming distance calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating hamming distance: {message}")

class JaccardDistanceError(Exception):
    """Raised during jaccard distance calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating jaccard distance: {message}")

class LevelGenerationError (Exception):
    """Raise this error if the generate level function fails"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error in generating a new level: {message}")

class NodeInsertionError (Exception):
    """Raise if there's a problem while storing a node"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error in storing the node: {message}")

class CosineSimDistanceError (Exception):
    """Raise if during comparing vector similarities"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while calculating distances between nodes and query: {message}")

class NeighbourSelectionError (Exception):
    """Raise when there's a problem while selecting a node's neighbour"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while selecting neighbours of a node: {message}")

class BidirectionalConnectionError (Exception):
    """Raise when creating bidirectional connections between nodes"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while creating bidirectional connections: {message}")

class SearchLayerError (Exception):
    """Raise when searching a layer for closest neighbours"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while searching a layer: {message}")

class InsertionError (Exception):
    """Raise when there's an error in inserting a vector"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while inserting the vector: {message}")

class SearchError (Exception):
    """Raise while performing K-NN search over the index"""
    def __init__(self, message: str) -> None:
        super().__init__(f"Error while search the vector over the index: {message}")

# def store_execution_time(attribute_name='_execution_time'):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper(self, *args, **kwargs):
#             start_time = time.perf_counter()
#             result = func(self, *args, **kwargs)
#             end_time = time.perf_counter()
#             elapsed_time = end_time - start_time
#             setattr(self, attribute_name, elapsed_time)
#             return result
#         return wrapper
#     return decorator