import mlx.core as mx

type Vector = mx.array

class NodeNotFoundError(Exception):
    """Raised when a node ID is not found in the node map."""
    def __init__(self, message: str):
        super().__init__(f"Error in finding the node: {message}")

class CosineSimilarityError(Exception):
    """Raised during cosine similarity calculation."""
    def __init__(self, message: str):
        super().__init__(f"Error calculating cosine similarity: {message}")

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