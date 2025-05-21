import json
import uuid
import math
import random
import datetime
import mlx.nn as nn
from .models import *
import mlx.core as mx
from typing import Literal, Optional, Callable

class Node :
    """Represents a node in the HNSW graph with vector data and neighbors across levels."""
    
    def __init__(self, vector: Vector, level: int) -> None:
        """
        Initialize a node with a vector and level.

        Args:
            vector (Vector): The vector representation of the node.
            level (int): The maximum level of the node in the graph.
        """
        self.max_level: int = level if level != 0 else 1
        self.vector: Vector = vector
        self.neighbours: dict[int, dict[str, None]] = {
            i: dict() for i in range(self.max_level+1)
        }
    
    def __eq__(self, other: object) -> bool:
        """
        Checks equality between nodes based on their vectors.

        Args:
            other (object): The other node to compare.

        Returns:
            bool: True if vectors are equal, else False.
        """
        if not isinstance(other, Node):
            return False
        return bool(self.vector == other.vector)
                    
class HNSW :
    """Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search."""
    
    def __init__(self, M: int, efc: int, max_level: int, dist: distance = 'Cosine') -> None:
        """
        Initialize the HNSW graph.

        Args:
            M (int): Maximum number of connections per node.
            efc (int): Size of the dynamic candidate list during construction.
            max_level (int): Maximum level allowed in the hierarchy.
        """
        self.M: int = M
        self.total_nodes: int = 0
        self.efconstruction: int = efc
        self.max_level: int = max_level
        self.nodeMap: dict[str, Node] = {}
        self.entry_point_id: Optional[str] = None
        self.timeMap: dict[str, list[float]] = {
            'bidirectional connection': list(),
            'search layer': list(),
            'insert': list(),
            'search': list(),
            'cosine distance': list(),
            'vector distance': list(),
            'search neighbours': list()
        }
        
        if dist == 'Manhattan':
            self.distance: Callable[[str, Vector], float] = self.__manhattan__
            
        elif dist == 'Eucilidean':
            self.distance: Callable[[str, Vector], float] = self.__euclidean__
            
        elif dist == 'Hamming':
            self.distance: Callable[[str, Vector], float] = self.__hamming__
            
        elif dist == 'Inner':
            self.distance: Callable[[str, Vector], float] = self.__inner_product__
            
        elif dist == 'Jaccard':
            self.distance: Callable[[str, Vector], float] = self.__jaccard__
            
        else:
            self.distance: Callable[[str, Vector], float] = self.__cos_sim__
    
    def __repr__(self) -> str:
        """
        For printing and debugging
        """
        return f"""
                HNSW (
                    M: {self.M}, 
                    efconstruction: {self.efconstruction}, 
                    distance: {self.distance}, 
                    max_level: {self.max_level},
                    total_nodes: {self.total_nodes}
                )
                """
    
    def __generate_level__ (self, ml: float, max_level: int) -> int:
        """
        Generate a random level for a new node using an exponential distribution.

        Args:
            ml (float): Multiplier for level distribution.
            max_level (int): Maximum level allowed.

        Returns:
            int: The generated level.
        """
        try:
            return min(int(-math.log(random.random())*ml), max_level)
        
        except Exception as error:
            raise LevelGenerationError(str(error))
    
    def __get_node_from_map__ (self, id: str) -> Node :
        """
        Retrieve a node by ID.

        Args:
            id (str): Node ID.

        Returns:
            Node: The corresponding node.
        """
        if id not in self.nodeMap:
            raise NodeNotFoundError(id)
        
        return self.nodeMap[id]
    
    def __cos_sim__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute cosine similarity between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Cosine similarity score.
        """
        try:
            start = datetime.datetime.now()
            node = self.__get_node_from_map__(nodeID)
            cos_sim = float(nn.losses.cosine_similarity_loss(
                mx.expand_dims(node.vector, axis=1).T, 
                mx.expand_dims(q, axis=1).T)
            )
            duration = datetime.datetime.now() - start
            self.timeMap['cosine distance'].append(duration.total_seconds())
            
            return cos_sim
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise CosineSimilarityError(str(error))
    
    def __euclidean__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute euclidean distance between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Euclidean distance.
        """
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sqrt(mx.sum((node.vector - q) ** 2)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise EuclideanDistanceError(str(error))
    
    def __manhattan__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute manhattan distance between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Manhattan distance.
        """
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(mx.abs(node.vector -q)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise ManhattanDistanceError(str(error))
    
    def __inner_product__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute inner product between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Negative inner product.
        """
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(node.vector * q))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise InnerProductError(str(error))
    
    def __hamming__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute hamming distance between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Hamming distance.
        """
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(mx.array((node.vector != q)).astype(mx.float16)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise HammingDistanceError(str(error))
    
    def __jaccard__ (self, nodeID: str, q: Vector) -> float:
        """
        Compute jaccard between a node's vector and a query vector.

        Args:
            nodeID (str): Node ID.
            q (Vector): Query vector.

        Returns:
            float: Jaccard distance.
        """
        try:
            
            node = self.__get_node_from_map__(nodeID)
            intersection = mx.sum(mx.logical_and(node.vector, q))
            union = mx.sum(mx.logical_or(node.vector, q))
            jaccard_similarity = intersection / union
            jaccard_distance = 1.0 - jaccard_similarity
            return float(jaccard_distance)
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise JaccardDistanceError(str(error))
    
    def __insert_node_into_map__ (self, node: Node) -> str :
        """
        Insert a new node into the graph and assign it a unique ID.

        Args:
            node (Node): Node to insert.

        Returns:
            str: Assigned node ID.
        """
        try:
            id: str = str(uuid.uuid4())
            self.nodeMap[id] = node
            return id
        
        except Exception as error:
            raise NodeInsertionError(str(error))
    
    def __get_vector_by_distance__ (self, vecIDs: dict[str, None], q: Vector, dist: Literal['nearest', 'furthest']) -> str :
        """
        Find a vector from a set that is closest or farthest to the query.

        Args:
            vecIDs (dict[str, None]): Candidate node IDs.
            q (Vector): Query vector.
            dist (Literal['nearest', 'furthest']): Distance type to compare.

        Returns:
            str: ID of the selected vector.
        """
        try:    
            start = datetime.datetime.now()        
            matrix = mx.stack(
                list(
                    map(
                        lambda id: self.nodeMap[id].vector, 
                        list(vecIDs.keys())
                    )
                )
            ).squeeze()
            
            matrix = mx.expand_dims(matrix, axis=1).T if len(matrix.shape) < 2 else matrix
            
            if self.distance == 'Manhattan':
                dists: Vector = mx.sum(mx.abs(matrix - q), axis=1)
                
            elif self.distance == 'Eucilidean':
                dists: Vector = mx.sqrt(mx.sum((matrix - q) ** 2, axis=1))
                
            elif self.distance == 'Hamming':
                dists: Vector = mx.sum(mx.array(matrix != q), axis=1)
                
            elif self.distance == 'Inner':
                dists: Vector = mx.sum(matrix * q, axis=1)
                
            elif self.distance == 'Jaccard':
                intersection = mx.sum(mx.logical_and(matrix, q), axis=1)
                union = mx.sum(mx.logical_or(matrix, q), axis=1)
                jaccard_similarity = intersection / union
                dists: Vector = 1.0 - jaccard_similarity
                
            else:
                dists: Vector = nn.losses.cosine_similarity_loss(matrix, mx.expand_dims(q, axis=1).T)
                
            idx = int(mx.argmax(dists)) if dist == 'nearest' else int(mx.argmin(dists))
            duration = datetime.datetime.now() - start
            self.timeMap['vector distance'].append(duration.total_seconds())
            
            return list(vecIDs.keys())[idx]
        
        except KeyError as error:
            raise NodeNotFoundError(str(error))
        
        except Exception as error:
            raise CosineSimDistanceError(str(error))
    
    def __select_neighbours__ (self, point: Vector, ids: dict[str, None], M: int) -> dict[str, None]:
        """
        Select top M closest neighbors to a given point.

        Args:
            point (Vector): Reference vector.
            ids (dict[str, None]): Candidate node IDs.
            M (int): Maximum neighbors to select.

        Returns:
            dict[str, None]: Selected neighbor IDs.
        """
        try:
            start = datetime.datetime.now()
            id_dist_map: dict[str, float] = {id: self.distance(id, point) for id in ids.keys()}
            sorted_keys: list[str] = sorted(id_dist_map.keys(), key=lambda k: id_dist_map[k], reverse=True)[:M]
            duration = datetime.datetime.now() - start
            self.timeMap['search neighbours'].append(duration.total_seconds())
            
            return {k: None for k in sorted_keys}
        
        except Exception as error:
            raise NeighbourSelectionError(str(error))

    def __bidirectional_connection__(self, nodes: dict[str, None], node: str, M: int, layer: int) -> None:
        """
        Establish bidirectional connections between a node and its neighbors.

        Args:
            nodes (dict[str, None]): Neighbor IDs.
            node (str): Node ID.
            M (int): Max number of neighbors.
            layer (int): Layer index.
        """
        try:
            start = datetime.datetime.now()
            for node1 in nodes.keys() :
                self.nodeMap[node1].neighbours[layer][node] = None
                self.nodeMap[node].neighbours[layer][node1] = None
                self.nodeMap[node1].neighbours[layer] = self.__select_neighbours__(
                    self.nodeMap[node1].vector, 
                    self.nodeMap[node1].neighbours[layer], 
                    M
                )
        
            self.nodeMap[node].neighbours[layer] = self.__select_neighbours__(
                    self.nodeMap[node].vector, 
                    self.nodeMap[node].neighbours[layer], 
                    M
                )
            duration = datetime.datetime.now() - start
            self.timeMap['bidirectional connection'].append(duration.total_seconds())
            
        except Exception as error:
            raise BidirectionalConnectionError(str(error))
        
    def __search_layer__ (self, q: Vector, ep: str, ef: int, layer: int) -> dict[str, None] :
        """
        Perform greedy search at a given layer.

        Args:
            q (Vector): Query vector.
            ep (str): Entry point ID.
            ef (int): Size of the candidate list.
            layer (int): Layer index.

        Returns:
            dict[str, None]: Found neighbor IDs.
        """
        try:
            start = datetime.datetime.now()
            visited: dict[str, None] = {ep: None}
            candidates: dict[str, None] = {ep: None}
            neighbours: dict[str, None] = {ep: None}
            while len(candidates) > 0 :
                c: str = self.__get_vector_by_distance__ (candidates, q, 'nearest')
                f: str = self.__get_vector_by_distance__ (neighbours, q, 'furthest')
                for e_id in self.nodeMap[c].neighbours[layer] :
                    if e_id not in visited.keys():
                        visited[e_id] = None
                        f: str = self.__get_vector_by_distance__ (neighbours, q, 'furthest')
                        if self.distance(e_id, q) < self.distance(f, q) or len(neighbours) < ef :
                            candidates[e_id] = None
                            neighbours[e_id] = None
                            if len(neighbours.keys()) > ef :
                                f: str = self.__get_vector_by_distance__ (neighbours, q, 'furthest')
                                del neighbours[f]
                if self.distance(c, q) < self.distance (f, q) :
                    break
                del candidates[c]
            duration = datetime.datetime.now() - start
            self.timeMap['search layer'].append(duration.total_seconds())
            return neighbours
        
        except Exception as error:
            raise SearchLayerError(str(error))
    
    def Insert (self, q: Vector) -> None:
        """
        Insert a vector into the HNSW graph.

        Args:
            q (Vector): The vector to insert.
        """
        try:
            start = datetime.datetime.now()
            ml = 1 / (-math.log(1 - (1 / self.M)))
            
            if self.entry_point_id is None:
                node_level: int = self.__generate_level__(ml, self.max_level)
                self.entry_point_id = self.__insert_node_into_map__(Node(q, node_level))
                self.total_nodes += 1
                return 
            
            ep: str = self.entry_point_id
            L: int = self.nodeMap[ep].max_level
            l: int = self.__generate_level__(ml, self.max_level)
            
            new_node_id: str = self.__insert_node_into_map__(Node(q, l)) 
            
            for l_c in range(L, l - 1, -1):
                W: dict[str, None] = self.__search_layer__(q, ep, 1, l_c)
                ep: str = self.__get_vector_by_distance__(W, q, 'nearest')
            
            for l_c in range(min(L, l), -1, -1):
                W: dict[str, None] = self.__search_layer__(q, ep, self.efconstruction, l_c)
                neighbours: dict[str, None] = self.__select_neighbours__(q, W, self.M) 
                self.__bidirectional_connection__(neighbours, new_node_id, self.M, l_c)
            
            if l > L :
                self.entry_point_id = new_node_id
            
            self.total_nodes += 1
            duration = datetime.datetime.now() - start
            
            self.timeMap['insert'].append(duration.total_seconds())
            
            
        except Exception as error:
            raise InsertionError(str(error))
    
    def Search (self, q: Vector, K: int, efsearch: int) -> list[tuple[float, Vector]]:
        """
        Perform approximate nearest neighbor search for a given query vector.
        
        Args:
            q (Vector): The query vector to search against the index.
            K (int): The number of nearest neighbors to return.
            efsearch (int): The size of the candidate list to explore during search.

        Returns:
            list[Vector]: A list of vectors representing the closest neighbors, sorted by similarity.
        """
        try:
            start = datetime.datetime.now()
            if K > efsearch:
                raise Exception('efsearch must be greater then or equal to K')
            
            if not self.entry_point_id :
                raise Exception('No entry point determined! Insert a point first')
            
            if self.entry_point_id not in self.nodeMap:
                raise NodeNotFoundError('No Entry point stored in the node map')
            
            ep: str = self.entry_point_id
            L: int = self.nodeMap[ep].max_level
            
            for l_c in range(L, 0, -1):
                W: dict[str, None] = self.__search_layer__(q, ep, efsearch, l_c)
                ep: str = self.__get_vector_by_distance__(W, q, 'nearest')
            
            nearest_ids: dict[str, None] = self.__search_layer__(q, ep, efsearch, 0)
            vec_dist_tuple: list[tuple[float, Vector]] = [(
                    float(self.distance(id, q)), 
                    self.nodeMap[id].vector
                ) for id in nearest_ids.keys()]
            
            sorted_vecs = sorted(vec_dist_tuple, key=lambda x: x[0], reverse=True)
            duration = datetime.datetime.now() - start
            self.timeMap['search'].append(duration.total_seconds())
            
            return sorted_vecs if len(sorted_vecs) < K else sorted_vecs[:K]
        
        except Exception as error:
            raise SearchError(str(error))

    def TimeReport (self) -> None:
        avg: dict[str, float] = {k: round(sum(self.timeMap[k]) / len(self.timeMap[k]), 4) for k in self.timeMap.keys()}
        print(json.dumps(avg, indent=2))
    
