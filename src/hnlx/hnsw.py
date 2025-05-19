import uuid
import math
import random
import mlx.nn as nn
from .models import *
import mlx.core as mx
from typing import Literal, Optional

class Node :
    """Represents a node in the HNSW graph with vector data and neighbors across levels."""
    
    def __init__(self, vector: Vector, level: int) -> None:
        """
        Initialize a node with a vector and level.

        Args:
            vector (Vector): The vector representation of the node.
            level (int): The maximum level of the node in the graph.
        """
        self.max_level: int = level
        self.vector: Vector = vector
        self.neighbours: dict[int, dict[str, None]] = {
            i: dict() for i in range(self.max_level)
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
    
    def __init__(self, M: int, efc: int, max_level: int) -> None:
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
            
            node = self.__get_node_from_map__(nodeID)
            return float(nn.losses.cosine_similarity_loss(node.vector, q))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise CosineSimilarityError(str(error))
    
    def __insert_node_into_map__ (self, node: Node) -> str :
        """
        Insert a new node into the graph and assign it a unique ID.

        Args:
            node (Node): Node to insert.

        Returns:
            str: Assigned node ID.
        """
        try:
            id: str = str(uuid.UUID())
            self.nodeMap[id] = node
            return id
        
        except Exception as error:
            raise NodeInsertionError(str(error))
    
    
    def __get_vector_by_cosine_sim__ (
        self, 
        vecIDs: dict[str, None], 
        q: Vector, 
        dist: Literal['nearest', 'furthest']
    ) -> str :
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
            matrix = mx.stack(
                list(
                    map(
                        lambda id: self.nodeMap[id].vector, 
                        list(vecIDs.keys())
                    )
                )
            ).squeeze()
        
            dists = nn.losses.cosine_similarity_loss(matrix, q)
            idx = int(mx.argmax(dists)) if dist == 'nearest' else int(mx.argmin(dists))
        
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
            id_dist_map: dict[str, float] = {id: self.__cos_sim__(id, point) for id in ids.keys()}
            sorted_keys: list[str] = sorted(id_dist_map.keys(), key=lambda k: id_dist_map[k], reverse=True)[:M]
            return {k: None for k in sorted_keys}
        
        except Exception as error:
            raise NeighbourSelectionError(str(error))

    def __bidirectional_connection__ (self, nodes: dict[str, None], node: str, M: int, layer: int) -> None:
        """
        Establish bidirectional connections between a node and its neighbors.

        Args:
            nodes (dict[str, None]): Neighbor IDs.
            node (str): Node ID.
            M (int): Max number of neighbors.
            layer (int): Layer index.
        """
        try:
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
            visited: dict[str, None] = {ep: None}
            candidates: dict[str, None] = {ep: None}
            neighbours: dict[str, None] = {ep: None}
            
            while len(candidates) > 0 :
                c: str = self.__get_vector_by_cosine_sim__ (candidates, q, 'nearest')
                f: str = self.__get_vector_by_cosine_sim__ (neighbours, q, 'furthest')
                for e_id in self.nodeMap[c].neighbours[layer] :
                    if e_id not in visited:
                        visited[e_id] = None
                        f: str = self.__get_vector_by_cosine_sim__ (neighbours, q, 'furthest')
                        if self.__cos_sim__(e_id, q) < self.__cos_sim__(f, q) or len(neighbours) < ef :
                            candidates[e_id] = None
                            neighbours[e_id] = None
                            if len(neighbours) > ef :
                                f: str = self.__get_vector_by_cosine_sim__ (neighbours, q, 'furthest')
                                del neighbours[f]
                if self.__cos_sim__(c, q) < self.__cos_sim__ (f, q) :
                    break
                del candidates[c]
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
            ml = 1 / (-math.log(1 - (1 / self.M)))
            
            if self.entry_point_id is None:
                node_level = self.__generate_level__(ml, self.max_level)
                self.entry_point_id = self.__insert_node_into_map__(Node(q, node_level))
                self.total_nodes += 1
                return 
            
            
            ep = self.entry_point_id
            L = self.nodeMap[ep].max_level
            l = self.__generate_level__(ml, self.max_level)
            
            new_node_id: str = self.__insert_node_into_map__(Node(q, l))
            
            for l_c in range(L, l - 1, -1):
                W: dict[str, None] = self.__search_layer__(q, ep, 1, l_c)
                ep: str = self.__get_vector_by_cosine_sim__(W, q, 'nearest')
            
            for l_c in range(min(L, l), -1, -1):
                W: dict[str, None] = self.__search_layer__(q, ep, self.efconstruction, l_c)
                neighbours: dict[str, None] = self.__select_neighbours__(q, W, self.M) 
                self.__bidirectional_connection__(neighbours, new_node_id, self.M, l_c)
            
            if l > L :
                self.entry_point_id = new_node_id
            
            self.total_nodes += 1
        
        except Exception as error:
            raise InsertionError(str(error))
            