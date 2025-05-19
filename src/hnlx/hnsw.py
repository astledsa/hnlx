import uuid
import math
import random
import mlx.nn as nn
from .models import *
import mlx.core as mx
from typing import Literal, Optional

class Node :
    
    def __init__(self, vector: Vector, level: int) -> None:
        self.max_level: int = level
        self.vector: Vector = vector
        self.neighbours: dict[int, dict[str, None]] = {
            i: dict() for i in range(self.max_level)
        }
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return bool(self.vector == other.vector)
        
    
class HNSW :
    
    def __init__(self, M: int, efc: int, max_level: int) -> None:
        self.M: int = M
        self.total_nodes: int = 0
        self.efconstruction: int = efc
        self.max_level: int = max_level
        self.nodeMap: dict[str, Node] = {}
        self.entry_point_id: Optional[str] = None
    
    def __generate_level__ (self, ml: float, max_level: int) -> int:
        try:
            return min(int(-math.log(random.random())*ml), max_level)
        
        except Exception as error:
            raise LevelGenerationError(str(error))
    
    def __get_node_from_map__ (self, id: str) -> Node :
        
        if id not in self.nodeMap:
            raise NodeNotFoundError(id)
        
        return self.nodeMap[id]
    
    def __cos_sim__ (self, nodeID: str, q: Vector) -> float:
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(nn.losses.cosine_similarity_loss(node.vector, q))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise CosineSimilarityError(str(error))
    
    def __insert_node_into_map__ (self, node: Node) -> str :
        
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
        
        try:
            id_dist_map: dict[str, float] = {id: self.__cos_sim__(id, point) for id in ids.keys()}
            sorted_keys: list[str] = sorted(id_dist_map.keys(), key=lambda k: id_dist_map[k], reverse=True)[:M]
            return {k: None for k in sorted_keys}
        
        except Exception as error:
            raise NeighbourSelectionError(str(error))

    def __bidirectional_connection__ (self, nodes: dict[str, None], node: str, M: int, layer: int) -> None:
        
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
            