import json
import uuid
import math
import random
import datetime
import mlx.nn as nn
from .models import *
import mlx.core as mx 
from typing import Literal, Optional, Callable
from collections.abc import Iterable

class Node :
    def __init__(self, vector: Vector, level: int) -> None:
        self.max_level: int = level if level != 0 else 1
        self.vector: Vector = vector
        self.neighbours: dict[int, dict[str, None]] = {
            i: dict() for i in range(self.max_level+1)
        }
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return bool(self.vector == other.vector)
                    
class HNSW :
    def __init__(self, M: int, efc: int, max_level: int, threshold:int, dist: distance = 'Cosine', pruning: bool = False) -> None:
        self.M: int = M
        self.pruning = pruning
        self.cache = list()
        self.threshold = threshold
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
    
    def __search_layer_break_condition__ (self, c: str, f: str, q: list[Vector]) -> object :
        cv: Vector = self.__get_node_from_map__(c).vector
        fv: Vector = self.__get_node_from_map__(f).vector
        qv: Vector = mx.stack(q)
        cv_qv = nn.losses.cosine_similarity_loss(mx.expand_dims(cv, axis=1).T, qv)
        fv_qv = nn.losses.cosine_similarity_loss(mx.expand_dims(fv, axis=1).T, qv)

        return mx.all(cv_qv < fv_qv).item()
    
    def __euclidean__ (self, nodeID: str, q: Vector) -> float:
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sqrt(mx.sum((node.vector - q) ** 2)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise EuclideanDistanceError(str(error))
    
    def __manhattan__ (self, nodeID: str, q: Vector) -> float:
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(mx.abs(node.vector -q)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise ManhattanDistanceError(str(error))
    
    def __inner_product__ (self, nodeID: str, q: Vector) -> float:
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(node.vector * q))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise InnerProductError(str(error))
    
    def __hamming__ (self, nodeID: str, q: Vector) -> float:
        try:
            
            node = self.__get_node_from_map__(nodeID)
            return float(mx.sum(mx.array((node.vector != q)).astype(mx.float16)))
            
        except NodeNotFoundError as e:
            raise e
        
        except Exception as error:
            raise HammingDistanceError(str(error))
    
    def __jaccard__ (self, nodeID: str, q: Vector) -> float:
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
        try:
            id: str = str(uuid.uuid4())
            self.nodeMap[id] = node
            return id
        
        except Exception as error:
            raise NodeInsertionError(str(error))
    
    def __get_vector_by_distance__ (self, vecIDs: dict[str, None], q: list[Vector], dist: Literal['nearest', 'furthest']) -> str :
        try:
            if len(list(vecIDs.keys())) == 1:
                return list(vecIDs.keys())[0]
             
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
            dists: Vector = mx.matmul(matrix, mx.stack(q).T) / mx.maximum(mx.expand_dims(mx.linalg.norm(matrix, axis=1), axis=1) * mx.expand_dims(mx.linalg.norm(mx.stack(q), axis=1), axis=0), 1e-8)
            idx = int(mx.argmax(mx.argmax(dists, axis=1))) if dist == 'nearest' else int(mx.argmin(mx.argmin(dists, axis=1)))
            duration = datetime.datetime.now() - start
            self.timeMap['vector distance'].append(duration.total_seconds())

            assert(len(list(vecIDs.keys())) > idx)
            return list(vecIDs.keys())[idx]
        
        except KeyError as error:
            raise NodeNotFoundError(str(error))
        
        except Exception as error:
            raise CosineSimDistanceError(str(error))
    
    def __select_neighbours__ (self, point: Vector, ids: dict[str, None], M: int) -> dict[str, None]:
        try:
            start = datetime.datetime.now()
            id_dist_map: dict[str, float] = {id: self.distance(id, point) for id in ids.keys()}
            sorted_keys: list[str] = sorted(id_dist_map.keys(), key=lambda k: id_dist_map[k], reverse=True)[:M]
            duration = datetime.datetime.now() - start
            self.timeMap['search neighbours'].append(duration.total_seconds())
            
            return {k: None for k in sorted_keys}
        
        except Exception as error:
            raise NeighbourSelectionError(str(error))
    
    def __select_neighbours_batch__(self, points: list[Vector], ids: list[list[str]], M: int) -> Optional[list[dict[str, None]]]:
        
        ks = [list(g) for g in ids]
        B,L,d = len(ks),max(map(len,ks)), self.__get_node_from_map__(ks[0][0]).vector.shape[0]
        X,mask = mx.zeros((B,L,d)),mx.zeros((B,L))
        for i,g in enumerate(ks):
            e = mx.stack([self.__get_node_from_map__(t).vector for t in g]); n=e.shape[0]
            X[i,:n],mask[i,:n]=e,1
        X /= mx.linalg.norm(X,axis=-1,keepdims=True)+1e-9
        V = mx.stack(points) 
        V /= mx.linalg.norm(V,axis=-1,keepdims=True)+1e-9
        S = mx.sum(X*V[:,None,:],axis=-1)+(mask-1)*1e9
        top = mx.argsort(-S,axis=1)[:,:M].tolist()
        if isinstance(top, Iterable):
            return [{g[j]: None for j in idx if j < len(g)}
                    for g, idx in zip(ks, top)
                ]
        else:
            return None

    def __bidirectional_connection__(self, nodes: dict[str, None], node: str, M: int, layer: int) -> None:
        try:
            start = datetime.datetime.now()

            for node1 in nodes.keys() :
                self.nodeMap[node1].neighbours[layer][node] = None
                self.nodeMap[node].neighbours[layer][node1] = None

            if self.pruning:
                str_nodes = list(nodes.keys())
                points: list[Vector] = list() 
                ids: list[list[str]] = list()

                for id in nodes.keys():
                    points.append(self.nodeMap[id].vector)
                    ids.append(list(self.nodeMap[id].neighbours[layer].keys()))

                all_neighbours = self.__select_neighbours_batch__(points, ids, M)

                if not all_neighbours:
                    print("Error in pruning")
                    return
                assert(len(all_neighbours) == len(nodes.keys()))

                for i in range(len(nodes.keys())):
                    self.nodeMap[str_nodes[i]].neighbours[layer] = all_neighbours[i]
                
                self.nodeMap[node].neighbours[layer] = self.__select_neighbours__(
                    self.nodeMap[node].vector, 
                    self.nodeMap[node].neighbours[layer], 
                    M
                )
        
            duration = datetime.datetime.now() - start
            self.timeMap['bidirectional connection'].append(duration.total_seconds())
            
        except Exception as error:
            raise BidirectionalConnectionError(str(error))
             
    def __search_layer__ (self, q: list[Vector], ep: str, ef: int, layer: int) -> dict[str, None] :
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
                        if self.__search_layer_break_condition__(e_id, f, q) or len(neighbours) < ef :
                            candidates[e_id] = None
                            neighbours[e_id] = None
                            if len(neighbours.keys()) > ef :
                                f: str = self.__get_vector_by_distance__ (neighbours, q, 'furthest')
                                del neighbours[f]
                if self.__search_layer_break_condition__(c, f, q):
                    break
                del candidates[c]
            duration = datetime.datetime.now() - start
            self.timeMap['search layer'].append(duration.total_seconds())
            return neighbours
        
        except Exception as error:
            raise SearchLayerError(str(error))
    
    def Insert (self, q: Vector) -> None:
        try:
            start = datetime.datetime.now()
            ml = 1 / (-math.log(1 - (1 / self.M)))
            
            if self.entry_point_id is None:
                node_level: int = self.__generate_level__(ml, self.max_level)
                self.entry_point_id = self.__insert_node_into_map__(Node(q, node_level))
                self.cache.append((self.entry_point_id, q, node_level))
                self.total_nodes += 1
                return 
            
            ep: str = self.entry_point_id
            l: int = self.__generate_level__(ml, self.max_level)
            new_node_id: str = self.__insert_node_into_map__(Node(q, l))
            self.cache.append((new_node_id, q, l))

            if len(self.cache) == self.threshold:
                max_lvl = max([x[2] for x in self.cache])
                for l_c in range(max_lvl, - 1, -1):
                    batch_vector = [x[1] for x in self.cache if x[2] >= l_c]
                    W: dict[str, None] = self.__search_layer__(batch_vector, ep, 1, l_c)
                    ep: str = self.__get_vector_by_distance__(W, batch_vector, 'nearest')
                
                for l_c in range(max_lvl, -1, -1):
                    batch_ids = [x[0] for x in self.cache if x[2] >= l_c]
                    batch_vector = [x[1] for x in self.cache if x[2] >= l_c]
                    W: dict[str, None] = self.__search_layer__(batch_vector, ep, self.efconstruction, l_c) 
                    neighbours = self.__select_neighbours_batch__(batch_vector, [list(W.keys())] * len(batch_vector), self.M)
                    if neighbours:
                        for n, id in zip(neighbours, batch_ids):
                            self.__bidirectional_connection__(n, id, self.M, l_c)
                    else:
                        print("Select layers batch function failed")
                        continue
                    ep: str = self.__get_vector_by_distance__(W, batch_vector, 'nearest')
                
                self.entry_point_id = max(self.cache, key=lambda x: x[2])[0]
                self.total_nodes += 1
                self.cache = list()

            duration = datetime.datetime.now() - start
            self.timeMap['insert'].append(duration.total_seconds())
            
        except Exception as error:
            raise InsertionError(str(error))
    
    def __search_layer_single__ (self, q: Vector, ep: str, ef: int, layer: int) -> dict[str, None] :
        try:
            start = datetime.datetime.now()
            visited: dict[str, None] = {ep: None}
            candidates: dict[str, None] = {ep: None}
            neighbours: dict[str, None] = {ep: None}
            while len(candidates) > 0 :
                c: str = self.__get_vector_by_distance_single__ (candidates, q, 'nearest')
                f: str = self.__get_vector_by_distance_single__ (neighbours, q, 'furthest')
                for e_id in self.nodeMap[c].neighbours[layer] :
                    if e_id not in visited.keys():
                        visited[e_id] = None
                        f: str = self.__get_vector_by_distance_single__ (neighbours, q, 'furthest')
                        if self.distance(e_id, q) < self.distance(f, q) or len(neighbours) < ef :
                            candidates[e_id] = None
                            neighbours[e_id] = None
                            if len(neighbours.keys()) > ef :
                                f: str = self.__get_vector_by_distance_single__ (neighbours, q, 'furthest')
                                del neighbours[f]
                if self.distance(c, q) < self.distance (f, q) :
                    break
                del candidates[c]
            duration = datetime.datetime.now() - start
            self.timeMap['search layer'].append(duration.total_seconds())
            return neighbours
        
        except Exception as error:
            raise SearchLayerError(str(error))

    def __get_vector_by_distance_single__ (self, vecIDs: dict[str, None], q: Vector, dist: Literal['nearest', 'furthest']) -> str :
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
            dists: Vector = nn.losses.cosine_similarity_loss(matrix, mx.expand_dims(q, axis=1).T)    
            idx = int(mx.argmax(dists)) if dist == 'nearest' else int(mx.argmin(dists))
            duration = datetime.datetime.now() - start
            self.timeMap['vector distance'].append(duration.total_seconds())
            
            return list(vecIDs.keys())[idx]
        
        except KeyError as error:
            raise NodeNotFoundError(str(error))
        
        except Exception as error:
            raise CosineSimDistanceError(str(error))

    def Search (self, q: Vector, K: int, efsearch: int) -> list[tuple[float, Vector]]:
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
                W: dict[str, None] = self.__search_layer_single__(q, ep, efsearch, l_c)
                ep: str = self.__get_vector_by_distance_single__(W, q, 'nearest')
            
            nearest_ids: dict[str, None] = self.__search_layer_single__(q, ep, efsearch, 0)
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
        avg: dict[str, float] = {
            k: round(sum(self.timeMap[k]) / len(self.timeMap[k]), 4)
            for k in self.timeMap.keys() 
            if len(self.timeMap[k]) > 0
        }
        print(json.dumps(avg, indent=2)) 
    
