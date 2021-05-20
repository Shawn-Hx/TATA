from typing import Optional, Type, List, Dict, Union, Set, Tuple, Any, Generator, Deque
from collections import deque


class Graph:
    def __init__(self, vertex_value_class: Optional[Type] = None, edge_value_class: Optional[Type] = None):
        self.idx_map: Dict[Union[int, str], int] = {}
        self.vertex_map: Dict[int, Union[int, str]] = {}
        self.vertexes: List[Any] = []
        self.to_edges: List[Dict[int, Any]] = []
        self.from_edges: List[Dict[int, Any]] = []
        self.vertex_value_class = vertex_value_class
        self.edge_value_class = edge_value_class

    def add_vertex(self, vertex: Union[int, str], vertex_value) -> None:
        if vertex in self.idx_map:
            return
        self.idx_map[vertex] = len(self.vertexes)
        self.vertex_map[len(self.vertexes)] = vertex
        if not vertex_value:
            if self.vertex_value_class:
                vertex_value = self.vertex_value_class()
            else:
                raise Exception("uninitialized vertex type")
        self.vertexes.append(vertex_value)
        self.to_edges.append(dict())
        self.from_edges.append(dict())

    def add_edge(self, from_vertex: Union[int, str], to_vertex: Union[int, str], edge_value: any = None) -> None:
        if from_vertex not in self.idx_map:
            self.add_vertex(from_vertex, edge_value)
        if to_vertex not in self.idx_map:
            self.add_vertex(to_vertex, edge_value)
        from_idx = self.idx_map[from_vertex]
        to_idx = self.idx_map[to_vertex]
        if not edge_value:
            if self.edge_value_class:
                edge_value = self.edge_value_class()
            else:
                raise Exception("uninitialized edge type")
        self.to_edges[from_idx][to_idx] = edge_value
        self.from_edges[to_idx][from_idx] = edge_value

    def get_vertex_value(self, vertex: Union[int, str]) -> Any:
        idx = self.idx_map[vertex]
        return self.vertexes[idx]

    def get_edge_value(self, from_vertex, to_vertex) -> Any:
        from_idx = self.idx_map[from_vertex]
        to_idx = self.idx_map[to_vertex]
        for idx, edge_value in self.to_edges[from_idx].items():
            if idx == to_idx:
                return edge_value

    def get_to_edge_number(self, from_vertex: int) -> int:
        from_idx = self.idx_map[from_vertex]
        edges = self.to_edges[from_idx]
        return len(edges)

    def iter_to_edges(self, from_vertex: int) -> Generator:
        from_idx = self.idx_map[from_vertex]
        edges = self.to_edges[from_idx].items()
        for to_idx, edge_value in edges:
            yield self.vertex_map[to_idx], edge_value

    def iter_from_edges(self, to_vertex: int) -> Generator:
        to_idx = self.idx_map[to_vertex]
        edges = self.from_edges[to_idx].items()
        for from_idx, edge_value in edges:
            yield self.vertex_map[from_idx], edge_value

    def topological_sort(self) -> List[Union[int, str]]:
        in_degrees = [len(v) for v in self.from_edges]

        que: Deque[int] = deque(idx for idx, in_degree in enumerate(in_degrees) if in_degree == 0)
        order = []
        while len(que) > 0:
            idx = que.popleft()
            order.append(self.vertex_map[idx])
            for to_vertex, _ in self.iter_to_edges(self.vertex_map[idx]):
                to_idx = self.idx_map[to_vertex]
                in_degrees[to_idx] -= 1
                if in_degrees[to_idx] == 0:
                    que.append(to_idx)
        return order
