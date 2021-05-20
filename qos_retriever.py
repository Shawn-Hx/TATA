from typing import *

from graph import *
from resource import Resource as Res
from tools.graph_helper import Graph


class VertexValue:
    def __init__(self, cpu_consumption: float = 0, memory_consumption: float = 0):
        self.throughput_capacity: float = 0
        self.throughput_now: float = 0
        self.throughput_back: float = 0
        self.cpu_consumption: float = cpu_consumption
        self.memory_consumption: float = memory_consumption
        self.latency_now: float = 0


class EdgeValue:
    def __init__(self, latency: float = 0):
        self.latency: float = latency


class Resource:
    def __init__(self, cpu_capacity: float = 0, memory_capacity: float = 0):
        self.cpu_capacity: float = cpu_capacity
        self.memory_capacity: float = memory_capacity
        self.linked_vertexes: List[VertexValue] = []
        self.memory_allocated = False

    def allocate_resource(self) -> bool:

        if not self.memory_allocated:
            for vertex in self.linked_vertexes:
                self.memory_capacity -= vertex.memory_consumption
            self.memory_allocated = True

        for vertex in self.linked_vertexes:
            if vertex.throughput_now < vertex.throughput_capacity:
                recycle_amount = vertex.throughput_capacity - vertex.throughput_now
                self.cpu_capacity += recycle_amount * vertex.cpu_consumption
                vertex.throughput_capacity -= recycle_amount

        cpu_consumption_sum = sum(vertex.cpu_consumption for vertex in self.linked_vertexes)
        for vertex in self.linked_vertexes:
            vertex.throughput_capacity += self.cpu_capacity / cpu_consumption_sum
        self.cpu_capacity = 0
        return self.memory_capacity >= 0


def get_qos(dsp_graph: DSPGraph, res: Res, placements: List[int], memory_restrict: bool = True) -> Tuple[float, float]:
    """
    :param dsp_graph:   DSP graph
    :param res:    Res
    :param placements:   array, index: vertex id, value: slot id
    :param memory_restrict:    ignore memory restrict if false
    :return:            delay and throughput
    """

    graph = Graph(VertexValue, EdgeValue)
    resources: Dict[int, Resource] = {slot_id: Resource(slot.cpu, slot.memory) for slot_id, slot in res.slots.items()}
    for operator_id, operator in dsp_graph.operators.items():
        vertex_value = VertexValue(operator.cpu, operator.memory)
        graph.add_vertex(operator_id, vertex_value)
        slot_id = placements[operator_id]
        resources[slot_id].linked_vertexes.append(vertex_value)
    for from_id, to_id in zip(dsp_graph.edges[0], dsp_graph.edges[1]):
        delay = res.matrix[placements[from_id]][placements[to_id]]
        graph.add_edge(from_id, to_id, EdgeValue(delay))

    order = graph.topological_sort()
    throughput = 0
    while True:
        for resource in resources.values():
            if not resource.allocate_resource() and memory_restrict:
                return -1, -1
        for vertex in order:
            vertex_value: VertexValue = graph.get_vertex_value(vertex)
            input_capacities = sum(graph.get_vertex_value(v).throughput_now / graph.get_to_edge_number(v) for v, e in
                                   graph.iter_from_edges(vertex))
            if input_capacities == 0:
                input_capacities = 9e8
            vertex_value.throughput_now = min(input_capacities, vertex_value.throughput_capacity)
            
        for vertex in reversed(order):
            vertex_value: VertexValue = graph.get_vertex_value(vertex)
            input_capacities = [graph.get_vertex_value(v).throughput_now / graph.get_to_edge_number(v) for v, e in
                                graph.iter_from_edges(vertex)]
            input_capacity_sum = sum(input_capacities)
            for i, (v, e) in enumerate(graph.iter_from_edges(vertex)):
                v_value: VertexValue = graph.get_vertex_value(v)
                v_value.throughput_back += (input_capacities[i] / input_capacity_sum) * vertex_value.throughput_now
                
        for vertex in order[:-1]:
            vertex_value: VertexValue = graph.get_vertex_value(vertex)
            vertex_value.throughput_now = vertex_value.throughput_back
            vertex_value.throughput_back = 0

        throughput_new = graph.get_vertex_value(order[-1]).throughput_now
        if throughput_new - throughput < 0.01:
            throughput = throughput_new
            break
        else:
            throughput = throughput_new

    for vertex in order:
        vertex_value: VertexValue = graph.get_vertex_value(vertex)
        to_edge_number = graph.get_to_edge_number(vertex)
        for v, e in graph.iter_to_edges(vertex):
            v_value = graph.get_vertex_value(v)
            v_value.latency_now += vertex_value.throughput_now/to_edge_number * (
                    vertex_value.latency_now + e.latency) / v_value.throughput_now
    latency = graph.get_vertex_value(order[-1]).latency_now
    return throughput, latency

