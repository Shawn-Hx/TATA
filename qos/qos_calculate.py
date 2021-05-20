from typing import *

__all__ = ['Vertex', 'Edge', 'Graph', 'Device', 'qos_calculate']

back_press_weighted = True
resource_average_allocate = True
# average_limit_up = True


class Vertex:

    def __init__(self, cpu_consumption: float = 0):
        # 当前节点的CPU消耗
        self.cpu_consumption: float = cpu_consumption
        # 节点流量的上限
        self.throughput_capacity: float = 0
        # 当前的节点流量
        self.throughput_now: float = 0
        # 节点流量的反压值
        self.throughput_back: float = 0
        # 节点目前的延迟
        self.latency_now: float = 0


class Edge:

    def __init__(self, latency: float = 0):
        self.latency: float = latency


class Graph:

    def __init__(self):
        self.vertexes: Dict[int, Vertex] = {}
        self.edges_out: Dict[int, Dict[int, Edge]] = {}
        self.edges_in: Dict[int, Dict[int, Edge]] = {}

    def add_vertex(self, vertex_id, vertex_value) -> None:
        if vertex_id not in self.vertexes:
            self.vertexes[vertex_id] = vertex_value
            self.edges_in[vertex_id] = {}
            self.edges_out[vertex_id] = {}

    def add_edge(self, vertex_from_id, vertex_to_id, edge_value) -> None:
        self.edges_out[vertex_from_id][vertex_to_id] = edge_value
        self.edges_in[vertex_to_id][vertex_from_id] = edge_value

    def __getitem__(self, vertex_id):
        return self.vertexes[vertex_id]

    def get_edge_value(self, vertex_from_id, vertex_to_id) -> Edge:
        return self.edges_out[vertex_from_id][vertex_to_id]


class Device:
    def __init__(self, cpu_capacity: float, device_id):
        self.cpu_capacity: float = cpu_capacity
        self.cpu_capacity_max = cpu_capacity
        self.linked_vertexes: List[Vertex] = []
        self.device_id = device_id

    def allocate_resource(self):
        for vertex in self.linked_vertexes:
            if vertex.throughput_now < vertex.throughput_capacity:

                recycle_amount = vertex.throughput_capacity - vertex.throughput_now
                self.cpu_capacity += recycle_amount * vertex.cpu_consumption
                vertex.throughput_capacity -= recycle_amount

        cpu_consumption_sum = sum(vertex.cpu_consumption for vertex in self.linked_vertexes)
        capacity = self.cpu_capacity / cpu_consumption_sum
        for vertex in self.linked_vertexes:
            if (vertex.throughput_capacity + capacity) * vertex.cpu_consumption > 100:
                capacity_vertex = max(0, 100/ vertex.cpu_consumption - vertex.throughput_capacity)
            else:
                capacity_vertex = capacity
            vertex.throughput_capacity += capacity_vertex
            self.cpu_capacity -= capacity_vertex * vertex.cpu_consumption


        # if resource_average_allocate:
        #     cpu_used = self.cpu_capacity / len(self.linked_vertexes)
        #     for vertex in self.linked_vertexes:
        #         # if cpu_used > 100:
        #         #     cpu_used = 100
        #         vertex.throughput_capacity += cpu_used / vertex.cpu_consumption
        #         self.cpu_capacity -= cpu_used
        #         capacity_up = 100
        #         # if average_limit_up:
        #         #     capacity_up = min(capacity_up, self.cpu_capacity_max / len(self.linked_vertexes))
        #         if vertex.throughput_capacity * vertex.cpu_consumption > capacity_up:
        #             vertex.throughput_capacity -= (vertex.throughput_capacity * vertex.cpu_consumption
        #                                            - self.cpu_capacity_max / len(self.linked_vertexes)) / \
        #                                           vertex.cpu_consumption
        # else:
        #     cpu_consumption_sum = sum(vertex.cpu_consumption for vertex in self.linked_vertexes)
        #     for vertex in self.linked_vertexes:
        #         capacity = self.cpu_capacity / cpu_consumption_sum
        #         # if capacity * vertex.cpu_consumption > 100:
        #         #     print("OK")
        #         #     capacity = 100 / vertex.cpu_consumption
        #         vertex.throughput_capacity += capacity
        #         self.cpu_capacity -= capacity * vertex.cpu_consumption
        #
        #         if vertex.throughput_capacity * vertex.cpu_consumption > 100:
        #             vertex.throughput_capacity -= (vertex.throughput_capacity * vertex.cpu_consumption
        #                                            - self.cpu_capacity_max / len(self.linked_vertexes)) / \
        #                                           vertex.cpu_consumption


def qos_calculate(graph: Graph, devices: Dict[int, Device]):
    throughput = 0
    while True:
        for resource in devices.values():
            resource.allocate_resource()

        for vertex_id in range(len(graph.vertexes)):
            input_capacity_list = sum(
                graph[v].throughput_now / len(graph.edges_out[v]) for v in graph.edges_in[vertex_id].keys()
            )
            if input_capacity_list == 0:
                graph[vertex_id].throughput_now = graph[vertex_id].throughput_capacity
            else:
                graph[vertex_id].throughput_now = min(input_capacity_list, graph[vertex_id].throughput_capacity)

        for vertex_id in reversed(range(len(graph.vertexes))):
            input_capacity_list = [
                graph[v].throughput_now / len(graph.edges_out[v]) for v in graph.edges_in[vertex_id].keys()
            ]
            input_capacity = sum(input_capacity_list)
            if back_press_weighted:
                for i, v_id in enumerate(graph.edges_in[vertex_id].keys()):
                    graph[v_id].throughput_back += (input_capacity_list[i] / input_capacity) \
                                                   * graph[vertex_id].throughput_now
            else:
                for i, v_id in enumerate(graph.edges_in[vertex_id].keys()):
                    graph[v_id].throughput_back += graph[vertex_id].throughput_now / len(graph.edges_in[vertex_id])
        for vertex_id in range(len(graph.vertexes) - 1):
            graph[vertex_id].throughput_now = graph[vertex_id].throughput_back
            graph[vertex_id].throughput_back = 0

        throughput_new = graph[len(graph.vertexes) - 1].throughput_now
        if throughput_new - throughput < 0.01:
            throughput = throughput_new
            break
        else:
            throughput = throughput_new

    for vertex_id in range(len(graph.vertexes)):
        to_edge_number = len(graph.edges_out[vertex_id])
        for v_id, e in graph.edges_out[vertex_id].items():
            graph[v_id].latency_now += graph[vertex_id].throughput_now / to_edge_number * (
                    graph[vertex_id].latency_now + e.latency) / graph[v_id].throughput_now
    latency = graph[len(graph.vertexes) - 1].latency_now
    return throughput, latency
