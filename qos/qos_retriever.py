from qos.qos_calculate import Device, Graph, qos_calculate, Vertex, Edge
from resource import Resource
from graph import DSPGraph
from typing import *

__all__ = ['get_qos']


def get_qos(dsp_graph: DSPGraph, res: Resource, placements: List[int], memory_restrict: bool = True) -> Tuple[float, float]:
    graph = Graph()
    devices = {}
    slot_to_devices = {}
    for slot_id, slot in res.slots.items():
        slot_to_devices[slot.id] = slot.device
        if slot.device not in devices:
            devices[slot.device] = Device(slot.cpu, slot.device)
        else:
            devices[slot.device].cpu_capacity += slot.cpu

    for operator_id, operator in dsp_graph.operators.items():
        vertex_value = Vertex(operator.cpu)
        graph.add_vertex(operator_id, vertex_value)
        slot_id = placements[operator_id]
        devices[slot_to_devices[slot_id]].linked_vertexes.append(vertex_value)
    for device_id, device in list(devices.items()):
        if len(device.linked_vertexes) == 0:
            del devices[device_id]
    for from_id, to_id in zip(dsp_graph.edges[0], dsp_graph.edges[1]):
        delay = res.matrix[placements[from_id]][placements[to_id]]
        graph.add_edge(from_id, to_id, Edge(delay))

    res_mem = {}
    for slot_id, slot in res.slots.items():
        res_mem[slot_id] = slot.memory
    if memory_restrict:
        for operator_id, operator in dsp_graph.operators.items():
            slot_id = placements[operator_id]
            res_mem[slot_id] -= operator.memory
            if res_mem[slot_id] < 0:
                return -1, -1

    throughput, latency = qos_calculate(graph, devices)
    return throughput, latency

