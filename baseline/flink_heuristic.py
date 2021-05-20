from collections import defaultdict

from graph import *
from resource import *


def select_with_location_preference(slot_pool, resource, preferred_slots, operator, slot_job_vertices):
    mem_required = operator.memory
    preferred_tm_ids = defaultdict(int)
    preferred_host_ids = defaultdict(int)
    for preferred_slot in preferred_slots:
        tm_id = resource.slots[preferred_slot].process
        preferred_tm_ids[tm_id] += 1
        host_id = resource.slots[preferred_slot].device
        preferred_host_ids[host_id] += 1
    candidate = None
    best_score = -1
    for slot_id in slot_pool:
        # sanity check
        if slot_job_vertices[slot_id] is not None and operator.vertex_id in slot_job_vertices[slot_id]:
            continue
        if slot_pool[slot_id] >= mem_required:
            local_weigh = preferred_tm_ids[resource.slots[slot_id].process]
            host_local_weigh = preferred_host_ids[resource.slots[slot_id].device]
            score = local_weigh * 10 + host_local_weigh
            if score > best_score:
                candidate = slot_id
                best_score = score

    return candidate


def select_without_location_preference(slot_pool, operator, slot_job_vertices):
    mem_required = operator.memory
    for slot_id in slot_pool:
        if slot_job_vertices[slot_id] is not None and operator.vertex_id in slot_job_vertices[slot_id]:
            continue
        if slot_pool[slot_id] >= mem_required:
            return slot_id
    return None


def require_new_slot(slot_pool, resource, mem_required):
    for id, slot in resource.slots.items():
        if id in slot_pool or slot.memory < mem_required:
            continue
        return slot
    return None


class FlinkHeuristic(object):
    def __init__(self):
        self.name = 'flink heuristic'

    def place(self, dsp_graph: DSPGraph, resource: Resource, debug=False):
        assert dsp_graph.max_parallelism <= resource.slot_num
        slot_pool = {}
        slot_job_vertices = {}
        placement = []
        for id in range(len(dsp_graph.operators)):
            operator = dsp_graph.operators[id]
            mem_required = operator.memory
            preferred_slots = set()
            # Find preferred locations base on inputs
            for upstream_op_id in operator.inputs:
                preferred_slots.add(placement[upstream_op_id])

            if len(preferred_slots) > 0:
                slot_id = select_with_location_preference(slot_pool, resource, preferred_slots, operator, slot_job_vertices)
            else:
                slot_id = select_without_location_preference(slot_pool, operator, slot_job_vertices)

            if slot_id is None:
                # Request new slot
                new_slot = require_new_slot(slot_pool, resource, mem_required)
                if new_slot is None:
                    if debug:
                        print(f"[flink heuristic] memory requirements cannot be satisfied "
                              f"for graph {dsp_graph.id} and resource {resource.id}")
                    return None
                slot_pool[new_slot.id] = new_slot.memory
                slot_id = new_slot.id

            if slot_id not in slot_job_vertices:
                slot_job_vertices[slot_id] = set()
            slot_job_vertices[slot_id].add(operator.vertex_id)

            slot_pool[slot_id] -= mem_required
            placement.append(slot_id)

        return placement
