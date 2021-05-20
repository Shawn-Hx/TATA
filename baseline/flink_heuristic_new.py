import random
from collections import defaultdict

from graph import *
from resource import *


class FlinkSlot(object):
    def __init__(self, id, mem, tm_id, host_id):
        self.id = id
        self.mem = mem
        self.tm_id = tm_id
        self.host_id = host_id


class FlinkHeuristicNew(object):
    def __init__(self, opts):
        self.name = 'flink heuristic'
        random.seed(opts.seed)

    # Locality:
    # - UNCONSTRAINED   0
    # - LOCAL           1
    # - HOST_LOCAL      2
    # - NON_LOCAL       3
    # - UNKNOWN         4
    def select_with_location_preference(self, slots, operator, preferred_slots):
        mem_required = operator.memory
        preferred_tm_ids = defaultdict(int)
        preferred_host_ids = defaultdict(int)
        for preferred_slot in preferred_slots:
            preferred_tm_ids[preferred_slot.tm_id] += 1
            preferred_host_ids[preferred_slot.host_id] += 1
        candidate = None
        best_score = -1
        locality = 3
        for slot in slots:
            if slot.mem >= mem_required:
                local_weigh = preferred_tm_ids[slot.tm_id]
                host_local_weigh = preferred_host_ids[slot.host_id]
                score = local_weigh * 10 + host_local_weigh
                if score > best_score:
                    candidate = slot
                    best_score = score
                    locality = 1 if local_weigh > 0 else 2 if host_local_weigh > 0 else 3
        if candidate is not None:
            return candidate, locality
        return None, -1

    def select_without_location_preference(self, slots, operator):
        for slot in slots:
            if slot.mem > operator.memory:
                return slot, -1
        return None, -1

    def select_best(self, slots, operator, preferred_slots):
        if len(slots) == 0:
            return None, -1
        if None in slots:
            print('debug')
        if len(preferred_slots) > 0:
            return self.select_with_location_preference(slots, operator, preferred_slots)
        else:
            return self.select_without_location_preference(slots, operator)

    def try_allocate_from_available(self, slot_pool, operator, preferred_slots):
        slot, locality = self.select_best(slot_pool, operator, preferred_slots)
        if slot is not None:
            slot_pool.remove(slot)
        return slot, locality

    def request_new_slot(self, free_slots, operator):
        if len(free_slots) == 0:
            print('debug')
        candidate_slots = []
        for slot in free_slots:
            if slot.mem > operator.memory:
                candidate_slots.append(slot)
        choose_slot = random.choice(candidate_slots)
        free_slots.remove(choose_slot)
        return choose_slot

    def place(self, dsp_graph: DSPGraph, resource: Resource):
        assert dsp_graph.max_parallelism <= resource.slot_num
        all_slots = {}
        slot_pool = []
        free_slots = []
        for id, slot in resource.slots.items():
            flink_slot = FlinkSlot(slot.id, slot.memory, slot.process, slot.device)
            all_slots[id] = flink_slot
            free_slots.append(flink_slot)

        slot_groups = defaultdict(set)  # slot contains job vertex ids
        placement = []
        for id in range(len(dsp_graph.operators)):
            operator = dsp_graph.operators[id]

            preferred_slots = set()
            # Find preferred locations base on inputs
            for upstream_op_id in operator.inputs:
                preferred_slots.add(all_slots[placement[upstream_op_id]])

            resolved_slots = []
            # sanity check
            for slot_id, groups in slot_groups.items():
                if len(groups) > 0 and operator.vertex_id not in groups:
                    resolved_slots.append(all_slots[slot_id])

            slot1, locality = self.select_best(resolved_slots, operator, preferred_slots)

            if slot1 is not None and locality == 1:
                placement.append(slot1.id)
                slot1.mem -= operator.memory
                slot_groups[slot1.id].add(operator.vertex_id)
                continue

            slot2, locality = self.try_allocate_from_available(slot_pool, operator, preferred_slots)
            if slot2 is not None and (locality == 1 or slot1 is None):
                placement.append(slot2.id)
                slot2.mem -= operator.memory
                slot_groups[slot2.id].add(operator.vertex_id)
                continue

            if slot1 is not None:
                placement.append(slot1.id)
                slot1.mem -= operator.memory
                slot_groups[slot1.id].add(operator.vertex_id)
                if slot2 is not None:
                    slot_pool.append(slot2)
                continue

            slot = self.request_new_slot(free_slots, operator)
            if slot is None:
                print('[flink] error, cannot fulfill memory requirements')
                return None
            placement.append(slot.id)
            slot.mem -= operator.memory
            slot_groups[slot.id].add(operator.vertex_id)

        return placement
