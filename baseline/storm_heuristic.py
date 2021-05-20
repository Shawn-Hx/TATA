from collections import defaultdict

from graph import *
from resource import *


class StormHeuristic(object):
    def __init__(self, opts):
        self.name = 'Storm heuristic (polling)'
        self.opts = opts

    def place(self, dsp_graph: DSPGraph, resource: Resource):
        num_slots = len(resource.slots)
        # count each device's number of slots
        device_slot_num = defaultdict(list)
        for _, slot in resource.slots.items():
            device_slot_num[slot.device].append(slot)
        # sort by number of slots (reversed)
        ordered_devices = sorted(device_slot_num.items(), key=lambda x: len(x), reverse=True)
        # element in ordered slots: (slot id, memory remained)
        ordered_slots = []
        for device, slots in ordered_devices:
            # sort by cpu computational capability (reversed)
            slots.sort(key=lambda x: x.cpu, reverse=True)
            for slot in slots:
                ordered_slots.append((slot.id, slot.memory))

        placement = []
        slot_index = 0
        for op_id in range(len(dsp_graph.operators)):
            operator = dsp_graph.operators[op_id]
            mem_required = operator.memory
            can_assign = False
            for i in range(num_slots):
                slot_index = (slot_index + i) % num_slots
                slot_id, mem_remained = ordered_slots[slot_index]
                if mem_required < mem_remained:
                    ordered_slots[slot_index] = (slot_id, mem_remained - mem_required)
                    placement.append(slot_id)
                    can_assign = True
                    break
            if not can_assign:
                print(f"[storm heuristic] memory requirements cannot be satisfied "
                      f"for graph {dsp_graph.id} and resource {resource.id}")
                return None
            slot_index += 1

        return placement
