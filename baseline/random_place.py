import random

from graph import *
from resource import *


class RandomStrategy(object):
    def __init__(self, opts):
        self.name = 'random strategy'
        self.opts = opts
        self.seed(self.opts)

    def seed(self, opts):
        random.seed(opts.seed)

    def place(self, dsp_graph: DSPGraph, resource: Resource):
        placement = []
        # try at most 10 times
        for _ in range(10):
            slot_mem = {}
            for slot_id, slot in resource.slots.items():
                slot_mem[slot_id] = slot.memory
            for id in range(len(dsp_graph.operators)):
                op = dsp_graph.operators[id]
                candidates = []
                for slot_id, mem in slot_mem.items():
                    if mem >= op.memory:
                        candidates.append(slot_id)
                if len(candidates) == 0:
                    placement.clear()
                    break
                choose_id = random.choice(candidates)
                slot_mem[choose_id] -= op.memory
                placement.append(choose_id)

            if len(placement) > 0:
                break

        if len(placement) == 0:
            print(f'[random strategy]: memory requirements cannot be satisfied '
                  f'for graph {dsp_graph.id} and resource {resource.id}')
            return None

        return placement
