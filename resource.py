import numpy as np


class Resource(object):
    def __init__(self, id, slot_num):
        self.id = id
        self.slot_num = slot_num
        self.slots = {}
        self.edges = [[], []]
        self.matrix = np.zeros([slot_num, slot_num], dtype=np.float)

    def add_slot(self, slot):
        self.slots[slot.id] = slot

    def add_edge(self, from_id, to_id, cost):
        self.edges[0].append(from_id)
        self.edges[1].append(to_id)
        self.matrix[from_id][to_id] = cost


class Slot(object):
    def __init__(self, id, cpu, memory, process, device):
        self.id = id
        self.cpu = cpu
        self.memory = memory
        self.process = process
        self.device = device
