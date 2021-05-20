class DSPGraph(object):
    def __init__(self, index, max_parallelism):
        self.id = index
        self.max_parallelism = max_parallelism
        self.operators = {}
        self.edges = [[], []]

    def add_node(self, operator):
        self.operators[operator.id] = operator

    def add_edge(self, from_id, to_id):
        self.edges[0].append(from_id)
        self.edges[1].append(to_id)
        self.operators[from_id].add_output(to_id)
        self.operators[to_id].add_input(from_id)


class Operator(object):
    def __init__(self, id, vertex_id, task_id, is_source, is_sink, cpu, memory):
        self.id = id
        self.vertex_id = vertex_id
        self.task_id = task_id
        self.is_source = is_source
        self.is_sink = is_sink
        self.cpu = cpu
        self.memory = memory
        self.inputs = []
        self.outputs = []

    def add_input(self, from_id):
        self.inputs.append(from_id)

    def add_output(self, to_id):
        self.outputs.append(to_id)
