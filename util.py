import os
import json
from sklearn.preprocessing import StandardScaler

from qos_retriever import *
from baseline.flink_heuristic import *
from baseline.flink_heuristic_new import *
from baseline.storm_heuristic import *
from baseline.random_place import *


class SampleData(object):
    def __init__(self, graph, resource, bl_throughputs, bl_delays):
        self.graph = graph
        self.resource = resource
        self.bl_throughputs = bl_throughputs
        self.bl_delays = bl_delays
        self.bl_throughput_mean = np.mean(bl_throughputs)
        self.bl_delay_mean = np.mean(bl_delays)


def get_reward(sample: SampleData, placement: list, alpha: float, mem_restrict):
    throughput, delay = get_qos(sample.graph, sample.resource, placement, memory_restrict=mem_restrict)
    if throughput == -1 and delay == -1:
        return None

    # t_betters = []
    # d_betters = []
    # for bl_throughput, bl_delay in zip(sample.bl_throughputs, sample.bl_delays):
    #     t_better = (throughput - bl_throughput) / bl_throughput
    #     d_better = (bl_delay - delay) / bl_delay
    #     t_betters.append(t_better)
    #     d_betters.append(d_better)
    #
    # t_better = np.mean(t_betters)
    # d_better = np.mean(d_betters)

    t_better = (throughput - sample.bl_throughput_mean) / sample.bl_throughput_mean
    d_better = (sample.bl_delay_mean - delay) / sample.bl_delay_mean

    return alpha * t_better + (1 - alpha) * d_better


def build_samples(graphs, resources, opts):
    """
    Build samples.
    A sample is composed of a dsp graph and the corresponding resource
    :param graphs:      dsp graphs
    :param resources:   resources set
    :return:            tuple list
    """
    # count slot number
    slot_num_dict = defaultdict(list)
    for r in resources:
        slot_num_dict[r.slot_num].append(r)
    sample_data_list = []
    baselines = []
    for baseline in opts.baselines:
        if baseline == 'storm':
            baselines.append(StormHeuristic(opts))
        elif baseline == 'flink':
            baselines.append(FlinkHeuristicNew(opts))
        elif baseline == 'random':
            baselines.append(RandomStrategy(opts))
        else:
            print(f'please specify correct baseline name: {baseline}')
            return

    for g in graphs:
        provide_resources = []
        # dsp.max_parallelism <= res.slot_num <= dsp.max+parallelism + ?
        max_slot_num_greater_than_max_parall = opts.max_slot_num_greater_than_max_parall
        for i in range(g.max_parallelism, g.max_parallelism + max_slot_num_greater_than_max_parall + 1):
            provide_resources.extend(slot_num_dict[i])
        sample_num = min(len(provide_resources), opts.resources_per_dag)
        selected_resources = random.sample(provide_resources, sample_num)
        for r in selected_resources:
            bl_throughputs = []
            bl_delays = []
            for baseline in baselines:
                bl_place = baseline.place(g, r)
                if bl_place is not None:
                    throughput, delay = get_qos(g, r, bl_place)
                    assert throughput != -1 and delay != -1
                    if throughput == -1 or delay == -1:
                        print('error! hx debug')
                    bl_throughputs.append(throughput)
                    bl_delays.append(delay)
            sample_data_list.append(SampleData(g, r, bl_throughputs, bl_delays))
    return sample_data_list


def build_one_sample(graph, resource):
    baseline = FlinkHeuristic()
    bl_place = baseline.place(graph, resource)
    if bl_place is not None:
        throughput, delay = get_qos(graph, resource, bl_place)
        assert throughput != -1 and delay != -1
        return SampleData(graph, resource, throughput, delay)
    return None


def data_split(data, ratio, shuffle=False):
    size = len(data)
    offset = int(size * ratio)
    if shuffle:
        random.shuffle(data)
    return data[:offset], data[offset:]


def data_augment(train_data, train_batch_size):
    n = len(train_data)
    if n % train_batch_size == 0:
        return train_data
    add_num = train_batch_size - n % train_batch_size
    train_data.extend(random.sample(train_data, add_num))
    return train_data


def parse_json_graph(data):
    graph = DSPGraph(data["id"], data["max_parallelism"])

    for op in data["operators"]:
        operator = Operator(
            op["id"],
            op["vertex_id"],
            op["task_id"],
            op["is_source"],
            op["is_sink"],
            op["cpu"],
            op["memory"]
        )
        graph.add_node(operator)

    for edge in data["edges"]:
        graph.add_edge(
            edge["from_id"],
            edge["to_id"],
        )
    return graph


def load_graphs(dirname='dsp_dataset'):
    graphs = []
    for filename in os.listdir(dirname):
        if filename.startswith("graph") and filename.endswith(".json"):
            data = json.load(open(os.path.join(dirname, filename)))
            # id = int(filename.split('.')[0].split('_')[1])
            graph = parse_json_graph(data)
            graphs.append(graph)

    return graphs


def parse_json_resource(data, id, costs):
    slot_num = len(data)
    resource = Resource(id, slot_num)
    for element in data:
        slot = Slot(
            element['id'],
            element['cpu'],
            element['memory'],
            element['process'],
            element['device'],
        )
        resource.add_slot(slot)
    for i in range(slot_num):
        for j in range(slot_num):
            slot_i = resource.slots[i]
            slot_j = resource.slots[j]
            if i == j:
                cost = costs[0]
            elif slot_i.process == slot_j.process:
                cost = costs[1]
            elif slot_i.device == slot_j.device:
                cost = costs[2]
            else:
                cost = costs[3]
            resource.add_edge(i, j, cost)
    return resource


def load_resources(costs, dirname='resource_dataset'):
    resources = []
    for filename in os.listdir(dirname):
        if filename.startswith('resource') and filename.endswith('.json'):
            data = json.load(open(os.path.join(dirname, filename)))
            id = int(filename.split('.')[0].split('_')[1])
            resource = parse_json_resource(data, id, costs)
            resources.append(resource)

    return resources


def get_op_feat(op):
    return [op.cpu, op.memory]


op_scaler = StandardScaler()


def build_graph_feature(samples, is_train=True):
    all_feats = []
    for sample in samples:
        graph = sample.graph
        op_feats = np.vstack([get_op_feat(graph.operators[i])
                              for i in range(len(graph.operators))])
        all_feats.append(op_feats)

    if is_train:
        feats = np.vstack([_ for _ in all_feats])
        op_scaler.fit(feats)

    for i in range(len(all_feats)):
        samples[i].graph.op_feats = op_scaler.transform(all_feats[i])


def get_slot_feat(slot):
    return [slot.cpu, slot.memory]


slot_scaler = StandardScaler()
edge_scaler = StandardScaler()


def build_res_feature(samples, is_train=True):
    all_slot_feats = []
    for sample in samples:
        res = sample.resource
        slot_feats = np.vstack([get_slot_feat(res.slots[i])
                                for i in range(res.slot_num)])
        all_slot_feats.append(slot_feats)

    if is_train:
        feats = np.vstack([_ for _ in all_slot_feats])
        slot_scaler.fit(feats)

    for i in range(len(all_slot_feats)):
        samples[i].resource.slot_feats = slot_scaler.transform(all_slot_feats[i])

    # edge feature
    all_edge_feats = []
    for sample in samples:
        res = sample.resource
        edge_attr = []
        for i in range(len(res.edges[0])):
            slot1 = res.edges[0][i]
            slot2 = res.edges[1][i]
            edge_attr.append([res.matrix[slot1][slot2]])
        all_edge_feats.append(edge_attr)
        # res.edge_feats = np.array(edge_attr, dtype=np.float)

    if is_train:
        feats = np.vstack([_ for _ in all_edge_feats])
        edge_scaler.fit(feats)

    for i in range(len(samples)):
        samples[i].resource.edge_feats = edge_scaler.transform(all_edge_feats[i])


def build_feature(samples, is_train=True):
    build_graph_feature(samples, is_train)
    build_res_feature(samples, is_train)


def load_single_dsp_graph(filename):
    return parse_json_graph(json.load(open(filename)))


def load_single_resourse(filename, costs):
    return parse_json_resource(json.load(open(filename)), 0, costs)


def select_graph_by_max_parallelism(graphs, max_parallelism):
    ans = []
    for g in graphs:
        if g.max_parallelism == max_parallelism:
            ans.append(g)
    return ans


def select_res_by_slot_num(resources, slot_num):
    ans = []
    for r in resources:
        if r.slot_num == slot_num:
            ans.append(r)
    return ans
