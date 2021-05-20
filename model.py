import math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch_geometric.nn import MessagePassing


class DAGConv(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr='mean', act='relu', **kwargs):
        super(DAGConv, self).__init__(aggr=aggr, **kwargs)
        self.linear = nn.Linear(in_dim, out_dim)
        self.updater_linear = nn.Linear(in_dim + out_dim, out_dim, bias=False)
        self.act = nn.Tanh() if act == 'tanh' else nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.updater_linear.reset_parameters()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        x_j = self.linear(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, inputs, x):
        new = torch.cat([x, inputs], dim=1)
        new = self.updater_linear(new)
        new = self.act(new)
        return new


class DAGEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, k, aggr='mean', act='relu'):
        super(DAGEncoder, self).__init__()
        self.up_conv1 = DAGConv(in_dim, out_dim // 2, aggr=aggr, act=act, flow='source_to_target')
        self.down_conv1 = DAGConv(in_dim, out_dim // 2, aggr=aggr, act=act, flow='target_to_source')
        self.up_conv2 = DAGConv(out_dim, out_dim // 2, aggr=aggr, act=act, flow='source_to_target')
        self.down_conv2 = DAGConv(out_dim, out_dim // 2, aggr=aggr, act=act, flow='target_to_source')
        self.k = k

    def forward(self, x, edge_index):
        x_u = self.up_conv1(x, edge_index)
        x_d = self.down_conv1(x, edge_index)
        x = torch.cat([x_u, x_d], dim=1)
        for _ in range(self.k - 1):
            x_u = self.up_conv2(x, edge_index)
            x_d = self.up_conv2(x, edge_index)
            x = torch.cat([x_u, x_d], dim=1)
        return x


class GraphAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, aggr='max'):
        super(GraphAggregator, self).__init__()
        self.aggr = aggr
        self.linear = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, ops):
        g = self.linear(ops)
        g = self.relu(g)
        if self.aggr == 'mean':
            g = torch.mean(g, 0)
        else:
            g, _ = torch.max(g, 0)
        return g


class ResourceConv(MessagePassing):
    def __init__(self, in_dim, out_dim, edge_dim, aggr='mean', act='relu', **kwargs):
        super(ResourceConv, self).__init__(aggr=aggr, **kwargs)
        self.linear = nn.Linear(in_dim + edge_dim, out_dim)
        self.update_linear = nn.Linear(in_dim + out_dim, out_dim, bias=False)
        self.act = nn.Tanh() if act == 'tanh' else nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.linear.reset_parameters()
        self.update_linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, edge_attr=edge_attr, x=x)

    def message(self, x_j, edge_attr):
        x_j = torch.cat([x_j, edge_attr], dim=1)
        x_j = self.linear(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, inputs, x):
        new = torch.cat([x, inputs], dim=1)
        new = self.update_linear(new)
        new = self.act(new)
        return new


class ResourceEncoder(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim, k, aggr='mean', act='relu'):
        super(ResourceEncoder, self).__init__()
        self.conv1 = ResourceConv(in_dim, out_dim, edge_dim, aggr=aggr, act=act)
        self.conv2 = ResourceConv(out_dim, out_dim, edge_dim, aggr=aggr, act=act)
        self.k = k

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        for _ in range(self.k - 1):
            x = self.conv2(x, edge_index, edge_attr)
        return x


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, query, key):
        e = torch.mm(key, query.t())
        alpha = self.softmax(e)
        c = alpha * key
        c = torch.sum(c, dim=0)
        return c.unsqueeze(0)


class LogProb(nn.Module):
    def __init__(self, query_dim, key_dim, tanh_clip):
        super(LogProb, self).__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.tanh_clip = tanh_clip
        self.tanh = nn.Tanh()
        self.w_q = nn.Linear(query_dim, key_dim, bias=False)
        self.w_k = nn.Linear(key_dim, key_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def reset_parameters(self):
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()

    def forward(self, query, key):
        q = self.w_q(query)
        k = self.w_k(key)

        if self.tanh_clip > 0:
            u = self.tanh_clip * self.tanh(torch.mm(k, q.t()) / math.sqrt(self.key_dim))
        else:
            u = torch.mm(k, q.t()) / math.sqrt(self.key_dim)
        probs = self.softmax(u.t())
        return probs


class PolicyNet(nn.Module):
    def __init__(self, dim, rnn_type, tanh_clip):
        super(PolicyNet, self).__init__()
        self.dim = dim
        if rnn_type == 'LSTM':
            self.rnn_cell = nn.LSTMCell(input_size=dim, hidden_size=dim)
            self.cell_state = nn.Parameter(torch.zeros(1, dim))
        elif rnn_type == 'GRU':
            self.rnn_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.temporal_attenion = AttentionLayer()
        self.log_prob = LogProb(query_dim=dim * 2, key_dim=dim, tanh_clip=tanh_clip)
        self.softmax = nn.Softmax()

        self.rnn_type = rnn_type
        self.saved_log_probs = []

    def _get_upstream_slots(self, id, edge_index, placement, slots_embed):
        upstream_ids = []
        for i in range(len(edge_index[0])):
            if edge_index[1][i] == id:
                upstream_ids.append(edge_index[0][i])
        if len(upstream_ids) == 0:
            return torch.zeros([1, self.dim], dtype=torch.float).to(slots_embed.device)
        upstream_slots = torch.cat([slots_embed[placement[id]] for id in upstream_ids]).to(slots_embed.device)
        return torch.max(upstream_slots, dim=0)[0]

    def _init_hidden(self, initial_hidden):
        if self.rnn_type == 'LSTM':
            return initial_hidden, self.cell_state
        return initial_hidden

    def forward(self, ops_embed, graph_embed, edge_index, slots_embed, is_train):
        num_ops = ops_embed.size(0)
        placement = []
        # initial_hidden = graph_embed.unsqueeze(0)
        initial_hidden = graph_embed.view(1, -1)
        hidden = self._init_hidden(initial_hidden)
        for i in range(num_ops):
            last_slots = self._get_upstream_slots(i, edge_index, placement, slots_embed)
            input = ops_embed[i].unsqueeze(0) + last_slots
            hidden = self.rnn_cell(input, hidden)
            output = hidden[0] if self.rnn_type == 'LSTM' else hidden
            c = self.temporal_attenion(output, ops_embed)
            cat = torch.cat([output, c], dim=1)
            probs = self.log_prob(cat, slots_embed)

            # if is_train:
            m = Categorical(probs)
            action = m.sample()
            self.saved_log_probs.append(m.log_prob(action))
            slot_id = action.item()
            # else:
            #     prob, index = torch.max(probs.squeeze(0), dim=0)
            #     slot_id = index.item()

            placement.append(slot_id)

        return placement


class Model(nn.Module):
    def __init__(self, op_dim, slot_dim, edge_dim, embed_dim,
                 dsp_iter=2, res_iter=2,
                 dsp_gcn_aggr='mean',
                 res_gcn_aggr='mean',
                 gcn_act='relu',
                 rnn_type='LSTM', tanh_clip=10):
        super(Model, self).__init__()
        self.dsp_encoder = DAGEncoder(op_dim, embed_dim, k=dsp_iter, aggr=dsp_gcn_aggr, act=gcn_act)
        self.aggregator = GraphAggregator(embed_dim, embed_dim)
        self.res_encoder = ResourceEncoder(slot_dim, edge_dim, embed_dim, k=res_iter, aggr=res_gcn_aggr, act=gcn_act)
        self.policy_net = PolicyNet(embed_dim, rnn_type, tanh_clip)

    def forward(self, op_feats, dsp_edge_index, slot_feats, res_edge_index, res_edge_attr, is_train=True):
        ops_embed = self.dsp_encoder(op_feats, dsp_edge_index)
        graph_embed = self.aggregator(ops_embed)
        slots_embed = self.res_encoder(slot_feats, res_edge_index, res_edge_attr)
        placement = self.policy_net(ops_embed, graph_embed, dsp_edge_index, slots_embed, is_train)
        return placement

    def get_log_probs(self):
        return self.policy_net.saved_log_probs

    def finish_episode(self):
        self.policy_net.saved_log_probs.clear()


