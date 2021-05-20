import numpy as np
import random
import torch

from util import get_reward


def get_placement(model, sample, opts, is_train=True):
    graph = sample.graph
    resource = sample.resource

    op_feats = torch.tensor(graph.op_feats, dtype=torch.float).to(opts.device)
    dsp_edge_index = torch.tensor(graph.edges, dtype=torch.long).to(opts.device)
    slot_feats = torch.tensor(resource.slot_feats, dtype=torch.float).to(opts.device)
    res_edge_index = torch.tensor(resource.edges, dtype=torch.long).to(opts.device)
    res_edge_attr = torch.tensor(resource.edge_feats, dtype=torch.float).to(opts.device)

    placement = model(op_feats, dsp_edge_index, slot_feats, res_edge_index, res_edge_attr, is_train)
    return placement


def validate(model, valid_data, opts):
    model.eval()
    total_size = len(valid_data)
    total_reward_with_restrict = 0
    total_reward_without_restrict = 0
    exceed_mem_cnt = 0
    for sample in valid_data:
        placement = get_placement(model, sample, opts, is_train=False)
        reward_with_restrict = get_reward(sample, placement, opts.alpha, mem_restrict=True)
        if reward_with_restrict is None:
            reward_with_restrict = opts.punishment
            exceed_mem_cnt += 1
        total_reward_with_restrict += reward_with_restrict
        reward_without_restrict = get_reward(sample, placement, opts.alpha, mem_restrict=False)
        total_reward_without_restrict += reward_without_restrict

    avg_reward_with_restrict = total_reward_with_restrict / total_size
    avg_reward_without_restrict = total_reward_without_restrict / total_size
    return avg_reward_with_restrict, exceed_mem_cnt, avg_reward_without_restrict


def train_epoch(train_data, valid_data, model, optimizer, lr_scheduler, epoch, opts):
    model.train()

    train_size = len(train_data)
    valid_size = len(valid_data)
    batch_size = opts.train_batch_size
    total_reward = 0
    exceed_mem_cnt = 0
    random.shuffle(train_data)
    for i in range(train_size // batch_size):
        batch_rewards = []
        batch_log_probs = []
        optimizer.zero_grad()
        for j in range(i * batch_size, (i + 1) * batch_size):
            sample = train_data[j]
            placement = get_placement(model, sample, opts, is_train=True)

            if epoch >= opts.mem_restrict_epoch_threshold:
                reward = get_reward(sample, placement, opts.alpha, mem_restrict=False)
            else:
                reward = get_reward(sample, placement, opts.alpha, mem_restrict=True)
            if reward is None:
                reward = opts.punishment
                exceed_mem_cnt += 1
            log_prob = torch.cat(model.get_log_probs()).sum()
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)

            model.finish_episode()

        baseline = np.mean(batch_rewards)
        R = torch.tensor(batch_rewards) - baseline
        batch_log_probs_stack = torch.stack(batch_log_probs)
        loss = - (batch_log_probs_stack * R).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts.grad_clip)
        optimizer.step()

        total_reward += np.sum(batch_rewards)

    lr_scheduler.step()

    train_avg_reward = total_reward / train_size
    valid_avg_reward, valid_exceed_mem_cnt, valid_avg_reward_without_restrict = validate(model, valid_data, opts)

    print(f"epoch {format(epoch, '4d')}\t "
          f"train avg reward: {format(train_avg_reward, '8.4f')}\t "
          f"{format(exceed_mem_cnt, '2d')}/{train_size}\t "
          f"valid avg reward: {format(valid_avg_reward, '8.4f')}\t "
          f"{format(valid_exceed_mem_cnt, '2d')}/{valid_size}\t "
          f"without mem restrict: {format(valid_avg_reward_without_restrict, '8.4f')}\t "
          )
    return valid_avg_reward

