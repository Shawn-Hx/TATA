import time
import torch
import torch.optim as optim

from model import Model
from train import train_epoch

from options import get_options
from util import *


def run(opts):
    # Set the random seed
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)

    # Set the device
    opts.device = torch.device(f'cuda:{opts.gpu_id}' if opts.use_cuda else 'cpu')

    # Load and prepare data
    train_graphs = load_graphs(dirname=opts.train_dsp_dataset_dir)
    valid_graphs = load_graphs(dirname=opts.valid_dsp_dataset_dir)
    resources = load_resources(opts.communicate_costs, dirname=opts.res_dataset_dir)

    train_data = build_samples(train_graphs, resources, opts)
    valid_data = build_samples(valid_graphs, resources, opts)

    # train_data, valid_data = data_split(total_data, opts.train_ratio, shuffle=True)
    build_feature(train_data, is_train=True)
    build_feature(valid_data, is_train=False)
    train_data = data_augment(train_data, opts.train_batch_size)

    # Initialize model
    model = Model(opts.op_dim,
                  opts.slot_dim,
                  opts.edge_dim,
                  opts.embed_dim,
                  opts.dsp_conv_iter,
                  opts.res_conv_iter,
                  opts.dsp_gcn_aggr,
                  opts.res_gcn_aggr,
                  opts.gcn_act,
                  opts.rnn_type,
                  opts.tanh_clip).to(opts.device)
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)

    if opts.save_model:
        model_dir = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.mkdir(os.path.join(opts.model_dir, model_dir))

    best_avg_reward = -1
    for epoch in range(1, opts.epochs + 1):
        valid_avg_reward = train_epoch(train_data, valid_data, model, optimizer, lr_scheduler, epoch, opts)
        if opts.save_model and epoch > opts.save_model_epoch_threshold and valid_avg_reward > best_avg_reward:
            best_avg_reward = valid_avg_reward
            torch.save(model, f'model/{model_dir}/best_model.pt')


if __name__ == '__main__':
    run(get_options())
