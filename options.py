import argparse
import torch


def get_options():
    parser = argparse.ArgumentParser(
        description='Deep reinforcement model for solving the streaming operator placement problem')

    # Data
    parser.add_argument('--train_dsp_dataset_dir', type=str, default='dsp_dataset/train',
                        help='dag dataset directory for training')
    parser.add_argument('--valid_dsp_dataset_dir', type=str, default='dsp_dataset/valid',
                        help='dag dataset directory for validation')
    parser.add_argument('--res_dataset_dir', type=str, default='resource_dataset',
                        help='resource dataset directory')
    parser.add_argument('--model_dir', type=str, default='model', help='save model directory')
    parser.add_argument('--resources_per_dag', type=int, default=5, help='...')
    parser.add_argument('--max_slot_num_greater_than_max_parall', type=int, default=3, help='...')


    # Mixed
    parser.add_argument('--baselines', type=str,
                        default=['flink', 'storm'], nargs='+',
                        help='baselines to train the model, choices: flink, storm and random')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='reward = alpha * throughput + (1-alpha) * delay')
    parser.add_argument('--punishment', type=float, default=-5,
                        help='punishment reward if memory requirement is not satisfied')
    parser.add_argument('--communicate_costs', type=float,
                        default=[1, 1.5, 2, 4], nargs='+',
                        help='different levels communication costs, which are used to estimate delay')

    # Model
    parser.add_argument('--op_dim', type=int, default=2, help='feature dimensions of an DSP operator')
    parser.add_argument('--slot_dim', type=int, default=2, help='feature dimensions of a slot')
    parser.add_argument('--edge_dim', type=int, default=1, help='feature dimensions of an edge in resources graph')
    parser.add_argument('--embed_dim', type=int, default=128, help='embedding vector dimensions of a slot/operator')
    parser.add_argument('--dsp_conv_iter', type=int, default=2, help='graph conv iteration times of DSP graph')
    parser.add_argument('--res_conv_iter', type=int, default=2, help='graph conv iteration times of resource graph')
    parser.add_argument('--dsp_gcn_aggr', choices=['mean', 'max', 'add'], default='mean',
                        help='aggregation scheme in dsp GCN')
    parser.add_argument('--res_gcn_aggr', choices=['mean', 'max', 'add'], default='mean',
                        help='aggregation scheme in resource GCN')
    parser.add_argument('--gcn_act', choices=['relu', 'tanh'], default='relu',
                        help='activation function in GCN')
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default='GRU',
                        help='the rnn type to use, LSTM or GRU')
    parser.add_argument('--tanh_clip', type=float, default=10,
                        help='Clip the parameters to within +- this value using tanh. '
                             'Set to 0 to not perform any clipping.')

    # Training
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='learning rate decay per epoch')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train')
    parser.add_argument('--train_batch_size', type=int, default=20, help='batch size')
    # parser.add_argument('--train_ratio', type=float, default=0.7, help='training data ratio')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='maximum L2 norm for gradient clipping')
    parser.add_argument('--mem_restrict_epoch_threshold', type=int, default=10000,
                        help='after how much epochs to ignore memory restriction')
    parser.add_argument('--save_model', type=bool, default=True, help='save model to file or not')
    parser.add_argument('--save_model_epoch_threshold', type=int, default=500,
                        help='after how much epochs to start save the model')
    parser.add_argument('--seed', type=int, default=1234, help='set random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA device')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')


    opts = parser.parse_args()
    opts.use_cuda = torch.cuda.is_available() and opts.cuda

    return opts
