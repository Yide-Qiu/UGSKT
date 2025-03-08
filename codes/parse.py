from gnns import *
from ours import *

def parse_method(args, c, d):
    if args.method == 'c3':
        model = C3(args, d, args.hidden_channels, c, activation=F.relu)
    elif args.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).cuda()
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, use_bn=args.use_bn, heads=args.gat_heads, out_heads=args.out_heads).cuda()
    elif args.method == 'mlp' or args.method == 'manireg':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).cuda()
    elif args.method == 'heat':
        model = GCN(in_channels=d,
                    hidden_channels=args.hidden_channels,
                    out_channels=c,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    use_bn=args.use_bn).cuda()
    elif args.method == 'sgc':
        # model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        model = SGCMem(in_channels=d, out_channels=c, hops=args.hops, use_bn=args.use_bn).cuda()
    elif args.method=='sgc2':
        model=SGC2(d,args.hidden_channels,c,args.hops,args.num_layers,args.dropout, use_bn=args.use_bn).cuda()
    elif args.method=='sign':
        model=SIGN(in_channels=d,hidden_channels=args.hidden_channels,
                    out_channels=c, hops=args.hops, num_layers=args.num_layers,
                    dropout=args.dropout,use_bn=args.use_bn).cuda()
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=1, help='number of distinct runs')
    parser.add_argument('--directed', action='store_true', help='set to not symmetrize adjacency')
    parser.add_argument('--train_prop', type=float, default=.5, help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25, help='validation label proportion')
    parser.add_argument('--protocol', type=str, default='semi', help='protocol for cora datasets, semi or supervised')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--rand_split_class', action='store_true', help='use random splits with a fixed number of labeled nodes for each class')
    parser.add_argument('--label_num_per_class', type=int, default=20, help='labeled nodes randomly selected')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'], help='evaluation metric')

    # model branch
    parser.add_argument('--method', type=str, default='c3')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--num_hops', type=int, default=5)
    parser.add_argument('--n_layers_1', type=int, default=2)
    parser.add_argument('--n_layers_2', type=int, default=1)
    parser.add_argument('--weight_style', type=str, default='HA')
    parser.add_argument('--HA_activation', type=str, default='leakyrelu')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--zero_inits', action='store_true')
    parser.add_argument('--position_emb', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--input_drop', type=float, default=0.0)
    parser.add_argument('--feat_drop', type=float, default=0.0)
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--edge_drop', type=float, default=0.0)
    parser.add_argument('--diffusion_drop', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--propagate_first', action='store_true')
    parser.add_argument('--negative_slope', type=float, default=0.2)

    # training
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=10000, help='mini batch training for large graphs')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience.')
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--save_att', action='store_true', help='whether to save attention (for visualization)')
    parser.add_argument('--model_dir', type=str, default='../../model/')

    # other gnn parameters (for baselines)
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')
    
