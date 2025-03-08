import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from logger import Logger
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_f1, load_fixed_splits
from eval import evaluate
from parse import parse_method, parser_add_main_args
from torch_geometric.utils import degree
import torch_sparse
from torch_sparse import SparseTensor
import warnings
warnings.filterwarnings('ignore')

import pdb

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)

# get the splits for all runs
if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.runs)]
elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m', 'ogbn-papers100M', 'ogbn-papers100M-sub']:
    split_idx_lst = [dataset.load_fixed_splits()
                     for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset, protocol=args.protocol)


### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]
print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

# whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']

# feature propagation (这里是按GCN的一致邻接阵做传播，很多解耦方案是直接在A上传播)
row, col = dataset.graph['edge_index'] 
dg = degree(col, n).float()
d_norm_in = (1. / dg[col]).sqrt()
d_norm_out = (1. / dg[row]).sqrt()
value = torch.ones_like(row) * d_norm_in * d_norm_out
value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(n, n))
x_list=[dataset.graph['node_feat']]
for _ in range(args.num_hops):
    x_list.append(torch_sparse.matmul(adj, x_list[-1]))
# pdb.set_trace()
### Load method ###
model = parse_method(args, c, d)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
print('MODEL:', model)

if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label
else:
    true_label = dataset.label

### Training loop ###
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    num_batch = n // args.batch_size + (n%args.batch_size>0)
    for epoch in range(args.epochs):
        model.cuda()
        model.train()
        # -------------------------------------- batch train ---------------------------------
        idx = torch.randperm(n)
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            train_mask_i = train_mask[idx_i]
            x_i = []
            for h in range(args.num_hops + 1):
                x_i.append(x_list[h][idx_i].cuda())
            y_i = true_label[idx_i].cuda()
            optimizer.zero_grad()
            out_i = model(x_i)
            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))
            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
            loss.backward()
            optimizer.step()
        # -------------------------------------- batch train ---------------------------------

        # -------------------------------------- batch test ---------------------------------
        if epoch % args.eval_step == 0:
            result = evaluate(model, x_list, split_idx, args, n, true_label)
            logger.add_result(run, result[:-1])

            if epoch % args.display_step == 0:
                print_str = f'Epoch: {epoch:02d}, ' + \
                            f'Loss: {loss:.4f}, ' + \
                            f'Train: {100 * result[0]:.2f}%, ' + \
                            f'Valid: {100 * result[1]:.2f}%, ' + \
                            f'Test: {100 * result[2]:.2f}%'
                print(print_str)
        # -------------------------------------- batch test ---------------------------------
    logger.print_statistics(run)

logger.print_statistics()