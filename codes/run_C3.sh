
python main.py --method c3  --dataset ogbn-arxiv --metric acc --lr 0.002 --hidden_channels 256 --num_heads 3 --num_hops 4 \
    --weight_style 'HA'  --n_layers_1 2 --n_layers_2 1 --HA_activation 'leakyrelu' --residual --zero_inits --position_emb \
    --batch_norm --dropout 0.75 --input_drop 0.1 --edge_drop 0.15 --attn_drop 0. --diffusion_drop 0. --bias --propagate_first \
    --negative_slope 0.2 --seed 123 --runs 5 --epochs 1000 --eval_step 9 > results/arxiv_v6.txt

python main-c3.py --method agdn --dataset ogbn-arxiv \
    --seed 0 \
    --n-label-iters 0 \
    --lr 0.002 \
    --model agdn \
    --mode test \
    --standard-loss \
    --n-layers 3 \
    --n-hidden 256 \
    --K 3 \
    --n-heads 3 \
    --dropout 0.75 \
    --input_drop 0.1 \
    --edge_drop 0.15 \
    --attn_drop 0.0 \
    --diffusion_drop 0.0 \
    --transition-matrix gat \
    --epochs 2000 \
    --n-runs 10 \
    --mask-rate 0. \
    --weight-style HA


CUDA_VISIBLE_DEVICES=1 python main-c3.py --method C3  --dataset ogbn-proteins --metric rocauc --lr 0.01 --hidden_channels 64 \
    --gnn_num_layers 2  --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 \
    --batch_size 10000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 1


CUDA_VISIBLE_DEVICES=2 python main-c3.py --method C3 --dataset amazon2m --metric acc --lr 0.01 --hidden_channels 256 \
    --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
    --trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
    --use_graph --graph_weight 0.5 \
    --batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 2


CUDA_VISIBLE_DEVICES=3 python main-c3.py --method C3  --dataset pokec --rand_split --metric acc --lr 0.01 --hidden_channels 64 \
--gnn_num_layers 2 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
--trans_num_layers 1 --trans_dropout 0. --trans_weight_decay 0. --trans_use_residual --trans_use_weight --trans_use_bn \
--use_graph --graph_weight 0.5 \
--batch_size 100000 --seed 123 --runs 5 --epochs 1000 --eval_step 9 --device 3











