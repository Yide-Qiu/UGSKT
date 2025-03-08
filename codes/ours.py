import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from mamba_ssm import Mamba
import pdb
import copy


class MambaConvBlock(nn.Module):
    def __init__(self, in_hidden, num_hops, out_hidden, d_state=16, d_conv=8, expand=2):
        super(MambaConvBlock, self).__init__()
        self.in_hidden = in_hidden
        self.num_hops = num_hops
        self.out_hidden = out_hidden

        self.convs = nn.ModuleList()
        for i in range(self.num_hops + 1):
            self.convs.append(Mamba(d_model=in_hidden, d_state=d_state, d_conv=d_conv, expand=expand))


    def forward(self, h_list):
        for i in range(self.num_hops + 1):
            h_list[i] = self.convs[i](h_list[i].view(1,-1,h_list[i].shape[1])) 
        return h_list



class AttBlock(nn.Module):
    def __init__(self, n_hidden, num_hops, n_classes, n_layers_2, num_heads, weight_style, HA_activation, activation,
                 residual, zero_inits, position_emb, batch_norm, feat_drop, attn_drop, edge_drop, diffusion_drop, bias,
                 propagate_first=False, negative_slope=0.2,):
        super(AttBlock, self).__init__()
        self.n_hidden = n_hidden
        self.num_hops = num_hops
        self.num_heads = num_heads
        self.n_classes = n_classes
        self.n_layers = n_layers_2
        self.weight_style = weight_style
        self.HA_activation = HA_activation
        self.residual = residual
        self.zero_inits = zero_inits
        self.batch_norm = batch_norm
        self.position_emb = position_emb
        
        # self.fcs = nn.ModuleList()
        # self.bns1 = nn.ModuleList()
        # for i in range(self.n_layers):
        #     in_hidden = n_hidden 
        #     out_hidden = n_hidden if i < self.n_layers - 1 else n_classes
        #     self.fcs.append(nn.Linear(in_hidden, out_hidden, bias=False))
        #     if i < self.n_layers_1 - 1:
        #         self.bns1.append(nn.BatchNorm1d(out_hidden))

        if propagate_first:
            propagate_feats = n_hidden
        else:
            propagate_feats = n_classes

        self.fc = nn.Linear(self.n_hidden, self.n_classes * num_heads, bias=False)
        if position_emb:
            self.fc_position_emb = nn.Parameter(torch.FloatTensor(size=(num_hops+1, num_heads, propagate_feats)))
        if weight_style in ["HA", "HA+HC"]:
            self.hop_attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
            self.hop_attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, propagate_feats)))
        if weight_style in ["HC", "HA+HC"]:
            self.weights = nn.Parameter(torch.FloatTensor(size=(1, num_heads, num_hops+1, propagate_feats)))

        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(num_hops + 1):
                self.offset.append(nn.Parameter(torch.zeros(size=(1, num_heads, n_classes))))
                self.scale.append(nn.Parameter(torch.ones(size=(1, num_heads, n_classes))))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.diffusion_drop = diffusion_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self.n_hidden, self.n_classes * num_heads, bias=False)
        else:
            self.register_buffer("res_fc", None)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size=(1, num_heads, self.n_classes)))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()
        self._activation = activation
    

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        if self.position_emb:
            nn.init.xavier_normal_(self.fc_position_emb)
        if self.weight_style in ["HA", "HA+HC"]:
            if self.zero_inits:
                nn.init.zeros_(self.hop_attn_l)
                nn.init.zeros_(self.hop_attn_r)
            else:
                nn.init.xavier_normal_(self.hop_attn_l, gain=gain)
                nn.init.xavier_normal_(self.hop_attn_r, gain=gain)
            # nn.init.xavier_normal_(self.hop_attn_bias_l, gain=gain)
            # nn.init.xavier_normal_(self.hop_attn_bias_r, gain=gain)
            # nn.init.uniform_(self.beta)
        if self.weight_style in ["HC", "HA+HC"]:
            nn.init.xavier_uniform_(self.weights, gain=gain)
            # nn.init.ones_(self.weights)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        if isinstance(self.bias, nn.Parameter):
            nn.init.zeros_(self.bias)


    def feat_trans(self, h, idx):
        if self.batch_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self.num_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self.num_heads, 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]
        if self.position_emb:
            h = h + self.fc_position_emb[[idx], :, :]
        return h


    def forword(self, h_list):
        hstack = h_list
        feat = h_list[0]
        hstack = [self.feat_trans(h, k) for k, h in enumerate(hstack)]
        hop_a = None
        if self.weight_style in ["HA", "HA+HC"]:
            hop_a_l = (hstack[0] * self.hop_attn_l).sum(dim=-1).unsqueeze(-1)
            hop_astack_r = [(feat_dst * self.hop_attn_r).sum(dim=-1).unsqueeze(-1) for feat_dst in hstack]
            hop_a = torch.cat([(a_r + hop_a_l) for a_r in hop_astack_r], dim=-1)
            if self.HA_activation == "sigmoid":
                hop_a = torch.sigmoid(hop_a)
            if self.HA_activation == "leakyrelu":
                hop_a = self.leaky_relu(hop_a)
            if self.HA_activation == "relu":
                hop_a = F.relu(hop_a)
            if self.HA_activation == "standardize":
                hop_a = (hop_a - hop_a.min(dim=2, keepdim=True)[0]) / (hop_a.max(dim=2, keepdim=True)[0] - hop_a.min(dim=2, keepdim=True)[0]).clamp(min=1e-9)

            hop_a = F.softmax(hop_a, dim=-1)
            # hop_a = self.attn_drop(hop_a)
            if not self.training:
                self.hop_a = hop_a
                
            rst = 0
            for i in range(hop_a.shape[2]):
                    
                if self.weight_style == "HA+HC":
                    rst += hstack[i] * hop_a[:, :, [i]] * self.weights[:, :, i, :]
                else:
                    rst += hstack[i] * hop_a[:, :, [i]]

        if self.weight_style == "HC":
            rst = 0
            for i in range(len(hstack)):
                rst += hstack[i] * self.weights[:, :, i, :]
        if self.weight_style == "mean":
            rst = 0
            for i in range(len(hstack)):
                rst += hstack[i] / len(hstack)

        # if self.propagate_first:
        #     rst = self.fc(rst)
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(feat)
            rst = rst + resval
        # bias
        if self.bias is not None:
            rst = rst + self.bias
        # activation
        if self._activation is not None:
            rst = self._activation(rst)
        return rst



class C3(nn.Module):
    def __init__(self, args, in_feats, n_hidden, n_classes, activation=F.relu):
        super(C3, self).__init__()

        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers_1 = args.n_layers_1
        self.n_layers_2 = args.n_layers_2
        self.num_hops = args.num_hops
        self.residual = args.residual
        # self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.bns1 = nn.ModuleList()
        for i in range(self.n_layers_1):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden
            self.convs.append(
                MambaConvBlock(
                    in_hidden, 
                    self.num_hops, 
                    out_hidden
                )
            )
            if i < self.n_layers_1 - 1:
                self.bns1.append(nn.BatchNorm1d(out_hidden))


        self.att = AttBlock(n_hidden, self.num_hops, n_classes, self.n_layers_2, args.num_heads, args.weight_style, args.HA_activation, 
                            activation, args.residual, args.zero_inits, args.position_emb, args.batch_norm, args.feat_drop, 
                            args.attn_drop, args.edge_drop, args.diffusion_drop, args.bias, args.propagate_first, args.negative_slope)

        self.input_dropout = nn.Dropout(args.input_drop)
        self.dropout = nn.Dropout(args.dropout)
        self.activation = activation


    def forward(self, x_list):
        h = x_list
        for k in range(len(h)):
            h[k] = self.input_dropout(h[k])
        h_last = h
        for i in range(self.n_layers_1):
            conv = self.convs[i](h)
            h = conv
            if i < self.n_layers_1 - 1:
                for k in range(len(h)):
                    if self.residual:
                        h[k] = h[k] + h_last[k]
                    h[k] = self.bns1[i](h[k])
                    h[k] = self.activation(h[k], inplace=True)
                    h[k] = self.dropout(h[k])
                h_last = h
        h = self.att(h)
        return h
  

    def reset_parameters(self):
        self.att.reset_parameters()
        # self.trans_conv.reset_parameters()
        # if self.use_graph:
        #     self.decp_conv.reset_parameters()













