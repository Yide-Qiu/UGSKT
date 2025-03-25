
# UNIVERSAL GRAPH STRUCTURAL KNOWLEDGE TRANSFER.

This is the official implementation of the paper **One for All: A Motif-Driven Graph Dual-Serialization Transfer Framework**.

## Requirements:

```
torch                         1.9.0
numpy                         1.19.2
optuna                        3.5.0
scikit-learn                  1.1.3
scipy                         1.6.2
torch-cluster                 1.6.1
torch-geometric               1.7.2
torch-scatter                 2.0.7
torch-sparse                  0.6.10
torch-spline-conv             1.2.2
ogb                           1.3.1

```

## Datasets:
We have constructed motif scores and generated structured tokens for 14 datasets: 
```
**Pubmed**,**Citeseer**,**Cora**
**ogbn-products**,**ogbn-arxiv**,**ogbn-proteins**,**ogbn-mag**
**ogbl-ppa**,**ogbl-ddi**,**ogbl-citation2**
**ogbg-ppa**,**ogbg-molpcba**,**ogbg-code2**
**UniKG**
```
Please click on the following link to download all datasets:
```
https://pan.quark.cn/s/a68121ffd314
```

## Pipeline
The overall task pipeline is shown below:

![Alt](./pipelinev4.png)


## Annotation
The overall framework of annotation is shown below:

![Alt](./annotationv2.png)

## Transfermodel
The overall framework of universal structure knowledge transfer is shown below:

![Alt](./transfermodelv3.png)

