

# 🌐 Universal Graph Structural Knowledge Transfer (UGSKT)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-%23EE4C2C.svg)](https://pytorch.org/)
[![OGB](https://img.shields.io/badge/OGB-1.3.1-%23007CBC.svg)](https://ogb.stanford.edu/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **"One for All: A Motif-Driven Graph Dual-Serialization Transfer Framework"**  
📄 Paper (Coming Soon) | 💻 Interactive Demo (Coming Soon)

# 🔥 Key Features

- ​​First universal framework​​ for cross-domain graph knowledge transfer using topological primitives
- STA-18 Benchmark​​ - Largest aligned topological-textual graph dataset (18 diverse graphs)
- 5.2% avg. performance gain​​ across 13 downstream tasks
- Parameter-efficient transfer​​ with <9% additional parameters
- Dual-stream ODE Network​​ with theoretical convergence guarantees
  
# 🚀 Quick Start

## Installation
```
conda create -n ugskt python=3.8
conda activate ugskt
pip install -r requirements.txt
```

## Requirements
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

## Datasets Overall
We have constructed motif scores and generated structured tokens for **18 datasets**: 

- **Small-scale Graph Datasets**:
  - Pubmed
  - Citeseer
  - Cora
- **OGB Node-level Datasets**:
  - ogbn-products
  - ogbn-arxiv
  - ogbn-proteins
  - ogbn-mag
- **OGB Edge-level Datasets**:
  - ogbl-ppa
  - ogbl-ddi
  - ogbl-citation2
- **OGB Graph-level Datasets**:
  - ogbg-ppa
  - ogbg-molpcba
  - ogbg-code2
- **Graph Structure Learning Datasets**:
  - Peptides
  - PascalVOC-SP
  - COCO-SP
  - MalNet-Tiny
- **The Largest Universal Heterogeneous Graph Dataset**:
  - UniKG


## Dataset Statistics

| Dataset               | #Nodes       | #Tokens          | Avg.Length | Disk Size | Generation Time | Task Type          | Metric     | Domain          |
|-----------------------|--------------|------------------|------------|-----------|-----------------|--------------------|------------|-----------------|
| Cora                  | 2,708        | 342,249         | 126.4      | 0.11 MB   | 7.07 s          | Node Classification| Accuracy   | Citation        |
| Citeseer              | 3,327        | 420,580         | 126.4      | 0.13 MB   | 4.65 s          | Node Classification| Accuracy   | Citation        |
| Pubmed                | 19,717       | 2,495,065       | 126.5      | 0.75 MB   | 108.94 s        | Node Classification| Accuracy   | Citation        |
| ogbn-proteins         | 132,534      | 17,028,774      | 127.7      | 5.06 MB   | 6.83 h          | Node Classification| ROCAUC     | Biology         |
| ogbn-arxiv            | 169,343      | 21,338,818      | 126.0      | 6.46 MB   | 10.26 min       | Node Classification| Accuracy   | Citation        |
| ogbn-mag              | 1,939,743    | 242,542,736     | 125.0      | 74.00 MB  | 4.56 h          | Node Classification| Accuracy   | Citation        |
| ogbn-products         | 2,449,029    | 312,639,453     | 127.7      | 93.42 MB  | 113.14 h        | Node Classification| Accuracy   | Product         |
| ogbl-ddi              | 4,267        | 549,291         | 128.7      | 0.16 MB   | 14.09 min       | Link Prediction    | Hits@30    | Biology         |
| ogbl-ppa              | 576,289      | 73,407,315      | 127.4      | 21.98 MB  | 3.01 h          | Link Prediction    | Hits@100   | Biology         |
| ogbl-citation2        | 2,927,963    | 36,996,424      | 126.4      | 111.69 MB | 12.81 h         | Link Prediction    | MRR        | Citation        |
| ogbg-molhiv           | 1,048,738    | 130,250,852     | 124.1      | 40.02 MB  | 6.64 min        | Graph Classification| ROCAUC     | Biology         |
| ogbg-molpcba          | 11,386,154   | 1,528,535,562   | 134.4      | 433.85 MB | 1.17 h          | Graph Classification| AP         | Biology         |
| ogbg-code2            | 56,683,173   | 7,528,675,744   | 132.8      | 2.11 GB   | 2.84 h          | Graph Classification| F1 score   | Code            |
| Peptides              | 2,344,231    | 299,346,341     | 127.7      | 89.45 MB  | 15.52 min       | Graph Cls & Reg    | AP & MAE   | Biology         |
| PascalVOC-SP          | 5,443,587    | 714,198,614     | 131.2      | 213.60 MB | 55.2 min        | Node Classification| F1 score   | Computer Vision |
| COCO-SP               | 58,795,093   | 7,619,844,052   | 129.6      | 2.22 GB   | 6.58 h          | Node Classification| F1 score   | Computer Vision |
| MalNet-Tiny           | 7,051,500    | 897,655,950     | 127.3      | 268.23 MB | 46.2 min        | Graph Classification| Accuracy   | Cybersecurity   |
| UniKG                 | 77,312,474   | 10,274,827,794  | 132.9      | 6.08 GB   | 214.22 h        | Node Classification| Accuracy   | Universal       |


## Download Preprocessed Datasets
```
https://pan.quark.cn/s/a68121ffd314
```

# 🏆 Benchmark Performance

| Dataset         | Task Type            | Base Model | Frozen Transfer | Full Fine-tune |
|-----------------|----------------------|------------|------------------|----------------|
| ogbn-arxiv      | Node Classification | 71.3%      | 73.8% (+3.5%)    | 75.1% (+5.2%)  |
| ogbl-ppa        | Link Prediction     | 0.776      | 0.891 (+14.8%)   | 0.917 (+18.2%) |
| ogbg-molpcba    | Graph Classification | 0.274 AP   | 0.281 AP (+2.5%) | 0.289 AP (+5.4%) |
| PascalVOC-SP    | Node Classification | 0.287 F1   | 0.368 F1 (+28.2%)| 0.420 F1 (+46.3%) |


# 🧠 Framework Architecture

## Dual-Serialization Process Pipeline
![Alt](./pipelinev4.png)

## Structural Annotation
![Alt](./annotationv2.png)

## Adaptive Transfer Mechanism
![Alt](./transfermodelv3.png)

# 📚 Citation

```
@article{oneforall2025,
  title={One for All: A Motif-Driven Graph Dual-Serialization Transfer Framework},
  author={Anonymous Authors},
  journal={Under Review},
  year={2025}
}
```

