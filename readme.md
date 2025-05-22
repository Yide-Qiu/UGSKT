

# 🌐 Universal Graph Structural Knowledge Transfer (UGSKT)

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-%23EE4C2C.svg)](https://pytorch.org/)
[![OGB](https://img.shields.io/badge/OGB-1.3.1-%23007CBC.svg)](https://ogb.stanford.edu/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official implementation of **"One for All: A Motif-Driven Graph Dual-Serialization Transfer Framework"**  
📄 Paper (Coming Soon) | 💻 Interactive Demo (Coming Soon)

# 🔥 Key Features

- ​​First universal framework​​ for cross-domain graph knowledge transfer using topological primitives
​​- STA-18 Benchmark​​ - Largest aligned topological-textual graph dataset (18 diverse graphs)
​​- 5.2% avg. performance gain​​ across 13 downstream tasks
​​- Parameter-efficient transfer​​ with <9% additional parameters
​​- Dual-stream ODE Network​​ with theoretical convergence guarantees


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

