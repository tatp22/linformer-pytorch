# Linformer Pytorch Implementation

A practical implementation of the [Linformer paper](https://arxiv.org/pdf/2006.04768.pdf).

## How to use
Simply pull the repo and install the necessary dependencies.

Code Example:

```python
from linformer_pytorch import Linformer

model = Linformer(...)
x = torch.randn(1, 8192, 1024)
model(x) #(1, 8192, 1024)
```

## Citations

```
@misc{wang2020linformer,
    title={Linformer: Self-Attention with Linear Complexity},
    author={Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
    year={2020},
    eprint={2006.04768},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
