# Linformer Pytorch Implementation

A practical implementation of the [Linformer paper](https://arxiv.org/pdf/2006.04768.pdf).

I am not the author of the paper.

## How to use
Simply pull the repo and install the necessary dependencies.

Code Example:

```python
from linformer_pytorch import Linformer
import torch

device = torch.device("cuda")
model = Linformer(
        input_size=16384,
        channels=128,
        dim_ff=128,
        ).cuda()
x = torch.randn(1, 16384, 128).cuda()
y = model(x)
print(y)
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
