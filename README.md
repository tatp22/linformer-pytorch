# Linformer Pytorch Implementation
A practical implementation of the [Linformer paper](https://arxiv.org/pdf/2006.04768.pdf). Still a work in progress.

I am not the author of the paper.

## How to use
Assuming you have `torch` installed from `pip`, and a GPU with enough memory:

```
git clone git@github.com:tatp22/linformer-pytorch.git
cd reformer-pytorch
python example.py
```

Copy the files into your project if need be. Will look into making this easily installable via `pip`.

Code Example:

```python
from linformer_pytorch import Linformer
import torch

device = torch.device("cuda")
model = Linformer(
        input_size=16384, # Dimension 1 of the input
        channels=128, # Dimension 2 of the input
        dim_k=128, # The second dimension of the P_bar matrix from the paper
        dim_ff=128, # Dimension in the feed forward network
        dropout_ff=0.15, # Dropout for feed forward network
        nhead=4, # Number of attention heads
        depth=2, # How many times to run the model
        dropout=0.1, # How much dropout to apply to P_bar after softmax
        activation="gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
        ).cuda()
x = torch.randn(1, 16384, 128).cuda()
y = model(x)
print(y)
```

## Things left to do
* ~~Change the `einsum`s to `matmul` for faster multiplication~~
* Fix a bug where the model is using too much memory. Probably has to do with the inner dimension.
* Add option to change the `E` and `F` downsampling matrices
* Run some benchmark tests to see what the performace is

## Disclaimer
This is the first time that I am reproducing a result from a paper, so some things may be wrong. If you see a problem, please open up an issue, and I will attempt to work on it.

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
