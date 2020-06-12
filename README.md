# Linformer Pytorch Implementation
A practical implementation of the [Linformer paper](https://arxiv.org/pdf/2006.04768.pdf).
Has not been emperically tested (i.e. if it performs well on any datasets), but the self attention mechanism works.
I am not the author of the paper.

## How to use
Assuming you have `torch` installed from `pip`, and a GPU with enough memory:

```
git clone git@github.com:tatp22/linformer-pytorch.git
cd linformer-pytorch
python example.py
```

Copy the files into your project if need be. Will look into making this easily installable via `pip`.

Code Example:

```python
from linformer_pytorch import Linformer
import torch

device = torch.device("cuda")
model = Linformer(
        input_size=262144, # Dimension 1 of the input
        channels=64, # Dimension 2 of the input
        dim_d=256, # The inner dimension of the attention heads
        dim_k=128, # The second dimension of the P_bar matrix from the paper
        dim_ff=128, # Dimension in the feed forward network
        dropout_ff=0.15, # Dropout for feed forward network
        nhead=4, # Number of attention heads
        depth=2, # How many times to run the model
        dropout=0.1, # How much dropout to apply to P_bar after softmax
        activation="gelu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
        ).cuda()
x = torch.randn(1, 262144, 64).cuda()
y = model(x)
print(y)
```

## Practical Tips
* Note that the Linformer has O(nk) time and space complexity. So, while it may be linear in n, make sure that your k is not too large as well. These are editable with `input_size` and `dim_k`, respectively.
* Speaking about k, the authors found that emperical evidence supports the fact that "the performance of Linformer model is mainly determined bythe projected dimension k instead of the ratio n/k". Therefore, even when increasing sequence lengths, it may be fine to keep a relatively low, constant k (the authors showed with k=256, that it still performed almost as good as a vanilla transformer).
* One more tip for k: The authors recommend that k = O(d/eps^2), if self attention wants to be approximated by full attention, with eps error.
* This code, so far, is pretty much only linear layers as well as matrix multiplications. So, libraries like `apex` should work with this, however, in practice, it has not been tested.
* In practice, I found that the memory and time requirements are more on the order of O(nkd), with n=`input_size`, k=`dim_k`, and d=`dim_d`.

## Future work
* ~~Change the `einsum`s to `matmul` for faster multiplication~~
* ~~Fix a bug where the model is using too much memory. Probably has to do with the inner dimension.~~
* Add option to change the `E` and `F` downsampling matrices
* Run some benchmark tests to see what the performace is
* Instead of matrix multiplication to bring the dimensions down to k (With EKW and FVW), try to do convolution, as mentioned in the paper, with a stride length and kernel size of n/k.
* In the paper, emperical studies showed that one can reduce the value of k when increasing depth. Add some option to decrease k more per layers, saving even more memory.

## Disclaimer
This is the first time that I am reproducing a result from a paper, so some things may be wrong. If you see a problem, please open up an issue, and I will attempt to work on it.

## Thanks
Thank you to [lucidrains](https://github.com/lucidrains), whose other sparse attention repositories helped me in designing this Linformer Repo.

## Citations

```bibtex
@misc{wang2020linformer,
    title={Linformer: Self-Attention with Linear Complexity},
    author={Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
    year={2020},
    eprint={2006.04768},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
["Listen with attention..."](https://www.youtube.com/watch?v=ZKirRqHtuBU)
