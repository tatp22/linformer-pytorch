# Linformer Pytorch Implementation
[![PyPI version](https://badge.fury.io/py/linformer-pytorch.svg)](https://badge.fury.io/py/linformer-pytorch)

![Linear Self Attention](./linformer.png)

A practical implementation of the [Linformer paper](https://arxiv.org/pdf/2006.04768.pdf). This is attention with only linear complexity in n, allowing for very long sequence lengths (1mil+) to be attended to on modern hardware.

This repo has not been empirically tested (i.e. if it performs well on any datasets), but the self attention mechanism works.

I am not the author of the paper.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zHenqau3rMo3oS_7EisfGsahSs-1_sok?usp=sharing) 1.23m tokens

## Install
```
pip install linformer-pytorch
```

Alternatively,

```
git clone https://github.com/tatp22/linformer-pytorch.git
cd linformer-pytorch
```

## Code example

```python
from linformer_pytorch import Linformer
import torch

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
        use_pos_emb=True, # Whether or not to use positional embeddings
        checkpoint_level="C0", # What checkpoint level to use. For more information, see below.
        parameter_sharing="layerwise", # What level of parameter sharing to use. For more information, see below.
        k_reduce_by_layer=0, # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
        ).cuda()
x = torch.randn(1, 262144, 64).cuda()
y = model(x)
print(y) # (1, 262144, 64)
```

## Checkpoint levels
As an attempt to further introduce memory savings, the concept of checkpoint levels have been introduced. The current three checkpoint levels are `C0`, `C1`, and `C2`. When going up checkpoint levels, one sacrifices speed for memory savings. That is, checkpoint level `C0` is the fastest, but takes up the most space on the GPU, while `C2` is the slowest, but takes up the least space on the GPU. The details of each checkpoint level are as follows:
* `C0`: No checkpointing. The models runs while keeping all of the attention heads and ff layers in the GPU memory.
* `C1`: Checkpoint each MultiHead attention as well as each ff layer. With this, increasing `depth` should have minimal impact on the memory.
* `C2`: Along with the optimizations at the `C1` level, checkpoint each head in each MultiHead Attention layer. With this, increasing `nhead` should have less of an impact on memory. However, concating the heads together with `torch.cat` still takes up a lot of memory, and this will hopefully be optimized out in the future.

Performance details are still unknown, but the option exists for users that want to try.

## Parameter Sharing
Another attempt to introduce memory savings in the paper was to introduce parameter sharing between projections. This is mentioned in section 4 of the paper; in particular, there were 4 different types of parameter sharing that the authors discussed, and all have been implemented in this repo. The first option takes up the most memory, and each further option reduces the necessary memory requirements.
* `none`: This is no parameter sharing. For every head and for every layer, a new `E` and a new `F` matrix is calculated for every head at each layer.
* `headwise`: Each layer has a unique `E` and `F` matrix. All heads in the layer share this matrix.
* `kv`: Each layer has a unique projection matrix `P`, and `E = F = P` for each layer. All heads share this projection matrix `P`.
* `layerwise`: There is one projection matrix `P`, and every head in every layer uses `E = F = P`.

As started in the paper, this means that for a 12 layer, 12 head network, there would be `288`, `24`, `12` and `1` different projection matrices, respectively.

Note that with the `k_reduce_by_layer` option, the `layerwise` option will not be effective, since it will use the dimension of `k` for the first layer. Therefore, if the value of `k_reduce_by_layer` value is greater than `0`, one should most likely not use the `layerwise` sharing option.

Also, note that according to the authors, in figure 3, this parameter sharing doesn't really affect the end result too much. So it may be best to just stick with `layerwise` sharing for everything, but the option exists for users to try it out.

## Padder
One slight problem with the current implementation of the Linformer is that your sequence length has to match the `input_size` flag of the model. The Padder pads the input size such that the tensor can be fed into the network. An example:

```python
from linformer_pytorch import Linformer, Padder
import torch

model = Linformer(
        input_size=512,
        channels=16,
        dim_d=32,
        dim_k=16,
        dim_ff=32,
        nhead=6,
        depth=3,
        checkpoint_level="C1",
        )
model = Padder(model)
x = torch.randn(1, 500, 16) # This does not match the input size!
y = model(x)
print(y) # (1, 500, 16)
```

## E and F matrices
*Please upgrade to the latest version of `linformer-pytorch`, or a version `>=0.3.1`, if you downloaded it from `pip`!* The way I calculated the E and F matrices before was that I simply used an identity matrix to downsample. However, I contacted the authors of the paper, who told me that they are actually learned parameters, and that using a `nn.Linear` layer with Xavier initialization is the way they used to compute these matrices.

## Practical Tips
* Note that the Linformer has O(nk) time and space complexity. So, while it may be linear in n, make sure that your k is not too large as well. These are editable with `input_size` and `dim_k`, respectively.
* Speaking about k, the authors found that empirical evidence supports the fact that "the performance of Linformer model is mainly determined by the projected dimension k instead of the ratio n/k". Therefore, even when increasing sequence lengths, it may be fine to keep a relatively low, constant k (the authors showed with k=256, that it still performed almost as good as a vanilla transformer).
* One more tip for k: The authors recommend that k = O(d/eps^2), if self attention wants to be approximated by full attention, with eps error.
* This code, so far, is pretty much only linear layers as well as matrix multiplications. So, libraries like `apex` should work with this, however, in practice, it has not been tested.
* In practice, I found that the memory and time requirements are more on the order of O(nkd), with n=`input_size`, k=`dim_k`, and d=`dim_d`.

## Future work
* ~~Add option to change the `E` and `F` downsampling matrices~~
* Run some benchmark tests to see what the performance is
* Instead of matrix multiplication to bring the dimensions down to k (With EKW and FVW), try to do convolution, as mentioned in the paper, with a stride length and kernel size of n/k.
* ~~In the paper, empirical studies showed that one can reduce the value of k when increasing depth, because the eigenvalues went up. Add some option to decrease k more per layers, saving even more memory.~~

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

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in neural information processing systems},
  pages={5998--6008},
  year={2017}
}
```
["Listen with attention..."](https://youtu.be/dRSOB-E0gPA?t=54)
