import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import Linformer

model = Linformer(
        input_size=510,
        channels=21,
        dim_d=26,
        dim_k=61,
        dim_ff=32,
        nhead=4,
        depth=3,
        activation="relu",
        checkpoint_level="C0",
        parameter_sharing="none",
        k_reduce_by_layer=1,
        include_ff=True,
        method="convolution",
        )
x = torch.randn(1, 510, 21)
y = model(x)
print(y) # (1, 510, 21)
