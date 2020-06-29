import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import Linformer

model = Linformer(
        input_size=512,
        channels=16,
        dim_k=16,
        dim_ff=32,
        nhead=4,
        depth=3,
        activation="relu",
        checkpoint_level="C1",
        parameter_sharing="none",
        k_reduce_by_layer=1,
        include_ff=True,
        )
x = torch.randn(1, 512, 16)
y = model(x)
print(y) # (1, 512, 16)
