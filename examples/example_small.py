import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import Linformer

model = Linformer(
        input_size=512,
        channels=16,
        dim_d=32,
        dim_k=16,
        dim_ff=32,
        nhead=6,
        depth=3,
        checkpoint_level="C2",
        parameter_sharing="none",
        )
x = torch.randn(1, 512, 16)
y = model(x)
print(y) # (1, 512, 16)
