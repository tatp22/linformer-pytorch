import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import Linformer, LinformerLM, Padder

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
print(y, y.shape) # (1, 500, 16)

model_lm = LinformerLM(
        num_tokens=10000,
        input_size=512,
        channels=16,
        dim_d=32,
        dim_k=16,
        dim_ff=32,
        nhead=6,
        depth=3,
        checkpoint_level="C1",
        )
model_lm = Padder(model_lm)
x = torch.randint(0, 10000, (1,510))
y = model_lm(x)
print(y, y.shape) # (1, 510, 10000)
