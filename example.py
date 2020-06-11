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
