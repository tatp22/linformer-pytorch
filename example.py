from linformer_pytorch import Linformer
import torch

device = torch.device("cuda")
model = Linformer(
        input_size=1024,
        channels=128,
        ).cuda()
x = torch.randn(1, 1024, 128).cuda()
y = model(x)
print(y)
