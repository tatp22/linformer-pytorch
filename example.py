from linformer_pytorch import Linformer
import torch

device = torch.device("cuda")
model = Linformer(
        input_size=16384,
        channels=128,
        dim_k=128,
        dim_ff=128,
        ).cuda()
x = torch.randn(1, 16384, 128).cuda()
y = model(x)
print(y)

# To see memory usage, uncomment the line below.
#print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
