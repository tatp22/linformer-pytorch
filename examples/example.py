import torch
import sys

sys.path.insert(0, "../")
from linformer_pytorch import Linformer

model = Linformer(
        input_size=262144,
        channels=64,
        dim_d=256,
        dim_k=64,
        dim_ff=128,
        ).cuda()
x = torch.randn(1, 262144, 64).cuda()
y = model(x)
print(y) # (1, 262144, 64)

# To see memory usage, uncomment the line below.
#print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
