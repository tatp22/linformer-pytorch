from linformer_pytorch import Linformer
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
x = torch.randn(1, 512, 16)
y = model(x)
print(y)

# To see memory usage, uncomment the line below.
#print('Allocated Memory:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
