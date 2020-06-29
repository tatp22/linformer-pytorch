import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import Linformer, Visualizer

model = Linformer(
        input_size=512,
        channels=16,
        dim_k=128,
        dim_ff=32,
        nhead=4,
        depth=3,
        activation="relu",
        checkpoint_level="C0",
        parameter_sharing="layerwise",
        k_reduce_by_layer=1,
        )
x = torch.randn(1, 512, 16)
y = model(x, visualize=True)
vis = Visualizer(model)
vis.plot_all_heads(title="All P_bar matrices",
                   show=True,
                   save_file=None,
                   figsize=(8,6),
                   n_limit=256)
print(y) # (1, 512, 16)
