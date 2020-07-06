import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import LinformerLM

encoder = LinformerLM(
    num_tokens=10000,
    input_size=512,
    channels=16,
    dim_k=16,
    dim_ff=32,
    nhead=4,
    depth=3,
    activation="relu",
    checkpoint_level="C0",
    parameter_sharing="none",
    k_reduce_by_layer=1,
    include_ff=True,
    return_emb=True,
    )
decoder = LinformerLM(
    num_tokens=10000,
    input_size=512,
    channels=16,
    dim_k=16,
    dim_ff=32,
    nhead=4,
    depth=3,
    activation="relu",
    checkpoint_level="C0",
    parameter_sharing="none",
    k_reduce_by_layer=1,
    include_ff=True,
    decoder_mode=True,
    )
x = torch.randint(1,10000,(1,512))
y = torch.randint(1,10000,(1,512))
enc_output = encoder(x)
print(enc_output.shape) # (1, 512, 128)
dec_output = decoder(y, embeddings=enc_output)
print(dec_output.shape) # (1, 512, 10000)
