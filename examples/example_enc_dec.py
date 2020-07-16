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
    k_reduce_by_layer=1,
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
    decoder_mode=True,
    )
x = torch.randint(1,10000,(1,512))
y = torch.randint(1,10000,(1,512))

x_mask = torch.ones_like(x).bool()
y_mask = torch.ones_like(y).bool()

enc_output = encoder(x, input_mask=x_mask)
print(enc_output.shape) # (1, 512, 16)
dec_output = decoder(y, embeddings=enc_output, input_mask=y_mask, embeddings_mask=x_mask)
print(dec_output.shape) # (1, 512, 10000)
