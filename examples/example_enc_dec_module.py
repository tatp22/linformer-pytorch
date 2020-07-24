import sys
import torch

sys.path.insert(0, "../")
from linformer_pytorch import LinformerEncDec

encdec = LinformerEncDec(
    enc_num_tokens=10000,
    enc_input_size=512,
    enc_channels=16,
    dec_num_tokens=10000,
    dec_input_size=512,
    dec_channels=16,
)

x = torch.randint(1,10000,(1,512))
y = torch.randint(1,10000,(1,512))

output = encdec(x,y)
print(output.shape) # (1, 512, 10000)
