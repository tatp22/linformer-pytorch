import torch
import torch.nn as nn

from linformer_pytorch import Linformer, LinformerLM

class Padder(nn.Module):
    """
    A padder for the Linformer. Currently just pads the input to the Linformer's `input_size` parameter if it is smaller.
    """
    def __init__(self, net):
        super(Padder, self).__init__()
        assert isinstance(net, (Linformer, LinformerLM)), "Only the Linformer and LinformerLM are supported"
        self.net = net

    def forward(self, tensor, **kwargs):
        batch_size, seq_len = tensor.shape[:2]
        padding_amount = self.net.input_size - seq_len
        if isinstance(self.net, Linformer):
            net_tensor = torch.zeros((batch_size, seq_len+padding_amount, tensor.shape[-1]), device=tensor.device)
        else:
            net_tensor = torch.zeros((batch_size, seq_len+padding_amount), device=tensor.device).type(tensor.type())
        net_tensor[:,:seq_len] = tensor
        net_tensor = self.net(net_tensor, **kwargs)
        tensor = net_tensor[:,:seq_len]
        return tensor
