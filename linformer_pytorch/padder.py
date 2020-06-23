import torch
import torch.nn as nn

from linformer_pytorch import Linformer

class Padder(nn.Module):
    """
    A padder for the Linformer. Currently just pads the input to the Linformer's `input_size` parameter if it is smaller.
    """
    def __init__(self, net):
        super(Padder, self).__init__()
        assert isinstance(net, Linformer), "Only the Linformer is supported"
        self.net = net

    def forward(self, tensor, **kwargs):
        batch_size, seq_len, ch = tensor.shape
        padding_amount = self.net.input_size - seq_len
        net_tensor = torch.zeros((batch_size, seq_len+padding_amount, ch), device=tensor.device)
        net_tensor[:,:seq_len,:] = tensor
        net_tensor = self.net(net_tensor, **kwargs)
        tensor = net_tensor[:,:seq_len,:]
        return tensor
