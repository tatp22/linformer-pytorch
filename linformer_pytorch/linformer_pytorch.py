import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None

def get_EF(input_size, dim):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    """
    EF = nn.Linear(input_size, dim)
    torch.nn.init.xavier_normal_(EF.weight)
    return EF

class PositionalEmbedding(nn.Module):
    """
    Standard positional embedding.
    From the paper "Attention is all you need".
    Changed the constant from 10k to 100k, since this may be better for longer sequence lengths.
    """
    def __init__(self, channels):
        super(PositionalEmbedding, self).__init__()
        inv_freq = 1. / (100000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        pos = torch.arange(tensor.shape[1], device=tensor.device).type(self.inv_freq.type())
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.cat((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return emb[None,:,:]

class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, channels, ff_dim, dropout=0.0, activation="gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        tensor = self.w_1(tensor)
        if self.activation is not None:
            tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)
        return tensor

class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(self, dim, dropout, E_proj, F_proj):
        super(LinearAttentionHead, self).__init__()
        self.w_k = nn.Linear(dim, dim)
        self.w_q = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        KW = self.w_k(K)
        KW = torch.transpose(KW, 1, 2)
        KW = self.E(KW)
        QW = self.w_q(Q)
        QW = torch.matmul(QW, KW)

        P_bar = QW/torch.sqrt(torch.tensor(self.dim).type(Q.type()))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        VW = self.w_v(V)
        VW = torch.transpose(VW, 1, 2)
        VW = self.F(VW)
        VW = torch.transpose(VW, 1, 2)
        out_tensor = torch.matmul(P_bar, VW)

        return out_tensor

class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, E_proj, F_proj):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.checkpoint_level = checkpoint_level
        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k)
            F_proj = get_EF(input_size, dim_k) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj

        for head in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k)
                F_proj = get_EF(input_size, dim_k)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj)
            self.heads.append(attn)
        self.w_o = nn.Linear(dim*nhead, channels)
        self.to_q = nn.Linear(channels, dim, bias=False)
        self.to_k = nn.Linear(channels, dim, bias=False)
        self.to_v = nn.Linear(channels, dim, bias=False)
        self.activation = get_act(activation)

    def forward(self, tensor, **kwargs):
        head_outputs = []
        for head in self.heads:
            Q = self.to_q(tensor)
            K = self.to_k(tensor)
            V = self.to_v(tensor)
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head,Q,K,V,**kwargs))
            else:
                head_outputs.append(head(Q,K,V,**kwargs))
        out = torch.cat(head_outputs, dim=2)
        if self.activation is not None:
            out = self.activation(out)
        out = self.w_o(out)
        return out

class Linformer(nn.Module):
    """
    My attempt at reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf
    """
    def __init__(self, input_size=8192, channels=128, dim_k=64, dim_ff=256, dim_d=None, dropout_ff=0.15, nhead=4, depth=1, dropout=0.1, activation="gelu", use_pos_emb=True, checkpoint_level="C0", parameter_sharing="layerwise", k_reduce_by_layer=0):
        super(Linformer, self).__init__()
        assert activation == "gelu" or activation == "relu", "Only gelu and relu activations supported for now"
        assert checkpoint_level == "C0" or checkpoint_level == "C1" or checkpoint_level == "C2", "Checkpoint level has to be either C0, C1, or C2."
        assert parameter_sharing == "none" or parameter_sharing == "headwise" or parameter_sharing == "kv" or parameter_sharing == "layerwise", "The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
        assert channels % nhead == 0 if dim_d is None else True, "If `dim_d` is not set to a custom value, `channels` must be divisible by `nhead`!"

        self.layers = nn.ModuleList()
        self.input_size = input_size
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.pos_emb = PositionalEmbedding(channels) if use_pos_emb else None
        self.depth = depth
        self.nhead = nhead

        head_dim = channels // nhead if dim_d is None else dim_d

        self.E = get_EF(input_size, dim_k)
        self.F = self.E

        get_attn = lambda curr_dim_k: MHAttention(input_size, head_dim, channels, curr_dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, self.E, self.F)
        get_ff = lambda: FeedForward(channels, dim_ff, dropout_ff)
        norm_attn = lambda: nn.LayerNorm(channels)
        norm_ff = lambda: nn.LayerNorm(channels)

        for index in range(depth):
            self.layers.append(nn.ModuleList([get_attn(max(1, dim_k - index*k_reduce_by_layer)),
                                norm_attn(),
                                get_ff(),
                                norm_ff()]))

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, channels)
        """
        bt, n, c = tensor.shape
        assert n == self.input_size, "This tensor is of the wrong size. Dimension 1 has to match the `input_size` flag"
        assert c == self.channels, "This tensor is of the wrong size. Dimension 2 has to match the `channels` flag"
        assert self.checkpoint_level == "C0" if kwargs else True, "Cannot run checkpointing when visualizing. Please set the checkpoint level to `C0`"

        if self.pos_emb is not None:
            tensor += self.pos_emb(tensor).type(tensor.type())

        for layer in self.layers:
            attn = layer[0]
            attn_norm = layer[1]
            ff = layer[2]
            ff_norm = layer[3]

            # Run attention
            before_attn_tensor = tensor.clone().detach()
            if self.checkpoint_level == "C1" or self.checkpoint_level == "C2":
                tensor = checkpoint(attn, tensor)
            else:
                tensor = attn(tensor, **kwargs)
            tensor += before_attn_tensor
            tensor = attn_norm(tensor)

            # Run ff
            before_ff_tensor = tensor.clone().detach()
            if self.checkpoint_level == "C1" or self.checkpoint_level == "C2":
                tensor = checkpoint(ff, tensor)
            else:
                tensor = ff(tensor)
            tensor += before_ff_tensor
            tensor = ff_norm(tensor)

        return tensor
