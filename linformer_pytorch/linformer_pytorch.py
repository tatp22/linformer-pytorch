import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

def identity(x, *args, **kwargs):
    return x

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    if activation == "relu":
        return F.relu
    return None

def get_linear(input_size, dim, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    """
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

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

class ProjectInOut(nn.Module):
    """
    Impelemenation taken from https://github.com/lucidrains/sinkhorn-transformer/blob/73da02958965e1a690cb301292c0a3c549687d44/sinkhorn_transformer/sinkhorn_transformer.py#L218
    """
    def __init__(self, fn, dim_in, dim_out, project_out=True):
        super(ProjectInOut, self).__init__()
        self.fn = fn
        self.project_in = nn.Linear(dim_in, dim_out)
        self.project_out = nn.Linear(dim_out, dim_in) if project_out else identity

    def forward(self, tensor, **kwargs):
        tensor = self.project_in(tensor)
        tensor = self.fn(tensor, **kwargs)
        tensor = self.project_out(tensor)
        return tensor

class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, channels, ff_dim, dropout, activation="gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = get_linear(channels, ff_dim)
        self.w_2 = get_linear(ff_dim, channels)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
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
    def __init__(self, dim, dropout, E_proj, F_proj, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        K = torch.transpose(K, 1, 2)
        if not self.full_attention:
            K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type()))
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = torch.transpose(V, 1, 2)
            V = self.F(V)
            V = torch.transpose(V, 1, 2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, E_proj, F_proj, full_attention, w_o_intermediate_dim=None):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim
        if parameter_sharing != "layerwise":
            E_proj = get_linear(input_size, dim_k)
            F_proj = get_linear(input_size, dim_k) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj

        self.get_linear = lambda: get_linear(channels, dim, bias=False)
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for head in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_linear(input_size, dim_k)
                F_proj = get_linear(input_size, dim_k)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, full_attention)
            self.heads.append(attn)
            self.to_q.append(self.get_linear())
            self.to_k.append(self.get_linear())
            self.to_v.append(self.get_linear())
        if w_o_intermediate_dim is None:
            self.w_o = get_linear(dim*nhead, channels)
        else:
            self.w_o_1 = get_linear(dim*nhead, w_o_intermediate_dim)
            self.w_o_2 = get_linear(w_o_intermediate_dim, channels)
        self.activation = get_act(activation)

    def forward(self, tensor, **kwargs):
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor)
            V = self.to_v[index](tensor)
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head,Q,K,V))
            else:
                head_outputs.append(head(Q,K,V,**kwargs))
        out = torch.cat(head_outputs, dim=2)
        if self.activation is not None:
            out = self.activation(out)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        return out

class Linformer(nn.Module):
    """
    My attempt at reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf
    """
    def __init__(self, input_size, channels, dim_k, dim_ff=256, dim_d=None, dropout_ff=0.15, nhead=4, depth=1, dropout=0.1, activation="gelu", use_pos_emb=True, checkpoint_level="C0", parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False, include_ff=True, w_o_intermediate_dim=None):
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

        self.E = get_linear(input_size, dim_k)
        self.F = self.E

        get_attn = lambda curr_dim_k: MHAttention(input_size, head_dim, channels, curr_dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, self.E, self.F, full_attention, w_o_intermediate_dim)
        get_ff = lambda: FeedForward(channels, dim_ff, dropout_ff)
        get_norm = lambda: nn.LayerNorm(channels)

        for index in range(depth):
            self.layers.append(nn.ModuleList([get_attn(max(1, dim_k - index*k_reduce_by_layer)), get_norm()]))
            if include_ff:
                self.layers.append(nn.ModuleList([get_ff(), get_norm()]))

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
            module = layer[0]
            norm = layer[1]

            before_module_tensor = tensor.clone().detach()
            if self.checkpoint_level == "C1" or self.checkpoint_level == "C2":
                tensor = checkpoint(module, tensor)
            else:
                tensor = module(tensor, **kwargs)
            tensor += before_module_tensor
            tensor = norm(tensor)

        return tensor

class LinformerLM(nn.Module):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    """
    def __init__(self, num_tokens, input_size, channels, dim_k=64, dim_ff=1024, dim_d=None, dropout_ff=0.1, nhead=4, depth=2, dropout=0.05, activation="gelu", checkpoint_level="C0", parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False, include_ff=True, w_o_intermediate_dim=None, emb_dim=None):
        super(LinformerLM, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = PositionalEmbedding(emb_dim)
        self.linformer = Linformer(input_size, channels, dim_k=dim_k, dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff, nhead=nhead, depth=depth, dropout=dropout, activation=activation, use_pos_emb=False, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing, k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff, w_o_intermediate_dim=w_o_intermediate_dim)

        if emb_dim != channels:
            self.linformer = ProjectInOut(self.linformer, emb_dim, channels)

        self.to_logits = get_linear(emb_dim, num_tokens)

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.linformer(tensor, **kwargs)
        tensor = self.to_logits(tensor)
        return tensor
