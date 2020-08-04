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

def gen_causal_mask(input_size, dim_k, full_attention=False):
    """
    Generates a causal mask of size (input_size, dim_k) for linformer
    Else, it generates (input_size, input_size) for full attention
    """
    if full_attention:
        return (torch.triu(torch.ones(input_size, input_size))==1).transpose(0,1)
    return (torch.triu(torch.ones(dim_k, input_size))==1).transpose(0,1)

def get_EF(input_size, dim, convolution=False, head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Also, includes an option for convolution, as proposed by the authors.
    """
    if convolution:
        conv = nn.Conv1d(head_dim, head_dim, kernel_size=int(input_size/dim), stride=int(input_size/dim))
        return conv
    lin = nn.Linear(input_size, dim, bias)
    torch.nn.init.xavier_normal_(lin.weight)
    return lin

class Residual(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    """
    def __init__(self, fn, input_channels=0, output_channels=0):
        super(Residual, self).__init__()
        self.fn = fn
        self.resample = nn.Linear(input_channels, output_channels) if input_channels != output_channels else None

    def forward(self, tensor, **kwargs):
        if self.resample is not None:
            return self.resample(tensor) + self.fn(tensor, **kwargs)
        return tensor + self.fn(tensor, **kwargs)

class PreNorm(nn.Module):
    """
    Implemenation taken from
    https://github.com/lucidrains/sinkhorn-transformer/blob/master/sinkhorn_transformer/sinkhorn_transformer.py
    """
    def __init__(self, channels, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(channels)

    def forward(self, tensor, **kwargs):
        tensor = self.norm(tensor)
        return self.fn(tensor, **kwargs)

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
    def __init__(self, input_channels, output_channels, ff_dim, dropout, activation="gelu"):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(input_channels, ff_dim)
        self.w_2 = nn.Linear(ff_dim, output_channels)
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
    def __init__(self, dim, dropout, E_proj, F_proj, causal_mask, full_attention=False):
        super(LinearAttentionHead, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.causal_mask = causal_mask

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:,:,None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:,:,None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1,2)
        if not self.full_attention:
            K = self.E(K)
        Q = torch.matmul(Q, K)

        P_bar = Q/torch.sqrt(torch.tensor(self.dim).type(Q.type()))
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, -1e10)
        P_bar = P_bar.softmax(dim=-1)

        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = V.transpose(1,2)
            V = self.F(V)
            V = V.transpose(1,2)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, activation, checkpoint_level,
                parameter_sharing, E_proj, F_proj, full_attention, causal_mask, w_o_intermediate_dim=None, decoder_mode=False, convolution=False):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()
        self.input_size = input_size
        self.dim_k = dim_k
        self.channels = channels
        self.causal_mask = causal_mask
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim
        if parameter_sharing != "layerwise":
            E_proj = get_EF(input_size, dim_k, convolution, dim)
            F_proj = get_EF(input_size, dim_k, convolution, dim) if parameter_sharing == "none" or parameter_sharing == "headwise" else E_proj

        self.decoder_mode = decoder_mode
        self.to_q = nn.ModuleList()
        self.to_k = nn.ModuleList()
        self.to_v = nn.ModuleList()

        for _ in range(nhead):
            if parameter_sharing == "none":
                E_proj = get_EF(input_size, dim_k, convolution, dim)
                F_proj = get_EF(input_size, dim_k, convolution, dim)
            attn = LinearAttentionHead(dim, dropout, E_proj, F_proj, causal_mask, full_attention)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias=False))
            self.to_k.append(nn.Linear(channels, dim, bias=False))
            self.to_v.append(nn.Linear(channels, dim, bias=False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim*nhead, channels)
        else:
            self.w_o_1 = nn.Linear(dim*nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.activation = get_act(activation)

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape
        assert not (self.decoder_mode and "embeddings" not in kwargs), "Embeddings must be supplied if decoding"
        assert not ("embeddings" in kwargs and (kwargs["embeddings"].shape[0], kwargs["embeddings"].shape[1], kwargs["embeddings"].shape[2]) != (batch_size, input_len, channels)), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if not self.decoder_mode else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if not self.decoder_mode else self.to_v[index](kwargs["embeddings"])
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head,Q,K,V))
            else:
                head_outputs.append(head(Q,K,V,**kwargs))
        out = torch.cat(head_outputs, dim=-1)
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
    def __init__(self, input_size, channels, dim_k, dim_ff=256, dim_d=None, dropout_ff=0.15, nhead=4, depth=1, dropout=0.1, activation="gelu", checkpoint_level="C0", parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False, include_ff=True, w_o_intermediate_dim=None, decoder_mode=False, causal=False, convolution=False, ff_intermediate=None):
        super(Linformer, self).__init__()
        assert activation == "gelu" or activation == "relu", "Only gelu and relu activations supported for now"
        assert checkpoint_level == "C0" or checkpoint_level == "C1" or checkpoint_level == "C2", "Checkpoint level has to be either C0, C1, or C2."
        assert parameter_sharing == "none" or parameter_sharing == "headwise" or parameter_sharing == "kv" or parameter_sharing == "layerwise", "The `parameter_sharing` flag has to be either 'none', 'headwise', 'kv', or 'layerwise'."
        assert channels % nhead == 0 if dim_d is None else True, "If `dim_d` is not set to a custom value, `channels` must be divisible by `nhead`!"
        assert not (ff_intermediate and parameter_sharing=="layerwise"), "Parameter sharing must not be layerwise if ff_intermediate is enabled!"
        assert not (ff_intermediate and decoder_mode), "Raising the dimension in the middle cannot be done in the decoder!"

        layers = nn.ModuleList()
        self.decoder_mode = decoder_mode
        self.input_size = input_size
        self.channels = channels
        self.checkpoint_level = checkpoint_level
        self.depth = depth
        self.nhead = nhead

        head_dim = channels // nhead if dim_d is None else dim_d

        E_proj = get_EF(input_size, dim_k, convolution, head_dim)
        causal_mask = gen_causal_mask(input_size, dim_k, full_attention) if causal else None
        # If we want causal but only with the encoder
        causal_enc = gen_causal_mask(input_size, dim_k, full_attention) if (causal and not decoder_mode) else None

        get_attn = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, E_proj, E_proj, full_attention, causal_enc, w_o_intermediate_dim, decoder_mode=False, convolution=convolution)
        get_attn_context = lambda attn_channels, curr_dim_k: MHAttention(input_size, head_dim, attn_channels, curr_dim_k, nhead, dropout, activation, checkpoint_level, parameter_sharing, E_proj, E_proj, full_attention, causal_mask, w_o_intermediate_dim, decoder_mode=True, convolution=convolution)
        get_ff = lambda input_channels, output_channels: FeedForward(input_channels, output_channels, dim_ff, dropout_ff)

        for index in range(depth):
            input_channels = ff_intermediate if (index != 0 and ff_intermediate is not None) and not decoder_mode else channels
            output_channels = ff_intermediate if (index != depth-1 and ff_intermediate is not None) and not decoder_mode else channels
            # TODO: Change the input and output channels here
            attn_layer = get_attn(input_channels, max(1, dim_k - index*k_reduce_by_layer))
            ff_layer = get_ff(input_channels, output_channels)

            attn_layer, ff_layer = map(lambda res_ch_in, res_ch_out, fn: Residual(PreNorm(input_channels, fn), res_ch_in, res_ch_out), (input_channels, input_channels), (input_channels, output_channels), (attn_layer, ff_layer))

            if include_ff:
                layers.extend([attn_layer, ff_layer])
            else:
                layers.extend([attn_layer])

            if not self.decoder_mode:
                continue

            attn_context = get_attn_context(channels, max(1, dim_k - index*k_reduce_by_layer))
            ff_context = get_ff(channels, channels)

            attn_context, ff_context = map(lambda fn: Residual(PreNorm(channels, fn)), (attn_context, ff_context))

            if include_ff:
                layers.extend([attn_context, ff_context])
            else:
                layers.extend([attn_context])

        self.seq = layers

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len, channels)
        """
        bt, n, c = tensor.shape
        assert n == self.input_size, "This tensor is of the wrong size. Dimension 1 has to match the `input_size` flag"
        assert c == self.channels, "This tensor is of the wrong size. Dimension 2 has to match the `channels` flag"
        assert self.checkpoint_level == "C0" if kwargs else True, "Cannot run checkpointing when using kwargs. Please set the checkpoint level to `C0`"
        assert "embeddings" not in kwargs or self.decoder_mode, "If decoding, needs to be initialized with `decoder_mode=True`"

        for layer in self.seq:
            if self.checkpoint_level != "C0":
                tensor = checkpoint(layer, tensor)
            else:
                tensor = layer(tensor, **kwargs)
        return tensor

class LinformerLM(nn.Module):
    """
    A wrapper function to accept LM tasks, inspired by https://github.com/lucidrains/sinkhorn-transformer
    """
    def __init__(self, num_tokens, input_size, channels,
                       dim_k=64, dim_ff=1024, dim_d=None,
                       dropout_ff=0.1, nhead=4, depth=2, ff_intermediate=None,
                       dropout=0.05, activation="gelu", checkpoint_level="C0",
                       parameter_sharing="layerwise", k_reduce_by_layer=0, full_attention=False,
                       include_ff=True, w_o_intermediate_dim=None, emb_dim=None,
                       return_emb=False, decoder_mode=False, causal=False, convolution=False):
        super(LinformerLM, self).__init__()
        emb_dim = channels if emb_dim is None else emb_dim

        self.input_size = input_size

        self.to_token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = PositionalEmbedding(emb_dim)
        self.linformer = Linformer(input_size, channels, dim_k=dim_k,
                                   dim_ff=dim_ff, dim_d=dim_d, dropout_ff=dropout_ff,
                                   nhead=nhead, depth=depth, dropout=dropout, ff_intermediate=ff_intermediate,
                                   activation=activation, checkpoint_level=checkpoint_level, parameter_sharing=parameter_sharing,
                                   k_reduce_by_layer=k_reduce_by_layer, full_attention=full_attention, include_ff=include_ff,
                                   w_o_intermediate_dim=w_o_intermediate_dim, decoder_mode=decoder_mode, causal=causal, convolution=convolution)

        if emb_dim != channels:
            self.linformer = ProjectInOut(self.linformer, emb_dim, channels)

        self.to_logits = identity if return_emb else nn.Linear(emb_dim, num_tokens)

    def forward(self, tensor, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        tensor = self.to_token_emb(tensor)
        tensor = self.pos_emb(tensor).type(tensor.type()) + tensor
        tensor = self.linformer(tensor, **kwargs)
        tensor = self.to_logits(tensor)
        return tensor

class LinformerEncDec(nn.Module):
    """
    A complete seq -> seq translation task. Complete with an encoder and a decoder module.
    """
    def __init__(self, enc_num_tokens, enc_input_size, enc_channels, dec_num_tokens, dec_input_size, dec_channels,
                       enc_dim_k=64, enc_dim_ff=1024, enc_dim_d=None, enc_ff_intermediate=None, dec_ff_intermediate=None,
                       enc_dropout_ff=0.1, enc_nhead=4, enc_depth=2, enc_dropout=0.05, enc_parameter_sharing="layerwise", enc_k_reduce_by_layer=0,
                       enc_full_attention=False, enc_include_ff=True, enc_w_o_intermediate_dim=None, enc_emb_dim=None, enc_convolution=False,
                       dec_dim_k=64, dec_dim_ff=1024, dec_dim_d=None, dec_dropout_ff=0.1, dec_nhead=4, dec_depth=2, dec_dropout=0.05,
                       dec_parameter_sharing="layerwise", dec_k_reduce_by_layer=0, dec_full_attention=False, dec_include_ff=True,
                       dec_w_o_intermediate_dim=None, dec_emb_dim=None, dec_convolution=False, activation="gelu", checkpoint_level="C0"):

        super(LinformerEncDec, self).__init__()
        self.encoder = LinformerLM(num_tokens=enc_num_tokens, input_size=enc_input_size, channels=enc_channels, dim_d=enc_dim_d, dim_ff=enc_dim_ff,
                                   dim_k=enc_dim_k, dropout_ff=enc_dropout_ff, nhead=enc_nhead, depth=enc_depth, dropout=enc_dropout,
                                   parameter_sharing=enc_parameter_sharing, k_reduce_by_layer=enc_k_reduce_by_layer, ff_intermediate=enc_ff_intermediate,
                                   full_attention=enc_full_attention, include_ff=enc_include_ff, w_o_intermediate_dim=enc_w_o_intermediate_dim,
                                   emb_dim=enc_emb_dim, return_emb=True, activation=activation, checkpoint_level=checkpoint_level, convolution=enc_convolution)
        self.decoder = LinformerLM(num_tokens=dec_num_tokens, input_size=dec_input_size, channels=dec_channels, dim_d=dec_dim_d, dim_ff=dec_dim_ff,
                                   dim_k=dec_dim_k, dropout_ff=dec_dropout_ff, nhead=dec_nhead, depth=dec_depth, dropout=dec_dropout, ff_intermediate=dec_ff_intermediate,
                                   parameter_sharing=dec_parameter_sharing, k_reduce_by_layer=dec_k_reduce_by_layer, convolution=dec_convolution,
                                   full_attention=dec_full_attention, include_ff=dec_include_ff, w_o_intermediate_dim=dec_w_o_intermediate_dim,
                                   emb_dim=dec_emb_dim, decoder_mode=True, causal=True, activation=activation, checkpoint_level=checkpoint_level)

    def forward(self, x, y=None, **kwargs):
        """
        Input is (batch_size, seq_len), and all items are ints from [0, num_tokens-1]
        """
        encoder_output = self.encoder(x, **kwargs)
        y = y if y is not None else x
        return self.decoder(y, embeddings=encoder_output)
