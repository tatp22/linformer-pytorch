import torch
import torch.nn as nn
import torch.nn.functional as F

def get_act(activation):
    if activation == "gelu":
        return F.gelu
    return F.relu

class FeedForward(nn.Module):
    """
    Standard Feed Forward Layer
    """
    def __init__(self, dim, mult=4, activation="gelu", dropout=0.0):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(dim, dim*mult)
        self.w_2 = nn.Linear(dim*mult, dim)
        self.activation = get_act(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tensor):
        tensor = self.w_1(tensor)
        if self.activation is not None:
            tensor = self.activation(tensor)
        tensor = self.dropout(tensor)
        tensor = self.w_2(tensor)

class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """
    def __init__(self, dim, dropout, E, F):
        super(LinearAttentionHead, self).__init__()
        self.w_k = nn.Linear(dim, dim)
        self.w_q = nn.Linear(dim, dim)
        self.w_v = nn.Linear(dim, dim)
        self.dim = dim
        self.E = E
        self.F = F
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        KW = self.w_k(K)
        KW = torch.einsum("nk,bnc->bkc", self.E, KW)
        QW = self.w_q(Q)
        QW = torch.einsum("bnc,bkc->bnkc", QW, KW)
        P_bar = QW/torch.sqrt(self.dim)
        P_bar = P_bar.softmax(dim=-1)
        P_bar = self.dropout(P_bar)

        VW = self.w_v(V)
        VW = torch.einsum("nk,bnc->bkc", self.F, VW)
        out_tensor = torch.einsum("bnkc,bkc->bnc", P_bar, VW)

        return out_tensor

class MHAttention(nn.Module):
    """
    Multihead attention, with each head being a Linformer Head
    This feeds directly into a feed forward head
    """
    def __init__(self, input_size, dim, channels, dim_k, nhead, dropout, activation):
        super(MHAttention, self).__init__()
        self.heads = nn.ModuleList()

        # For now, let's use the same projection matrix for all of them, as mentioned in the "Additional Efficiency Techniques" section
        self.EF = torch.eye(input_size, dim_k)

        for head in range(nhead):
            attn = LinearAttentionHead(dim, dropout, activation, self.EF, self.EF)
            self.heads.append(attn)
        self.w_o = nn.Linear(dim*nhead, dim_ff)
        self.to_q = nn.Linear(channels, dim, bias=False)
        self.to_k = nn.Linear(channels, dim, bias=False)
        self.to_v = nn.Linear(channels, dim, bias=False)

    def forward(self, tensor):
        head_outputs = []
        for head in self.heads:
            Q = self.to_q(tensor)
            K = self.to_k(tensor)
            V = self.to_v(tensor)
            head_outputs.append(head(Q,K,V))
        out = torch.cat(head_outputs, dim=2)
        out = self.w_o(out)
        return out

class Linformer(nn.Module):
    """
    My attempt at reproducing the Linformer Paper
    https://arxiv.org/pdf/2006.04768.pdf
    """
    def __init__(self, input_size=8192, channels=128, dim_k=256, dim_ff=2048, dropout_ff=0.15, nhead=8, depth=6, dropout=0.1, activation="gelu"):
        super(Linformer, self).__init__()

        self.layers = nn.ModuleList()
        self.input_size = input_size
        dim_d = input_size // nhead

        get_attn = lambda: MHAttention(input_size, dim_d, channels, dim_k, dim_ff, nhead, dropout, activation)
        get_ff = lambda: FeedForward(dim_ff, dropout_ff)
        norm_attn = lambda: nn.LayerNorm(dim_d)
        norm_ff = lambda: nn.LayerNorm(dim_ff)

        for index in range(depth):
            self.layers.append([get_attn(),
                                norm_attn(),
                                get_ff(),
                                norm_ff()])

    def forward(self, tensor):
        """
        Input is (batch_size, seq_len, channels)
        """
        for layer in self.layers:
            attn = layer[0]
            attn_norm = layer[1]
            ff = layer[2]
            ff_norm = layer[3]

            # Run attention
            before_attn_tensor = tensor.clone().detach()
            tensor = attn(tensor)
            tensor += before_attn_tensor
            tensor = attn_norm(tensor)

             # Run ff
            before_ff_tensor = tensor.clone().detach()
            tensor = ff(tensor)
            tensor += before_ff_tensor
            tensor = ff_norm(tensor)

        return tensor
