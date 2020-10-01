import torch
import torch.nn as nn
import numpy as np

import matplotlib.colors as col
import matplotlib.pyplot as plt

from linformer_pytorch import Linformer, MHAttention

class Visualizer():
    """
    A way to visualize the attention heads for each layer
    """
    def __init__(self, net):
        assert isinstance(net, (Linformer, MHAttention)), "Only the Linformer and MHAttention is supported"
        self.net = net

    def get_head_visualization(self, depth_no, max_depth, head_no, n_limit, axs):
        """
        Returns the visualization for one head in the Linformer or MHAttention
        """
        if isinstance(self.net, Linformer):
            depth_to_use = 2*depth_no if 2*(max_depth+1) == len(self.net.seq) else depth_no
            curr_mh_attn = self.net.seq[depth_to_use].fn
            curr_head = curr_mh_attn.heads[head_no]
        else:
            curr_head = self.net.fn.heads[head_no]

        arr = curr_head.P_bar[0].detach().cpu().numpy()
        assert arr is not None, "Cannot visualize a None matrix!"

        if n_limit is not None:
            arr = arr[:n_limit, :]

        # Remove axis ticks
        axs[depth_no, head_no].set_xticks([])
        axs[depth_no, head_no].set_yticks([])

        pcm = axs[depth_no, head_no].imshow(arr, cmap="Reds", aspect="auto", norm=col.Normalize())
        if head_no == 0:
            axs[depth_no, head_no].set_ylabel("Layer {}".format(depth_no+1), fontsize=20)

        if depth_no == max_depth:
            axs[depth_no, head_no].set_xlabel("Head {}".format(head_no+1), fontsize=20)

        return pcm

    def plot_all_heads(self, title="Visualization of Attention Heads", show=True, save_file=None, figsize=(8,6), n_limit=None):
        """
        Showcases all of the heads on a grid. It shows the P_bar matrices for each head,
        which turns out to be an NxK matrix for each of them.
        """

        if isinstance(self.net, Linformer):
            self.depth = self.net.depth
            self.heads = self.net.nhead
        else:
            self.depth = 1
            self.heads = self.net.nhead

        fig, axs = plt.subplots(self.depth, self.heads, figsize=figsize)
        axs = axs.reshape((self.depth, self.heads)) # In case depth or nheads are 1, bug i think

        fig.suptitle(title, fontsize=26)

        for d_idx in range(self.depth):
            for h_idx in range(self.heads):
                pcm = self.get_head_visualization(d_idx, self.depth-1, h_idx, n_limit, axs)

        if show:
            plt.show()

        if save_file is not None:
            fig.savefig(save_file)
