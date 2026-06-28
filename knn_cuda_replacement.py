"""
Drop-in replacement for knn_cuda.KNN using native PyTorch ops.
Matches the API of unlimblue/KNN_CUDA (now unavailable on GitHub).
"""
import torch
import torch.nn as nn


class KNN(nn.Module):
    def __init__(self, k, transpose_mode=True):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, ref, query):
        """
        transpose_mode=True:
            ref:   [bs, nr, dim]
            query: [bs, nq, dim]
            returns dist: [bs, nq, k], idx: [bs, nq, k]
        """
        if not self.transpose_mode:
            ref = ref.transpose(1, 2)
            query = query.transpose(1, 2)

        dist = torch.cdist(query, ref, p=2)
        dist_k, idx_k = torch.topk(dist, k=self.k, dim=-1, largest=False)

        if not self.transpose_mode:
            dist_k = dist_k.transpose(1, 2)
            idx_k = idx_k.transpose(1, 2)

        return dist_k, idx_k
