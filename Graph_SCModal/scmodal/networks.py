import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class encoder(nn.Module):
    def __init__(self, n_input, n_latent):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(
            n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(
            n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(
            self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(
            self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        z = F.linear(h, self.W_2, self.b_2)
        return z


class encoder_graphical(nn.Module):
    """
    Simple GNN-like encoder that ingests:
      - gene_expr: Tensor [N, G] (per-node gene expression)
      - coords:    Tensor [N, C] (spatial coordinates, e.g. (x,y))
    and produces:
      - z: Tensor [N, n_latent]

    Parameters
    ----------
    n_gene_input : int
        Number of gene features (G).
    coord_dim : int
        Dimensionality of coordinates (C), typically 2.
    n_latent : int
        Output latent dimension.
    n_hidden : int, optional
        Hidden size used inside the GNN (default 512 to match your other encoders).
    k : int, optional
        Number of nearest neighbours used to build adjacency (default 8).
    use_rbf : bool, optional
        If True, use RBF-weighted edges; otherwise edges are binary.
    rbf_sigma : float, optional
        Sigma value for RBF kernel (if use_rbf=True).
    eps : float, optional
        Small value to avoid division by zero in normalization.
    """

    def __init__(self, n_gene_input, coord_dim, n_latent,
                 n_hidden=512, k=8, use_rbf=True, rbf_sigma=1.0, eps=1e-8):
        super(encoder_graphical, self).__init__()
        self.n_gene_input = n_gene_input
        self.coord_dim = coord_dim
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        self.k = k
        self.use_rbf = use_rbf
        self.rbf_sigma = float(rbf_sigma)
        self.eps = eps

        # Gene feature projection: G -> hidden
        self.W_g1 = nn.Parameter(torch.Tensor(
            self.n_hidden, self.n_gene_input).normal_(0.0, 0.1))
        self.b_g1 = nn.Parameter(torch.Tensor(self.n_hidden).normal_(0.0, 0.1))

        # Coord projection: C -> hidden (so coords can contribute to messages)
        self.W_c1 = nn.Parameter(torch.Tensor(
            self.n_hidden, self.coord_dim).normal_(0.0, 0.1))
        self.b_c1 = nn.Parameter(torch.Tensor(self.n_hidden).normal_(0.0, 0.1))

        # Message combination -> hidden (after aggregation)
        # We'll use a second linear to mix aggregated messages
        self.W_msg = nn.Parameter(torch.Tensor(
            self.n_hidden, self.n_hidden).normal_(0.0, 0.1))
        self.b_msg = nn.Parameter(torch.Tensor(
            self.n_hidden).normal_(0.0, 0.1))

        # Final projection to latent
        self.W_z = nn.Parameter(torch.Tensor(
            self.n_latent, self.n_hidden).normal_(0.0, 0.1))
        self.b_z = nn.Parameter(torch.Tensor(self.n_latent).normal_(0.0, 0.1))

        # optional skip / self transform
        self.W_self = nn.Parameter(torch.Tensor(
            self.n_hidden, self.n_hidden).normal_(0.0, 0.1))
        self.b_self = nn.Parameter(
            torch.Tensor(self.n_hidden).normal_(0.0, 0.1))

    def _build_adj_knn(self, coords):
        """
        Build a dense adjacency matrix (N x N) using k-NN on coords.
        Returns adjacency matrix A (N,N) where A[i,j] = weight (0 if not neighbor).
        """
        # coords: [N, C]
        device = coords.device
        N = coords.size(0)
        # pairwise squared distances [N, N]
        # use torch.cdist for numerical stability; result is distances (not squared)
        dists = torch.cdist(coords, coords, p=2.0)  # [N, N]
        # For each row find k+1 smallest (including self), then exclude self
        k = min(self.k + 1, N)  # +1 to include self for easiest masking
        vals, idx = torch.topk(dists, k=k, largest=False)  # smallest distances
        # Build adjacency
        A = torch.zeros((N, N), device=device, dtype=coords.dtype)
        # idx: [N, k], vals: [N, k] where vals[:,0] == 0 for self (distance to self)
        for i in range(N):
            neighbors = idx[i]  # indices of nearest (including self)
            # set edges for all neighbors except self
            for j_idx, j in enumerate(neighbors):
                if j == i:
                    continue
                if self.use_rbf:
                    dist = vals[i, j_idx]
                    weight = torch.exp(- (dist**2) /
                                       (2.0 * (self.rbf_sigma**2) + 1e-12))
                else:
                    weight = 1.0
                A[i, j] = weight
        # Optionally symmetrize: A = (A + A.T) / 2
        A = (A + A.t()) / 2.0
        return A

    def _normalize_adj(self, A):
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}
        A: [N, N]
        """
        deg = torch.sum(A, dim=1)  # [N]
        deg_inv_sqrt = torch.pow(deg + self.eps, -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        return A_norm

    def forward(self, gene_expr, coords):
        """
        gene_expr: Tensor [N, G]
        coords: Tensor [N, C]
        returns z: Tensor [N, n_latent]
        """
        # Basic checks
        assert gene_expr.dim() == 2 and coords.dim() == 2
        N = gene_expr.size(0)
        assert coords.size(0) == N

        device = gene_expr.device
        coords = coords.to(device)

        # Project gene features into hidden space
        # h_g = ReLU(gene_expr @ W_g1^T + b_g1)
        h_g = F.linear(gene_expr, self.W_g1, self.b_g1)  # [N, hidden]
        h_g = F.relu(h_g)

        # Project coordinates into hidden space
        h_c = F.linear(coords, self.W_c1, self.b_c1)  # [N, hidden]
        h_c = F.relu(h_c)

        # Combine node features (elementwise add)
        h = h_g + h_c  # [N, hidden]

        # Build adjacency (based on coords)
        A = self._build_adj_knn(coords)  # [N, N]
        A_norm = self._normalize_adj(A)  # [N, N]

        # Aggregate neighbor messages: A_norm @ h
        agg = A_norm @ h  # [N, hidden]

        # Transform aggregated messages
        msg = F.linear(agg, self.W_msg, self.b_msg)  # [N, hidden]
        msg = F.relu(msg)

        # Add self information (skip)
        self_feat = F.linear(h, self.W_self, self.b_self)
        self_feat = F.relu(self_feat)

        h_final = msg + self_feat  # [N, hidden]
        h_final = F.relu(h_final)

        # Project to latent
        z = F.linear(h_final, self.W_z, self.b_z)  # [N, latent]
        return z


class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(
            n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(
            n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(
            self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(
            self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x


class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(
            n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(
            n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(
            n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(
            n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(
            1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)
