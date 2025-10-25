import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import kmeans, sinkhorn_algorithm


class CosineVectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim,
                 beta = 0.25, kmeans_init = False, kmeans_iters = 10,
                 use_sk=False, sk_epsilon=None, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_sk = use_sk
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        # 初始化码本
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        
        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)
    

    def get_codebook(self):
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x):
        B, D = x.shape
        latent = x.view(B, D)

        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()  # [K, D]

        # Cosine similarity for index selection
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        sim = torch.matmul(latent_norm, codebook_norm.t())  # [B, K]
        distances = 1 - sim  # 越小表示越接近

        if self.use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(distances)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any():
                print("Warning: Sinkhorn returned NaN, falling back to argmin")
                indices = torch.argmin(distances, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(distances, dim=-1)


        # Get codebook vectors
        codebook_vec = F.embedding(indices, codebook)  # [B, D]

        # Compute projection scalar: w = (x · c) / ||c||^2
        dot_product = torch.sum(latent * codebook_vec, dim=-1, keepdim=True)  # [B, 1]
        norm_sq = torch.sum(codebook_vec * codebook_vec, dim=-1, keepdim=True)
        scalar = dot_product / (norm_sq + 1e-8)                                # [B, 1]
        proj_vec = scalar * codebook_vec

        # Loss based on projection
        commitment_loss = F.mse_loss(proj_vec.detach(), x)
        codebook_loss = F.mse_loss(proj_vec, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        x_q = x + (proj_vec - x).detach()

        indices = indices.view(B)        # [B]
        scalar = scalar.view(B)          # [B]

        return x_q, loss, indices, scalar
    

    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances