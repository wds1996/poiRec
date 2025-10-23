import torch
import torch.nn as nn
import torch.nn.functional as F
from .cvq import CosineVectorQuantizer


class ResidualVectorQuantizer(nn.Module):

    def __init__(self, n_e_list, e_dim, use_sk=False, sk_epsilons=None, beta=0.25,
                 kmeans_init=False, kmeans_iters=100, sk_iters=100, use_linear=0):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_sk = use_sk
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        if use_sk and sk_epsilons is not None:
            self.vq_layers = nn.ModuleList([
                CosineVectorQuantizer(n_e, e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    use_sk=use_sk,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    use_linear=use_linear
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ])
        else:
            self.vq_layers = nn.ModuleList([
                CosineVectorQuantizer(
                    n_e, e_dim,
                    beta=self.beta,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    use_linear=use_linear
                )
                for n_e in n_e_list
            ])


    def forward(self, x):
        all_losses = []
        all_indices = []
        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, codebook_vec = quantizer(residual)
            
            # x_q = x_q + x_res
            # residual = residual - x_res
            
            # 计算投影系数
            dot_product = torch.sum(residual * codebook_vec, dim=-1, keepdim=True)
            norm_sq = torch.sum(codebook_vec * codebook_vec, dim=-1, keepdim=True)
            norm_sq = torch.clamp(norm_sq, min=1e-5)
            scalar = dot_product / norm_sq  # [B, 1]
            # 投影分量
            projection_component = scalar * codebook_vec  # [B, D]
            # 累积重建
            x_q = x_q + projection_component
            # 更新残差
            residual = residual - projection_component
            
            all_losses.append(loss)
            all_indices.append(indices)
        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

    @torch.no_grad()
    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)