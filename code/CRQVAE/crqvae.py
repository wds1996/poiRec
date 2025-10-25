import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLPLayers
from .rq import ResidualVectorQuantizer


class CRQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                #  loss_type="mse",
                 quant_loss_weight=0.25,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=100,
                 use_sk=False,
                 sk_epsilons=None,
                 sk_iters=100,
                 use_linear=0,
                 temperature=0.1,   # 对比学习温度参数
        ):
        super(CRQVAE, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        # self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.use_sk = use_sk
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.temperature = temperature

        # 编码器
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        # 残差向量量化器
        self.rq = ResidualVectorQuantizer(
            num_emb_list, e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            use_sk=self.use_sk,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear
        )

        # 基于下游推荐任务进行损失估计，无需再使用重构损失
        # self.decode_layer_dims = self.encode_layer_dims[::-1]
        # self.decoder = MLPLayers(layers=self.decode_layer_dims,
        #                                dropout=self.dropout_prob,bn=self.bn)

    def forward(self, x):
        x = self.encoder(x)
        x_q, rq_loss, codes = self.rq(x)
        return x_q, rq_loss, codes
    

    def contrastive_loss(self, x_q, pos_mask, anchor_local_idx):
        """计算 masked InfoNCE 损失"""
        device = x_q.device
        anchor_vecs = x_q[anchor_local_idx]  # [B, D]
        z_norm = F.normalize(anchor_vecs, dim=1)
        all_norm = F.normalize(x_q, dim=1)
        sim = torch.matmul(z_norm, all_norm.T) / self.temperature  # [B, N]

        # 数值稳定
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim_stable = sim - sim_max
        exp_sim = torch.exp(sim_stable)
        denom = exp_sim.sum(dim=1)  # [B]

        # numerator: sum over all positives
        num = (exp_sim * pos_mask.float()).sum(dim=1)  # [B]
        
        # 处理无正样本的情况
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        cl_loss = -torch.log((num[has_pos] + 1e-12) / (denom[has_pos] + 1e-12)).mean()
        return cl_loss

    def compute_loss(self, x_q, rq_loss, pos_mask, anchor_local_idx):
        cl_loss = self.contrastive_loss(x_q, pos_mask, anchor_local_idx)
        total_loss = self.quant_loss_weight * rq_loss + cl_loss
        return total_loss, rq_loss, cl_loss

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, (indices, scalars) = self.rq(x_e)
        # return indices.cpu(), scalars.cpu()  # [B, L], [B, L]
        indices_cpu = [idx.cpu().tolist() for idx in indices]
        return indices_cpu