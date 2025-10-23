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
        if self.use_linear == 1:
            return self.codebook_projection(self.embedding.weight)
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # 获取量化后的嵌入向量
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x):
        # Flatten input
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            codebook = self.codebook_projection(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        # # Calculate the L2 Norm between latent and Embedded weights
        # d = torch.sum(latent**2, dim=1, keepdim=True) + \
        #     torch.sum(embeddings_weight**2, dim=1, keepdim=True).t()- \
        #     2 * torch.matmul(latent, embeddings_weight.t())
        # indices = torch.argmin(d, dim=-1)
        
        # if use_sk and self.sk_epsilon > 0:
        #     d_soft = self.center_distance_for_constraint(d)
        #     d_soft = d_soft.double()
        #     Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
        # else:
        #     Q = F.softmax(-d, dim=-1)
        
        # ——————————————————————修改——————————————————————       
        # 归一化，用余弦相似度替代欧氏距离
        latent_norm = F.normalize(latent, dim=1)
        codebook_norm = F.normalize(codebook, dim=1)
        # 相似度矩阵：B × K
        sim = torch.matmul(latent_norm, codebook_norm.t())  # cosine similarity
        d = 1-sim  # 越相似距离越小
        
        if self.use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            d_soft = self.center_distance_for_constraint(d)
            d_soft = d_soft.double()
            Q = sinkhorn_algorithm(d_soft, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any():
                print("Warning: Sinkhorn returned NaN, falling back to argmin")
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        else:
            indices = torch.argmin(d, dim=-1)
        # ——————————————————————修改结束——————————————————————

        # 获取量化后的向量
        codebook_vec = F.embedding(indices, codebook).view(x.shape)  # [B, D]
        x_q = codebook_vec

        # 量化损失
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # 直通估计器
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices, codebook_vec


    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    

