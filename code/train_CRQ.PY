import os
import numpy as np
import random
from collections import defaultdict
from typing import List, Tuple, Optional
from CRQVAE.crqvae import CRQVAE
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 1) 配置
DATA_PATH = "data/NYC/cleaned_poi_transition_matrix.npy"          # 交互矩阵
BATCH_SIZE = 128
EMBED_DIM = 140                # item 原始 embedding dim（或 encoder 输入 dim）
EPOCHS = 10
LR = 1e-4
K_THRESHOLD = 2                # 交互次数阈值
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MASKED_INFO_NCE = True     # True -> 改进 InfoNCE (multi-positive masked)
NUM_ITEMS = 5135              # item 数量（即交互矩阵的行/列数）



# 2) 加载/构建 pos_pairs & item->positives 映射
def load_interaction_matrix(path: str, k: int):
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        M = np.loadtxt(path, delimiter=",").astype(int)
    pos_pairs = np.argwhere(M >= k)
    return pos_pairs, M


def build_pos_dict(pos_pairs: np.ndarray) -> dict:
    pos_dict = defaultdict(list)
    for a, b in pos_pairs:
        pos_dict[int(a)].append(int(b))
    return pos_dict

# 3) Dataset: 随机采样 anchors，并为每个 anchor 找一个 positive（或跳过）
class ItemPairDataset(Dataset):
    def __init__(self, num_items: int, pos_pairs: np.ndarray, pos_dict: dict,
                 batch_size: int = 128, ensure_positive_rate: float = 0.9):
        self.num_items = num_items
        self.pos_pairs = pos_pairs
        self.pos_dict = pos_dict
        self.all_items = np.arange(num_items)
        self.batch_size = batch_size
        self.ensure_positive_rate = ensure_positive_rate
        self.items = self.all_items

    def __len__(self):
        return max(1, (self.num_items // self.batch_size))

    def __getitem__(self, idx):
        B = self.batch_size
        anchors = np.random.choice(self.items, size=B, replace=False)
        positives = np.empty(B, dtype=np.int64)
        for i, a in enumerate(anchors):
            pos_list = self.pos_dict.get(int(a), None)
            if pos_list and len(pos_list) > 0:
                positives[i] = np.random.choice(pos_list)
            else:
                # 没有正样本时用随机替代（也可选择跳过，或返回标记）
                if random.random() < self.ensure_positive_rate:
                    # 采样随机 item 作为 pseudo-positive（注意：它可能并非真实正样本）
                    negatives = np.delete(self.items, np.where(self.items == a))
                    positives[i] = np.random.choice(negatives)
                else:
                    positives[i] = -1  # 标记无正样本
        return anchors, positives


# 4) collate_fn: 把 (anchors, positives) -> batch tensors, batch_idx, pos_mask
def collate_pairs(batch: List[Tuple[np.ndarray, np.ndarray]]):
    # flatten
    anchors_list = []
    positives_list = []
    for anchors, positives in batch:
        anchors_list.append(anchors)
        positives_list.append(positives)
    anchors = np.concatenate(anchors_list)   # shape [B_total]
    positives = np.concatenate(positives_list)

    # 构造 batch_item_ids 为 anchors + positives 的 unique 列表（保持顺序）
    combined = np.concatenate([anchors, positives[positives >= 0]])  # 去掉标记 -1
    # 保持顺序的 unique
    seen = dict()
    batch_item_ids = []
    for item in combined:
        if item not in seen:
            seen[item] = True
            batch_item_ids.append(int(item))
    batch_item_ids = np.array(batch_item_ids, dtype=np.int64)
    id2local = {int(item): idx for idx, item in enumerate(batch_item_ids)}
    # 构造 local idx
    anchor_local_idx = np.array([id2local[int(a)] for a in anchors], dtype=np.int64)
    positive_local_idx = np.array([id2local[int(p)] if p >= 0 else -1 for p in positives], dtype=np.int64)

    # 构造 pos_mask: shape [num_anchors, num_items_in_batch]
    B_total = anchors.shape[0]
    N_batch_items = batch_item_ids.shape[0]
    pos_mask = np.zeros((B_total, N_batch_items), dtype=np.bool_)
    for i in range(B_total):
        pj = positive_local_idx[i]
        if pj >= 0:
            pos_mask[i, pj] = True

    # Convert to tensors
    batch_item_ids_t = torch.LongTensor(batch_item_ids)
    anchor_local_idx_t = torch.LongTensor(anchor_local_idx)
    positive_local_idx_t = torch.LongTensor(positive_local_idx)
    pos_mask_t = torch.from_numpy(pos_mask)

    return batch_item_ids_t, anchor_local_idx_t, positive_local_idx_t, pos_mask_t

# 5) Trainer（核心训练流程）
class Trainer:
    def __init__(self, model: nn.Module, embeddings: Optional[torch.Tensor],
                 pos_dict: dict, pos_pairs_np: np.ndarray,
                 batch_size=BATCH_SIZE, device=DEVICE,
                 masked_info_nce=USE_MASKED_INFO_NCE, lr=LR):

        self.model = model.to(device)
        self.device = device
        self.embeddings = embeddings.to(device) if embeddings is not None else None
        self.batch_size = batch_size
        self.masked_info_nce = masked_info_nce
        self.pos_dict = pos_dict
        # pos_pairs 全局 tensor，注册在 trainer 里用于 compute_loss 映射
        self.pos_pairs = torch.LongTensor(pos_pairs_np) if pos_pairs_np is not None else None

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def sample_batch_indices(self):
        anchors = np.random.choice(np.arange(len(self.embeddings)), size=self.batch_size, replace=False)
        positives = np.empty(self.batch_size, dtype=np.int64)
        for i, a in enumerate(anchors):
            pos_list = self.pos_dict.get(int(a), None)
            if pos_list and len(pos_list) > 0:
                positives[i] = np.random.choice(pos_list)
            else:
                # fallback: 随机选择一个非自身 item
                negs = np.delete(np.arange(len(self.embeddings)), int(a))
                positives[i] = np.random.choice(negs)
        return anchors, positives

    @staticmethod
    def build_pos_mask_from_global(batch_item_ids: torch.LongTensor,
                                   anchor_local_idx: torch.LongTensor,
                                   positive_local_idx: torch.LongTensor,
                                   pos_pairs_global: Optional[torch.LongTensor]):
        num_anchors = anchor_local_idx.size(0)
        num_items_in_batch = batch_item_ids.size(0)
        pos_mask = torch.zeros((num_anchors, num_items_in_batch), dtype=torch.bool, device=batch_item_ids.device)

        if pos_pairs_global is None:
            # fallback: 使用 provided positive_local_idx
            for i in range(num_anchors):
                pj = int(positive_local_idx[i].item())
                if pj >= 0:
                    pos_mask[i, pj] = True
            return pos_mask

        # 建立 batch_item_id -> local index 映射 (字典)
        b_ids = batch_item_ids.cpu().numpy().tolist()
        id2local = {int(b): idx for idx, b in enumerate(b_ids)}

        # 筛选 pos_pairs_global: 仅保留 both endpoints in batch_item_ids
        pos_np = pos_pairs_global.cpu().numpy()
        # 这里可以更高效：把 pos_pairs_preprocessed 分桶为 dict: item->list (推荐预处理)
        # 为通用性，我们做一次过滤
        mask_bool = [(int(a) in id2local and int(b) in id2local) for a, b in pos_np]
        if not any(mask_bool):
            # no positives inside batch
            # fallback to simple positives if available
            for i in range(num_anchors):
                pj = int(positive_local_idx[i].item())
                if pj >= 0:
                    pos_mask[i, pj] = True
            return pos_mask

        selected = pos_np[[i for i, v in enumerate(mask_bool) if v]]
        # 把全局 ids 映射为 local
        for a_global, b_global in selected:
            ai = id2local[int(a_global)]
            bi = id2local[int(b_global)]
            # find all positions where batch anchors correspond to ai
            # anchor_local_idx gives local indices of each anchor in the batch_item list
            # we need to set pos_mask[row_index_where_anchor_local==ai, bi] = True
            # anchor_local_idx is length num_anchors, values are local positions
            rows = (anchor_local_idx == ai).nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() > 0:
                pos_mask[rows, bi] = True
        return pos_mask


    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        iters = 0

        for batch in dataloader:
            batch_item_ids, anchor_local_idx, positive_local_idx, pos_mask_simple = batch
            batch_item_ids = batch_item_ids.to(self.device)
            anchor_local_idx = anchor_local_idx.to(self.device)
            positive_local_idx = positive_local_idx.to(self.device)
            pos_mask_simple = pos_mask_simple.to(self.device)

            if self.embeddings is None:
                raise ValueError("请提供原始 item embeddings 或让 model 能直接根据 item id 生成输入")
            batch_feats = self.embeddings[batch_item_ids].to(self.device)  # [num_items_in_batch, feat_dim]
            x_q_all, rq_loss_all, _ = self.model(batch_feats)
            anchor_vecs = x_q_all[anchor_local_idx]                # [num_anchors, D]
            if self.pos_pairs is not None:
                pos_mask = self.build_pos_mask_from_global(batch_item_ids, anchor_local_idx, positive_local_idx, self.pos_pairs)
            else:
                pos_mask = pos_mask_simple  # fallback

            # 5) 计算对比损失（masked 或标准）
            z_norm = F.normalize(anchor_vecs, dim=1)          # [num_anchors, D]
            all_norm = F.normalize(x_q_all, dim=1)            # [num_items_in_batch, D]
            sim = torch.matmul(z_norm, all_norm.T) / self.model.temperature  # [num_anchors, num_items_in_batch]

            # 数值稳定：使用 logsumexp
            sim_max, _ = torch.max(sim, dim=1, keepdim=True)
            sim_stable = sim - sim_max

            exp_sim = torch.exp(sim_stable)   # [num_anchors, num_items_in_batch]
            denom = exp_sim.sum(dim=1)        # [num_anchors]

            if self.masked_info_nce:
                # numerator: sum over positive positions in mask
                num = (exp_sim * pos_mask.float()).sum(dim=1)  # [num_anchors]
                # 若某些 anchor 没有 positives (num==0)，用对角 fallback 或跳过
                zero_mask = (num <= 0)
                if zero_mask.any():
                    # fallback: use the element corresponding to positive_local_idx (if provided)
                    # positive_local_idx may be -1 for missing; handle carefully
                    fallback_vals = []
                    for i in range(anchor_local_idx.size(0)):
                        pj = int(positive_local_idx[i].item())
                        if pj >= 0:
                            fallback_vals.append(sim[i, pj].exp().item())
                        else:
                            fallback_vals.append(1e-8)
                    fallback_tensor = torch.tensor(fallback_vals, device=denom.device, dtype=denom.dtype)
                    num = torch.where(zero_mask, fallback_tensor, num)
            else:
                num = torch.zeros_like(denom)
                for i in range(anchor_local_idx.size(0)):
                    pj = int(positive_local_idx[i].item())
                    if pj >= 0:
                        num[i] = exp_sim[i, pj]
                    else:
                        num[i] = 1e-8

            # 防止为0
            num = torch.clamp(num, min=1e-12)
            denom = torch.clamp(denom, min=1e-12)

            # loss per anchor: -log(num/denom)
            cl_loss_per_anchor = -torch.log(num / denom)
            cl_loss = cl_loss_per_anchor.mean()

            # 总损失 =  weight * RQ量化损失 + cl_loss
            if torch.is_tensor(rq_loss_all) and rq_loss_all.numel() == 1:
                rq_loss = rq_loss_all
            else:
                rq_loss = rq_loss_all.mean()

            total_loss = self.model.quant_loss_weight * rq_loss + cl_loss

            # 7) 反向与优化
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            total_loss_val = total_loss.item()
            total_loss += total_loss_val
            total_cl_loss += cl_loss.item()
            iters += 1

        avg_loss = total_loss / max(1, iters)
        avg_cl_loss = total_cl_loss / max(1, iters)
        print(f"Epoch {epoch} | avg total loss: {avg_loss:.6f} | avg cl loss: {avg_cl_loss:.6f}")

    def fit(self, dataset: Dataset, epochs: int = EPOCHS):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_pairs)
        for epoch in range(1, epochs + 1):
            self.train_epoch(dataloader, epoch)



if __name__ == "__main__":
    # 0) 读取 pos_pairs
    pos_pairs_np, M = load_interaction_matrix(DATA_PATH, K_THRESHOLD)
    print(f"Total pos pairs: {pos_pairs_np.shape[0]}")

    # 1) 构建 pos_dict
    pos_dict = build_pos_dict(pos_pairs_np)

    # 2) item 原始 embedding（示例）
    item_embeddings = torch.randn(NUM_ITEMS, EMBED_DIM) * 0.1

    # 3) 实例化 Dataset
    dataset = ItemPairDataset(num_items=NUM_ITEMS, pos_pairs=pos_pairs_np, pos_dict=pos_dict, batch_size=BATCH_SIZE)

    # 4) 模型
    model = CRQVAE(
        in_dim=EMBED_DIM,
        num_emb_list=[128, 64],
        e_dim=64,
        layers=[64, 64],
        use_sk=False,
        pos_pairs=pos_pairs_np,
        temperature=0.07
    ).to(DEVICE)

    # 5) Trainer
    trainer = Trainer(model=model, embeddings=item_embeddings, pos_dict=pos_dict, pos_pairs_np=pos_pairs_np,
                      batch_size=BATCH_SIZE, device=DEVICE, masked_info_nce=USE_MASKED_INFO_NCE, lr=LR)

    # 6) 训练
    trainer.fit(dataset, epochs=50)
