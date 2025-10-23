import os
import numpy as np
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from CRQVAE.crqvae import CRQVAE
import pickle
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="Train CRQ-VAE for POI Semantic Quantization")

    # 数据路径
    parser.add_argument("--data_path", type=str, default="data/NYC/cleaned_poi_transition_matrix.npy", help="Path to POI transition matrix (.npy)")
    parser.add_argument("--embedding_path", type=str, default="data/NYC/poi_Emb_dict.pkl", help="Path to POI embedding dict (.pkl)")
    parser.add_argument("--model_save_path", type=str, default="checkpoints/nyc_crqvae/best_crqvae.pth", help="Path to save the best model checkpoint")
    # 训练超参
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for contrastive learning")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    # 模型配置
    parser.add_argument("--num_emb_list", type=int, nargs='+', default=[64, 64, 64], help="Codebook sizes for each residual layer")
    parser.add_argument("--e_dim", type=int, default=64, help="Dimension of quantized embedding")
    parser.add_argument("--layers", type=int, nargs='+', default=[256, 128], help="Hidden layers of encoder MLP")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=True, help="use bn or not")
    parser.add_argument("--quant_loss_weight", type=float, default=0.25, help="Weight for RQ loss (regularization)")
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight inside RQ")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--use_sk', type=bool, default=True, help="use sinkhorn or not")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.005, 0.01], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=100, help="max sinkhorn iters")
    parser.add_argument("--use-linear", type=int, default=0, help="use-linear")
    # 对比学习
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for contrastive loss")
    parser.add_argument("--k_threshold", type=int, default=2, help="Minimum co-visit count to form a positive pair")
    
    return parser.parse_args()



def load_interaction_matrix(path: str, k: int):
    if path.endswith(".npy"):
        M = np.load(path)
    else:
        M = np.loadtxt(path, delimiter=",").astype(int)
    pos_pairs = np.argwhere(M >= k)
    return pos_pairs, M

def build_pos_dict(pos_pairs: np.ndarray, max_item_id: int) -> dict:
    pos_dict = defaultdict(list)
    for a, b in pos_pairs:
        if a < max_item_id and b < max_item_id:  # 防止越界
            pos_dict[int(a)].append(int(b))
    return pos_dict

def load_poi_embeddings(embedding_path: str):
    """
    Args:
        embedding_path (str): Path to .pkl (dict {pid: vec}) or .npy ([N, D]) file.
    Returns:
        ids (List[int]): List of POI IDs, length = N
        embeddings (torch.Tensor): Float tensor of shape [N, D]
        num_items (int): N
        embed_dim (int): D
    """
    if embedding_path.endswith('.npy'):
        # Load as numpy array
        emb_array = np.load(embedding_path)
        assert emb_array.ndim == 2, f"Expected 2D array in {embedding_path}, got {emb_array.ndim}D"
        num_items, embed_dim = emb_array.shape
        # Implicit IDs: 0, 1, 2, ..., N-1
        ids = list(range(num_items))
        embeddings = torch.from_numpy(emb_array).float()
        
    elif embedding_path.endswith('.pkl'):
        # Load as dict {pid: vector}
        with open(embedding_path, 'rb') as f:
            emb_dict = pickle.load(f)
        assert isinstance(emb_dict, dict), f"Expected dict in {embedding_path}, got {type(emb_dict)}"
        assert len(emb_dict) > 0, "Empty embedding dict!"
        # Ensure all keys are integers
        if not all(isinstance(k, int) for k in emb_dict.keys()):
            raise ValueError("All POI IDs in .pkl must be integers.")
        # Sort by ID to ensure deterministic order
        sorted_items = sorted(emb_dict.items(), key=lambda x: x[0])
        ids = [item[0] for item in sorted_items]
        vectors = [item[1] for item in sorted_items]
        # Infer embed_dim from first vector
        first_vec = vectors[0]
        if isinstance(first_vec, torch.Tensor):
            embed_dim = first_vec.shape[0]
        elif isinstance(first_vec, np.ndarray):
            embed_dim = first_vec.shape[0]
        elif isinstance(first_vec, (list, tuple)):
            embed_dim = len(first_vec)
        else:
            raise ValueError(f"Unsupported vector type: {type(first_vec)}")
        # Convert all to torch.Tensor
        tensor_list = []
        for vec in vectors:
            if isinstance(vec, torch.Tensor):
                t = vec.float()
            else:
                t = torch.tensor(vec, dtype=torch.float32)
            assert t.shape[0] == embed_dim, f"Vector dim mismatch: {t.shape[0]} vs {embed_dim}"
            tensor_list.append(t)
        embeddings = torch.stack(tensor_list, dim=0)  # [N, D]
        num_items = len(ids)
        embed_dim = embeddings.shape[1]
        
    else:
        raise ValueError(f"Unsupported file format: {embedding_path}. Use .pkl or .npy")
    
    # Final sanity check
    assert len(ids) == embeddings.shape[0], "ID list length != embedding rows"
    assert embeddings.dtype == torch.float32, "Embedding dtype must be float32"
    
    return ids, embeddings, num_items, embed_dim
    
class Trainer:
    def __init__(self, model: nn.Module, embeddings: torch.Tensor, pos_dict: dict, args):
        self.model = model.to(args.device)
        self.device = args.device
        self.embeddings = embeddings.to(self.device)
        self.pos_dict = pos_dict
        self.batch_size = args.batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.num_items = embeddings.shape[0]
        self.model_save_path = args.model_save_path
        self.best_cl_loss = float('inf')  # 初始化为无穷大

    def _build_batch(self):
        """高效构建 batch 和 pos_mask"""
        # 采样 anchors
        anchors = np.random.choice(self.num_items, size=self.batch_size, replace=False)
        
        # 收集所有可能的 items: anchors + 所有正样本
        all_items_set = set(anchors.tolist())
        for a in anchors:
            all_items_set.update(self.pos_dict.get(int(a), []))
        all_items = list(all_items_set)
        id2local = {item: idx for idx, item in enumerate(all_items)}
        
        # 构建 pos_mask: [B, N_batch]
        B = len(anchors)
        N = len(all_items)
        pos_mask = torch.zeros((B, N), dtype=torch.bool, device=self.device)
        anchor_local = torch.tensor([id2local[a] for a in anchors], device=self.device)
        
        for i, a in enumerate(anchors):
            for p in self.pos_dict.get(int(a), []):
                if p in id2local:
                    pos_mask[i, id2local[p]] = True
        
        batch_item_ids = torch.tensor(all_items, device=self.device)
        return batch_item_ids, anchor_local, pos_mask

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        total_rq_loss = 0.0   
        total_cl_loss = 0.0   
        num_batches = 1000  # 每 epoch 训练 1000 batches
        
        for _ in range(num_batches):
            # 构建 batch
            batch_item_ids, anchor_local_idx, pos_mask = self._build_batch()
            batch_feats = self.embeddings[batch_item_ids]  # [N, EMBED_DIM]
            
            # 前向
            x_q_all, rq_loss, _ = self.model(batch_feats)
            
            # 计算损失
            total_loss_batch, rq_loss, cl_loss = self.model.compute_loss(
                x_q_all, rq_loss, pos_mask, anchor_local_idx
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            self.optimizer.step()
            
            total_loss += total_loss_batch.item()
            total_rq_loss += rq_loss.item()
            total_cl_loss += cl_loss.item()
        
        avg_total_loss = total_loss / num_batches
        avg_rq_loss = total_rq_loss / num_batches
        avg_cl_loss = total_cl_loss / num_batches

        # === 保存最佳模型（基于 cl_loss）===
        if avg_cl_loss < self.best_cl_loss:
            self.best_cl_loss = avg_cl_loss
            # 确保父目录存在
            save_dir = os.path.dirname(self.model_save_path)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"🎉 New best model saved at epoch {epoch} with cl_loss: {avg_cl_loss:.6f}")

        print(f"Epoch {epoch} | avg total loss: {avg_total_loss:.6f} | avg rq loss: {avg_rq_loss:.6f} | avg cl loss: {avg_cl_loss:.6f} | best cl loss: {self.best_cl_loss:.6f}")


    def fit(self, epochs: int):
        for epoch in range(1, epochs + 1):
            self.train_epoch(epoch)


if __name__ == "__main__":
    
    args = parse_args()
    # 创建保存目录
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # 打印配置
    print("Training CRQ-VAE with args:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # 加载 embedding，自动获取维度
    print(f"\nLoading POI embeddings from {args.embedding_path}...")
    poi_ids, poi_embeddings, NUM_ITEMS, EMBED_DIM = load_poi_embeddings(args.embedding_path)
    print(f"Auto-detected: NUM_ITEMS = {NUM_ITEMS}, EMBED_DIM = {EMBED_DIM}")

    # 验证是否 L2 归一化（允许微小误差）
    norms = torch.norm(poi_embeddings, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5):
        print("⚠️ Warning: Embeddings are NOT L2-normalized! Normalizing now for cosine similarity.")
        poi_embeddings = F.normalize(poi_embeddings, p=2, dim=1)
    else:
        print("✅ Embeddings are L2-normalized.")

    # 加载交互矩阵
    pos_pairs_np, M = load_interaction_matrix(args.data_path, args.k_threshold)
    print(f"Loaded interaction matrix from {args.data_path}, shape: {M.shape}")
    print(f"Total pos pairs: {pos_pairs_np.shape[0]}")

    # 构建 pos_dict
    pos_dict = build_pos_dict(pos_pairs_np, NUM_ITEMS)

    # 初始化模型
    model = CRQVAE(
        in_dim=EMBED_DIM,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        quant_loss_weight=args.quant_loss_weight,
        beta=args.beta,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        use_sk=args.use_sk,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
        use_linear=args.use_linear,
        temperature=args.temperature,
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        embeddings=poi_embeddings,
        pos_dict=pos_dict,
        args=args
    )
    trainer.fit(epochs=args.epochs)