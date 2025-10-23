import torch
import pickle
from CRQVAE.crqvae import CRQVAE
import argparse
import os
import pandas as pd
import numpy as np

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


def get_quantization():
    
    args = parse_args()
    # 加载 embedding，自动获取维度
    print(f"\nLoading POI embeddings from {args.embedding_path}...")
    poi_ids, poi_embeddings, NUM_ITEMS, EMBED_DIM = load_poi_embeddings(args.embedding_path)
    print(f"Auto-detected: NUM_ITEMS = {NUM_ITEMS}, EMBED_DIM = {EMBED_DIM}")
    poi_embeddings = poi_embeddings.to(args.device)
    
    # 加载模型
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
    ckpt = args.model_save_path
    if not os.path.exists(ckpt):
        print("No checkpoint found. Run training first.")
        return
    # 加载权重
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(args.device)
    model.eval()
    
    # 批量推理（避免 OOM）
    batch_size = 128
    all_indices = []
    
    with torch.no_grad():
        for i in range(0, len(poi_embeddings), batch_size):
            batch_emb = poi_embeddings[i:i + batch_size]
            indices_batch = model.get_indices(batch_emb)  # Shape: [B, num_heads]
            all_indices.extend(indices_batch)
    
    # 构建 POI ID → 量化码字 映射
    poi_quantized = {}
    for idx, poi_id in enumerate(poi_ids):
        poi_quantized[poi_id] = list(all_indices[idx])  # tuple -> list
    
    # 示例：打印前5个
    print("\n🎯 POI ID → Quantized Code (M codebook indices):")
    for i, (poi_id, codes) in enumerate(poi_quantized.items()):
        if i >= 5:
            break
        print(f"POI {poi_id}: {codes}")
    
    # 保存结果到 CSV
    df = pd.DataFrame(list(poi_quantized.items()), columns=['pid', 'semitic_codes'])    
    df.to_csv("data/NYC/poi_semitic_codes.csv", index=False)

if __name__ == "__main__":
    get_quantization()