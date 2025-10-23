import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

# ===============================
# 1. 数据读取
# ===============================
df = pd.read_csv("data/NYC/NYC.csv")

# 构建用户序列 (pid 已经是连续编号)
user_seqs = df.groupby("uid")["pid"].apply(list).to_dict()
vocab_size = df["pid"].max() + 1
num_users = df["uid"].max() + 1
print(f"用户数: {num_users}, POI数: {vocab_size}")

# 划分训练/测试
train_seqs = {}
test_data = {}
for uid, seq in user_seqs.items():
    if len(seq) > 20:
        train_seqs[uid] = seq[:-1]   
        test_data[uid] = (seq[-20:])  

# ===============================
# 2. Dataset + Dataloader
# ===============================
SEQ_LEN = 20  # 每条数据长度

class POISeqDataset(Dataset):
    def __init__(self, user_seqs, seq_len=SEQ_LEN):
        self.data = []
        for seq in user_seqs.values():
            if len(seq) <= seq_len:  # 如果序列过短，直接跳过
                continue
            for i in range(0, len(seq) - seq_len):
                self.data.append((uid, seq[i:i+seq_len], seq[i+seq_len]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, seq, target = self.data[idx]
        return (
            torch.tensor(uid, dtype=torch.long),
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )

dataset = POISeqDataset(train_seqs, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# ===============================
# 3. Transformer 模型
# ===============================
class POITransformer(nn.Module):
    def __init__(self, num_users, vocab_size, embed_dim=64, n_heads=2, n_layers=2, hidden_dim=128, max_len=SEQ_LEN):
        super().__init__()
        self.embed_dim = embed_dim
        self.poi_embed = nn.Embedding(vocab_size, embed_dim)      # POI embedding (要提取的)
        self.pos_embed = nn.Embedding(max_len + 1, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.user_embed = nn.Embedding(num_users, embed_dim)      # User embedding (仅训练用)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, uids, seq):
        B, L = seq.shape
        device = seq.device
        poi_emb = self.poi_embed(seq)  # [B, L, D]
        positions = torch.arange(L + 1, device=device).unsqueeze(0)  # [1, L+1]
        pos_emb = self.pos_embed(positions)  # [1, L+1, D]
        user_emb = self.user_embed(uids).unsqueeze(1)  # [B, 1, D]
        x = torch.cat([user_emb, poi_emb], dim=1)  # [B, L+1, D]
        x = x + pos_emb  # [B, L+1, D]
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # 预测下一个 POI
        return out

# ===============================
# 4. 训练
# ===============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = POITransformer(
    num_users=num_users,
    vocab_size=vocab_size,
    embed_dim=64,
    n_heads=2,
    n_layers=2,
    hidden_dim=128,
    max_len=SEQ_LEN
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

def evaluate(model, test_data, k=1):
    model.eval()
    hits, recalls, total = 0, 0, 0
    with torch.no_grad():
        for uid, seq in test_data.items():
            if len(seq) < 2:
                continue
            inp, target = seq[:-1], seq[-1]
            pad_len = max(0, SEQ_LEN - len(inp))
            padded_seq = [0] * pad_len + inp[-SEQ_LEN:]
            seq_tensor = torch.tensor([padded_seq], dtype=torch.long, device=device)
            uid_tensor = torch.tensor([uid], dtype=torch.long, device=device)

            output = model(uid_tensor, seq_tensor)
            topk = torch.topk(output, k=k, dim=1).indices.squeeze(0).tolist()

            if target in topk:
                hits += 1
            recalls += (1 if target in topk else 0)
            total += 1
    hit_rate = hits / total if total > 0 else 0
    recall = recalls / total if total > 0 else 0
    return hit_rate, recall


EPOCHS = 30
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for uids, seqs, targets in dataloader:
        uids, seqs, targets = uids.to(device), seqs.to(device), targets.to(device)
        outputs = model(uids, seqs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

    if (epoch + 1) % 5 == 0:
        hit_rate_1, recall_1 = evaluate(model, test_data, k=1)
        hit_rate_5, recall_5 = evaluate(model, test_data, k=5)
        print(f"测试结果: HitRate@1 = {hit_rate_1:.4f}" f", Recall@1 = {recall_1:.4f}")
        print(f"测试结果: HitRate@5 = {hit_rate_5:.4f}" f", Recall@5 = {recall_5:.4f}")

# ===============================
# 6. 提取 POI embedding
# ===============================
poi_embeddings = model.poi_embed.weight.detach().cpu().numpy()
item_embeddings_dict = {pid: poi_embeddings[pid] for pid in range(vocab_size)}

with open("data/NYC/cf_emb.pkl", "wb") as f:
    pickle.dump(item_embeddings_dict, f)

print(f"保存完成: {len(item_embeddings_dict)} 个 POI embedding, 维度={poi_embeddings.shape[1]}")
