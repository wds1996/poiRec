# build_poi_semantic_vectors.py
import pandas as pd
import numpy as np
import pickle
import ast
import argparse
import os

def normalize_vector(x):
    """L2 归一化，避免除零"""
    norm = np.linalg.norm(x)
    return x / (norm + 1e-8)

def parse_time_dict(x):
    """安全解析时间字典字符串"""
    if pd.isna(x) or x == '' or x == '{}':
        return {}
    try:
        return ast.literal_eval(x)
    except:
        return {}

def extract_time_features(time_dict):
    """提取时间行为特征：24维分布 + 3维统计量"""
    vec = np.zeros(24)
    for h in range(24):
        vec[h] = time_dict.get(h, 0)
    
    if vec.sum() == 0:
        return np.zeros(27)  # 24 + 3
    
    hist = vec / vec.sum()
    hours = np.arange(24)
    mean_hour = (hours * hist).sum()
    variance = ((hours - mean_hour) ** 2 * hist).sum()
    peak_hour = hours[np.argmax(hist)]
    
    return np.concatenate([
        hist,
        [mean_hour / 24.0,          # 归一化平均小时
         variance / (24.0**2),      # 归一化方差
         peak_hour / 24.0]          # 归一化峰值小时
    ])

def extract_time_features2(time_dict):
    """改进：傅里叶特征 + 多峰检测"""
    if not time_dict:
        return np.zeros(9)
    
    hours = np.array(list(time_dict.keys()))
    counts = np.array(list(time_dict.values()))
    total = counts.sum()
    
    if total == 0:
        return np.zeros(9)
    
    # 归一化权重（访问概率）
    weights = counts / total  # shape: (n_hours,)

    # === 1. 傅里叶特征（捕捉周期性）===
    fourier_features = []
    max_freq = 4  # 可捕捉 24h, 12h, 8h, 6h 周期
    for k in range(1, max_freq + 1):
        sin_val = np.sum(weights * np.sin(2 * np.pi * k * hours / 24.0))
        cos_val = np.sum(weights * np.cos(2 * np.pi * k * hours / 24.0))
        fourier_features.extend([sin_val, cos_val])
    fourier_features = np.array(fourier_features)  # shape: (8,)

    # === 2. 多峰检测（是否有多个显著高峰）===
    # 简化策略：检查 top-2 小时是否都显著高于平均
    sorted_items = sorted(time_dict.items(), key=lambda item: item[1], reverse=True)
    if len(sorted_items) >= 2:
        top1_count = sorted_items[0][1]
        top2_count = sorted_items[1][1]
        avg_count = total / 24.0
        # 若 top2 也明显高于平均，则视为多峰
        multi_peak = 1.0 if (top2_count > 1.2 * avg_count) else 0.0
    else:
        multi_peak = 0.0

    return np.concatenate([fourier_features, [multi_peak]])  # shape: (9,)


def latlon_to_3d(lat, lon):
    """经纬度转3D笛卡尔坐标（单位向量）"""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    return np.array([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad)
    ])

def main(csv_path, category_pkl, cf_pkl, output_dir):
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载数据
    print("Loading POI data...")
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
    
    # 3. 加载预计算的 category 向量
    print("Loading pre-computed category embeddings...")
    with open(category_pkl, 'rb') as f:
        category_to_embedding = pickle.load(f)
    
    # 获取 category 向量维度
    sample_cat_vec = next(iter(category_to_embedding.values()))
    cat_dim = len(sample_cat_vec)
    print(f"Category vector dimension: {cat_dim}")
    
    # 4. 加载 CF 向量
    print("Loading CF embeddings...")
    with open(cf_pkl, 'rb') as f:
        cf_embeddings = pickle.load(f)
    
    # 获取 CF 向量维度
    sample_cf_vec = next(iter(cf_embeddings.values()))
    cf_dim = len(sample_cf_vec)
    print(f"CF vector dimension: {cf_dim}")
    
    # 5. 生成完整 POI 向量（L2 归一化融合）
    print("Building full POI vectors with L2 normalization...")
    full_vectors = []
    valid_pids = []

    for _, row in df.iterrows():
        try:
            # CF 向量（缺失用零向量）
            cf_vec = cf_embeddings.get(row['pid'], np.zeros(cf_dim))
            cf_vec = normalize_vector(cf_vec)
           
            # Category 向量（缺失用零向量，不再跳过）
            cat_vec = category_to_embedding.get(row['category'], np.zeros(cat_dim))
            cat_vec = normalize_vector(cat_vec)
            
            # 地理空间向量 (3D，已是单位向量，但仍归一化确保数值稳定)
            spatial_vec = latlon_to_3d(row['latitude'], row['longitude'])
            spatial_vec = normalize_vector(spatial_vec)
            
            # 时间行为向量 (27D)
            # time_dict = parse_time_dict(row['visit_time_and_count'])
            # time_vec = extract_time_features(time_dict)
            # time_vec = normalize_vector(time_vec)

            # 时间行为向量 (9D: 8 Fourier + 1 multi-peak)
            time_dict = parse_time_dict(row['visit_time_and_count'])
            time_vec = extract_time_features2(time_dict)
            time_vec = normalize_vector(time_vec)
            
            # 拼接所有归一化后的子向量
            concatenated = np.concatenate([cf_vec, cat_vec, spatial_vec, time_vec])
            
            # 整体 L2 归一化（关键！适配相似度聚类）
            full_vec = normalize_vector(concatenated)
            
            full_vectors.append(full_vec)
            valid_pids.append(row['pid'])
            
        except Exception as e:
            print(f"Error processing pid {row['pid']}: {e}")
            continue
    
    full_vectors = np.array(full_vectors)
    print(f"Generated vectors for {len(full_vectors)} POIs")
    print(f"Total vector dimension: {full_vectors.shape[1]}")
    
    # 6. 保存结果
    poi_vector_dict = dict(zip(valid_pids, full_vectors))
    output_path = os.path.join(output_dir, "poi_Emb_dict.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(poi_vector_dict, f)
    
    print(f"Done! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build POI semantic vectors for similarity-based RQ-VAE")
    parser.add_argument("--csv_path", default="data/NYC/poi_info.csv", help="Path to POI CSV file")
    parser.add_argument("--category_pkl", default="data/NYC/category_to_embedding.pkl", help="Path to category_to_embedding.pkl")
    parser.add_argument("--cf_pkl", default="data/NYC/cf_emb.pkl", help="Path to CF_embedding.pkl")
    parser.add_argument("--output_dir", default="data/NYC", help="Output directory")

    args = parser.parse_args()
    main(args.csv_path, args.category_pkl, args.cf_pkl, args.output_dir)