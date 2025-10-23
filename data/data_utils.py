import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import csv
from openlocationcode import openlocationcode as olc
import json

class data_filter:
    def __init__(self, dataflod, min_user_interactions, min_poi_interactions):
        self.dataflod = dataflod
        self.min_user_interactions = min_user_interactions
        self.min_poi_interactions = min_poi_interactions

    def get_pluscode(self, latitude, longitude):
        if pd.isna(latitude) or pd.isna(longitude):
            return "INVALID"
        try:
            code = olc.encode(float(latitude), float(longitude))
            return code[:6]
        except:
            return "INVALID"

    def format_time(self, row):
        # 示例输入: "Mon Oct 19 20:06:23 +0800 2025" — 但 tz_offset 已单独给出
        # 忽略字符串中的时区，只用 tz_offset
        time_str = row['time']
        # 拆分出日期部分和年份
        parts = time_str.split()
        # 重建为 "Oct 19 20:06:23 2025"
        clean_time_str = f"{parts[1]} {parts[2]} {parts[3]} {parts[5]}"
        naive_dt = datetime.strptime(clean_time_str, "%b %d %H:%M:%S %Y")
        
        # 使用 tz_offset（分钟）创建时区
        tz = timezone(timedelta(minutes=int(row['tz_offset'])))
        localized_dt = naive_dt.replace(tzinfo=tz)
        return localized_dt.strftime("%Y-%m-%d %H:%M")

    def save_mapping(self, mapping, file_path):
        with open(file_path, "w", newline="") as uidfile:
            writer = csv.writer(uidfile)
            writer.writerow(["original_uid", "new_uid"])
            for original_uid, new_uid in mapping.items():
                writer.writerow([original_uid, new_uid])

    def filter_data(self):
        os.makedirs(f"data/{self.dataflod}", exist_ok=True)
        
        input_file = f"data/{self.dataflod}.txt"
        df = pd.read_csv(input_file, delimiter="\t", header=None, 
                         names=['uid', 'pid', '_', 'category', 'latitude', 'longitude', 'tz_offset', 'time'])

        # 转换经纬度为数值
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])

        # 过滤 POI
        poi_counts = df['pid'].value_counts()
        valid_pids = poi_counts[poi_counts >= self.min_poi_interactions].index
        df = df[df['pid'].isin(valid_pids)]

        # 过滤用户
        user_counts = df['uid'].value_counts()
        valid_uids = user_counts[user_counts >= self.min_user_interactions].index
        df = df[df['uid'].isin(valid_uids)].copy()

        # 生成 region
        df["region"] = df.apply(lambda row: self.get_pluscode(row['latitude'], row['longitude']), axis=1)

        # 映射 ID
        uid_map, pid_map, cid_map, region_map = {}, {}, {}, {}
        df['new_uid'] = df['uid'].apply(lambda x: uid_map.setdefault(x, len(uid_map)))
        df['new_pid'] = df['pid'].apply(lambda x: pid_map.setdefault(x, len(pid_map)))
        df['new_cid'] = df['cid'].apply(lambda x: cid_map.setdefault(x, len(cid_map)))
        df['new_region'] = df['region'].apply(lambda x: region_map.setdefault(x, len(region_map)))
        df['formatted_time'] = df.apply(self.format_time, axis=1)

        # 保存
        output_file = f"data/{self.dataflod}/{self.dataflod}.csv"
        df[['new_uid', 'new_pid', 'category', 'new_cid', 'new_region', 'latitude', 'longitude', 'formatted_time']].to_csv(
            output_file, index=False, header=['uid', 'pid', 'cid', 'category', 'region', 'latitude', 'longitude', 'time']
        )

        self.save_mapping(uid_map, f"data/{self.dataflod}/uidmap.csv")
        self.save_mapping(pid_map, f"data/{self.dataflod}/pidmap.csv")
        self.save_mapping(cid_map, f"data/{self.dataflod}/cidmap.csv")

        print("处理完成！")



class data_split:
    def __init__(self, datafold, train_ratio=0.8):
        self.datafold = datafold
        self.train_ratio = train_ratio

    def remove_users_pois_test(self, df_train, df_test):
        # 仅保留测试集中在训练集中出现的用户和POI
        valid_users = set(df_train['uid'].unique())
        valid_pois = set(df_train['pid'].unique())
        df_test = df_test[
            df_test['uid'].isin(valid_users) &
            df_test['pid'].isin(valid_pois)
        ]
        return df_test

    def split_data(self):
        file_name = f"data/{self.datafold}/{self.datafold}.csv"
        df = pd.read_csv(file_name)

        # 确保包含所需列
        df = df[['uid', 'pid', 'cid', 'category', 'latitude', 'longitude', 'time']]
        
        # 转换时间并排序
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time').reset_index(drop=True)

        # 按比例划分
        train_size = int(self.train_ratio * len(df))
        train_df = df.iloc[:train_size].copy()
        test_df = df.iloc[train_size:].copy()

        # 过滤测试集
        test_df = self.remove_users_pois_test(train_df, test_df)

        # 为测试用户保留完整历史（用于序列模型）
        test_uids = test_df['uid'].unique()
        expanded_test_df = df[df['uid'].isin(test_uids)].copy()

        # 保存
        train_df.to_csv(f'data/{self.datafold}/train_data.csv', index=False)
        expanded_test_df.to_csv(f'data/{self.datafold}/test_data.csv', index=False)

        print(f"Split done: {len(train_df)} train, {len(test_df)} test interactions.")


class poi_info:
    def __init__(self, datafold):
        self.datafold = datafold

    def extract_poi_info(self):
        file_path = f'data/{self.datafold}/{self.datafold}.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"错误：{file_path} 文件未找到。")

        df = pd.read_csv(file_path)

        # 确保 time 列可解析
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])  # 移除无法解析的时间

        poi_info_data = []

        for pid, group in df.groupby('pid'):
            # 取第一条记录的元信息（假设同一 POI 的 category/region/经纬度一致）
            row0 = group.iloc[0]
            category = row0['category']
            region = row0['region']
            latitude = row0['latitude']
            longitude = row0['longitude']

            # 统计每小时访问次数
            hours = group['time'].dt.hour
            hour_counts = hours.value_counts().to_dict()  # {hour: count}

            # 可选：只保留 count > 1 的小时（按你的需求）
            filtered_hour_counts = {int(h): int(c) for h, c in hour_counts.items() if c > 1}

            # 按访问次数降序排序
            sorted_hour_counts = dict(
                sorted(filtered_hour_counts.items(), key=lambda x: x[1], reverse=True)
            )

            poi_info_data.append({
                'pid': pid,
                'category': category,
                'region': region,
                'latitude': latitude,
                'longitude': longitude,
                'visit_time_and_count': sorted_hour_counts
            })

        # 创建 DataFrame 并保存
        poi_info_df = pd.DataFrame(poi_info_data)
        output_path = f'data/{self.datafold}/poi_info.csv'
        poi_info_df.to_csv(output_path, index=False)

        print(f"成功创建 {output_path}，共 {len(poi_info_df)} 个 POI")


def generate_check_in_sequences(datafold, datafile):
    # 读取CSV文件到pandas DataFrame
    df = pd.read_csv(f"data/{datafold}/{datafile}.csv")
    # 确保时间列是datetime类型
    df['time'] = pd.to_datetime(df['time'])
    # 按uid分组并聚合数据
    def aggregate(group):
        group = group.sort_values('time')
        pid_list = group['pid'].tolist()
        return pd.Series({
            'pid_sequence': pid_list,
        })
    check_in_sequences = df.groupby('uid').apply(aggregate).reset_index()
    
    # 保存结果
    output_file = f"data/{datafold}/{datafile.split('_')[0]}_sequences.csv"
    check_in_sequences.to_csv(output_file, index=False)
    print(f"Check-in sequences saved to {output_file}")
