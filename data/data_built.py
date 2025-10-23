import pandas as pd
import csv
import json
import ast


def romove_users_pois_test(df_train, df_test):
        users_train = df_train['uid'].unique()
        df_test = df_test[df_test['uid'].isin(users_train)]
        users_test = df_test['uid'].unique()
        df_train = df_train[df_train['uid'].isin(users_test)]

        pois_train = df_train['pid'].unique()
        df_test = df_test[df_test['pid'].isin(pois_train)]
        return df_test

def dataset_split(datafold):
    file_name = f"POI_data/{datafold}/{datafold}.csv"
    df = pd.read_csv(file_name)

    df = df[['uid', 'pid', 'cid', 'category', 'latitude', 'longitude', 'time']] # uid,pid,cid,category,latitude,longitude,time,timestamp,normalized_time
    # 按照时间排序
    df = df.sort_values(by='time')
    # 计算80%数据的索引
    train_size = int(0.8 * len(df))
    # 将前80%作为训练集
    train_df = df[:train_size]
    # 将后20%作为测试集
    test_df = df[train_size:]
    # 移除测试集中不在训练集中的用户和POI
    test_df = romove_users_pois_test(train_df, test_df)
    # 将训练集和测试集合并
    new_df = pd.concat([train_df, test_df], ignore_index=True)
    # 获取测试集中所有需要保留的 uid
    test_uids = test_df['uid'].unique()
    # 过滤原始 df，只保留那些 Uid 出现在 test_df 中的记录
    expanded_df = new_df[new_df['uid'].isin(test_uids)]
    # 保存训练集和测试集
    train_df.to_csv(f'POI_data/{datafold}/train_data.csv', index=False)
    # 大模型需要长历史序列输入，故测试集使用扩展后的数据
    expanded_df.to_csv(f'POI_data/{datafold}/test_data.csv', index=False)


def checkin_sequence(file_path, max_length=50):
    # 读取CSV文件到pandas DataFrame
    df = pd.read_csv(file_path)
    # 确保时间列是datetime类型
    df['time'] = pd.to_datetime(df['time'])
    # 按uid分组并聚合数据
    def aggregate(group):
        pid_list = list(group['pid'])
        category_list = list(group['category'])
        time_list = list(group['time'])
        # 按时间排序
        # sorted_indices = sorted(range(len(time_list)), key=lambda i: time_list[i])
        # pid_list = [pid_list[i] for i in sorted_indices]
        # category_list = [category_list[i] for i in sorted_indices]
        # time_list = [time_list[i] for i in sorted_indices]
        
        pid_list = pid_list[:max_length] if len(pid_list) > max_length else pid_list
        category_list = category_list[:max_length] if len(category_list) > max_length else category_list
        time_list = time_list[:max_length] if len(time_list) > max_length else time_list

        return pd.Series({
            'pid_list': pid_list,
            'category_list': category_list,
            'time_list': [t.strftime('%Y-%m-%d %H:%M') for t in time_list], # 格式化时间
        })

    df_grouped = df.groupby('uid', group_keys=False).apply(aggregate, include_groups=False).reset_index()
    return df_grouped


def save_checkin2json(df, output_file_path):
    data = []
    for _, row in df.iterrows():
        uid = row['uid']
        pid_list = ast.literal_eval(row['pid_list']) if isinstance(row['pid_list'], str) else row['pid_list']
        category = ast.literal_eval(row['category_list']) if isinstance(row['category_list'], str) else row['category_list']
        time_seq = ast.literal_eval(row['time_list']) if isinstance(row['time_list'], str) else row['time_list']
        
        if len(pid_list) < 2:
            continue  # 跳过检查点少于2的用户

        record = {
            "uid": uid,
            "pid_list": pid_list[:-1],  # 除了最后一个POI
            "category": category[:-1],
            "time": time_seq[:-1],  # 除了最后一个时间
            "next_time": time_seq[-1],  # 最后一个时间作为next_time
            "target_pid": pid_list[-1]  # 最后一个POI作为目标
        }
        data.append(record)

    json_records = []
    for record in data:
        uid = record["uid"]
        pid_list = record["pid_list"]
        category = record["category"]
        time_seq = record["time"]
        next_time = record["next_time"]
        target_pid = record["target_pid"]

        # 构造 input 字符串
        # input_text = (
        #     f"The user{uid} has recently POI check-in records: {pid_list}, with corresponding check-in times: {time_seq}."
        # )

        # sequence = [
        #     f"poi{poi}" + f' (belong to {category[i]})' + f' at {time_seq[i]}, ' if i < len(pid_list) - 1 else
        #     f"poi{poi}" + f' (belong to {category[i]})' + f' at {time_seq[i]}.'
        #     for i, poi in enumerate(pid_list)
        # ]
        prompt = f"Below is the user's historical POI visit trajectory. Your task is to predict the next most likely POI number that the user will visit at current time."
        sequence = [
            f"POI{poi} <embedding>" + f' at {time_seq[i]}, ' if i < len(pid_list) - 1 else
            f"POI{poi} <embedding>" + f' at {time_seq[i]}.'
            for i, poi in enumerate(pid_list)
        ]

        # 构造 input 字符串
        input_text = f"User_{uid} visited: " + "".join(sequence) + f" Now is {next_time}, user_{uid} is likely to visit?"

        record = {
            "prompt": prompt,
            "input": input_text,
            "target": target_pid
        }
        json_records.append(record)


    with open(output_file_path, mode='w', encoding='utf-8') as file:
        json.dump(json_records, file, ensure_ascii=False, indent=2)

    print(f"转换完成，结果保存至: {output_file_path}")



datafold = 'NYC'
max_length = 50
dataset_split(datafold)

train_file_path = f'POI_data/{datafold}/train_data.csv'
test_file_path = f'POI_data/{datafold}/test_data.csv'

df_train_seq = checkin_sequence(train_file_path, max_length)
df_test_seq = checkin_sequence(test_file_path, max_length)

output_train_json = f'POI_data/{datafold}/train_data.json'
output_test_json = f'POI_data/{datafold}/test_data.json'

save_checkin2json(df_train_seq, output_train_json)
save_checkin2json(df_test_seq, output_test_json)
