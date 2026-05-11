import pandas as pd
import numpy as np

# 读取两个文件
static_df = pd.read_csv('cohort_final_static.csv')
lstm_df = pd.read_csv('lstm_48h_l1_features.csv')

# 提取两个文件中的hadm_id
static_hadm_ids = static_df['hadm_id'].unique()
lstm_hadm_ids = lstm_df['hadm_id'].unique()

# 找出在static中但不在lstm中的hadm_id
missing_hadm_ids = set(static_hadm_ids) - set(lstm_hadm_ids)
print(f"需要补全的患者数量: {len(missing_hadm_ids)}")

if len(missing_hadm_ids) > 0:
    # 获取特征列名（排除hadm_id和hour）
    feature_cols = [col for col in lstm_df.columns if col not in ['hadm_id', 'hour']]

    # 创建补全数据
    complete_data = []

    # 对于每个缺失的hadm_id，创建48小时的数据
    for hadm_id in missing_hadm_ids:
        for hour in range(48):  # 0到47小时
            # 创建一行数据，所有特征值为0
            row = {'hadm_id': hadm_id, 'hour': hour}
            # 将所有特征列设为0
            for col in feature_cols:
                row[col] = 0.0
            complete_data.append(row)

    # 将补全数据转换为DataFrame
    complete_df = pd.DataFrame(complete_data)

    # 确保列顺序与原文件一致
    complete_df = complete_df[lstm_df.columns]

    # 合并原始数据和补全数据
    final_df = pd.concat([lstm_df, complete_df], ignore_index=True)

    # 按hadm_id和hour排序
    final_df = final_df.sort_values(['hadm_id', 'hour']).reset_index(drop=True)

    # 保存结果
    final_df.to_csv('lstm_48h_l1_features_complete.csv', index=False)
    print(f"补全完成！总数据量: {len(final_df)} 行")
    print(f"保存为: lstm_48h_l1_features_complete.csv")

    # 验证
    final_hadm_ids = final_df['hadm_id'].unique()
    print(f"补全后hadm_id数量: {len(final_hadm_ids)}")
    print(f"原始static hadm_id数量: {len(static_hadm_ids)}")

    # 检查是否所有患者都在
    if set(final_hadm_ids) == set(static_hadm_ids):
        print("✅ 所有患者数据已补全！")
    else:
        print("❌ 补全失败，仍有缺失患者")

    # 显示前几行补全的数据
    print("\n补全数据示例:")
    if len(missing_hadm_ids) > 0:
        sample_hadm_id = list(missing_hadm_ids)[0]
        sample_data = final_df[final_df['hadm_id'] == sample_hadm_id].head(5)
        print(sample_data)
else:
    print("✅ 无需补全，所有患者数据已存在")