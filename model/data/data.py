#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_timeseries_21d_with_abnormal_time.py

生成21天窗口期聚合特征，并添加"首次异常值时间"特征
Outputs: timeseries_21d_agg.csv

Usage:
    python build_timeseries_21d_with_abnormal_time.py --data_dir /path/to/csv --out_dir /path/to/out
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from tqdm import tqdm
import math
import warnings
from collections import defaultdict

# -------------------------
# CONFIGURABLE PARAMETERS
# -------------------------
DEFAULT_CONFIG = {
    # expected CSV filenames (in data_dir)
    "admissions": "admissions.csv",
    "patients": "patients.csv",
    "labevents": "labevents.csv",

    # outcome window (days)
    "outcome_days": 21,

    # 21-day window for full-aggregation
    "window_days": 21,

    # streaming chunk sizes
    "chunksize_lab": 500000,

    # whether to compute median by storing values
    "compute_median": True,
    "median_store_limit": int(5e6),

    # abnormal flag patterns (case insensitive)
    "abnormal_flag_patterns": ["abnormal", "high", "low", "elevated", "decreased", "positive"],

    # CSV writing options
    "csv_compression": None
}


# -------------------------
# Helpers
# -------------------------
def ensure_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def safe_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')


def is_abnormal_flag(flag_str, config):
    """Check if flag indicates abnormal value"""
    if pd.isna(flag_str):
        return False

    flag_lower = str(flag_str).lower()

    # Check against abnormal patterns
    for pattern in config['abnormal_flag_patterns']:
        if pattern.lower() in flag_lower:
            return True

    return False


# -------------------------
# Build 21-day aggregates with abnormal time
# -------------------------
def build_21d_aggregates_with_abnormal_time(labevents_path, cohort_df, config):
    """
    构建21天窗口期聚合特征，包括首次异常值时间

    Args:
        labevents_path: labevents.csv路径
        cohort_df: 队列DataFrame
        config: 配置字典

    Returns:
        DataFrame with aggregated statistics including first_abnormal_time
    """
    print("Aggregating labs across full 21-day window with abnormal time detection ...")

    # 准备hadm_id集合和时间映射
    hadm_set = set(cohort_df['hadm_id'].astype(str).tolist())
    adm_time_map = cohort_df.set_index('hadm_id')['admittime'].to_dict()
    window_seconds = config['window_days'] * 86400

    # 初始化存储结构
    stats = defaultdict(lambda: {
        'count': 0,
        'sum': 0.0,
        'sum_sq': 0.0,
        'min': float('inf'),
        'max': float('-inf'),
        'first_time': None,
        'first_val': None,
        'last_time': None,
        'last_val': None,
        'first_abnormal_time': None  # 新增：首次异常时间
    })

    # 用于中位数计算的值存储
    store_values = config['compute_median']
    values_storage = defaultdict(list) if store_values else None
    total_values_stored = 0

    # 流式处理labevents
    chunksize = config['chunksize_lab']
    chunk_iter = pd.read_csv(labevents_path, dtype=str, chunksize=chunksize)

    for chunk_idx, chunk in enumerate(tqdm(chunk_iter, desc="Processing labevents")):
        # 列名检测（处理不同版本的MIMIC-IV）
        col_mapping = {}
        for col in chunk.columns:
            col_lower = col.lower()
            if col_lower == 'hadm_id':
                col_mapping['hadm_id'] = col
            elif col_lower in ['itemid', 'item_id']:
                col_mapping['itemid'] = col
            elif col_lower in ['valuenum', 'value']:
                col_mapping['valuenum'] = col
            elif col_lower in ['charttime', 'chart_time']:
                col_mapping['charttime'] = col
            elif col_lower == 'flag':
                col_mapping['flag'] = col

        # 检查必要列是否存在
        required_cols = ['hadm_id', 'itemid', 'valuenum', 'charttime']
        missing_cols = [col for col in required_cols if col not in col_mapping]
        if missing_cols:
            raise ValueError(f"Missing required columns in labevents: {missing_cols}")

        # 筛选队列中的hadm_id
        chunk = chunk[chunk[col_mapping['hadm_id']].astype(str).isin(hadm_set)]
        if chunk.empty:
            continue

        # 数据类型转换
        chunk = chunk.copy()
        chunk[col_mapping['itemid']] = pd.to_numeric(
            chunk[col_mapping['itemid']], errors='coerce'
        ).astype('Int64')

        chunk[col_mapping['valuenum']] = pd.to_numeric(
            chunk[col_mapping['valuenum']], errors='coerce'
        )

        chunk[col_mapping['charttime']] = safe_to_datetime(
            chunk[col_mapping['charttime']]
        )

        # 移除缺失值
        chunk = chunk.dropna(subset=[
            col_mapping['itemid'],
            col_mapping['valuenum'],
            col_mapping['charttime']
        ])

        if chunk.empty:
            continue

        # 添加flag列（如果存在）
        has_flag = 'flag' in col_mapping
        if has_flag:
            chunk['is_abnormal'] = chunk[col_mapping['flag']].apply(
                lambda x: is_abnormal_flag(x, config)
            )

        # 处理每一行数据
        for _, row in chunk.iterrows():
            hadm_id = str(row[col_mapping['hadm_id']])
            itemid = int(row[col_mapping['itemid']])
            valuenum = float(row[col_mapping['valuenum']])
            charttime = row[col_mapping['charttime']]

            # 检查是否在21天窗口期内
            adm_time = adm_time_map.get(hadm_id)
            if adm_time is None:
                continue

            delta_seconds = (charttime - adm_time).total_seconds()
            if delta_seconds < 0 or delta_seconds > window_seconds:
                continue

            # 更新统计信息
            key = (hadm_id, itemid)
            stat = stats[key]

            # 基础统计
            stat['count'] += 1
            stat['sum'] += valuenum
            stat['sum_sq'] += valuenum * valuenum
            stat['min'] = min(stat['min'], valuenum)
            stat['max'] = max(stat['max'], valuenum)

            # 首次和末次值
            if stat['first_time'] is None or charttime < stat['first_time']:
                stat['first_time'] = charttime
                stat['first_val'] = valuenum

            if stat['last_time'] is None or charttime > stat['last_time']:
                stat['last_time'] = charttime
                stat['last_val'] = valuenum

            # 首次异常时间（如果存在flag且为异常）
            if has_flag and row.get('is_abnormal', False):
                if (stat['first_abnormal_time'] is None or
                        charttime < stat['first_abnormal_time']):
                    stat['first_abnormal_time'] = charttime

            # 存储值用于中位数计算
            if store_values and values_storage is not None:
                if total_values_stored < config['median_store_limit']:
                    values_storage[key].append(valuenum)
                    total_values_stored += 1
                elif store_values:
                    print(f"Warning: Reached median store limit ({config['median_store_limit']}). "
                          f"Stopping median value storage.")
                    store_values = False
                    values_storage = None

        # 定期清理内存（每处理10个chunk）
        if chunk_idx % 10 == 0:
            del chunk
            import gc
            gc.collect()

    print(f"Processed {len(stats)} unique (hadm_id, itemid) combinations")
    print(f"Total values stored for median: {total_values_stored}")

    # 构建结果DataFrame
    rows = []
    for (hadm_id, itemid), stat in stats.items():
        if stat['count'] == 0:
            continue

        # 计算统计量
        mean_val = stat['sum'] / stat['count'] if stat['count'] > 0 else np.nan

        # 计算标准差
        if stat['count'] > 1:
            variance = (stat['sum_sq'] / stat['count']) - (mean_val ** 2)
            std_val = math.sqrt(max(0, variance))
        else:
            std_val = 0.0

        # 计算中位数（如果启用了值存储）
        median_val = None
        if config['compute_median'] and values_storage is not None:
            key = (hadm_id, itemid)
            if key in values_storage and values_storage[key]:
                median_val = float(np.median(values_storage[key]))

        # 构建行数据
        row_data = {
            'hadm_id': hadm_id,
            'itemid': itemid,
            'count': stat['count'],
            'mean': mean_val,
            'std': std_val,
            'min': stat['min'] if stat['min'] != float('inf') else np.nan,
            'max': stat['max'] if stat['max'] != float('-inf') else np.nan,
            'first_time': stat['first_time'],
            'first_val': stat['first_val'],
            'last_time': stat['last_time'],
            'last_val': stat['last_val'],
            'first_abnormal_time': stat['first_abnormal_time'],
            'median': median_val
        }

        rows.append(row_data)

    # 创建DataFrame
    df_agg = pd.DataFrame(rows)

    # 按hadm_id和itemid排序
    df_agg = df_agg.sort_values(['hadm_id', 'itemid']).reset_index(drop=True)

    print(f"Generated {len(df_agg)} aggregated rows with abnormal time detection")
    return df_agg


# -------------------------
# 新增函数：提取首次异常时间单独表
# -------------------------
def extract_first_abnormal_times(labevents_path, cohort_df, config):
    """
    提取每个患者每个指标的首次异常时间（独立函数，用于验证或单独使用）

    Returns:
        DataFrame with columns: hadm_id, itemid, first_abnormal_time
    """
    print("Extracting first abnormal times for each patient and itemid ...")

    hadm_set = set(cohort_df['hadm_id'].astype(str).tolist())
    adm_time_map = cohort_df.set_index('hadm_id')['admittime'].to_dict()
    window_seconds = config['window_days'] * 86400

    # 存储首次异常时间
    abnormal_times = {}

    chunksize = config['chunksize_lab']
    chunk_iter = pd.read_csv(labevents_path, dtype=str, chunksize=chunksize)

    for chunk in tqdm(chunk_iter, desc="Extracting abnormal times"):
        # 检测列名
        col_mapping = {}
        for col in chunk.columns:
            col_lower = col.lower()
            if col_lower == 'hadm_id':
                col_mapping['hadm_id'] = col
            elif col_lower in ['itemid', 'item_id']:
                col_mapping['itemid'] = col
            elif col_lower in ['charttime', 'chart_time']:
                col_mapping['charttime'] = col
            elif col_lower == 'flag':
                col_mapping['flag'] = col

        if 'flag' not in col_mapping:
            print("Warning: No 'flag' column found in labevents. Cannot extract abnormal times.")
            return pd.DataFrame(columns=['hadm_id', 'itemid', 'first_abnormal_time'])

        # 筛选队列患者
        chunk = chunk[chunk[col_mapping['hadm_id']].astype(str).isin(hadm_set)]
        if chunk.empty:
            continue

        # 数据类型转换
        chunk = chunk.copy()
        chunk[col_mapping['itemid']] = pd.to_numeric(
            chunk[col_mapping['itemid']], errors='coerce'
        ).astype('Int64')

        chunk[col_mapping['charttime']] = safe_to_datetime(
            chunk[col_mapping['charttime']]
        )

        chunk['is_abnormal'] = chunk[col_mapping['flag']].apply(
            lambda x: is_abnormal_flag(x, config)
        )

        # 只保留异常记录
        abnormal_chunk = chunk[chunk['is_abnormal']].copy()

        if abnormal_chunk.empty:
            continue

        # 处理异常记录
        for _, row in abnormal_chunk.iterrows():
            hadm_id = str(row[col_mapping['hadm_id']])
            itemid = int(row[col_mapping['itemid']])
            charttime = row[col_mapping['charttime']]

            # 检查时间窗口
            adm_time = adm_time_map.get(hadm_id)
            if adm_time is None:
                continue

            delta_seconds = (charttime - adm_time).total_seconds()
            if delta_seconds < 0 or delta_seconds > window_seconds:
                continue

            # 更新首次异常时间
            key = (hadm_id, itemid)
            if key not in abnormal_times or charttime < abnormal_times[key]:
                abnormal_times[key] = charttime

    # 转换为DataFrame
    rows = []
    for (hadm_id, itemid), abnormal_time in abnormal_times.items():
        rows.append({
            'hadm_id': hadm_id,
            'itemid': itemid,
            'first_abnormal_time': abnormal_time
        })

    df_abnormal = pd.DataFrame(rows)
    print(f"Extracted {len(df_abnormal)} first abnormal time records")

    return df_abnormal


# -------------------------
# Orchestration
# -------------------------
def run_pipeline(data_dir, out_dir, config):
    """运行完整的数据处理流水线"""
    # 准备文件路径
    admissions_path = os.path.join(data_dir, config['admissions'])
    patients_path = os.path.join(data_dir, config['patients'])
    labevents_path = os.path.join(data_dir, config['labevents'])

    for path in [admissions_path, patients_path, labevents_path]:
        ensure_file_exists(path)

    os.makedirs(out_dir, exist_ok=True)

    # 1. 加载队列数据（简化版本，假设已有cohort_candidates.csv）
    # 如果没有，可以从admissions和patients构建
    print("Loading cohort data ...")

    # 检查是否已有cohort文件
    cohort_path = os.path.join(out_dir, 'cohort_candidates.csv')
    if os.path.exists(cohort_path):
        cohort_df = pd.read_csv(cohort_path)
        cohort_df['admittime'] = safe_to_datetime(cohort_df['admittime'])
        print(f"Loaded existing cohort: {len(cohort_df)} patients")
    else:
        # 简单构建队列（实际应用中可能需要更复杂的筛选）
        admissions = pd.read_csv(admissions_path, low_memory=False)
        patients = pd.read_csv(patients_path, low_memory=False)

        # 简单的队列构建：成年患者，首次住院
        admissions['admittime'] = safe_to_datetime(admissions['admittime'])

        # 合并年龄信息
        if 'anchor_age' in patients.columns:
            patients['anchor_age'] = pd.to_numeric(patients['anchor_age'], errors='coerce')
            admissions = admissions.merge(
                patients[['subject_id', 'anchor_age']],
                on='subject_id',
                how='left'
            )
            admissions['age'] = admissions['anchor_age']

        # 筛选成年患者，取首次住院
        admissions = admissions[admissions['age'] >= 18].copy()
        admissions = admissions.sort_values(['subject_id', 'admittime'])
        cohort_df = admissions.groupby('subject_id', as_index=False).first()

        print(f"Built cohort: {len(cohort_df)} patients")

    # 2. 构建21天聚合特征（包含首次异常时间）
    df_21d = build_21d_aggregates_with_abnormal_time(
        labevents_path,
        cohort_df,
        config
    )

    # 3. 保存为CSV格式
    csv_path = os.path.join(out_dir, 'timeseries_21d_agg.csv')

    # 配置CSV保存选项
    csv_kwargs = {'index': False}
    if config['csv_compression']:
        csv_kwargs['compression'] = config['csv_compression']

    df_21d.to_csv(csv_path, **csv_kwargs)
    print(f"Saved 21-day aggregates to: {csv_path}")
    print(f"File size: {os.path.getsize(csv_path) / (1024 * 1024):.2f} MB")

    # 4. 可选：单独保存首次异常时间表
    abnormal_times_df = extract_first_abnormal_times(labevents_path, cohort_df, config)
    if not abnormal_times_df.empty:
        abnormal_path = os.path.join(out_dir, 'first_abnormal_times.csv')
        abnormal_times_df.to_csv(abnormal_path, index=False)
        print(f"Saved first abnormal times to: {abnormal_path}")

    # 5. 输出统计信息
    print("\n=== Dataset Statistics ===")
    print(f"Total unique patients (hadm_id): {df_21d['hadm_id'].nunique()}")
    print(f"Total unique lab items: {df_21d['itemid'].nunique()}")
    print(f"Total records: {len(df_21d)}")

    # 异常时间统计
    abnormal_count = df_21d['first_abnormal_time'].notna().sum()
    abnormal_percent = abnormal_count / len(df_21d) * 100 if len(df_21d) > 0 else 0
    print(f"Records with abnormal flag: {abnormal_count} ({abnormal_percent:.1f}%)")

    # 样本数据预览
    print("\n=== Sample Data (first 5 rows) ===")
    print(df_21d.head().to_string())

    # 列信息
    print("\n=== Column Information ===")
    for col in df_21d.columns:
        non_null = df_21d[col].notna().sum()
        null_pct = (1 - non_null / len(df_21d)) * 100 if len(df_21d) > 0 else 100
        dtype = df_21d[col].dtype
        print(f"{col:25} {dtype:15} Non-null: {non_null:8} ({null_pct:5.1f}% null)")

    return df_21d


# -------------------------
# CLI / Run options
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build 21-day timeseries aggregates with first abnormal time"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory containing MIMIC CSV files'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default=None,
        help='Output directory for results'
    )
    parser.add_argument(
        '--median_store_limit',
        type=int,
        default=None,
        help='Override median store limit (default: 5,000,000)'
    )
    parser.add_argument(
        '--no_median',
        action='store_true',
        help='Disable median computation to save memory'
    )
    parser.add_argument(
        '--abnormal_patterns',
        type=str,
        default=None,
        help='Comma-separated list of abnormal flag patterns (e.g., "abnormal,high,low")'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 设置路径
    if args.data_dir is None or args.out_dir is None:
        # 默认路径（可以修改）
        DATA_DIR = "../../data"  # 修改为您的数据目录
        OUT_DIR = "../../data/output"  # 修改为输出目录

        if not os.path.exists(DATA_DIR):
            print(f"Error: Data directory does not exist: {DATA_DIR}")
            sys.exit(1)
    else:
        DATA_DIR = args.data_dir
        OUT_DIR = args.out_dir

    # 更新配置
    config = DEFAULT_CONFIG.copy()

    if args.median_store_limit is not None:
        config['median_store_limit'] = args.median_store_limit

    if args.no_median:
        config['compute_median'] = False

    if args.abnormal_patterns:
        config['abnormal_flag_patterns'] = [
            p.strip() for p in args.abnormal_patterns.split(',')
        ]

    # 运行流水线
    print("=" * 60)
    print("21-Day Aggregates with First Abnormal Time Detection")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Abnormal patterns: {config['abnormal_flag_patterns']}")
    print(f"Compute median: {config['compute_median']}")
    print(f"Median store limit: {config['median_store_limit']:,}")
    print("=" * 60)

    try:
        df_result = run_pipeline(DATA_DIR, OUT_DIR, config)
        print("\n✓ Processing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
