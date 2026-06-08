# 数据后处理工具使用指南

## 工具概览

```
data_process/
├── merge_datasets.py       # 多个 LeRobot v3.0 数据集合并（新增）
├── lerobot_to_hdf5.py      # LeRobot v3.0 → HDF5 格式转换
├── delete_episode.py       # 单条/批量 episode 删除
└── __init__.py             # 工具包初始化
```

> **环境**: `lerobot` conda，**工作目录**: `D:\VLA\kortex_code\collect_data`

---

## 1. 合并多个数据集

将两个或多个结构兼容的 LeRobot v3.0 数据集合并为一个数据集。

合并规则：
- episode 索引自动重编号（连续化）
- frame 索引自动重编号（全局连续）
- video 文件顺序复制并重排编号
- state/action 统计信息重新计算
- 自动校验所有数据集的 features 兼容性

### 合并两个数据集

```powershell
cd D:\VLA\kortex_code\collect_data

python data_process/merge_datasets.py --datasets D:/VLA/data/simu_data1 D:/VLA/data/simu_data2 --output_dir D:/VLA/data/simu_data_merged --yes
```

### 合并三个数据集（自定义 parquet 文件大小）

```powershell
cd D:\VLA\kortex_code\collect_data

python data_process/merge_datasets.py --datasets D:/VLA/data/simu_data1 D:/VLA/data/simu_data2 D:/VLA/data/simu_data3 --output_dir D:/VLA/data/simu_data_merged --max_data_file_size_mb 200 --yes
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--datasets` | 要合并的数据集根目录（空格分隔，至少2个） | 必填 |
| `--output_dir` | 合并后的输出目录 | 必填 |
| `--repo_id` | 合并数据集的 repo_id | `simu_data_merged` |
| `--max_data_file_size_mb` | 每个 parquet 文件最大 MB | 200 |
| `--yes` | 跳过确认（输出目录已存在时自动清空） | 否 |

---

## 2. 删除废数据

### 删除单个 episode（输出到新目录，保留原数据）

```powershell
cd D:\VLA\kortex_code\collect_data

python data_process/delete_episode.py --data_root D:/VLA/data/simu_data --episode 3 --output_dir D:/VLA/data/simu_data_cleaned --yes
```

### 批量删除多个 episode

```powershell
python data_process/delete_episode.py --data_root D:/VLA/data/simu_data --episodes 2 5 10 15 --output_dir D:/VLA/data/simu_data_cleaned --yes
```

### 原地删除（自动备份原数据到 `_backup` 目录）

```powershell
python data_process/delete_episode.py --data_root D:/VLA/data/simu_data --episode 3 --inplace --yes
```

---

## 3. 转换为 HDF5（供 ACT 训练）

```powershell
cd D:\VLA\kortex_code\collect_data

# 转换全部 episode
python data_process/lerobot_to_hdf5.py --lerobot_root D:/VLA/data/simu_data_merged --output_dir D:/VLA/data/simu_hdf5 --clean

# 只转换指定 episode
python data_process/lerobot_to_hdf5.py --lerobot_root D:/VLA/data/simu_data_merged --output_dir D:/VLA/data/simu_hdf5 --episodes 0 1 2
```

---

## 典型工作流

```powershell
cd D:\VLA\kortex_code\collect_data

# Step 1: 合并多个采集批次
python data_process/merge_datasets.py --datasets D:/VLA/data/simu_data1 D:/VLA/data/simu_data2 --output_dir D:/VLA/data/simu_data_merged --yes

# Step 2: 检查并删除异常 episode
python data_process/delete_episode.py --data_root D:/VLA/data/simu_data_merged --episode 5 --output_dir D:/VLA/data/simu_data_cleaned --yes

# Step 3: 转换为 HDF5 供训练
python data_process/lerobot_to_hdf5.py --lerobot_root D:/VLA/data/simu_data_cleaned --output_dir D:/VLA/data/simu_hdf5 --clean
```
