"""
数据集检查工具 - 列出所有 episode 信息，帮助定位废数据。

功能：
1. 打印所有 episode 的帧数、任务、起始结束帧等信息
2. 自动检测异常 episode（帧数异常、全零action等）
3. 支持导出检查报告

用法：
    python -m scripts.data_process.inspect_dataset \
        --data_root D:/VLA/data/simu_data

    或在代码中:
    from scripts.data_process.inspect_dataset import inspect_dataset
    inspect_dataset("D:/VLA/data/simu_data")
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def print_episodes_summary(dataset: LeRobotDataset) -> pd.DataFrame:
    """打印所有 episode 摘要信息，返回 DataFrame。

    Args:
        dataset: 已加载的 LeRobotDataset

    Returns:
        包含每个 episode 信息的 DataFrame
    """
    meta = dataset.meta
    episodes = meta.episodes
    if episodes is None:
        print("无法读取 episodes 元数据")
        return pd.DataFrame()

    records = []
    for i in range(meta.total_episodes):
        ep = episodes[i]
        frame_from = int(ep.get("index_from", -1))
        frame_to = int(ep.get("index_to", -1))
        frame_count = frame_to - frame_from + 1
        task = ep.get("task", "")
        task_idx = ep.get("task_index", -1)
        try:
            task_idx = int(task_idx)
        except (ValueError, TypeError):
            pass

        records.append({
            "episode": i,
            "frames": frame_count,
            "frame_from": frame_from,
            "frame_to": frame_to,
            "task": task,
            "task_index": task_idx,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("无 episode 数据")
        return df

    print("=" * 80)
    print(f"数据集: {meta.repo_id}")
    print(f"总 episodes: {meta.total_episodes}  总帧数: {meta.total_frames}")
    print(f"Robot: {meta.robot_type}  FPS: {meta.fps}")
    print(f"Features: {list(meta.features.keys())}")
    print("=" * 80)
    print()

    col_widths = {
        "episode": 8,
        "frames": 8,
        "frame_from": 10,
        "frame_to": 10,
        "task_index": 10,
    }
    header = (
        f"{'episode':>8}  {'frames':>8}  {'from':>10}  {'to':>10}"
        f"  {'task_idx':>10}  task"
    )
    print(header)
    print("-" * len(header))

    for _, row in df.iterrows():
        task_text = str(row["task"])[:60]
        print(
            f"{int(row['episode']):>8}  {int(row['frames']):>8}  "
            f"{int(row['frame_from']):>10}  {int(row['frame_to']):>10}"
            f"  {row['task_index']:>10}  {task_text}"
        )

    print()
    stats = df["frames"].describe()
    print(f"帧数统计: min={int(stats['min'])}, max={int(stats['max'])}, "
          f"mean={stats['mean']:.1f}, median={df['frames'].median():.0f}, "
          f"std={stats['std']:.1f}")

    return df


def find_suspicious_episodes(dataset: LeRobotDataset) -> list[int]:
    """自动检测可疑 episode（可能为废数据）。

    检测规则：
    1. 帧数明显少于平均值（少于均值 50%）
    2. 帧数为 0 或 1
    3. action 全为零（可能是录制错误）

    Args:
        dataset: 已加载的 LeRobotDataset

    Returns:
        可疑 episode 索引列表
    """
    meta = dataset.meta
    episodes = meta.episodes

    if episodes is None:
        return []

    frame_counts = []
    for i in range(meta.total_episodes):
        ep = episodes[i]
        frame_from = int(ep.get("index_from", 0))
        frame_to = int(ep.get("index_to", 0))
        frame_counts.append(frame_to - frame_from + 1)

    median_frames = np.median(frame_counts) if frame_counts else 0
    suspicious = []

    for i in range(meta.total_episodes):
        fc = frame_counts[i]
        reasons = []

        if fc <= 1:
            reasons.append("empty/single-frame")
        elif median_frames > 0 and fc < median_frames * 0.5:
            reasons.append(f"too few frames ({fc} vs median {median_frames:.0f})")

        if reasons:
            suspicious.append(i)
            print(f"[SUSPICIOUS] Episode {i}: frames={fc}, reasons: {', '.join(reasons)}")

    # 检查全零 action（采样检查前100帧）
    if not suspicious:
        try:
            for i in range(meta.total_episodes):
                sample = dataset[i]  # 取第一帧
                if sample is not None and "action" in sample:
                    action = np.asarray(sample["action"])
                    if np.allclose(action, 0) and frame_counts[i] > 2:
                        suspicious.append(i)
                        print(f"[SUSPICIOUS] Episode {i}: all-zero action during full episode")
        except Exception:
            pass

    if not suspicious:
        print("未发现明显可疑 episode")

    return suspicious


def inspect_dataset(data_root: str, repo_id: str = "simu_data") -> LeRobotDataset:
    """加载并检查数据集。

    Args:
        data_root: 数据集根目录
        repo_id: 数据集 ID

    Returns:
        加载的 LeRobotDataset 对象
    """
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {root}")
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"数据集 info.json 不存在: {info_path}")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        revision="local",
    )

    df = print_episodes_summary(dataset)
    print()
    find_suspicious_episodes(dataset)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="检查 LeRobot 数据集，列出所有 episode 信息"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="D:/VLA/data/simu_data",
        help="数据集根目录",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="simu_data",
        help="数据集 ID",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="导出 episode 摘要到 JSON 文件",
    )

    args = parser.parse_args()

    dataset = inspect_dataset(args.data_root, repo_id=args.repo_id)

    if args.export:
        meta = dataset.meta
        episodes = meta.episodes
        export_data = []
        if episodes is not None:
            for i in range(meta.total_episodes):
                ep = episodes[i]
                frame_from = int(ep.get("index_from", -1))
                frame_to = int(ep.get("index_to", -1))
                export_data.append({
                    "episode": i,
                    "frames": frame_to - frame_from + 1,
                    "frame_from": frame_from,
                    "frame_to": frame_to,
                    "task": str(ep.get("task", "")),
                    "task_index": int(ep.get("task_index", -1)),
                })
        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"\n摘要已导出到: {args.export}")


if __name__ == "__main__":
    main()
