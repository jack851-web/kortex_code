"""
删除指定 episode - 基于 LeRobot API 的 episode 删除工具。

功能：
1. 删除指定 episode 到新目录（安全，保留原始数据）
2. 原地删除并覆盖（谨慎使用）

核心 API: lerobot.datasets.dataset_tools.delete_episodes

用法：
    # 删除 episode 2, 5 到新目录
    python -m scripts.data_process.delete_episodes \
        --data_root D:/VLA/data/simu_data \
        --episodes  29 \
        --output_dir D:/VLA/data/simu_data_cleaned

    # 在原地删除（覆盖原数据）
    python -m scripts.data_process.delete_episodes \
        --data_root D:/VLA/data/simu_data \
        --episodes 3 \
        --inplace

    # 在代码中使用:
    from scripts.data_process.delete_episodes import delete_episodes_from_dataset
    delete_episodes_from_dataset("D:/VLA/data/simu_data", [3], "D:/VLA/data/simu_data_cleaned")
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes


def delete_episodes_from_dataset(
    data_root: str,
    episode_indices: list[int],
    output_dir: str,
    repo_id: str = "simu_data",
    new_repo_id: str | None = None,
) -> LeRobotDataset:
    """删除指定 episode 到新目录（保留原始数据）。

    Args:
        data_root: 原始数据集根目录
        episode_indices: 要删除的 episode 索引列表
        output_dir: 新数据集输出目录
        repo_id: 原始数据集 ID
        new_repo_id: 新数据集 ID（默认与原始相同）

    Returns:
        新数据集对象
    """
    os.environ["HF_HUB_OFFLINE"] = "1"

    root = Path(data_root)
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"数据集不存在: {root / 'meta' / 'info.json'}")

    out = Path(output_dir)

    if new_repo_id is None:
        new_repo_id = repo_id

    print(f"加载数据集: {data_root}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        revision="local",
    )

    total_before = dataset.meta.total_episodes
    total_frames_before = dataset.meta.total_frames
    print(f"当前: {total_before} episodes, {total_frames_before} frames")
    print(f"待删除 episode: {episode_indices}")
    print(f"输出目录: {out}")

    # 确认
    episodes_info = dataset.meta.episodes
    for idx in episode_indices:
        ep = episodes_info[idx]
        frame_from = int(ep.get("index_from", 0))
        frame_to = int(ep.get("index_to", 0))
        task = ep.get("task", "")
        print(f"  删除 episode {idx}: frames={frame_to - frame_from + 1}, task={task}")

    print(f"\n执行删除...")
    new_dataset = delete_episodes(
        dataset=dataset,
        episode_indices=episode_indices,
        output_dir=out,
        repo_id=new_repo_id,
    )

    total_after = new_dataset.meta.total_episodes
    total_frames_after = new_dataset.meta.total_frames
    print(f"\n完成!")
    print(f"  删除前: {total_before} episodes, {total_frames_before} frames")
    print(f"  删除后: {total_after} episodes, {total_frames_after} frames")
    print(f"  已删除: {total_before - total_after} episodes, "
          f"{total_frames_before - total_frames_after} frames")
    print(f"  新数据集: {out}")

    return new_dataset


def delete_episodes_inplace(
    data_root: str,
    episode_indices: list[int],
    repo_id: str = "simu_data",
) -> LeRobotDataset:
    """原地删除 episode（先用临时目录操作，再覆盖原目录）。

    Args:
        data_root: 数据集根目录
        episode_indices: 要删除的 episode 索引列表
        repo_id: 数据集 ID

    Returns:
        新数据集对象
    """
    root = Path(data_root)
    temp_dir = root.parent / f"{root.name}_temp_delete"

    print(f"=== 原地删除模式 ===")
    print(f"将在临时目录构建新数据集: {temp_dir}")
    print(f"构建完成后替换原目录")

    if temp_dir.exists():
        print(f"清理旧的临时目录: {temp_dir}")
        shutil.rmtree(temp_dir)

    try:
        new_dataset = delete_episodes_from_dataset(
            data_root=str(root),
            episode_indices=episode_indices,
            output_dir=str(temp_dir),
            repo_id=repo_id,
        )

        backup_dir = root.parent / f"{root.name}_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        print(f"\n备份原数据到: {backup_dir}")
        shutil.move(str(root), str(backup_dir))

        print(f"移动新数据到: {root}")
        shutil.move(str(temp_dir), str(root))

        print(f"\n删除备份（可选，建议验证后手动删除）: {backup_dir}")
        print(f"如需恢复: 删除 {root} 后重命名 {backup_dir} -> {root.name}")

        # 重新加载
        os.environ["HF_HUB_OFFLINE"] = "1"
        dataset = LeRobotDataset(
            repo_id=repo_id,
            root=root,
            revision="local",
        )
        return dataset

    except Exception as e:
        print(f"原地删除失败: {e}")

        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="删除 LeRobot 数据集中的指定 episode"
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
        "--episodes",
        type=int,
        nargs="+",
        required=True,
        help="要删除的 episode 索引（空格分隔），如: --episodes 2 5 10",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（不指定则原地删除需加 --inplace）",
    )
    parser.add_argument(
        "--new_repo_id",
        type=str,
        default=None,
        help="新数据集 ID",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="原地删除模式（替换原目录，会自动备份）",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过确认提示",
    )

    args = parser.parse_args()

    if args.output_dir is None and not args.inplace:
        print("错误: 必须指定 --output_dir 或 --inplace")
        print("  --output_dir: 输出到新目录（保留原数据）")
        print("  --inplace: 原地删除替换原数据（会自动备份）")
        sys.exit(1)

    if args.output_dir and args.inplace:
        print("错误: --output_dir 和 --inplace 不能同时使用")
        sys.exit(1)

    if not args.yes:
        print(f"将删除 episode: {args.episodes}")
        if args.inplace:
            print("模式: 原地删除（会自动备份原数据）")
        else:
            print(f"输出目录: {args.output_dir}")
        confirm = input("确认? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("已取消")
            sys.exit(0)

    if args.inplace:
        delete_episodes_inplace(
            data_root=args.data_root,
            episode_indices=args.episodes,
            repo_id=args.repo_id,
        )
    else:
        delete_episodes_from_dataset(
            data_root=args.data_root,
            episode_indices=args.episodes,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            new_repo_id=args.new_repo_id,
        )


if __name__ == "__main__":
    main()
