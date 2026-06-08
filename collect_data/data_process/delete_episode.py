r"""
单条 Episode 删除工具

基于 LeRobot API 从数据集中删除指定 episode。
支持安全模式（输出到新目录）和原地模式（自动备份）。

删除后 episode 索引自动重编号（连续化），parquet 和 video 切片自动更新。

用法:
    cd D:\VLA\kortex_code\collect_data

    # 安全模式：删除 episode 5，输出到新目录
    python data_process/delete_episode.py \
        --data_root D:/VLA/data/simu_data \
        --episode 5 \
        --output_dir D:/VLA/data/simu_data_cleaned

    # 原地模式：删除 episode 3（自动备份原数据）
    python data_process/delete_episode.py \
        --data_root D:/VLA/data/simu_data \
        --episode 3 --inplace

    # 批量删除
    python data_process/delete_episode.py \
        --data_root D:/VLA/data/simu_data \
        --episodes 2 5 10 15 \
        --output_dir D:/VLA/data/simu_data_cleaned

    # 跳过确认
    python data_process/delete_episode.py \
        --data_root D:/VLA/data/simu_data \
        --episode 5 \
        --output_dir D:/VLA/data/simu_data_cleaned --yes

    # 查看删除影响 (dry-run)
    python data_process/delete_episode.py \
        --data_root D:/VLA/data/simu_data \
        --episode 3 --dry_run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes


def print_episode_info(dataset: LeRobotDataset, episode_idx: int):
    """打印单个 episode 的详细信息"""
    ep = dataset.meta.episodes[episode_idx]
    frame_from = int(ep.get("index_from", 0))
    frame_to = int(ep.get("index_to", 0))
    frame_count = frame_to - frame_from + 1
    task = ep.get("task", "")
    print(f"  episode {episode_idx}: {frame_count} frames, task={task}")


def delete_episodes_safe(data_root: str, episode_indices: list[int],
                         output_dir: str, repo_id: str = "simu_data",
                         new_repo_id: str | None = None) -> LeRobotDataset:
    """安全删除：输出到新目录，保留原始数据。"""
    os.environ["HF_HUB_OFFLINE"] = "1"

    root = Path(data_root)
    out = Path(output_dir)

    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"数据集不存在: {root / 'meta' / 'info.json'}")
    if out.exists():
        print(f"输出目录已存在，将被清空: {out}")
        shutil.rmtree(out)

    dataset = LeRobotDataset(repo_id=repo_id, root=root, revision="local")

    total_before = dataset.meta.total_episodes
    print(f"当前: {total_before} episodes, {dataset.meta.total_frames} frames")
    print(f"待删除 episode: {episode_indices}")

    for idx in episode_indices:
        print_episode_info(dataset, idx)

    print(f"\n执行删除到: {out}")
    new_dataset = delete_episodes(
        dataset=dataset,
        episode_indices=episode_indices,
        output_dir=out,
        repo_id=new_repo_id or repo_id,
    )

    total_after = new_dataset.meta.total_episodes
    print(f"完成: {total_before}→{total_after} episodes "
          f"({total_before - total_after} 条已删除)")
    print(f"新数据集: {out}")
    return new_dataset


def delete_episodes_inplace(data_root: str, episode_indices: list[int],
                            repo_id: str = "simu_data") -> LeRobotDataset:
    """原地删除：先构建到临时目录，再替换原目录（自动备份）。"""
    root = Path(data_root)
    temp_dir = root.parent / f"{root.name}_temp_delete"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    try:
        new_dataset = delete_episodes_safe(
            data_root=str(root),
            episode_indices=episode_indices,
            output_dir=str(temp_dir),
            repo_id=repo_id,
        )

        backup_dir = root.parent / f"{root.name}_backup_{len(episode_indices)}eps"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

        print(f"\n备份原数据: {backup_dir}")
        shutil.move(str(root), str(backup_dir))
        print(f"替换原数据: {root}")
        shutil.move(str(temp_dir), str(root))

        print(f"\n原数据备份在: {backup_dir}")
        print(f"验证后如需删除备份: rmdir /s/q {backup_dir}")

        os.environ["HF_HUB_OFFLINE"] = "1"
        return LeRobotDataset(repo_id=repo_id, root=root, revision="local")

    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description="单条/批量 Episode 删除工具")
    parser.add_argument("--data_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--repo_id", type=str, default="simu_data", help="数据集 ID")
    parser.add_argument("--episode", type=int, default=None, help="要删除的单个 episode 索引")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="要删除的多个 episode 索引（空格分隔）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录（安全模式）")
    parser.add_argument("--new_repo_id", type=str, default=None, help="新数据集 ID")
    parser.add_argument("--inplace", action="store_true", help="原地删除（自动备份）")
    parser.add_argument("--yes", action="store_true", help="跳过确认")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅预览要删除的 episode，不执行")

    args = parser.parse_args()

    episode_indices = []
    if args.episode is not None:
        episode_indices.append(args.episode)
    if args.episodes is not None:
        episode_indices.extend(args.episodes)

    episode_indices = sorted(set(episode_indices))

    if not episode_indices:
        print("错误: 请用 --episode N 或 --episodes N M K 指定要删除的 episode")
        sys.exit(1)

    root = Path(args.data_root)
    if not (root / "meta" / "info.json").exists():
        print(f"错误: 数据集不存在: {root / 'meta' / 'info.json'}")
        sys.exit(1)

    os.environ["HF_HUB_OFFLINE"] = "1"
    dataset = LeRobotDataset(
        repo_id=args.repo_id, root=root, revision="local"
    )

    max_ep = dataset.meta.total_episodes - 1
    for idx in episode_indices:
        if idx < 0 or idx > max_ep:
            print(f"错误: episode {idx} 超出范围 (0 ~ {max_ep})")
            sys.exit(1)

    print(f"=== {'预览' if args.dry_run else '删除'}模式 ===")
    print(f"数据集: {args.data_root}")
    print(f"当前: {dataset.meta.total_episodes} episodes, "
          f"{dataset.meta.total_frames} frames")

    for idx in episode_indices:
        print_episode_info(dataset, idx)

    if args.dry_run:
        print(f"\n将删除 {len(episode_indices)} 条 episode")
        print("实际执行时去掉 --dry_run")
        return

    if not args.yes:
        if args.inplace:
            print("模式: 原地删除（会自动备份）")
        elif args.output_dir:
            print(f"模式: 输出到新目录 {args.output_dir}")
        else:
            print("错误: 必须指定 --output_dir 或 --inplace")
            sys.exit(1)

        confirm = input("\n确认删除? [y/N]: ")
        if confirm.lower() not in ("y", "yes"):
            print("已取消")
            return

    if args.inplace:
        delete_episodes_inplace(str(root), episode_indices, args.repo_id)
    else:
        delete_episodes_safe(str(root), episode_indices, args.output_dir,
                             args.repo_id, args.new_repo_id)


if __name__ == "__main__":
    main()
