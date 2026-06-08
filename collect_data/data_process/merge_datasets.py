r"""
多数据集合并工具 — 将多个 LeRobot v3.0 格式数据集合并为单个数据集

支持合并两个或多个结构兼容的数据集（相同相机、相同状态/动作维度、相同 FPS）。
合并后 episode 索引和 frame 索引自动重编号（连续化），video/parquet 文件自动重排。

用法:
    cd D:\VLA\kortex_code\collect_data

    # 合并两个数据集
    python data_process/merge_datasets.py ^
        --datasets D:/VLA/data/simu_data1 D:/VLA/data/simu_data2 ^
        --output_dir D:/VLA/data/simu_data_merged

    # 合并三个数据集（指定最大 parquet 文件大小）
    python data_process/merge_datasets.py ^
        --datasets D:/VLA/data/simu_data1 D:/VLA/data/simu_data2 D:/VLA/data/simu_data3 ^
        --output_dir D:/VLA/data/simu_data_merged ^
        --max_data_file_size_mb 200
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_parquets(data_root: Path, subdir: str) -> pd.DataFrame:
    """加载指定子目录下所有 chunk/*.parquet 文件。"""
    all_dfs = []
    target = data_root / subdir
    if not target.exists():
        return pd.DataFrame()
    for chunk_dir in sorted(target.glob("chunk-*")):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                if len(df) > 0:
                    all_dfs.append(df)
                    logger.info(f"    {pf.relative_to(data_root)}: {len(df)} 行")
            except Exception as e:
                logger.warning(f"    跳过 {pf.relative_to(data_root)}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


def collect_video_files(data_root: Path, video_key: str) -> list[Path]:
    """收集某个相机的所有视频文件路径（按文件索引排序）。"""
    files = []
    vdir = data_root / "videos" / video_key
    if not vdir.exists():
        return files
    for chunk_dir in sorted(vdir.glob("chunk-*")):
        for vf in sorted(chunk_dir.glob("*.mp4")):
            files.append(vf)
    return files


def validate_compatibility(datasets: list[dict]) -> None:
    """验证所有数据集的 features 兼容。"""
    ref = datasets[0]
    ref_features = ref["info"]["features"]
    ref_fps = ref["info"]["fps"]
    ref_robot = ref["info"].get("robot_type", "")

    video_keys_ref = sorted(k for k, v in ref_features.items() if v.get("dtype") == "video")
    state_feat = ref_features.get("observation.state", {})
    action_feat = ref_features.get("action", {})

    for i, ds in enumerate(datasets[1:], 1):
        feat = ds["info"]["features"]
        fps = ds["info"]["fps"]
        robot = ds["info"].get("robot_type", "")

        vk = sorted(k for k, v in feat.items() if v.get("dtype") == "video")
        if vk != video_keys_ref:
            logger.error(f"数据集 {i} 相机列表不匹配: {vk} vs {ref_features}")
            logger.error(f"  期望: {video_keys_ref}")
            logger.error(f"  实际: {vk}")
            sys.exit(1)

        sf = feat.get("observation.state", {})
        af = feat.get("action", {})
        if sf.get("shape") != state_feat.get("shape"):
            logger.error(f"数据集 {i} observation.state shape 不匹配")
            sys.exit(1)
        if af.get("shape") != action_feat.get("shape"):
            logger.error(f"数据集 {i} action shape 不匹配")
            sys.exit(1)
        if fps != ref_fps:
            logger.error(f"数据集 {i} fps 不匹配: {fps} vs {ref_fps}")
            sys.exit(1)
        if robot and ref_robot and robot != ref_robot:
            logger.warning(f"数据集 {i} robot_type 不同: {robot} vs {ref_robot}")

    logger.info("所有数据集兼容性检查通过")


def recalc_state_action_stats(df: pd.DataFrame) -> dict:
    """从合并后的 DataFrame 重新计算 state 和 action 的统计值。"""
    stats = {}
    for key in ["observation.state", "action"]:
        col = df[key]
        stacked = np.stack(col.values, axis=0)
        stats[key] = {
            "min": stacked.min(axis=0).tolist(),
            "max": stacked.max(axis=0).tolist(),
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
            "count": [stacked.shape[0]],
            "q01": np.percentile(stacked, 1, axis=0).tolist(),
            "q10": np.percentile(stacked, 10, axis=0).tolist(),
            "q50": np.percentile(stacked, 50, axis=0).tolist(),
            "q90": np.percentile(stacked, 90, axis=0).tolist(),
            "q99": np.percentile(stacked, 99, axis=0).tolist(),
        }
    return stats


def build_video_file_map(datasets: list[dict], video_keys: list[str]) -> dict:
    """构建 (dataset_idx, video_key, old_file_index) -> new_file_index 映射。"""
    mapping = {}
    for vk in video_keys:
        mapping[vk] = {}
        cum_count = 0
        for ds_idx, ds in enumerate(datasets):
            vfiles = collect_video_files(ds["root"], vk)
            for old_idx in range(len(vfiles)):
                mapping[vk][(ds_idx, old_idx)] = cum_idx = old_idx + cum_count
            cum_count += len(vfiles)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="多 LeRobot v3.0 数据集合并工具")
    parser.add_argument("--datasets", type=str, nargs="+", required=True,
                        help="要合并的数据集根目录路径（空格分隔）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="合并后的输出目录")
    parser.add_argument("--repo_id", type=str, default="simu_data_merged",
                        help="合并数据集的 repo_id")
    parser.add_argument("--max_data_file_size_mb", type=int, default=200,
                        help="每个 parquet 文件的最大 MB 数")
    parser.add_argument("--yes", action="store_true",
                        help="跳过确认")
    args = parser.parse_args()

    if len(args.datasets) < 2:
        logger.error("至少需要两个数据集才能合并")
        sys.exit(1)

    out = Path(args.output_dir)
    if out.exists():
        if args.yes:
            shutil.rmtree(out)
            logger.info(f"清空输出目录: {out}")
        else:
            logger.error(f"输出目录已存在: {out}，使用 --yes 覆盖或指定其他目录")
            sys.exit(1)

    # ==== Step 0: 加载所有数据集元信息 ====
    datasets = []
    for dspath in args.datasets:
        root = Path(dspath)
        if not (root / "meta" / "info.json").exists():
            logger.error(f"数据集不存在: {root / 'meta' / 'info.json'}")
            sys.exit(1)
        with open(root / "meta" / "info.json", "r") as f:
            info = json.load(f)
        datasets.append({"root": root, "info": info})

    logger.info(f"合并 {len(datasets)} 个数据集: {[d['root'].name for d in datasets]}")

    validate_compatibility(datasets)

    video_keys = sorted(
        k for k, v in datasets[0]["info"]["features"].items()
        if v.get("dtype") == "video"
    )
    logger.info(f"视频 keys: {video_keys}")

    # ==== Step 1: 合并所有 parquet 数据 ====
    logger.info("加载并合并 parquet 数据...")
    all_frames = []
    cum_episode_offset = 0
    cum_frame_offset = 0

    episode_remap = {}  # (ds_idx, old_ep_idx) -> new_ep_idx
    frame_remap_info = []  # per-dataset frame offset info

    for ds_idx, ds in enumerate(datasets):
        df = load_all_parquets(ds["root"], "data")
        if len(df) == 0:
            logger.error(f"数据集 {ds['root'].name} 无 parquet 数据")
            sys.exit(1)

        # Remap episode_index
        old_eps = sorted(df["episode_index"].unique())
        for old_ep in old_eps:
            episode_remap[(ds_idx, old_ep)] = old_ep + cum_episode_offset

        df["episode_index"] = df["episode_index"].map(
            lambda x: episode_remap[(ds_idx, x)]
        )

        # Remap index (frame index)
        df["index"] = df["index"] + cum_frame_offset

        # Remap frame_index (internal per-episode counter)
        # Keep original value — it's per-episode, not global

        all_frames.append(df)

        n_frames = len(df)
        n_eps = len(old_eps)
        logger.info(f"  {ds['root'].name}: {n_frames} 帧, {n_eps} episodes "
                    f"(ep {cum_episode_offset}–{cum_episode_offset + n_eps - 1})")

        frame_remap_info.append({
            "ds_idx": ds_idx,
            "n_frames": n_frames,
            "n_eps": n_eps,
            "frame_offset": cum_frame_offset,
            "episode_offset": cum_episode_offset,
        })
        cum_episode_offset += n_eps
        cum_frame_offset += n_frames

    merged_df = pd.concat(all_frames, ignore_index=True)
    total_frames = len(merged_df)
    total_episodes = cum_episode_offset
    logger.info(f"合并后: {total_frames} 帧, {total_episodes} episodes")

    # ==== Step 2: 写入合并后的 parquet 数据 ====
    logger.info("写入合并后的 parquet 文件...")
    out_data_dir = out / "data" / "chunk-000"
    out_data_dir.mkdir(parents=True, exist_ok=True)

    # Estimate rows per file based on column types (rough: ~500 bytes/row)
    rows_per_file = max(1, int(args.max_data_file_size_mb * 1024 * 1024 / 500))
    parquet_file_count = 0
    parquet_file_ranges = []  # (file_idx, start_frame, end_frame_exclusive)

    start = 0
    while start < total_frames:
        end = min(start + rows_per_file, total_frames)
        out_path = out_data_dir / f"file-{parquet_file_count:03d}.parquet"
        merged_df.iloc[start:end].to_parquet(out_path, index=False)
        parquet_file_ranges.append((parquet_file_count, start, end))
        logger.info(f"  file-{parquet_file_count:03d}.parquet: {start}–{end-1} ({end - start} 行)")
        parquet_file_count += 1
        start = end

    # Build parquet file lookup: frame_index -> file_idx
    def find_parquet_file(frame_idx: int) -> int:
        for fidx, fstart, fend in parquet_file_ranges:
            if fstart <= frame_idx < fend:
                return fidx
        return parquet_file_ranges[-1][0]

    # ==== Step 3: 复制视频文件 ====
    logger.info("复制视频文件...")
    video_file_map = build_video_file_map(datasets, video_keys)

    for vk in video_keys:
        out_vdir = out / "videos" / vk / "chunk-000"
        out_vdir.mkdir(parents=True, exist_ok=True)

        for ds_idx, ds in enumerate(datasets):
            src_files = collect_video_files(ds["root"], vk)
            for old_idx, src_vf in enumerate(src_files):
                new_idx = video_file_map[vk].get((ds_idx, old_idx))
                if new_idx is None:
                    continue
                dst = out_vdir / f"file-{new_idx:03d}.mp4"
                shutil.copy2(str(src_vf), str(dst))

        n_copied = len(list(out_vdir.glob("*.mp4")))
        logger.info(f"  {vk}: {n_copied} 个文件")

    # Build video file lookup: (ds_idx, video_key, old_video_file_idx) -> new_video_file_idx
    # Already in video_file_map

    def get_new_video_file_idx(ds_idx: int, vk: str, old_file_idx: int) -> int:
        return video_file_map[vk].get((ds_idx, old_file_idx), 0)

    # ==== Step 4: 重建 episodes 元数据 ====
    logger.info("重建 episodes 元数据...")
    all_episodes = []

    for ds_idx, ds in enumerate(datasets):
        eps_df = load_all_parquets(ds["root"], "meta/episodes")
        if len(eps_df) == 0:
            logger.warning(f"  数据集 {ds['root'].name} 无 episodes 元数据")
            continue

        ds_frame_offset = frame_remap_info[ds_idx]["frame_offset"]
        ds_ep_offset = frame_remap_info[ds_idx]["episode_offset"]

        for row_idx in range(len(eps_df)):
            row = eps_df.iloc[row_idx].copy()

            old_ep_idx = int(row["episode_index"])
            new_ep_idx = old_ep_idx + ds_ep_offset
            row["episode_index"] = new_ep_idx

            # Update frame range
            old_from = int(row["dataset_from_index"])
            old_to = int(row["dataset_to_index"])
            row["dataset_from_index"] = old_from + ds_frame_offset
            row["dataset_to_index"] = old_to + ds_frame_offset

            # Determine new data file index
            global_start = old_from + ds_frame_offset
            new_data_file = find_parquet_file(global_start)
            row["data/chunk_index"] = 0
            row["data/file_index"] = new_data_file

            # Determine new video file indices
            for vk in video_keys:
                old_vf = int(row.get(f"videos/{vk}/chunk_index", 0)), \
                         int(row.get(f"videos/{vk}/file_index", 0))
                old_chunk, old_vf_idx = old_vf
                new_vf_idx = get_new_video_file_idx(ds_idx, vk, old_vf_idx)
                row[f"videos/{vk}/chunk_index"] = 0
                row[f"videos/{vk}/file_index"] = new_vf_idx

            # Episodes file reference (to be updated after we write episodes)
            row["meta/episodes/chunk_index"] = 0
            # file_index will be set later

            all_episodes.append(row)

    if not all_episodes:
        logger.error("无 episodes 元数据可合并")
        sys.exit(1)

    episodes_df = pd.DataFrame(all_episodes).reset_index(drop=True)
    logger.info(f"  共 {len(episodes_df)} 条 episode 记录")

    # Write episodes parquet
    out_eps_dir = out / "meta" / "episodes" / "chunk-000"
    out_eps_dir.mkdir(parents=True, exist_ok=True)

    # Split episodes into multiple parquet files if needed (same size logic)
    eps_rows_per_file = max(1, int(10 * 1024 * 1024 / 2000))  # ~10MB per file
    eps_file_count = 0
    eps_start = 0
    while eps_start < len(episodes_df):
        eps_end = min(eps_start + eps_rows_per_file, len(episodes_df))
        chunk = episodes_df.iloc[eps_start:eps_end].copy()
        # Update meta/episodes/file_index for each row
        chunk["meta/episodes/file_index"] = eps_file_count
        out_eps_path = out_eps_dir / f"file-{eps_file_count:03d}.parquet"
        chunk.to_parquet(out_eps_path, index=False)
        logger.info(f"  episodes file-{eps_file_count:03d}.parquet: "
                    f"{eps_start}–{eps_end-1} ({eps_end - eps_start} 行)")
        eps_file_count += 1
        eps_start = eps_end

    # Also update meta/episodes/file_index in the original episodes_df for reference
    # (the actual written files already have correct values)

    # ==== Step 5: tasks.parquet ====
    logger.info("合并 tasks...")
    all_tasks = []
    for ds_idx, ds in enumerate(datasets):
        src_tasks = ds["root"] / "meta" / "tasks.parquet"
        if src_tasks.exists():
            try:
                tdf = pd.read_parquet(src_tasks)
                all_tasks.append(tdf)
            except Exception as e:
                logger.warning(f"  跳过 tasks: {e}")

    if all_tasks:
        tasks_combined = pd.concat(all_tasks, ignore_index=True).drop_duplicates(
            subset=["task_index"]
        ).reset_index(drop=True)
        tasks_combined.to_parquet(out / "meta" / "tasks.parquet", index=False)
        logger.info(f"  tasks: {len(tasks_combined)} 条")
    else:
        logger.warning("  无 tasks 文件，创建空文件")
        pd.DataFrame({"task_index": [0], "task": ["default"]}).to_parquet(
            out / "meta" / "tasks.parquet", index=False
        )

    # ==== Step 6: stats.json（重新计算 state/action，视频 stats 从第一个数据集复制）====
    logger.info("计算统计信息...")
    state_action_stats = recalc_state_action_stats(merged_df)

    # Load video stats from first dataset
    src_stats_path = datasets[0]["root"] / "meta" / "stats.json"
    if src_stats_path.exists():
        with open(src_stats_path, "r") as f:
            src_stats = json.load(f)
    else:
        src_stats = {}

    combined_stats = {}
    # Copy all stats from first dataset
    combined_stats.update(src_stats)
    # Override state and action stats with recalculated values
    for key in ["observation.state", "action"]:
        if key in state_action_stats:
            combined_stats[key] = state_action_stats[key]

    with open(out / "meta" / "stats.json", "w") as f:
        json.dump(combined_stats, f, indent=2, default=str)
    logger.info("  stats.json 已生成")

    # ==== Step 7: info.json ====
    logger.info("写入 info.json...")
    ref_info = datasets[0]["info"]
    out_info = dict(ref_info)
    out_info["total_episodes"] = total_episodes
    out_info["total_frames"] = total_frames
    out_info["splits"] = {"train": f"0:{total_episodes}"}
    out_info["total_tasks"] = 1
    out_info["total_videos"] = total_episodes
    out_info["total_chunks"] = 1
    out_info["chunks_size"] = 2000

    with open(out / "meta" / "info.json", "w") as f:
        json.dump(out_info, f, indent=4)
    logger.info(f"  total_episodes={total_episodes}, total_frames={total_frames}")

    # ==== Summary ====
    logger.info("=" * 60)
    logger.info(f"合并完成！输出: {out}")
    logger.info(f"  episodes: {total_episodes}")
    logger.info(f"  frames: {total_frames}")
    logger.info(f"  parquet 文件: {parquet_file_count}")
    logger.info(f"  episodes 文件: {eps_file_count}")
    for vk in video_keys:
        nv = len(list((out / "videos" / vk / "chunk-000").glob("*.mp4")))
        logger.info(f"  {vk} 视频: {nv} 个文件")


if __name__ == "__main__":
    main()
