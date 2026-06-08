"""
数据集合并工具 - 将分散的多个 data parquet / video mp4 文件合并成少量大文件。

使用场景：
- delete_episodes 后数据集文件碎片化（很多小文件）
- 需要减少文件数量以提高加载效率

用法：
    python -m scripts.data_process.consolidate_dataset \
        --data_root D:/VLA/data/simu_data_cleaned \
        --output_dir D:/VLA/data/simu_data_merged

工作原理：
    1. 合并所有 data/chunk-000/file-xxx.parquet -> 少量大 parquet
    2. 合并每个相机的 videos/.../file-xxx.mp4 -> 少量大 mp4
    3. 更新 meta/episodes, meta/info.json, meta/stats.json, meta/tasks.parquet
"""
import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import av
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DATA_DIR,
    EPISODES_DIR,
    VIDEO_DIR,
    get_file_size_in_mb,
    update_chunk_file_indices,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_info


def _data_chunk_dir(root: Path, chunk_index: int) -> Path:
    return root / DATA_DIR / f"chunk-{chunk_index:03d}"


def _video_chunk_dir(root: Path, video_key: str, chunk_index: int) -> Path:
    return root / VIDEO_DIR / video_key / f"chunk-{chunk_index:03d}"


def _episodes_chunk_dir(root: Path, chunk_index: int) -> Path:
    return root / EPISODES_DIR / f"chunk-{chunk_index:03d}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def consolidate_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    repo_id: str | None = None,
    max_data_file_size_mb: int | None = None,
    max_video_file_size_mb: int | None = None,
) -> LeRobotDataset:
    """合并碎片化的数据集文件。

    Args:
        data_root: 原始数据集根目录
        output_dir: 输出目录
        repo_id: 数据集 ID
        max_data_file_size_mb: 单 data parquet 最大大小（MB），默认 200
        max_video_file_size_mb: 单 video mp4 最大大小（MB），默认 500

    Returns:
        合并后的 LeRobotDataset
    """
    os.environ["HF_HUB_OFFLINE"] = "1"

    src_root = Path(data_root)
    out_root = Path(output_dir)

    if not (src_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"数据集不存在: {src_root / 'meta' / 'info.json'}")

    with open(src_root / "meta" / "info.json", "r") as f:
        src_info = json.load(f)

    if repo_id is None:
        repo_id = src_info.get("codebase_version", "v3.0")  # fallback

    total_episodes = src_info["total_episodes"]
    fps = src_info["fps"]
    features = src_info["features"]
    robot_type = src_info.get("robot_type", "unknown")
    video_keys = [k for k, v in features.items() if v.get("dtype") == "video"]
    data_file_size_mb = max_data_file_size_mb or 200
    video_file_size_mb = max_video_file_size_mb or 500

    logger.info(f"数据集: {repo_id}, episodes={total_episodes}, "
                f"video_keys={video_keys}")
    logger.info(f"源目录: {src_root}")
    logger.info(f"输出目录: {out_root}")

    if out_root.exists():
        logger.warning(f"输出目录已存在，将被清空: {out_root}")
        shutil.rmtree(out_root)

    # ============================================================
    # Step 1: 创建新 metadata
    # ============================================================
    logger.info("Step 1: 创建新 metadata")
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        root=out_root,
        use_videos=len(video_keys) > 0,
        chunks_size=DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=data_file_size_mb,
        video_files_size_in_mb=video_file_size_mb,
    )

    # ============================================================
    # Step 2: 合并 data parquet 文件
    # ============================================================
    logger.info("Step 2: 合并 data parquet 文件")

    src_data_dir = _data_chunk_dir(src_root, 0)
    parquet_files = sorted(src_data_dir.glob("*.parquet"))
    logger.info(f"  源 parquet 文件数: {len(parquet_files)}")

    all_dfs = []
    for pf in tqdm(parquet_files, desc="读取 parquet"):
        try:
            df = pd.read_parquet(pf)
            all_dfs.append(df)
        except Exception as e:
            logger.warning(f"跳过损坏文件 {pf.name}: {e}")

    if not all_dfs:
        raise RuntimeError("没有可读取的 parquet 文件")

    merged_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"  合并后总帧数: {len(merged_df)}")

    # 按新大小切分成多个 parquet
    chunk_idx = 0
    file_idx = 0
    start = 0
    total_rows = len(merged_df)
    rows_per_file = int(data_file_size_mb * 1024 * 1024 / 500)  # 约500字节/行

    dst_data_dir = _data_chunk_dir(out_root, chunk_idx)
    dst_data_dir.mkdir(parents=True, exist_ok=True)

    episode_data_meta = {}

    while start < total_rows:
        end = min(start + rows_per_file, total_rows)
        chunk_df = merged_df.iloc[start:end]

        out_path = dst_data_dir / f"file-{file_idx:03d}.parquet"
        chunk_df.to_parquet(out_path, index=False)

        file_size_mb = get_file_size_in_mb(out_path)
        logger.info(f"  写入 {out_path.name}: {len(chunk_df)} rows, {file_size_mb:.1f} MB")

        # 记录每个 episode 在这个文件中的信息
        for ep_idx in chunk_df["episode_index"].unique():
            ep_df = chunk_df[chunk_df["episode_index"] == ep_idx]
            episode_data_meta[int(ep_idx)] = {
                "meta/episodes/chunk_index": chunk_idx,
                "meta/episodes/file_index": file_idx,
                "meta/episodes/index_from": int(ep_df["index"].min()),
                "meta/episodes/index_to": int(ep_df["index"].max()),
            }

        start = end
        chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, DEFAULT_CHUNK_SIZE)
        dst_data_dir = _data_chunk_dir(out_root, chunk_idx)
        dst_data_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Step 3: 合并 video mp4 文件
    # ============================================================
    video_episode_meta = {}
    if video_keys:
        logger.info(f"Step 3: 合并 video mp4 文件 ({len(video_keys)} 个相机)")

        for video_key in video_keys:
            logger.info(f"  处理相机: {video_key}")

            src_video_dir = _video_chunk_dir(src_root, video_key, 0)
            video_files = sorted(src_video_dir.glob("*.mp4"))
            logger.info(f"    源 mp4 文件数: {len(video_files)}")

            if not video_files:
                logger.warning(f"  没有找到视频文件: {video_key}")
                continue

            # 先合并为一个大的临时视频
            temp_concat = out_root / f"_temp_{video_key}_concat.mp4"
            logger.info(f"    合并 {len(video_files)} 个视频 -> {temp_concat.name}...")
            concatenate_video_files(video_files, temp_concat, overwrite=True)

            concat_size = get_file_size_in_mb(temp_concat)
            logger.info(f"    合并后大小: {concat_size:.1f} MB")

            # 如果合并后不超过上限，直接使用单文件
            if concat_size <= video_file_size_mb:
                dst_video_dir = _video_chunk_dir(out_root, video_key, 0)
                dst_video_dir.mkdir(parents=True, exist_ok=True)
                final_path = dst_video_dir / "file-000.mp4"
                shutil.move(str(temp_concat), str(final_path))
                logger.info(f"    单文件输出: {final_path}")

                # 记录 episode 视频元数据
                all_frame_to_episode = _map_frames_to_episodes(merged_df)
                num_frames = _get_video_frame_count(final_path)
                ep_boundaries = _compute_episode_video_boundaries(
                    all_frame_to_episode, total_episodes, num_frames, fps
                )
                for ep_idx in range(total_episodes):
                    if ep_idx in ep_boundaries:
                        video_episode_meta.setdefault(ep_idx, {})[video_key] = {
                            "chunk_index": 0,
                            "file_index": 0,
                            **ep_boundaries[ep_idx],
                        }
            else:
                # 需要切成多个文件
                num_files = max(1, int(concat_size / video_file_size_mb) + 1)
                total_frames = _get_video_frame_count(temp_concat)
                frames_per_file = total_frames // num_files

                logger.info(f"    切分为 {num_files} 个文件，每个约 {frames_per_file} 帧")

                all_frame_to_episode = _map_frames_to_episodes(merged_df)
                current_chunk = 0
                current_file = 0

                for part_idx in range(num_files):
                    start_frame = part_idx * frames_per_file
                    end_frame = min(
                        (part_idx + 1) * frames_per_file, total_frames
                    ) if part_idx < num_files - 1 else total_frames

                    dst_video_dir = _video_chunk_dir(out_root, video_key, current_chunk)
                    dst_video_dir.mkdir(parents=True, exist_ok=True)
                    part_path = (
                        dst_video_dir / f"file-{current_file:03d}.mp4"
                    )

                    _extract_video_segment(temp_concat, part_path, start_frame, end_frame)
                    part_size = get_file_size_in_mb(part_path)
                    logger.info(f"      写入 {part_path.name}: frames {start_frame}-{end_frame}, "
                                f"{part_size:.1f} MB")

                    # 记录 episode 边界
                    ep_boundaries = _compute_episode_video_boundaries_range(
                        all_frame_to_episode, total_episodes, fps,
                        start_frame, end_frame, part_idx * frames_per_file
                    )
                    for ep_idx in range(total_episodes):
                        if ep_idx in ep_boundaries:
                            video_episode_meta.setdefault(ep_idx, {})[video_key] = {
                                "chunk_index": current_chunk,
                                "file_index": current_file,
                                **ep_boundaries[ep_idx],
                            }

                    current_chunk, current_file = update_chunk_file_indices(
                        current_chunk, current_file, DEFAULT_CHUNK_SIZE
                    )

                temp_concat.unlink()

    # ============================================================
    # Step 4: 复制 tasks.parquet
    # ============================================================
    logger.info("Step 4: 复制 tasks.parquet")
    src_tasks = src_root / "meta" / "tasks.parquet"
    if src_tasks.exists():
        tasks_df = pd.read_parquet(src_tasks)
        tasks_df_reindexed = tasks_df.copy()
        if "episode" in tasks_df_reindexed.columns:
            # delete_episodes 已经重映射了 episode index，不需要再改
            pass
        write_tasks(tasks_df_reindexed, out_root)

    # ============================================================
    # Step 5: 构建 episodes 元数据
    # ============================================================
    logger.info("Step 5: 构建 episodes 元数据")

    # 读取原始 episodes
    src_eps_dir = _episodes_chunk_dir(src_root, 0)
    src_eps_files = sorted(src_eps_dir.glob("*.parquet"))
    all_eps_df_list = []
    for ef in src_eps_files:
        try:
            all_eps_df_list.append(pd.read_parquet(ef))
        except Exception as e:
            logger.warning(f"跳过 {ef.name}: {e}")
    eps_df = pd.concat(all_eps_df_list, ignore_index=True) if all_eps_df_list else pd.DataFrame()

    # 构建新的 episodes 记录 -> 单个 parquet
    episodes_records = []
    for ep_idx in range(total_episodes):
        record = {}

        if ep_idx in episode_data_meta:
            record.update(episode_data_meta[ep_idx])

        if video_keys and ep_idx in video_episode_meta:
            for vk, vmeta in video_episode_meta[ep_idx].items():
                record[f"videos/{vk}/chunk_index"] = vmeta["chunk_index"]
                record[f"videos/{vk}/file_index"] = vmeta["file_index"]
                record[f"videos/{vk}/timestamp_from"] = vmeta["timestamp_from"]
                record[f"videos/{vk}/timestamp_to"] = vmeta["timestamp_to"]

        # 从原始 episodes 复制 task 等信息
        if not eps_df.empty and ep_idx < len(eps_df):
            src_ep = eps_df.iloc[ep_idx]
            for col in eps_df.columns:
                if col not in record and not col.startswith("meta/episodes/") and not col.startswith("videos/"):
                    record[col] = src_ep[col]

        # 确保基础字段
        record["episode_index"] = ep_idx
        if "task_index" not in record:
            record["task_index"] = 0
        if "task" not in record:
            record["task"] = ""

        episodes_records.append(record)

    eps_out_df = pd.DataFrame(episodes_records)

    dst_eps_dir = _episodes_chunk_dir(out_root, 0)
    dst_eps_dir.mkdir(parents=True, exist_ok=True)
    eps_out_df.to_parquet(dst_eps_dir / "file-000.parquet", index=False)
    logger.info(f"  写入 episodes: {len(eps_out_df)} 条")

    # ============================================================
    # Step 6: 更新 info.json 和 stats.json
    # ============================================================
    logger.info("Step 6: 更新 info.json 和 stats.json")

    # 直接读取 LeRobotDatasetMetadata.create() 已写入的正确 info.json，
    # 只更新 episode/frame 计数，避免覆盖 video_path 等关键字段
    existing_info_path = out_root / "meta" / "info.json"
    with open(existing_info_path, "r") as f:
        new_info = json.load(f)

    new_info["total_episodes"] = total_episodes
    new_info["total_frames"] = int(merged_df["index"].max() - merged_df["index"].min() + 1)
    new_info["total_tasks"] = 1

    data_chunk_counts = {}
    for pf in sorted(_data_chunk_dir(out_root, 0).glob("*.parquet")):
        chunk_idx = 0
        data_chunk_counts[chunk_idx] = data_chunk_counts.get(chunk_idx, 0) + 1
    new_info["total_chunks"] = max(data_chunk_counts.values()) if data_chunk_counts else 1

    # 计算 total_videos
    total_videos = 0
    if video_keys:
        for vk in video_keys:
            vc_dir = _video_chunk_dir(out_root, vk, 0)
            if vc_dir.exists():
                total_videos += len(list(vc_dir.glob("*.mp4")))
    new_info["total_videos"] = total_videos

    with open(existing_info_path, "w") as f:
        json.dump(new_info, f, indent=4, ensure_ascii=False)

    logger.info(f"  info.json 已更新: episodes={total_episodes}, frames={new_info['total_frames']}")

    src_stats = src_root / "meta" / "stats.json"
    dst_stats = out_root / "meta" / "stats.json"
    if src_stats.exists():
        dst_stats.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_stats, dst_stats)

    # ============================================================
    # Step 7: 加载并返回
    # ============================================================
    logger.info("Step 7: 加载合并后的数据集")

    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=out_root,
        revision="local",
    )

    logger.info(f"完成! 输出: {out_root}")
    logger.info(f"  Episodes: {dataset.meta.total_episodes}")
    logger.info(f"  Frames: {dataset.meta.total_frames}")

    return dataset


def _map_frames_to_episodes(merged_df: pd.DataFrame) -> dict[int, int]:
    """映射 frame_index -> episode_index"""
    mapping = {}
    for ep_idx in merged_df["episode_index"].unique():
        ep_df = merged_df[merged_df["episode_index"] == ep_idx]
        for _, row in ep_df.iterrows():
            mapping[int(row["frame_index"])] = int(ep_idx)
    return mapping


def _get_video_frame_count(video_path: Path) -> int:
    """获取视频帧数"""
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            return stream.frames if stream.frames > 0 else int(
                stream.duration * stream.average_rate / stream.time_base
            )
    except Exception:
        return 0


def _compute_episode_video_boundaries(
    frame_to_ep: dict[int, int], total_episodes: int,
    total_video_frames: int, fps: float
) -> dict[int, dict]:
    """计算每个 episode 在视频中的帧范围"""
    result = {}
    current_ep = None
    start_video_frame = 0
    data_frame_idx = 0

    for video_frame in range(total_video_frames):
        ep = frame_to_ep.get(data_frame_idx, current_ep)
        if ep is not None and ep != current_ep:
            if current_ep is not None:
                result[current_ep] = {
                    "timestamp_from": start_video_frame / fps,
                    "timestamp_to": (video_frame - 1) / fps,
                }
            current_ep = ep
            start_video_frame = video_frame
        # 视频帧率与数据帧率可能不同，做近似映射
        progress = video_frame / max(total_video_frames, 1)
        data_frame_idx = int(progress * len(frame_to_ep))
        data_frame_idx = min(data_frame_idx, max(frame_to_ep.keys()) if frame_to_ep else 0)

    if current_ep is not None:
        result[current_ep] = {
            "timestamp_from": start_video_frame / fps,
            "timestamp_to": (total_video_frames - 1) / fps,
        }

    return result


def _compute_episode_video_boundaries_range(
    frame_to_ep: dict[int, int], total_episodes: int, fps: float,
    segment_start: int, segment_end: int, offset: int
) -> dict[int, dict]:
    """计算分片视频中每个 episode 的边界"""
    result = {}
    current_ep = None
    start_local = 0

    for video_frame in range(segment_start, segment_end):
        progress = (video_frame + offset) / max(len(frame_to_ep) * 2, 1)
        data_idx = int(progress * len(frame_to_ep))
        data_idx = min(data_idx, max(frame_to_ep.keys()) if frame_to_ep else 0)
        ep = frame_to_ep.get(data_idx, current_ep)
        local_frame = video_frame - segment_start

        if ep is not None and ep != current_ep:
            if current_ep is not None:
                result[current_ep] = {
                    "timestamp_from": (start_local + offset) / fps,
                    "timestamp_to": (local_frame - 1 + offset) / fps,
                }
            current_ep = ep
            start_local = local_frame

    if current_ep is not None:
        result[current_ep] = {
            "timestamp_from": (start_local + offset) / fps,
            "timestamp_to": (segment_end - segment_start - 1 + offset) / fps,
        }

    return result


def _extract_video_segment(
    src_path: Path, dst_path: Path, start_frame: int, end_frame: int
):
    """从视频中提取帧范围并重新编码"""
    input_container = av.open(str(src_path))
    output_container = av.open(str(dst_path), "w")

    try:
        in_stream = input_container.streams.video[0]
        out_stream = output_container.add_stream("libx264", rate=float(in_stream.average_rate))
        out_stream.width = in_stream.width
        out_stream.height = in_stream.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "23"}

        frame_idx = 0
        for packet in input_container.demux(in_stream):
            for frame in packet.decode():
                if start_frame <= frame_idx < end_frame:
                    for enc_packet in out_stream.encode(frame):
                        output_container.mux(enc_packet)
                frame_idx += 1
                if frame_idx >= end_frame:
                    break
            if frame_idx >= end_frame:
                break

        for enc_packet in out_stream.encode():
            output_container.mux(enc_packet)
    finally:
        input_container.close()
        output_container.close()


def main():
    parser = argparse.ArgumentParser(
        description="合并 LeRobot 数据集的碎片化文件"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="输入数据集根目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出数据集根目录",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="数据集 ID",
    )
    parser.add_argument(
        "--max_data_file_size_mb",
        type=int,
        default=200,
        help="单 data parquet 最大大小（MB）",
    )
    parser.add_argument(
        "--max_video_file_size_mb",
        type=int,
        default=500,
        help="单 video mp4 最大大小（MB）",
    )

    args = parser.parse_args()

    consolidate_dataset(
        data_root=args.data_root,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_data_file_size_mb=args.max_data_file_size_mb,
        max_video_file_size_mb=args.max_video_file_size_mb,
    )


if __name__ == "__main__":
    main()
