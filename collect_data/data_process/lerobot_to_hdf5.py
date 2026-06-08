r"""
LeRobot v3.0 → HDF5 格式转换工具

将 LeRobot v3.0 格式数据集（parquet + mp4）转换为 act-plus-plus 兼容的 HDF5 格式。
每条 episode 输出一个独立的 .hdf5 文件。

特性:
- 自动从 meta/info.json 读取相机名和 FPS
- 支持多 chunk 的 parquet 和 video 自动合并
- 自动计算 qvel (state差分/时间)

用法:
    cd D:\VLA\kortex_code\collect_data
    python data_process/lerobot_to_hdf5.py \
        --lerobot_root D:/VLA/data/simu_data \
        --output_dir D:/VLA/data/simu_hdf5

    # 只转换某些 episode
    python data_process/lerobot_to_hdf5.py \
        --lerobot_root D:/VLA/data/simu_data \
        --output_dir D:/VLA/data/simu_hdf5 \
        --episodes 0 1 2

    # 清除目标目录已有数据
    python data_process/lerobot_to_hdf5.py \
        --lerobot_root D:/VLA/data/simu_data \
        --output_dir D:/VLA/data/simu_hdf5 \
        --clean
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import cv2


def load_all_parquets(lerobot_root: Path) -> pd.DataFrame:
    """加载数据集的所有 parquet 文件（支持多 chunk）。"""
    data_root = lerobot_root / "data"
    all_dfs = []
    for chunk_dir in sorted(data_root.glob("chunk-*")):
        for pf in sorted(chunk_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(pf)
                all_dfs.append(df)
                print(f"  加载 {pf.relative_to(lerobot_root)}: {len(df)} 行")
            except Exception as e:
                print(f"  跳过损坏文件 {pf.relative_to(lerobot_root)}: {e}")
    if not all_dfs:
        raise FileNotFoundError(f"在 {data_root} 下未找到有效的 parquet 文件")
    return pd.concat(all_dfs, ignore_index=True)


def load_all_video_frames(lerobot_root: Path, camera_name: str, num_frames: int) -> list:
    """加载一个相机的所有视频文件帧（支持多 chunk 多文件 MP4 合并）。"""
    video_root = lerobot_root / "videos" / f"observation.images.{camera_name}"
    frames = []
    for chunk_dir in sorted(video_root.glob("chunk-*")):
        for video_file in sorted(chunk_dir.glob("*.mp4")):
            cap = cv2.VideoCapture(str(video_file))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if len(frames) >= num_frames:
                    cap.release()
                    return frames[:num_frames]
            cap.release()
    if not frames:
        print(f"  警告: 相机 {camera_name} 未找到视频文件")
    return frames


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot v3.0 → HDF5 格式转换"
    )
    parser.add_argument("--lerobot_root", type=str, required=True,
                        help="LeRobot v3.0 数据集根目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出 HDF5 目录")
    parser.add_argument("--episodes", type=int, nargs="*", default=None,
                        help="指定转换的 episode 索引（空格分隔），默认全部")
    parser.add_argument("--is_sim", action="store_true", default=True,
                        help="标记为仿真数据")
    parser.add_argument("--clean", action="store_true",
                        help="清除输出目录中已有的 hdf5 文件")
    args = parser.parse_args()

    lerobot_root = Path(args.lerobot_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (lerobot_root / "meta" / "info.json").exists():
        print(f"错误: 未找到 {lerobot_root / 'meta' / 'info.json'}")
        sys.exit(1)

    if args.clean:
        for h5 in output_dir.glob("*.hdf5"):
            h5.unlink()
            print(f"  删除旧文件: {h5.name}")

    with open(lerobot_root / "meta" / "info.json", "r") as f:
        info = json.load(f)

    fps = info["fps"]
    camera_names = [
        k.replace("observation.images.", "")
        for k in info["features"]
        if k.startswith("observation.images.")
    ]
    total_episodes = info["total_episodes"]
    total_frames = info["total_frames"]
    print(f"数据集: {total_episodes} episodes, {total_frames} frames, {fps} FPS")
    print(f"相机: {camera_names}")

    # 加载所有 parquet（支持多 chunk 自动合并）
    print("加载 parquet 数据...")
    df = load_all_parquets(lerobot_root)
    print(f"共 {len(df)} 行")

    # 按 episode_index 分组
    episode_frames = df.groupby("episode_index").apply(lambda x: x.index.tolist()).to_dict()
    available_eps = sorted(episode_frames.keys())

    if args.episodes is not None:
        selected_eps = [e for e in args.episodes if e in available_eps]
        missing = [e for e in args.episodes if e not in available_eps]
        if missing:
            print(f"警告: episode {missing} 不存在，跳过")
    else:
        selected_eps = available_eps

    if not selected_eps:
        print("没有可转换的 episode")
        print(f"可用 episode: {available_eps}")
        sys.exit(0)

    print(f"将转换 {len(selected_eps)} 个 episode: {selected_eps}")

    # 预加载所有相机视频帧（支持多 chunk 多文件 MP4 自动合并）
    print("解码视频...")
    cached_video_frames = {}
    for cam in camera_names:
        frames = load_all_video_frames(lerobot_root, cam, total_frames)
        if frames:
            cached_video_frames[cam] = np.stack(frames, axis=0)
            print(f"  {cam}: {cached_video_frames[cam].shape}")
        else:
            print(f"  {cam}: 无视频帧")

    for ep_idx in selected_eps:
        frame_idxs = episode_frames[ep_idx]
        T = len(frame_idxs)
        print(f"  Episode {ep_idx}: {T} frames")

        states = np.stack(df.loc[frame_idxs, "observation.state"].values, axis=0).astype(np.float32)
        actions = np.stack(df.loc[frame_idxs, "action"].values, axis=0).astype(np.float32)

        qvel = np.zeros_like(states)
        if T > 1:
            dt = 1.0 / fps
            qvel[1:] = (states[1:] - states[:-1]) / dt
            qvel[0] = qvel[1]

        images = {}
        for cam in camera_names:
            if cam in cached_video_frames:
                images[cam] = cached_video_frames[cam][frame_idxs]
            else:
                images[cam] = np.zeros((T, 480, 640, 3), dtype=np.uint8)

        output_path = output_dir / f"episode_{ep_idx}.hdf5"
        with h5py.File(str(output_path), "w") as root:
            root.attrs["sim"] = args.is_sim
            root.attrs["compress"] = False
            root.create_dataset("/observations/qpos", data=states)
            root.create_dataset("/observations/qvel", data=qvel)
            root.create_dataset("/action", data=actions)
            for cam in camera_names:
                root.create_dataset(f"/observations/images/{cam}", data=images[cam])

        print(f"    已保存: {output_path}")

    print(f"\n完成! HDF5 文件位于: {output_dir}")
    print(f"共 {len(selected_eps)} 个文件")


if __name__ == "__main__":
    main()
