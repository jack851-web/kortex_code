"""
真实机器人数据收集器 - LeRobot API 直录版

核心设计：
- 使用 LeRobotDataset.create() + streaming_encoding 直接写入 LeRobot v3.0 格式
- 每帧调用 add_frame()，视频实时流式编码为 MP4
- observation.state: [6关节角(度), 1夹爪] = 7维 float32
- action: [6关节增量(度), 1夹爪增量] = 7维 float32
- end_episode 时调用 save_episode()，停止时调用 finalize()
- 保持外部接口与旧版完全兼容
"""
import json
import time
import gc
import numpy as np
import threading
from typing import Dict, List, Any
from pathlib import Path


MAX_EPISODE_FRAMES = 5000
CAMERA_KEY_PREFIX = "observation.images"


def _build_features(camera_names: List[str], image_height: int = 480, image_width: int = 640) -> dict:
    """根据相机列表构建 LeRobot features 字典"""
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dj1", "dj2", "dj3", "dj4", "dj5", "dj6", "dgripper"],
        },
    }
    for cam_name in camera_names:
        key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
        features[key] = {
            "dtype": "video",
            "shape": (image_height, image_width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


class RealDataCollector:
    """真实机器人数据收集器 - LeRobot API 直录版

    使用 LeRobotDataset.create(streaming_encoding=True) 直接写入 LeRobot v3.0 格式。
    外部接口与旧版完全兼容。

    数据源：
    - broker 模式：从 MessageBroker 订阅 RealPublisher 发布的数据
    - 直接模式：直接调用 RealInterface.get_camera_images()
    """

    def __init__(self, real_interface, data_root: str, fps: int = 20, video_fps: int = 30,
                 broker=None, repo_id: str = "kortex_real_dataset",
                 image_size: tuple = (480, 640),
                 use_videos: bool = True):
        self._real = real_interface
        self._data_root = Path(data_root)
        self._fps = fps
        self._video_fps = video_fps
        self._use_videos = use_videos
        self._image_height, self._image_width = image_size
        self._episode_count = 0
        self._is_collecting = False
        self._is_recording = False
        self._collect_thread = None
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()
        self._repo_id = repo_id

        # 任务数据保存点
        self._task_save_point = 0
        self._last_task_end_point = 0

        # 帧计数
        self._frame_count = 0
        self._current_episode_info = {}
        self._camera_names = []

        # 上一帧的 state（用于计算增量 action）
        self._prev_state = None

        # LeRobot 数据集实例
        self._dataset = None

        # 任务描述
        self._task_description = ""

        # Broker 模式
        self._broker = broker
        self._latest_broker_images = None
        self._latest_broker_joints = None
        self._latest_broker_cartesian = None
        self._latest_broker_gripper = None

        if self._broker is not None:
            from scripts.core.topic_defs import REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER
            self._broker.subscribe(REAL_IMAGES, self._on_broker_images)
            self._broker.subscribe(REAL_JOINTS, self._on_broker_joints)
            self._broker.subscribe(REAL_CARTESIAN, self._on_broker_cartesian)
            self._broker.subscribe(REAL_GRIPPER, self._on_broker_gripper)

        self._save_progress()

    # ================================================================
    # Broker 回调
    # ================================================================

    def _on_broker_images(self, images):
        if images is not None:
            self._latest_broker_images = {k: np.array(v, copy=True) for k, v in images.items()}

    def _on_broker_joints(self, joints):
        if joints is not None:
            self._latest_broker_joints = np.copy(joints)

    def _on_broker_cartesian(self, cartesian):
        if cartesian is not None:
            self._latest_broker_cartesian = np.copy(cartesian)

    def _on_broker_gripper(self, gripper):
        if gripper is not None:
            self._latest_broker_gripper = float(gripper)

    def _init_lerobot_dataset(self):
        """创建或恢复 LeRobot 数据集"""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        features = _build_features(
            self._camera_names,
            image_height=self._image_height,
            image_width=self._image_width,
        )

        if not self._use_videos:
            for cam_name in self._camera_names:
                key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                features[key]["dtype"] = "image"

        try:
            self._dataset = LeRobotDataset.create(
                repo_id=self._repo_id,
                fps=self._fps,
                features=features,
                root=self._data_root,
                robot_type="kortex",
                use_videos=self._use_videos,
                streaming_encoding=self._use_videos,
                vcodec="libsvtav1",
                metadata_buffer_size=10,
                encoder_queue_maxsize=30,
            )
            print(f"[RealDataCollector] New LeRobot dataset created at {self._data_root}")
        except FileExistsError:
            print(f"[RealDataCollector] Loading existing LeRobot dataset from {self._data_root}")
            self._dataset = LeRobotDataset(
                repo_id=self._repo_id,
                root=self._data_root,
            )
            self._dataset.episode_buffer = self._dataset.create_episode_buffer()

    def start_collection(self):
        if self._is_collecting:
            return
        self._stop_event.clear()
        self._is_collecting = True
        self._collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._collect_thread.start()
        print("[RealDataCollector] Started (LeRobot mode)")

    def stop_collection(self):
        if not self._is_collecting:
            return
        self._is_collecting = False
        self._stop_event.set()

        if self._collect_thread:
            self._collect_thread.join(timeout=2.0)

        self._is_recording = False

        if self._dataset is not None and self._dataset.episode_buffer is not None:
            if self._dataset.episode_buffer.get("size", 0) > 0:
                try:
                    self._dataset.save_episode()
                    print("[RealDataCollector] Saved remaining episode buffer on stop")
                except Exception as e:
                    print(f"[RealDataCollector] Warning: failed to save remaining buffer: {e}")
            try:
                self._dataset.finalize()
            except Exception as e:
                print(f"[RealDataCollector] Warning: finalize error: {e}")
            self._dataset = None

        print("[RealDataCollector] Stopped")

    def _collect_loop(self):
        print(f"[RealDataCollector] Loop started (target FPS: {self._fps})")

        while not self._stop_event.is_set():
            loop_start = time.time()

            try:
                if not self._is_recording:
                    time.sleep(0.01)
                    continue

                if self._frame_count >= MAX_EPISODE_FRAMES:
                    time.sleep(0.01)
                    continue

                timestamp = time.time()

                # 获取数据
                if self._broker is not None and self._latest_broker_images is not None:
                    images = self._latest_broker_images
                    joints = self._latest_broker_joints if self._latest_broker_joints is not None else np.zeros(6)
                    gripper = self._latest_broker_gripper if self._latest_broker_gripper is not None else 0.0
                else:
                    images = self._real.get_camera_images()
                    joints, cartesian, gripper = self._real.get_full_state()

                with self._data_lock:
                    # 构建 state 向量: [6关节角(度), 1夹爪]
                    state_vec = np.array(
                        list(joints) + [float(gripper)],
                        dtype=np.float32,
                    )

                    # 构建 action: 增量
                    if self._prev_state is not None:
                        action_vec = (state_vec - self._prev_state).astype(np.float32)
                    else:
                        action_vec = np.zeros(7, dtype=np.float32)
                    self._prev_state = state_vec.copy()

                    # 构建 LeRobot frame
                    frame = {
                        "observation.state": state_vec,
                        "action": action_vec,
                        "task": self._task_description,
                    }

                    # 图像：OpenCV 输出 BGR，LeRobot 需要 RGB
                    import cv2
                    for cam_name, img in images.items():
                        key = f"{CAMERA_KEY_PREFIX}.{cam_name}"
                        if isinstance(img, np.ndarray):
                            if img.ndim == 3 and img.shape[2] == 3:
                                # BGR -> RGB
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                frame[key] = img_rgb.astype(np.uint8)
                            else:
                                frame[key] = img
                        else:
                            frame[key] = img

                    self._dataset.add_frame(frame)
                    self._frame_count += 1

            except Exception as e:
                print(f"[RealDataCollector] Error: {e}")
                import traceback
                traceback.print_exc()

            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / self._fps) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_episode(self, episode_id: int, camera_names: list, task_info: Dict[str, Any]):
        """开始一个新的 episode"""
        self._episode_count = episode_id
        self._camera_names = camera_names

        # 解析任务描述
        self._task_description = self._parse_task_description(task_info)

        with self._data_lock:
            self._frame_count = 0
            self._prev_state = None
            self._current_episode_info = {
                "episode_id": episode_id,
                "task_id": task_info.get("task_id", episode_id),
                "task_name": task_info.get("task_name", ""),
                "description": task_info.get("description", ""),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

            if self._dataset is None:
                self._init_lerobot_dataset()

            self._dataset.episode_buffer = self._dataset.create_episode_buffer()

        print(f"[RealDataCollector] Episode {episode_id} started (LeRobot mode, task: {self._task_description})")

    def start_recording(self):
        with self._data_lock:
            self._task_save_point = self._frame_count
        self._is_recording = True
        print(f"[RealDataCollector] Recording started, save point: frame={self._task_save_point}")

    def stop_recording(self):
        self._is_recording = False
        with self._data_lock:
            self._last_task_end_point = self._frame_count
        print("[RealDataCollector] Recording stopped")

    def delete_last_episode(self) -> bool:
        """删除最后一个已保存的 episode（用于重做任务时清理旧数据）

        注意：此方法会重新编码视频文件，确保数据一致性。

        Returns:
            bool: 是否成功删除
        """
        import pandas as pd
        import av
        import shutil

        with self._data_lock:
            if self._dataset is None:
                print("[RealDataCollector] No dataset to delete from")
                return False

            # 获取当前 episode 数量
            total_episodes = self._dataset.meta.total_episodes
            if total_episodes == 0:
                print("[RealDataCollector] No episodes to delete")
                return False

            last_episode_idx = total_episodes - 1
            print(f"[RealDataCollector] Deleting episode {last_episode_idx} (with video re-encoding)...")

            try:
                # 1. 从 parquet 数据中删除该 episode 的行
                data_dir = self._data_root / "data" / "chunk-000"
                if data_dir.exists():
                    for parquet_file in data_dir.glob("*.parquet"):
                        try:
                            df = pd.read_parquet(parquet_file)
                        except Exception as e:
                            print(f"[RealDataCollector] Warning: Failed to read {parquet_file.name}: {e}")
                            try:
                                parquet_file.unlink()
                                print(f"[RealDataCollector] Deleted corrupted file: {parquet_file.name}")
                            except Exception:
                                pass
                            continue
                        if "episode_index" in df.columns:
                            original_len = len(df)
                            df = df[df["episode_index"] != last_episode_idx]
                            if len(df) < original_len:
                                df.to_parquet(parquet_file, index=False)
                                print(f"[RealDataCollector] Removed episode {last_episode_idx} from {parquet_file.name}")

                # 2. 重新编码视频（移除被删除episode的帧）
                self._reencode_videos_without_episode(last_episode_idx)

                # 3. 更新 meta/info.json
                info_path = self._data_root / "meta" / "info.json"
                if info_path.exists():
                    with open(info_path, "r") as f:
                        info = json.load(f)
                    info["total_episodes"] = total_episodes - 1
                    # 重新计算 total_frames
                    total_frames = 0
                    if data_dir.exists():
                        for pf in data_dir.glob("*.parquet"):
                            try:
                                pdf = pd.read_parquet(pf)
                                total_frames += len(pdf)
                            except Exception:
                                pass
                    info["total_frames"] = total_frames
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=2)
                    print(f"[RealDataCollector] Updated info.json: total_episodes = {total_episodes - 1}")

                # 4. 更新 meta/tasks.parquet
                tasks_path = self._data_root / "meta" / "tasks.parquet"
                if tasks_path.exists():
                    try:
                        tasks_df = pd.read_parquet(tasks_path)
                        if "episode" in tasks_df.columns:
                            tasks_df = tasks_df[tasks_df["episode"] != last_episode_idx]
                            tasks_df.to_parquet(tasks_path, index=False)
                            print(f"[RealDataCollector] Removed episode {last_episode_idx} from tasks.parquet")
                    except Exception as e:
                        print(f"[RealDataCollector] Warning: Failed to update tasks.parquet: {e}")

                # 5. 更新 meta/episodes parquet
                self._update_episodes_parquet_after_delete(last_episode_idx)

                # 6. 更新 dataset 元数据
                self._dataset.meta.total_episodes = total_episodes - 1

                print(f"[RealDataCollector] Episode {last_episode_idx} deleted successfully")
                return True

            except Exception as e:
                print(f"[RealDataCollector] Failed to delete episode: {e}")
                import traceback
                traceback.print_exc()
                return False

    def _reencode_videos_without_episode(self, episode_to_delete: int):
        """重新编码视频，移除指定episode的帧"""
        import av
        import shutil
        import pandas as pd

        # 获取所有视频目录
        videos_dir = self._data_root / "videos"
        if not videos_dir.exists():
            return

        # 读取parquet获取每个episode的帧范围
        data_dir = self._data_root / "data" / "chunk-000"
        if not data_dir.exists():
            return

        # 收集所有episode的帧范围
        episode_frames = {}  # episode_index -> list of frame indices
        for parquet_file in sorted(data_dir.glob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_file)
            except Exception:
                continue  # 跳过损坏的文件
            if "episode_index" in df.columns and "frame_index" in df.columns:
                for ep_idx in df["episode_index"].unique():
                    if ep_idx not in episode_frames:
                        episode_frames[ep_idx] = []
                    ep_df = df[df["episode_index"] == ep_idx]
                    episode_frames[ep_idx].extend(ep_df["frame_index"].tolist())

        if not episode_frames:
            return

        # 确定要保留的帧索引（全局）
        frames_to_keep = set()
        for ep_idx, frames in episode_frames.items():
            if ep_idx != episode_to_delete:
                frames_to_keep.update(frames)

        # 对每个视频key进行处理
        for video_key_dir in videos_dir.iterdir():
            if not video_key_dir.is_dir():
                continue
            video_key = video_key_dir.name

            # 处理每个视频文件
            for chunk_dir in video_key_dir.glob("chunk-*"):
                for video_file in sorted(chunk_dir.glob("*.mp4")):
                    print(f"[RealDataCollector] Re-encoding {video_file}...")

                    # 创建临时文件
                    temp_video = video_file.with_suffix(".temp.mp4")

                    try:
                        # 使用PyAV解码和重新编码
                        with av.open(str(video_file)) as input_container:
                            input_stream = input_container.streams.video[0]

                            # 创建输出容器
                            with av.open(str(temp_video), 'w') as output_container:
                                output_stream = output_container.add_stream('libx264', rate=30)
                                output_stream.width = input_stream.width
                                output_stream.height = input_stream.height
                                output_stream.pix_fmt = 'yuv420p'
                                output_stream.options = {'crf': '23'}

                                frame_idx = 0
                                for packet in input_container.demux(input_stream):
                                    for frame in packet.decode():
                                        if frame_idx in frames_to_keep:
                                            # 重新编码这帧
                                            frame.pict_type = None  # 重置帧类型让编码器决定
                                            output_container.mux(output_stream.encode(frame))
                                        frame_idx += 1

                                # 刷新编码器
                                for packet in output_stream.encode():
                                    output_container.mux(packet)

                        # 替换原文件
                        temp_video.replace(video_file)
                        print(f"[RealDataCollector] Re-encoded {video_file.name} ({frame_idx} frames, kept {len(frames_to_keep)})")

                    except Exception as e:
                        print(f"[RealDataCollector] Failed to re-encode {video_file}: {e}")
                        if temp_video.exists():
                            temp_video.unlink()

    def _update_episodes_parquet_after_delete(self, deleted_episode_idx: int):
        """删除episode后更新meta/episodes parquet文件"""
        import pandas as pd

        episodes_dir = self._data_root / "meta" / "episodes"
        if not episodes_dir.exists():
            return

        for chunk_dir in episodes_dir.glob("chunk-*"):
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(parquet_file)
                except Exception as e:
                    print(f"[RealDataCollector] Warning: Failed to read {parquet_file.name}: {e}")
                    continue
                if "episode_index" in df.columns:
                    original_len = len(df)
                    df = df[df["episode_index"] != deleted_episode_idx]
                    if len(df) < original_len:
                        df.to_parquet(parquet_file, index=False)
                        print(f"[RealDataCollector] Updated {parquet_file.name}")

    def discard_current_task(self):
        """丢弃当前任务的数据"""
        with self._data_lock:
            save_point = self._task_save_point
            self._frame_count = save_point
            self._prev_state = None

            if self._dataset is not None:
                ep_idx = self._dataset.meta.total_episodes
                self._dataset.episode_buffer = self._dataset.create_episode_buffer(ep_idx)

            print(f"[RealDataCollector] Current task data discarded, reverted to frame={save_point}")

    def end_episode(self, episode_id: int, success: bool = True):
        """结束 episode，保存到 LeRobot 数据集"""
        self._is_recording = False

        with self._data_lock:
            self._current_episode_info["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self._current_episode_info["success"] = success

            t0 = time.time()
            num_frames = self._frame_count

            if self._dataset is not None and num_frames > 0:
                try:
                    self._dataset.save_episode()
                    self._patch_info_json()  # 补全 total_videos 等字段
                    print(f"[RealDataCollector] Episode {episode_id} saved to LeRobot ({num_frames} frames)")
                except Exception as e:
                    print(f"[RealDataCollector] Save error: {e}")
                    import traceback
                    traceback.print_exc()
            elif num_frames == 0:
                print(f"[RealDataCollector] Episode {episode_id}: No frames collected, skipping save")

            elapsed = time.time() - t0
            print(f"[RealDataCollector] Episode {episode_id} completed in {elapsed:.1f}s")

            self._frame_count = 0
            self._prev_state = None
            self._task_save_point = 0
            self._last_task_end_point = 0

            if self._dataset is not None:
                self._dataset.episode_buffer = self._dataset.create_episode_buffer()

            gc.collect()

            return num_frames > 0

    def _patch_info_json(self):
        """补全 info.json 和缺失的元数据文件（rerun 可视化需要）"""
        info_path = self._data_root / "meta" / "info.json"
        if not info_path.exists():
            return

        try:
            with open(info_path, "r") as f:
                info = json.load(f)

            patched = False

            # total_videos: 视频特征数 × episodes
            if "total_videos" not in info:
                video_count = sum(1 for feat in info.get("features", {}).values()
                                  if isinstance(feat, dict) and feat.get("dtype") == "video")
                info["total_videos"] = video_count * info.get("total_episodes", 0)
                patched = True

            # total_chunks: 实际数据目录中的 chunk 文件夹数
            if "total_chunks" not in info:
                data_dir = self._data_root / "data"
                if data_dir.exists():
                    chunks = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
                    info["total_chunks"] = len(chunks)
                else:
                    info["total_chunks"] = 0
                patched = True

            if patched:
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
                print(f"[RealDataCollector] Patched info.json")

            # 生成 episodes parquet（rerun 需要）
            self._write_episodes_parquet()

        except Exception as e:
            print(f"[RealDataCollector] Warning: failed to patch metadata: {e}")

    def _write_episodes_parquet(self):
        """生成 meta/episodes/chunk-000/file-000.parquet（v3 格式，rerun 可视化需要）"""
        import pandas as pd

        episodes_dir = self._data_root / "meta" / "episodes"
        episodes_path = episodes_dir / "chunk-000" / "file-000.parquet"
        if episodes_path.exists():
            return

        tasks_path = self._data_root / "meta" / "tasks.parquet"
        if not tasks_path.exists():
            return

        try:
            tasks_df = pd.read_parquet(tasks_path)
            total_episodes = len(tasks_df)

            rows = []
            for idx, row in tasks_df.iterrows():
                ep_idx = int(row.get("episode", idx))
                task_desc = str(row.get("task", "")) or "Grasp the object"
                length = self._estimate_episode_length(idx, total_episodes)

                rows.append({
                    "episode_index": ep_idx,
                    "tasks": [task_desc],
                    "length": length,
                    "meta/episodes/chunk_index": 0,
                    "meta/episodes/file_index": 0,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                })

            df = pd.DataFrame(rows)
            df.to_parquet(episodes_path, index=False)

            # 删除旧的 .jsonl
            legacy_jsonl = self._data_root / "meta" / "episodes.jsonl"
            if legacy_jsonl.exists():
                legacy_jsonl.unlink()

            print(f"[RealDataCollector] Generated {episodes_path} ({total_episodes} episodes)")
        except Exception as e:
            print(f"[RealDataCollector] Warning: failed to generate episodes parquet: {e}")

    def _estimate_episode_length(self, episode_index, total_episodes):
        """估算单个 episode 的帧数"""
        stats_path = self._data_root / "meta" / "stats.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                state_key = "observation.state"
                if state_key in stats:
                    return stats[state_key].get("count", 0) // total_episodes
            except Exception:
                pass
        return 0

    def finalize(self):
        """关闭数据集"""
        if self._dataset is not None:
            try:
                self._dataset.finalize()
                print("[RealDataCollector] Dataset finalized")
            except Exception as e:
                print(f"[RealDataCollector] Finalize error: {e}")
            # 补全 rerun 需要的字段
            self._patch_info_json()
            self._dataset = None

    def _parse_task_description(self, task_info: Dict[str, Any]) -> str:
        """从 task_info 解析出任务描述字符串"""
        if "tasks" in task_info:
            tasks = task_info["tasks"]
            if isinstance(tasks, list) and len(tasks) > 0:
                last_task = tasks[-1]
                desc = last_task.get("task_name", "") or last_task.get("description", "")
                if desc:
                    return desc

        desc = task_info.get("task_name", "") or task_info.get("description", "")
        return desc if desc else "Grasp the object"

    def _save_progress(self):
        self._data_root.mkdir(parents=True, exist_ok=True)
        progress_file = self._data_root / "progress.json"
        with open(progress_file, "w") as f:
            json.dump({"episode_count": self._episode_count}, f, indent=2)

    def load_progress(self) -> int:
        progress_file = self._data_root / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                self._episode_count = json.load(f).get("episode_count", 0)
        return self._episode_count
