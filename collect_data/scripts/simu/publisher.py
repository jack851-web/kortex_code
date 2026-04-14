"""
MuJoCo 仿真数据发布者

在独立线程中运行，以固定频率采集仿真状态并发布到 MessageBroker。
这是 Qt 与 MuJoCo 之间的唯一桥梁——Qt 和 DataCollector 都从 broker 订阅，
不再直接访问 SimuInterface。

关键设计:
- 分段锁策略：状态数据短锁采集，相机图像逐相机短锁渲染
  避免长时间持锁导致 GLFW 画面不更新（GLFW 使用非阻塞锁，拿不到锁就跳帧）
- 发布的 images 是 RGB 格式（MuJoCo 原生）
- 发布的 joints 是弧度制（MuJoCo 原生）
- DataCollector 订阅后直接使用，无需再调用 SimuInterface
"""
import threading
import time
import gc
import numpy as np
from typing import Optional, Dict, List

try:
    import mujoco
except ImportError:
    mujoco = None

from scripts.core.message_bus import MessageBroker
from scripts.core.topic_defs import (
    SIMU_IMAGES, SIMU_JOINTS, SIMU_GRIPPER,
    SIMU_TCP_POSE, SIMU_OBJECT_POS, SIMU_STATUS, SIMU_CARTESIAN,
    ALL_SIMU_TOPICS,
)


class SimuPublisher:
    """MuJoCo 仿真数据发布者

    在独立线程中运行，以固定频率采集仿真状态并发布到 MessageBroker。
    这是 Qt 与 MuJoCo 之间的唯一桥梁。
    """

    def __init__(self, simu_interface, broker: MessageBroker, fps: int = 20):
        self._simu = simu_interface
        self._broker = broker
        self._fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 注册所有仿真话题
        for topic_name in ALL_SIMU_TOPICS:
            self._broker.create_topic(topic_name)

        # 统计
        self._publish_count = 0
        self._last_publish_time = 0.0
        self._error_count = 0

    def start(self):
        """启动发布线程"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True, name="SimuPublisher")
        self._thread.start()
        print(f"[SimuPublisher] Started (fps={self._fps})")

    def stop(self):
        """停止发布线程"""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[SimuPublisher] WARNING: thread did not stop within timeout")
            self._thread = None

        # 线程停止后，清理残留的 renderer（正常情况下线程已清理自己的）
        # 这里作为兜底，防止资源泄漏
        if self._simu is not None:
            self._cleanup_remaining_renderers()

        print("[SimuPublisher] Stopped")

    def _cleanup_remaining_renderers(self):
        """清理残留的 renderer（线程停止后调用，作为兜底）"""
        if not self._simu._thread_renderers:
            return

        print(f"[SimuPublisher] Cleaning up {len(self._simu._thread_renderers)} remaining renderers...")
        for tid, renderer in list(self._simu._thread_renderers.items()):
            try:
                # MjrContext 可以在任何线程释放
                if hasattr(renderer, '_mjr_context') and renderer._mjr_context is not None:
                    try:
                        renderer._mjr_context.free()
                    except Exception:
                        pass
                    renderer._mjr_context = None
                if hasattr(renderer, '_scene') and renderer._scene is not None:
                    renderer._scene = None
                # GLContext 已经随着线程结束而失效，置空即可
                if hasattr(renderer, '_gl_context'):
                    renderer._gl_context = None
            except Exception as e:
                print(f"[SimuPublisher] Error cleaning renderer: {e}")
        self._simu._thread_renderers.clear()

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def publish_count(self) -> int:
        return self._publish_count

    def _publish_loop(self):
        """发布循环：采集仿真状态 → 发布到话题"""
        interval = 1.0 / self._fps if self._fps > 0 else 0.05
        my_tid = threading.current_thread().ident

        try:
            while not self._stop_event.is_set():
                try:
                    loop_start = time.time()
                    self._publish_once()
                    self._publish_count += 1

                    elapsed = time.time() - loop_start
                    sleep_time = max(0.0, interval - elapsed)
                    self._stop_event.wait(sleep_time)

                except Exception as e:
                    self._error_count += 1
                    self._broker.publish(SIMU_STATUS, f"error: {e}")
                    if self._error_count <= 3:
                        print(f"[SimuPublisher] Error in publish loop: {e}")
                    self._stop_event.wait(0.1)
        finally:
            # 线程退出前，在本线程内安全清理自己创建的 renderer
            # 这是关键：GLFW 窗口必须在创建它的线程中销毁
            self._cleanup_my_renderer(my_tid)

    def _cleanup_my_renderer(self, tid):
        """清理当前线程创建的 renderer（必须在线程内调用）"""
        if self._simu is None:
            return
        if tid not in self._simu._thread_renderers:
            return

        renderer = self._simu._thread_renderers.get(tid)
        if renderer is None:
            return

        print(f"[SimuPublisher] Cleaning up my renderer (tid={tid})...")
        try:
            import mujoco
            # 释放 MjrContext
            if hasattr(renderer, '_mjr_context') and renderer._mjr_context is not None:
                try:
                    renderer._mjr_context.free()
                except Exception:
                    pass
                renderer._mjr_context = None
            # 释放 MjvScene
            if hasattr(renderer, '_scene') and renderer._scene is not None:
                renderer._scene = None
            # 释放 GLContext（在创建线程中可以安全销毁）
            if hasattr(renderer, '_gl_context') and renderer._gl_context is not None:
                try:
                    # 调用 free() 会销毁 GLFW 窗口
                    renderer._gl_context.free()
                except Exception as e:
                    print(f"[SimuPublisher] GLContext.free() error: {e}")
                renderer._gl_context = None
        except Exception as e:
            print(f"[SimuPublisher] Error cleaning renderer: {e}")
        finally:
            # 从缓存中移除
            self._simu._thread_renderers.pop(tid, None)
            print(f"[SimuPublisher] Renderer cleaned up (tid={tid})")

    def _publish_once(self):
        """采集一帧仿真数据并发布

        采用分段锁策略：
        1. 短锁：快速采集轻量状态数据（joints/gripper/tcp/object_pos）
        2. 逐相机短锁：每个相机单独获取锁渲染，渲染间隙释放锁让 GLFW viewer 有机会渲染
        这样避免长时间持锁导致 GLFW 画面不更新。
        """
        simu = self._simu
        if simu is None or simu._model is None or simu._data is None:
            self._broker.publish(SIMU_STATUS, "idle")
            return

        try:
            # === 第1段：短锁采集状态数据 ===
            with simu._lock:
                # 模型重载期间 _model 可能为 None
                if simu._model is None or simu._data is None:
                    self._broker.publish(SIMU_STATUS, "idle")
                    return

                # 确保派生量（site_xpos, xpos 等）是最新的
                mujoco.mj_forward(simu._model, simu._data)

                # 关节状态（弧度）
                joints = np.copy(simu._data.qpos[:6]) if len(simu._data.qpos) >= 6 else np.zeros(6)

                # 夹爪状态
                gripper = self._get_gripper_raw(simu)

                # TCP 位姿
                tcp_pos, tcp_rot = self._get_tcp_raw(simu)

                # 物体位置
                obj_pos = self._get_object_pos_raw(simu)

                # 笛卡尔位姿
                cartesian = self._get_cartesian_raw(simu)

            # === 第2段：逐相机短锁渲染（释放锁间隙让 GLFW viewer 渲染）===
            images = self._render_all_cameras_per_camera(simu)

            # === 发布到 broker ===
            self._broker.publish(SIMU_IMAGES, images)
            self._broker.publish(SIMU_JOINTS, joints)
            self._broker.publish(SIMU_GRIPPER, gripper)
            self._broker.publish(SIMU_TCP_POSE, (tcp_pos, tcp_rot))
            self._broker.publish(SIMU_OBJECT_POS, obj_pos)
            self._broker.publish(SIMU_CARTESIAN, cartesian)
            self._broker.publish(SIMU_STATUS, "running")

            self._last_publish_time = time.time()

        except Exception as e:
            self._broker.publish(SIMU_STATUS, f"error: {e}")
            raise

    def _render_all_cameras_per_camera(self, simu) -> Dict[str, np.ndarray]:
        """逐相机短锁渲染，每个相机单独获取锁，渲染间隙释放锁让 GLFW viewer 有机会渲染

        这是解决 GLFW 画面不更新问题的关键：之前的 _render_all_cameras 在一次
        长锁中渲染所有相机，导致 GLFW 非阻塞锁获取失败，画面无法更新。
        """
        camera_names = simu._camera_names
        if not camera_names:
            return {}

        images = {}

        # 进程渲染器模式：不走本地渲染
        if simu._use_process_renderer and simu._render_process:
            # 补齐黑帧
            h, w = simu._render_height, simu._render_width
            for cam_name in camera_names:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            return images

        # 本地渲染器模式：逐相机短锁渲染
        # 获取/创建 Renderer 需要加锁，因为涉及 GLFW 操作
        with simu._lock:
            renderer = self._get_or_create_renderer(simu)
        if renderer is None or mujoco is None:
            h, w = simu._render_height, simu._render_width
            for cam_name in camera_names:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            return images

        for cam_name in camera_names:
            try:
                # 每个相机单独获取锁，渲染后立即释放
                with simu._lock:
                    # 模型重载期间 _model 可能为 None，跳过渲染
                    if simu._model is None:
                        h, w = simu._render_height, simu._render_width
                        images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
                        continue

                    cam_id = mujoco.mj_name2id(simu._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    if cam_id >= 0:
                        mujoco.mj_forward(simu._model, simu._data)
                        img = self._render_with_retry(renderer, simu, cam_id)
                        if img is not None:
                            images[cam_name] = np.copy(img)
                        else:
                            h, w = simu._render_height, simu._render_width
                            images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
                    else:
                        h, w = simu._render_height, simu._render_width
                        images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[SimuPublisher] render error for {cam_name}: {e}")
                h, w = simu._render_height, simu._render_width
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

            # 每个相机渲染后主动让出，给 GLFW viewer 渲染机会
            time.sleep(0.001)

        # 补齐缺失的相机（黑帧）
        h, w = simu._render_height, simu._render_width
        for cam_name in camera_names:
            if cam_name not in images:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        return images

    def _get_or_create_renderer(self, simu):
        """获取或创建当前线程的渲染器"""
        import mujoco

        tid = threading.current_thread().ident
        # 检查是否有缓存的渲染器
        if tid in simu._thread_renderers:
            return simu._thread_renderers[tid]

        # 使用主渲染器（如果是在主线程创建的）
        if simu._renderer is not None and simu._renderer_thread_id == tid:
            return simu._renderer

        # 创建新的线程渲染器
        try:
            renderer = mujoco.Renderer(simu._model, height=simu._render_height, width=simu._render_width)
            simu._thread_renderers[tid] = renderer
            return renderer
        except Exception as e:
            print(f"[SimuPublisher] Failed to create renderer: {e}")
            return None

    def _render_with_retry(self, renderer, simu, cam_id):
        """带重试的渲染：scene 初始化失败时重建 renderer 再试一次"""
        import mujoco
        for attempt in range(2):
            try:
                renderer.update_scene(simu._data, camera=cam_id)
                return renderer.render()
            except Exception as e:
                if attempt == 0 and 'mjv_updateScene' in str(e):
                    # 首次 scene 初始化失败，重建 renderer
                    try:
                        new_renderer = mujoco.Renderer(
                            simu._model,
                            height=simu._render_height,
                            width=simu._render_width,
                        )
                        # 替换线程缓存的旧 renderer
                        tid = threading.current_thread().ident
                        if tid in simu._thread_renderers:
                            simu._thread_renderers[tid] = new_renderer
                        elif hasattr(simu, '_renderer') and simu._renderer is renderer:
                            simu._renderer = new_renderer
                        return self._render_with_retry(new_renderer, simu, cam_id)
                    except Exception as e2:
                        print(f"[SimuPublisher] renderer rebuild failed: {e2}")
                else:
                    print(f"[SimuPublisher] render error (attempt {attempt + 1}): {e}")
                    return None
        return None

    def _get_gripper_raw(self, simu) -> float:
        """获取夹爪原始状态（在锁内调用）"""
        try:
            if not simu._gripper_indices:
                return 0.0
            values = []
            for idx in simu._gripper_indices:
                qpos_idx = simu._model.jnt_qposadr[idx]
                values.append(simu._data.qpos[qpos_idx])
            return float(np.mean(values)) if values else 0.0
        except Exception:
            return 0.0

    def _get_tcp_raw(self, simu):
        """获取 TCP 位姿（在锁内调用）"""
        try:
            tcp_id = mujoco.mj_name2id(simu._model, mujoco.mjtObj.mjOBJ_SITE, simu._tcp_site_name)
            if tcp_id < 0:
                return np.zeros(3), np.eye(3)
            pos = np.copy(simu._data.site_xpos[tcp_id])
            rot = np.copy(simu._data.site_xmat[tcp_id].reshape(3, 3))
            return pos, rot
        except Exception:
            return np.zeros(3), np.eye(3)

    def _get_object_pos_raw(self, simu) -> np.ndarray:
        """获取物体位置（在锁内调用）"""
        try:
            body_name = simu._active_object_body_name
            body_id = mujoco.mj_name2id(simu._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                return np.zeros(3)
            return np.copy(simu._data.xpos[body_id])
        except Exception:
            return np.zeros(3)

    def _get_cartesian_raw(self, simu) -> np.ndarray:
        """获取笛卡尔位姿（在锁内调用）"""
        try:
            tcp_pos, tcp_rot = self._get_tcp_raw(simu)
            # 将旋转矩阵转为欧拉角 (XYZ 外旋)
            sy = np.sqrt(tcp_rot[0, 0] ** 2 + tcp_rot[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(tcp_rot[2, 1], tcp_rot[2, 2])
                y = np.arctan2(-tcp_rot[2, 0], sy)
                z = np.arctan2(tcp_rot[1, 0], tcp_rot[0, 0])
            else:
                x = np.arctan2(-tcp_rot[1, 2], tcp_rot[1, 1])
                y = np.arctan2(-tcp_rot[2, 0], sy)
                z = 0.0
            return np.concatenate([tcp_pos, np.rad2deg([x, y, z])])
        except Exception:
            return np.zeros(6)
