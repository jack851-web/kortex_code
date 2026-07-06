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
import logging
import traceback
import numpy as np
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

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

    def __init__(self, simu_interface, broker: MessageBroker, fps: int = 20, auto_step: bool = True):
        self._simu = simu_interface
        self._broker = broker
        self._fps = fps
        self._auto_step = auto_step  # 是否自动推进仿真（无外部控制循环时设为 True）
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

    def set_auto_step(self, auto_step: bool):
        """设置是否自动推进仿真

        当有外部控制循环（TeleopController/GraspExecutor）时，应设为 False，
        由控制循环负责 step()，避免双重推进导致 simtime 回溯和锁竞争。
        当控制循环停止时，应设为 True，让 Publisher 自动推进仿真保持画面更新。
        """
        self._auto_step = auto_step

    def start(self):
        """启动发布线程"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True, name="SimuPublisher")
        self._thread.start()
        logger.info(f"Started (fps={self._fps})")

    def stop(self):
        """停止发布线程"""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                logger.warning("thread did not stop within timeout")
            self._thread = None

        # 线程停止后，清理残留的 renderer（正常情况下线程已清理自己的）
        # 这里作为兜底，防止资源泄漏
        if self._simu is not None:
            self._cleanup_remaining_renderers()

        logger.info("Stopped")

    def _cleanup_remaining_renderers(self):
        """清理残留的 renderer（线程停止后调用，作为兜底）"""
        if not self._simu.get_thread_renderer_tids():
            return

        logger.info("Cleaning up remaining renderers...")
        self._simu.clear_thread_renderers()

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
                        logger.error(f"Error in publish loop: {e}")
                    self._stop_event.wait(0.1)
        finally:
            # 线程退出前，在本线程内安全清理自己创建的 renderer
            # 这是关键：GLFW 窗口必须在创建它的线程中销毁
            self._cleanup_my_renderer(my_tid)

    def _cleanup_my_renderer(self, tid):
        """清理当前线程创建的 renderer（必须在线程内调用）"""
        if self._simu is None:
            return
        if not self._simu.has_thread_renderer(tid):
            return

        renderer = self._simu.get_thread_renderer(tid)
        if renderer is None:
            return

        logger.info(f"Cleaning up my renderer (tid={tid})...")
        try:
            # 释放 MjrContext
            if hasattr(renderer, '_mjr_context') and renderer._mjr_context is not None:
                try:
                    renderer._mjr_context.free()
                except Exception:
                    logger.debug("Failed to free mjr_context in publisher", exc_info=True)
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
                    logger.warning(f"GLContext.free() error: {e}")
                renderer._gl_context = None
        except Exception as e:
            logger.error(f"Error cleaning renderer: {e}")
        finally:
            # 从缓存中移除
            self._simu.pop_thread_renderer(tid)
            logger.info(f"Renderer cleaned up (tid={tid})")

    def _publish_once(self):
        """采集一帧仿真数据并发布

        采用数据副本分离策略：
        1. 短锁：快速采集状态数据 + 创建数据副本（用于无锁渲染）
        2. 无锁渲染：用副本数据渲染相机图像，不持主锁

        重要：不在 Publisher 中调用 mj_step 推进仿真！
        仿真推进由 step() / TeleopController / GraspExecutor 等控制循环负责。
        Publisher 只读取当前状态，避免与控制循环的 step() 产生 simtime 回溯和锁竞争。
        """
        simu = self._simu
        if simu is None or simu._model is None or simu._data is None:
            self._broker.publish(SIMU_STATUS, "idle")
            return

        try:
            # === 第1段：短锁采集状态数据 + 创建数据副本 ===
            data_copy = None
            with simu._lock:
                # 模型重载期间 _model 可能为 None
                if simu._model is None or simu._data is None:
                    self._broker.publish(SIMU_STATUS, "idle")
                    return

                # 仅在 auto_step 模式下推进仿真
                # 当有外部控制循环（TeleopController/GraspExecutor）时，auto_step=False，
                # 由控制循环负责 step()，避免双重推进导致 simtime 回溯和锁竞争
                if self._auto_step:
                    # 在锁内直接调用 mj_step 并更新 tick（与 simu.step() 保持一致）
                    simu._sync_gripper_tips()
                    mujoco.mj_step(simu._model, simu._data, nstep=1)
                    simu._tick += 1

                # 更新派生量（site_xpos, xpos 等），不改变 data.time
                mujoco.mj_forward(simu._model, simu._data)

                # 关节状态（弧度）- 通过 jnt_qposadr 正确索引，不假设 qpos 连续排列
                joints = np.zeros(6)
                for i, joint_idx in enumerate(simu._joint_indices):
                    if i >= 6:
                        break
                    qpos_idx = simu._model.jnt_qposadr[joint_idx]
                    joints[i] = simu._data.qpos[qpos_idx]

                # 夹爪状态
                gripper = self._get_gripper_raw(simu)

                # TCP 位姿
                tcp_pos, tcp_rot = self._get_tcp_raw(simu)

                # 物体位置
                obj_pos = self._get_object_pos_raw(simu)

                # 笛卡尔位姿
                cartesian = self._get_cartesian_raw(simu)

                # 创建数据副本用于无锁渲染（约1ms，显著减少渲染时的锁竞争）
                if simu._camera_names:
                    try:
                        data_copy = mujoco.MjData(simu._model)
                        mujoco.mj_copyData(data_copy, simu._model, simu._data)
                    except Exception:
                        data_copy = None

            # === 第2段：无锁渲染（用副本数据，不持主锁）===
            if data_copy is not None:
                images = self._render_all_cameras_with_copy(simu, data_copy)
            else:
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
                logger.warning(f"render error for {cam_name}: {e}")
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

    def _render_all_cameras_with_copy(self, simu, data_copy) -> Dict[str, np.ndarray]:
        """使用数据副本无锁渲染所有相机

        关键优化：渲染时不持 simu._lock，消除与控制循环的锁竞争。
        渲染线程使用独立的 data_copy，不影响主仿真的 data。

        Args:
            simu: 仿真接口
            data_copy: 主仿真数据的副本（在锁内创建）
        """
        camera_names = simu._camera_names
        if not camera_names:
            return {}

        images = {}

        # 进程渲染器模式：不走本地渲染
        if simu._use_process_renderer and simu._render_process:
            h, w = simu._render_height, simu._render_width
            for cam_name in camera_names:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            return images

        # 本地渲染器模式：无锁渲染
        # 获取/创建 Renderer 仍需短锁（涉及 GLFW 操作）
        with simu._lock:
            renderer = self._get_or_create_renderer(simu)
        if renderer is None or mujoco is None or simu._model is None:
            h, w = simu._render_height, simu._render_width
            for cam_name in camera_names:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            return images

        # 缓存 model 引用，避免渲染过程中被其他线程置 None
        model = simu._model

        # 无锁渲染所有相机（使用 data_copy）
        for cam_name in camera_names:
            try:
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id >= 0:
                    # 用副本数据更新场景并渲染，不持主锁
                    renderer.update_scene(data_copy, camera=cam_id)
                    img = renderer.render()
                    images[cam_name] = np.copy(img)
                else:
                    h, w = simu._render_height, simu._render_width
                    images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)
            except Exception as e:
                logger.warning(f"render error for {cam_name}: {e}")
                h, w = simu._render_height, simu._render_width
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        # 补齐缺失的相机（黑帧）
        h, w = simu._render_height, simu._render_width
        for cam_name in camera_names:
            if cam_name not in images:
                images[cam_name] = np.zeros((h, w, 3), dtype=np.uint8)

        return images

    def _get_or_create_renderer(self, simu):
        """获取或创建当前线程的渲染器"""
        tid = threading.current_thread().ident
        # 检查是否有缓存的渲染器
        if simu.has_thread_renderer(tid):
            return simu.get_thread_renderer(tid)

        # 使用主渲染器（如果是在主线程创建的）
        if simu._renderer is not None and simu._renderer_thread_id == tid:
            return simu._renderer

        # 创建新的线程渲染器
        try:
            renderer = mujoco.Renderer(simu._model, height=simu._render_height, width=simu._render_width)
            simu.set_thread_renderer(tid, renderer)
            return renderer
        except Exception as e:
            logger.error(f"Failed to create renderer: {e}")
            return None

    def _render_with_retry(self, renderer, simu, cam_id):
        """带重试的渲染：scene 初始化失败时重建 renderer 再试一次

        使用迭代而非递归，避免重建后仍失败导致无限递归栈溢出。
        最多重建 renderer 1 次，总共最多尝试 3 次。
        """
        current_renderer = renderer
        rebuilt = False
        for attempt in range(3):
            try:
                current_renderer.update_scene(simu._data, camera=cam_id)
                return current_renderer.render()
            except Exception as e:
                if not rebuilt and 'mjv_updateScene' in str(e):
                    # 首次 scene 初始化失败，重建 renderer（仅重建一次）
                    try:
                        new_renderer = mujoco.Renderer(
                            simu._model,
                            height=simu._render_height,
                            width=simu._render_width,
                        )
                        # 替换线程缓存的旧 renderer
                        tid = threading.current_thread().ident
                        if simu.has_thread_renderer(tid):
                            simu.set_thread_renderer(tid, new_renderer)
                        elif hasattr(simu, '_renderer') and simu._renderer is current_renderer:
                            simu._renderer = new_renderer
                        current_renderer = new_renderer
                        rebuilt = True
                        continue  # 用新 renderer 重试
                    except Exception as e2:
                        logger.error(f"renderer rebuild failed: {e2}")
                        return None
                else:
                    logger.warning(f"render error (attempt {attempt + 1}): {e}")
                    return None
        return None

    def _get_gripper_raw(self, simu) -> float:
        """获取夹爪状态（在锁内调用），返回 0~1 归一化值

        0 = 张开, 1 = 闭合
        只读取 RIGHT_BOTTOM 的 qpos，映射方式与 SimuInterface.get_gripper_state 一致：
        gripper = 1 - qpos / max_open
        （max_open 从 actuator ctrlrange 动态读取，默认 0.8）
        """
        try:
            if not simu._gripper_indices:
                return 0.0
            idx = simu._gripper_indices[0]
            joint_name = mujoco.mj_id2name(simu._model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            if 'RIGHT' not in joint_name:
                for i in simu._gripper_indices:
                    tmp_name = mujoco.mj_id2name(simu._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if 'RIGHT' in tmp_name:
                        idx = i
                        break
            qpos_idx = simu._model.jnt_qposadr[idx]
            joint_pos = simu._data.qpos[qpos_idx]
            max_open = getattr(simu, '_gripper_max_open', 0.8)
            return float(np.clip(1 - joint_pos / max_open, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"_get_gripper_raw error: {e}")
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
        except Exception as e:
            logger.warning(f"_get_tcp_raw error: {e}")
            return np.zeros(3), np.eye(3)

    def _get_object_pos_raw(self, simu) -> np.ndarray:
        """获取物体位置（在锁内调用）"""
        try:
            body_name = simu._active_object_body_name
            body_id = mujoco.mj_name2id(simu._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                return np.zeros(3)
            return np.copy(simu._data.xpos[body_id])
        except Exception as e:
            logger.warning(f"_get_object_pos_raw error: {e}")
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
        except Exception as e:
            logger.warning(f"_get_cartesian_raw error: {e}")
            return np.zeros(6)
