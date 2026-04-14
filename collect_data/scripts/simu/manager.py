"""
MuJoCo 仿真生命周期管理器

负责：创建/销毁 SimuInterface、启动/停止 Publisher、场景重建。
每个任务彻底 stop_simulation() + start_simulation()（方案B），销毁重建 MuJoCo。

核心需求：
- 关闭/结束任务时 MuJoCo 结束
- 新开任务时 MuJoCo 新开
- 保证资源干净释放，无 OpenGL 泄漏
"""
import gc
import time
import threading
import numpy as np
from typing import Optional, List, Dict, Any

from scripts.core.message_bus import MessageBroker
from scripts.core.topic_defs import ALL_SIMU_TOPICS
from .publisher import SimuPublisher


class SimuManager:
    """MuJoCo 仿真生命周期管理器

    负责:
    - 创建/销毁 SimuInterface
    - 启动/停止 SimuPublisher
    - 场景重建（每个任务彻底销毁重建）
    - 管理仿真相机配置
    """

    def __init__(self, broker: MessageBroker):
        self._broker = broker
        self._simu = None
        self._publisher: Optional[SimuPublisher] = None
        self._lock = threading.Lock()
        self._simu_config: Dict[str, Any] = {}  # 保存仿真配置，用于重建

    @property
    def simu(self):
        """获取当前 SimuInterface（可能为 None）"""
        return self._simu

    @property
    def is_running(self) -> bool:
        return self._simu is not None and self._publisher is not None and self._publisher.is_running

    def start_simulation(self, xml_path: str, camera_names: List[str],
                         use_ik: bool = False, fps: int = 20,
                         show_viewer: bool = False,
                         use_process_renderer: bool = False) -> bool:
        """启动仿真：创建 SimuInterface + Publisher

        Args:
            xml_path: MuJoCo XML 场景路径
            camera_names: 相机名称列表
            use_ik: 是否启用 IK
            fps: 发布频率
            show_viewer: 是否显示被动查看器
            use_process_renderer: 是否使用进程渲染器

        Returns:
            是否成功启动
        """
        # 1. 确保旧的已清理
        self.stop_simulation()

        # 2. 保存配置（用于重建）
        self._simu_config = {
            'xml_path': xml_path,
            'camera_names': list(camera_names),
            'use_ik': use_ik,
            'fps': fps,
            'show_viewer': show_viewer,
            'use_process_renderer': use_process_renderer,
        }

        # 3. 创建新的 SimuInterface
        from .interface import SimuInterface
        self._simu = SimuInterface(xml_path, camera_names=camera_names, use_ik=use_ik)
        if not self._simu.initialize(show_viewer=show_viewer):
            print("[SimuManager] Failed to initialize SimuInterface")
            self._simu = None
            return False

        # 4. 设置相机
        self._simu.set_display_cameras(camera_names)

        # 5. 启动进程渲染器（如果需要）
        if use_process_renderer:
            try:
                self._simu.start_render_process(camera_names)
            except Exception as e:
                print(f"[SimuManager] Failed to start render process: {e}")

        # 6. 启动发布者
        self._publisher = SimuPublisher(self._simu, self._broker, fps=fps)
        self._publisher.start()

        print(f"[SimuManager] Simulation started: xml={xml_path}, cameras={camera_names}, ik={use_ik}")
        return True

    def stop_simulation(self):
        """停止仿真：关闭 Publisher → 释放 SimuInterface → 清除话题缓存"""
        # 1. 停止发布者
        if self._publisher is not None:
            self._publisher.stop()
            self._publisher = None

        # 2. 释放 SimuInterface
        if self._simu is not None:
            try:
                self._simu.close_glfw_viewer()
            except Exception as e:
                print(f"[SimuManager] close_glfw_viewer error: {e}")

            try:
                self._simu.stop_process_renderer()
            except Exception as e:
                print(f"[SimuManager] stop_process_renderer error: {e}")

            try:
                self._simu.disconnect()
            except Exception as e:
                print(f"[SimuManager] disconnect error: {e}")

            self._simu = None

        # 3. 强制 GC 回收 OpenGL/MuJoCo 资源
        gc.collect()
        time.sleep(0.2)
        gc.collect()  # 二次 GC 确保循环引用也被回收

        # 4. 清除仿真相关话题的缓存（避免 GUI 显示过期数据）
        #    使用 clear_latest 而非 publish(None)，避免触发订阅者回调中 float(None) 等错误
        for topic_name in ALL_SIMU_TOPICS:
            topic = self._broker.get_topic(topic_name)
            if topic is not None:
                topic.clear_latest()

        print("[SimuManager] Simulation stopped and resources cleaned")

    def restart_simulation(self, xml_path: str = None, camera_names: List[str] = None,
                           use_ik: bool = None, fps: int = None,
                           show_viewer: bool = None,
                           use_process_renderer: bool = None) -> bool:
        """重启仿真（任务切换时调用）

        使用保存的配置作为默认值，可覆盖任何参数。
        """
        cfg = dict(self._simu_config)  # 复制上次的配置
        if xml_path is not None:
            cfg['xml_path'] = xml_path
        if camera_names is not None:
            cfg['camera_names'] = list(camera_names)
        if use_ik is not None:
            cfg['use_ik'] = use_ik
        if fps is not None:
            cfg['fps'] = fps
        if show_viewer is not None:
            cfg['show_viewer'] = show_viewer
        if use_process_renderer is not None:
            cfg['use_process_renderer'] = use_process_renderer

        return self.start_simulation(**cfg)

    def start_task_simulation(self, task_config: dict) -> bool:
        """为特定任务启动仿真

        在任务开始时调用，使用任务配置重建场景。

        Args:
            task_config: 包含以下键的字典:
                - base_scene_xml: 基础场景 XML 路径
                - object_model_xml: 物体模型 XML 路径
                - object_body_name: 物体 body 名称
                - plate_model_xml: 放置目标模型 XML 路径（可选）
                - plate_body_name: 放置目标 body 名称（可选）
                - camera_names: 相机名称列表
                - use_ik: 是否启用 IK
                - object_position: 物体初始位置
                - initial_joints_deg: 初始关节角度（度）

        Returns:
            是否成功
        """
        # 停止旧的仿真
        self.stop_simulation()

        base_xml = task_config.get('base_scene_xml', '')
        camera_names = task_config.get('camera_names', ['agentview'])
        use_ik = task_config.get('use_ik', True)
        fps = task_config.get('fps', 20)
        show_viewer = task_config.get('show_viewer', False)
        use_process_renderer = task_config.get('use_process_renderer', False)

        if not base_xml:
            print("[SimuManager] No base scene XML provided")
            return False

        # 如果有物体模型，先构建合并的 XML（在 SimuInterface 创建之前）
        # 这样可以避免后续 reload_scene_with_object 时 Publisher 线程活跃导致 GLFW 窗口泄漏
        object_model_xml = task_config.get('object_model_xml', '')
        object_body_name = task_config.get('object_body_name', 'cube')
        plate_model_xml = task_config.get('plate_model_xml', '')
        plate_body_name = task_config.get('plate_body_name', '')

        final_xml = base_xml  # 默认使用基础 XML
        if object_model_xml:
            # 先创建临时 SimuInterface 仅用于构建合并 XML（不加载模型/不创建 renderer）
            from .interface import SimuInterface
            tmp_simu_for_build = SimuInterface.__new__(SimuInterface)
            tmp_simu_for_build._generated_scene_xml_path = None
            tmp_simu_for_build._active_object_body_name = object_body_name
            try:
                if plate_model_xml:
                    generated = tmp_simu_for_build._build_scene_with_objects(
                        base_xml, object_model_xml, object_body_name,
                        plate_model_xml, plate_body_name
                    )
                else:
                    generated = tmp_simu_for_build._build_scene_with_object(
                        base_xml, object_model_xml, object_body_name
                    )
                if generated:
                    final_xml = generated
                    print(f"[SimuManager] Using generated scene XML: {generated}")
            except Exception as e:
                print(f"[SimuManager] Failed to build scene XML, using base XML: {e}")

        # 启动新仿真（使用合并后的 XML，只需一次 initialize）
        if not self.start_simulation(final_xml, camera_names, use_ik, fps, show_viewer, use_process_renderer):
            return False

        # 设置物体属性和初始状态
        if object_model_xml and self._simu is not None:
            try:
                # 设置物体位置
                object_position = task_config.get('object_position')
                if object_position is not None:
                    self._simu.set_object_position(object_body_name, np.array(object_position), reset_z=True)

                # 设置初始关节
                initial_joints = task_config.get('initial_joints_deg')
                if initial_joints is not None:
                    self._simu.set_joint_target(np.array(initial_joints[:6], dtype=float))
                    self._simu.step(300)

                # 设置放置目标位置
                plate_position = task_config.get('plate_position')
                if plate_position is not None and plate_body_name:
                    self._simu.set_object_position(plate_body_name, np.array(plate_position), reset_z=True)

                # 预热渲染
                for _ in range(3):
                    self._simu.get_camera_images(camera_names)
                    time.sleep(0.03)

            except Exception as e:
                print(f"[SimuManager] Failed to setup task scene: {e}")
                import traceback
                traceback.print_exc()
                return False

        print(f"[SimuManager] Task simulation started successfully")
        return True

    def apply_initial_joints(self, joints_deg: Optional[np.ndarray] = None):
        """应用初始关节角度"""
        if self._simu is None:
            return
        if joints_deg is not None:
            self._simu.set_joint_target(np.asarray(joints_deg[:6], dtype=float))
            self._simu.step(300)
