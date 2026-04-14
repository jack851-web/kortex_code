"""
仿真渲染进程 - 在独立进程中运行 MuJoCo 渲染
通过 multiprocessing.Queue 与主进程通信
"""
import multiprocessing as mp
import numpy as np
import mujoco
import time
from typing import Dict, List, Optional


def render_worker(
    xml_path: str,
    camera_names: List[str],
    width: int,
    height: int,
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    joint_queue: mp.Queue,
    object_queue: mp.Queue = None,
    gripper_queue: mp.Queue = None,
    object_body_name: str = "cube",
):
    """
    渲染工作进程

    Args:
        xml_path: MuJoCo XML 文件路径
        camera_names: 需要渲染的相机名称列表
        width: 图像宽度
        height: 图像高度
        command_queue: 接收命令的队列 (start, stop, render)
        result_queue: 发送渲染结果的队列
        joint_queue: 接收关节位置的队列
        object_queue: 接收物块位置的队列
        gripper_queue: 接收夹爪位置的队列
        object_body_name: 物体 body 名称
    """
    # 在子进程中初始化 MuJoCo
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    
    # 为每个相机创建独立的 renderer
    renderers = {}
    for cam_name in camera_names:
        renderers[cam_name] = mujoco.Renderer(model, height=height, width=width)
    
    running = True
    current_joints = None
    current_object_pos = None
    current_gripper = None

    # 获取夹爪关节索引
    right_bottom_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "RIGHT_BOTTOM")
    left_bottom_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "LEFT_BOTTOM")
    right_tip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "RIGHT_TIP")
    left_tip_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "LEFT_TIP")

    def _drain_latest(queue, latest_value):
        if queue is None:
            return latest_value
        try:
            while not queue.empty():
                latest_value = queue.get_nowait()
        except Exception:
            pass
        return latest_value

    def _apply_latest_state():
        nonlocal current_joints, current_object_pos, current_gripper

        current_joints = _drain_latest(joint_queue, current_joints)
        current_object_pos = _drain_latest(object_queue, current_object_pos)
        current_gripper = _drain_latest(gripper_queue, current_gripper)

        if current_joints is not None:
            joints_rad = np.deg2rad(current_joints)
            for i in range(min(len(joints_rad), model.nq)):
                data.qpos[i] = joints_rad[i]

        if current_gripper is not None:
            right_bottom_pos = 0.8 * (1 - current_gripper)
            left_bottom_pos = -0.8 * (1 - current_gripper)

            if right_bottom_idx >= 0:
                qpos_idx = model.jnt_qposadr[right_bottom_idx]
                data.qpos[qpos_idx] = right_bottom_pos
            if left_bottom_idx >= 0:
                qpos_idx = model.jnt_qposadr[left_bottom_idx]
                data.qpos[qpos_idx] = left_bottom_pos
            if right_tip_idx >= 0:
                qpos_idx = model.jnt_qposadr[right_tip_idx]
                data.qpos[qpos_idx] = 0.0
            if left_tip_idx >= 0:
                qpos_idx = model.jnt_qposadr[left_tip_idx]
                data.qpos[qpos_idx] = 0.0

        if current_object_pos is not None:
            try:
                body_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
                if body_idx >= 0:
                    joint_adr = model.body_jntadr[body_idx]
                    if joint_adr >= 0:
                        qpos_adr = model.jnt_qposadr[joint_adr]
                        data.qpos[qpos_adr:qpos_adr+3] = current_object_pos
            except Exception:
                pass

        mujoco.mj_forward(model, data)

    while running:
        _apply_latest_state()

        try:
            cmd = command_queue.get(timeout=0.001)
            if cmd == "stop":
                running = False
            elif cmd == "render":
                # 在真正渲染前再次吸收一遍最新状态，避免队列中的新场景/新物体滞后一帧
                _apply_latest_state()

                images = {}
                for cam_name in camera_names:
                    cam_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    if cam_idx >= 0:
                        renderers[cam_name].update_scene(data, camera=cam_idx)
                    else:
                        renderers[cam_name].update_scene(data)

                    image = renderers[cam_name].render()
                    images[cam_name] = image.copy()

                result_queue.put(images)
        except Exception:
            pass

        time.sleep(0.001)
    
    # 清理
    for renderer in renderers.values():
        del renderer


class SimuRenderProcess:
    """
    仿真渲染进程管理器
    在独立进程中运行 MuJoCo 渲染，避免缓冲区共享问题
    """
    
    def __init__(self, xml_path: str, camera_names: List[str], width: int = 640, height: int = 480, object_body_name: str = "cube"):
        self._xml_path = xml_path
        self._camera_names = camera_names
        self._width = width
        self._height = height
        self._object_body_name = object_body_name or "cube"
        
        self._command_queue: Optional[mp.Queue] = None
        self._result_queue: Optional[mp.Queue] = None
        self._joint_queue: Optional[mp.Queue] = None
        self._object_queue: Optional[mp.Queue] = None
        self._gripper_queue: Optional[mp.Queue] = None
        self._process: Optional[mp.Process] = None
        self._running = False

    def start(self):
        """启动渲染进程"""
        if self._running:
            return

        # 创建队列
        self._command_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._joint_queue = mp.Queue()
        self._object_queue = mp.Queue()
        self._gripper_queue = mp.Queue()

        # 启动进程
        self._process = mp.Process(
            target=render_worker,
            args=(
                self._xml_path,
                self._camera_names,
                self._width,
                self._height,
                self._command_queue,
                self._result_queue,
                self._joint_queue,
                self._object_queue,
                self._gripper_queue,
                self._object_body_name,
            ),
        )
        self._process.start()
        self._running = True
        print(f"[SimuRenderProcess] Started with cameras: {self._camera_names}")
    
    def stop(self):
        """停止渲染进程"""
        if not self._running:
            return
        
        try:
            self._command_queue.put("stop")
        except:
            pass
        
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
        
        try:
            self._command_queue.close()
            self._result_queue.close()
            self._joint_queue.close()
            self._object_queue.close()
            self._gripper_queue.close()
        except:
            pass
        
        self._command_queue = None
        self._result_queue = None
        self._joint_queue = None
        self._object_queue = None
        self._gripper_queue = None
        
        self._running = False
        print("[SimuRenderProcess] Stopped")
    
    def update_joints(self, joint_positions: np.ndarray):
        """更新关节位置"""
        if self._running:
            # 清空旧数据
            try:
                while not self._joint_queue.empty():
                    self._joint_queue.get_nowait()
            except:
                pass
            self._joint_queue.put(joint_positions.copy())
    
    def update_object_position(self, position: np.ndarray):
        """更新物块位置"""
        if self._running and self._object_queue is not None:
            # 清空旧数据
            try:
                while not self._object_queue.empty():
                    self._object_queue.get_nowait()
            except:
                pass
            self._object_queue.put(position.copy())

    def update_gripper(self, gripper_position: float):
        """更新夹爪位置"""
        if self._running and self._gripper_queue is not None:
            # 清空旧数据
            try:
                while not self._gripper_queue.empty():
                    self._gripper_queue.get_nowait()
            except:
                pass
            self._gripper_queue.put(gripper_position)
    
    def render(self) -> Optional[Dict[str, np.ndarray]]:
        """
        请求渲染一帧
        
        Returns:
            字典 {camera_name: image_array} 或 None
        """
        if not self._running:
            return None
        
        # 清空旧结果
        try:
            while not self._result_queue.empty():
                self._result_queue.get_nowait()
        except:
            pass
        
        # 发送渲染命令
        self._command_queue.put("render")
        
        # 等待结果
        try:
            return self._result_queue.get(timeout=1.0)
        except:
            return None
    
    def get_images(self) -> Dict[str, np.ndarray]:
        """获取图像 - 兼容接口"""
        result = self.render()
        if result is None:
            return {name: np.zeros((self._height, self._width, 3), dtype=np.uint8) 
                    for name in self._camera_names}
        return result
    
    def is_running(self) -> bool:
        return self._running and self._process.is_alive()
