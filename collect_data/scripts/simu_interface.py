"""
仿真机器人接口
支持两种模式:
1. 实机同步模式: 接收真实机器人关节数据并同步到仿真
2. 纯仿真模式: 使用 IK 根据笛卡尔坐标控制仿真机械臂
"""
import numpy as np
import mujoco
import mujoco.viewer
import glfw
import time
import threading
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any, Tuple, Callable

from pathlib import Path
import sys

# Qt 延迟导入（仅在需要时使用）
try:
    from PyQt5.QtWidgets import QApplication
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# 添加 IK 模块路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'kortex_simu'))
try:
    from ik import MuJoCoIK, SimulationController
    IK_AVAILABLE = True
except ImportError as e:
    print(f"[SimuInterface] IK module not available: {e}")
    IK_AVAILABLE = False


class SimuInterface:
    """MuJoCo仿真接口
    
    支持两种模式:
    - 实机同步模式: 通过 update_from_real() 接收真实机器人数据
    - 纯仿真模式: 通过 move_to_cartesian() 使用 IK 控制
    """
    
    _glfw_initialized = False
    
    def __init__(self, xml_path: Optional[str] = None, camera_names: Optional[List[str]] = None, 
                 use_ik: bool = False):
        self._model = None
        self._data = None
        self._xml_path = xml_path
        self._viewer = None
        
        # GLFW 查看器 (用于 Mock 模式)
        self._glfw_window = None
        self._glfw_ctx = None
        self._glfw_scn = None
        self._glfw_cam = None
        self._glfw_opt = None
        self._glfw_pert = None
        self._glfw_viewport = None
        self._glfw_viewer_running = False
        self._glfw_timer = None
        
        self._joint_names = ['J0', 'J1', 'J2', 'J3', 'J4', 'J5']
        # 只控制根部夹爪关节，不控制指尖
        self._gripper_joint_names = ['RIGHT_BOTTOM', 'LEFT_BOTTOM']
        self._gripper_tip_names = ['RIGHT_TIP', 'LEFT_TIP']
        self._joint_indices = []
        self._gripper_indices = []
        self._gripper_tip_indices = []
        self._body_names = []
        self._camera_names = camera_names or []
        self._camera_viewer_running = False
        self._camera_viewer_thread = None
        self._renderer = None
        self._render_width = 640
        self._render_height = 480
        # 进程渲染器
        self._render_process = None
        self._use_process_renderer = False
        self._process_render_lock = threading.Lock()
        self._renderer_thread_id = None
        self._renderer_thread_name = None

        # 物块物理控制标志

        self._object_physics_enabled = True  # 让物理引擎控制物块
        # 线程锁 - 保护 MuJoCo 操作（RLock 允许同线程重入，避免 IK 路径死锁）
        self._lock = threading.RLock()

        
        # IK 相关
        self._use_ik = use_ik and IK_AVAILABLE
        self._ik_solver: Optional[MuJoCoIK] = None
        self._sim_controller: Optional[SimulationController] = None
        self._tcp_site_name = "tcp"

        # 动态物体配置
        self._active_object_body_name = "cube"
        self._generated_scene_xml_path: Optional[str] = None



    def set_active_object_body_name(self, object_body_name: str):
        self._active_object_body_name = object_body_name or "cube"

    def get_active_object_body_name(self) -> str:
        return self._active_object_body_name

    def _build_scene_with_object(self, base_scene_xml: str, object_model_xml: str, object_body_name: Optional[str] = None) -> Optional[str]:
        try:
            base_text = Path(base_scene_xml).read_text(encoding="utf-8")
            object_root = ET.parse(object_model_xml).getroot()

            asset_elem = object_root.find("asset")
            worldbody_elem = object_root.find("worldbody")
            if worldbody_elem is None:
                print(f"[SimuInterface] Invalid object xml (missing worldbody): {object_model_xml}")
                return None

            selected_body = None
            if object_body_name:
                selected_body = worldbody_elem.find(f".//body[@name='{object_body_name}']")

            if selected_body is None:
                selected_body = worldbody_elem.find("body")

            if selected_body is None:
                print(f"[SimuInterface] Invalid object xml (no body found): {object_model_xml}")
                return None

            asset_xml = ""
            if asset_elem is not None:
                object_root_dir = Path(object_model_xml).parent
                for child in list(asset_elem):
                    file_attr = child.attrib.get("file")
                    if file_attr:
                        file_path = Path(file_attr)
                        if not file_path.is_absolute():
                            child.set("file", str((object_root_dir / file_path).resolve()).replace("\\", "/"))

                asset_children = [ET.tostring(child, encoding="unicode") for child in list(asset_elem)]
                asset_xml = "\n".join(asset_children)


            body_xml = ET.tostring(selected_body, encoding="unicode")

            if "<!-- DYNAMIC_OBJECT_ASSET -->" not in base_text or "<!-- DYNAMIC_OBJECT_BODY -->" not in base_text:
                print("[SimuInterface] Base scene xml missing DYNAMIC_OBJECT placeholders")
                return None

            composed_xml = base_text.replace("<!-- DYNAMIC_OBJECT_ASSET -->", asset_xml)
            composed_xml = composed_xml.replace("<!-- DYNAMIC_OBJECT_BODY -->", body_xml)

            generated_path = Path(base_scene_xml).with_name("task_pick_place.generated.xml")
            generated_path.write_text(composed_xml, encoding="utf-8")
            self._generated_scene_xml_path = str(generated_path)
            return str(generated_path)

        except Exception as e:
            print(f"[SimuInterface] build scene with object failed: {e}")
            return None

    def reload_scene_with_object(self, base_scene_xml: str, object_model_xml: str, object_body_name: Optional[str] = None, show_viewer: bool = False) -> bool:
        generated_xml = self._build_scene_with_object(base_scene_xml, object_model_xml, object_body_name)
        if not generated_xml:
            return False

        if object_body_name:
            self._active_object_body_name = object_body_name

        glfw_was_running = self._glfw_viewer_running

        result = self.initialize(generated_xml, show_viewer=False)

        if result and glfw_was_running:
            self.start_glfw_viewer()

        return result

    def reload_scene_with_objects(
        self, 
        base_scene_xml: str, 
        object_model_xml: str, 
        object_body_name: Optional[str] = None,
        plate_model_xml: Optional[str] = None,
        plate_body_name: Optional[str] = None,
        show_viewer: bool = False
    ) -> bool:
        """加载场景同时包含抓取物体和放置目标（碟子）"""
        generated_xml = self._build_scene_with_objects(
            base_scene_xml, object_model_xml, object_body_name,
            plate_model_xml, plate_body_name
        )
        if not generated_xml:
            return False

        if object_body_name:
            self._active_object_body_name = object_body_name

        glfw_was_running = self._glfw_viewer_running
        
        result = self.initialize(generated_xml, show_viewer=False)
        
        if result and glfw_was_running:
            self.start_glfw_viewer()
        
        return result

    def _build_scene_with_objects(
        self, 
        base_scene_xml: str, 
        object_model_xml: str, 
        object_body_name: Optional[str] = None,
        plate_model_xml: Optional[str] = None,
        plate_body_name: Optional[str] = None,
    ) -> Optional[str]:
        """构建包含两个物体的场景XML"""
        print(f"[SimuInterface] _build_scene_with_objects called:")
        print(f"  - base_scene_xml: {base_scene_xml}")
        print(f"  - object_model_xml: {object_model_xml}")
        print(f"  - object_body_name: {object_body_name}")
        print(f"  - plate_model_xml: {plate_model_xml}")
        print(f"  - plate_body_name: {plate_body_name}")
        
        try:
            base_text = Path(base_scene_xml).read_text(encoding="utf-8")
            
            asset_xml = ""
            body_xml = ""
            
            if object_model_xml:
                object_root = ET.parse(object_model_xml).getroot()
                asset_elem = object_root.find("asset")
                worldbody_elem = object_root.find("worldbody")
                
                if worldbody_elem is not None:
                    selected_body = None
                    if object_body_name:
                        selected_body = worldbody_elem.find(f".//body[@name='{object_body_name}']")
                    if selected_body is None:
                        selected_body = worldbody_elem.find("body")
                    
                    if selected_body is not None:
                        if asset_elem is not None:
                            object_root_dir = Path(object_model_xml).parent
                            for child in list(asset_elem):
                                file_attr = child.attrib.get("file")
                                if file_attr:
                                    file_path = Path(file_attr)
                                    if not file_path.is_absolute():
                                        child.set("file", str((object_root_dir / file_path).resolve()).replace("\\", "/"))
                            asset_children = [ET.tostring(child, encoding="unicode") for child in list(asset_elem)]
                            asset_xml = "\n".join(asset_children)
                        
                        body_xml = ET.tostring(selected_body, encoding="unicode")
            
            plate_asset_xml = ""
            plate_body_xml = ""
            if plate_model_xml:
                print(f"[SimuInterface] Loading plate model from: {plate_model_xml}")
                if not Path(plate_model_xml).exists():
                    print(f"[SimuInterface] WARNING: Plate model file not found: {plate_model_xml}")
                else:
                    try:
                        plate_root = ET.parse(plate_model_xml).getroot()
                        plate_asset_elem = plate_root.find("asset")
                        plate_worldbody_elem = plate_root.find("worldbody")
                        
                        if plate_worldbody_elem is not None:
                            plate_selected_body = None
                            if plate_body_name:
                                plate_selected_body = plate_worldbody_elem.find(f".//body[@name='{plate_body_name}']")
                                print(f"[SimuInterface] Looking for plate body: '{plate_body_name}', found: {plate_selected_body is not None}")
                            if plate_selected_body is None:
                                plate_selected_body = plate_worldbody_elem.find("body")
                                print(f"[SimuInterface] Using first body as fallback: {plate_selected_body is not None}")
                            
                            if plate_selected_body is not None:
                                if plate_asset_elem is not None:
                                    plate_root_dir = Path(plate_model_xml).parent
                                    for child in list(plate_asset_elem):
                                        file_attr = child.attrib.get("file")
                                        if file_attr:
                                            file_path = Path(file_attr)
                                            if not file_path.is_absolute():
                                                child.set("file", str((plate_root_dir / file_path).resolve()).replace("\\", "/"))
                                    plate_asset_children = [ET.tostring(child, encoding="unicode") for child in list(plate_asset_elem)]
                                    plate_asset_xml = "\n".join(plate_asset_children)
                                    print(f"[SimuInterface] Plate asset loaded: {len(plate_asset_children)} assets")
                                
                                plate_body_xml = ET.tostring(plate_selected_body, encoding="unicode")
                                print(f"[SimuInterface] Plate body loaded successfully")
                            else:
                                print(f"[SimuInterface] WARNING: No plate body found in model")
                        else:
                            print(f"[SimuInterface] WARNING: No worldbody in plate model")
                    except Exception as e:
                        print(f"[SimuInterface] ERROR parsing plate model: {e}")
            
            composed_xml = base_text
            print(f"[SimuInterface] DEBUG: asset_xml length={len(asset_xml)}, plate_asset_xml length={len(plate_asset_xml)}")
            print(f"[SimuInterface] DEBUG: body_xml length={len(body_xml)}, plate_body_xml length={len(plate_body_xml)}")
            if "<!-- DYNAMIC_OBJECT_ASSET -->" in composed_xml:
                composed_xml = composed_xml.replace("<!-- DYNAMIC_OBJECT_ASSET -->", asset_xml)
                print(f"[SimuInterface] Replaced DYNAMIC_OBJECT_ASSET")
            else:
                print(f"[SimuInterface] WARNING: DYNAMIC_OBJECT_ASSET not found in base XML")
            if "<!-- DYNAMIC_OBJECT_BODY -->" in composed_xml:
                composed_xml = composed_xml.replace("<!-- DYNAMIC_OBJECT_BODY -->", body_xml)
                print(f"[SimuInterface] Replaced DYNAMIC_OBJECT_BODY")
            else:
                print(f"[SimuInterface] WARNING: DYNAMIC_OBJECT_BODY not found in base XML")
            if "<!-- PLATE_OBJECT_ASSET -->" in composed_xml:
                composed_xml = composed_xml.replace("<!-- PLATE_OBJECT_ASSET -->", plate_asset_xml)
                print(f"[SimuInterface] Replaced PLATE_OBJECT_ASSET")
            else:
                print(f"[SimuInterface] WARNING: PLATE_OBJECT_ASSET not found in base XML")
            if "<!-- PLATE_OBJECT_BODY -->" in composed_xml:
                composed_xml = composed_xml.replace("<!-- PLATE_OBJECT_BODY -->", plate_body_xml)
                print(f"[SimuInterface] Replaced PLATE_OBJECT_BODY")
            else:
                print(f"[SimuInterface] WARNING: PLATE_OBJECT_BODY not found in base XML")

            generated_path = Path(base_scene_xml).with_name("task_pick_place.generated.xml")
            generated_path.write_text(composed_xml, encoding="utf-8")
            self._generated_scene_xml_path = str(generated_path)
            return str(generated_path)

        except Exception as e:
            print(f"[SimuInterface] build scene with objects failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def initialize(self, xml_path: Optional[str] = None, show_viewer: bool = True) -> bool:
        if xml_path is not None:
            self._xml_path = xml_path

        if self._xml_path is None:
            print("[SimuInterface] Error: No XML path provided")
            return False

        try:
            # 先清理 GLFW 查看器（必须在替换 model/data 之前）
            glfw_was_running = self._glfw_viewer_running
            if glfw_was_running:
                self.close_glfw_viewer()
            # 额外确保清理所有 GLFW 渲染上下文引用
            self._glfw_ctx = None
            self._glfw_scn = None
            self._glfw_cam = None
            self._glfw_opt = None
            self._glfw_pert = None
            self._glfw_viewport = None

            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None
            self._renderer_thread_id = None
            self._renderer_thread_name = None

            if self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception:
                    pass
                self._viewer = None

            print(f"[SimuInterface] Loading model from: {self._xml_path}")


            self._model = mujoco.MjModel.from_xml_path(self._xml_path)
            self._data = mujoco.MjData(self._model)
            
            # 获取关节索引
            self._joint_indices = []
            for name in self._joint_names:
                idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if idx >= 0:
                    self._joint_indices.append(idx)
                    qpos_idx = self._model.jnt_qposadr[idx]
                    # 读取当前值
                    current_val = self._data.qpos[qpos_idx]
                    print(f"[SimuInterface] Joint {name}: id={idx}, qpos_idx={qpos_idx}, current_val={current_val:.4f} ({np.rad2deg(current_val):.2f} deg)")
            print(f"[SimuInterface] Joint indices: {self._joint_indices}")
            
            # 打印所有关节信息
            print(f"[SimuInterface] Total joints: {self._model.njnt}, nq: {self._model.nq}")
            try:
                for i in range(self._model.njnt):
                    name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    qpos_idx = self._model.jnt_qposadr[i]
                    jnt_type = self._model.jnt_type[i]
                    print(f"[SimuInterface] Joint[{i}]: name={name}, type={jnt_type}, qpos_idx={qpos_idx}")
            except Exception as e:
                print(f"[SimuInterface] Error printing joint info: {e}")
            
            # 打印前 20 个 qpos 值
            try:
                print(f"[SimuInterface] First 20 qpos values: {self._data.qpos[:min(20, self._model.nq)]}")
            except Exception as e:
                print(f"[SimuInterface] Error printing qpos: {e}")
            
            # 打印执行器映射
            try:
                self.print_actuator_mapping()
            except Exception as e:
                print(f"[SimuInterface] Error printing actuator mapping: {e}")
            
            # 获取夹爪关节索引
            self._gripper_indices = []
            for name in self._gripper_joint_names:
                idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if idx >= 0:
                    self._gripper_indices.append(idx)
            print(f"[SimuInterface] Gripper indices: {self._gripper_indices}")

            # 获取夹爪指尖关节索引
            self._gripper_tip_indices = []
            for name in self._gripper_tip_names:
                idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if idx >= 0:
                    self._gripper_tip_indices.append(idx)
            print(f"[SimuInterface] Gripper tip indices: {self._gripper_tip_indices}")

            # 获取所有body名称
            self._body_names = []
            for i in range(self._model.nbody):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
                if name:
                    self._body_names.append(name)
            
            # 检测可用相机
            camera_names = []
            for i in range(self._model.ncam):
                name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    camera_names.append(name)
            print(f"[SimuInterface] Found cameras: {camera_names}")
            
            # 初始化仿真
            mujoco.mj_forward(self._model, self._data)
            
            # 初始化 IK 求解器（纯仿真模式）
            if self._use_ik and IK_AVAILABLE:
                try:
                    self._ik_solver = MuJoCoIK(self._model, self._data, self._tcp_site_name)
                    print(f"[SimuInterface] IK solver initialized")
                    print(f"[SimuInterface] IK joints: {self._ik_solver.joint_names}")
                except Exception as e:
                    print(f"[SimuInterface] Failed to initialize IK: {e}")
                    self._use_ik = False
            
            # 启动 MuJoCo 查看器
            if show_viewer:
                self.start_viewer()
            
            print(f"[SimuInterface] Model initialized successfully")
            print(f"[SimuInterface] Bodies: {self._model.nbody}, Joints: {self._model.njnt}")
            print(f"[SimuInterface] IK mode: {self._use_ik}")

            # 若进程渲染器已启用（如动态换物体后），重启以加载新的XML
            if self._use_process_renderer:
                cam_names = list(self._camera_names)
                self.stop_process_renderer()
                self.start_process_renderer(cam_names)
                print("[SimuInterface] Render process restarted for new scene")
            
            return True

            
        except Exception as e:
            print(f"[SimuInterface] Error initializing model: {e}")
            return False
    
    def start_viewer(self):
        """启动 MuJoCo 查看器 (被动模式)"""
        try:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            # 设置相机视角
            self._viewer.cam.azimuth = 45
            self._viewer.cam.elevation = -30
            self._viewer.cam.distance = 2.0
            print("[SimuInterface] MuJoCo viewer started (passive mode)")
        except Exception as e:
            print(f"[SimuInterface] Failed to start viewer: {e}")
            import traceback
            traceback.print_exc()
            self._viewer = None
    
    def start_glfw_viewer(self, width: int = 1200, height: int = 900, 
                          title: str = "MuJoCo Simulation") -> bool:
        """启动基于 GLFW 的独立 MuJoCo 查看器窗口 (用于 Mock 模式)
        
        这个查看器在独立窗口中运行，使用 QTimer 进行渲染
        """
        print(f"[SimuInterface] start_glfw_viewer called: width={width}, height={height}, title={title}")
        print(f"[SimuInterface] _model={self._model is not None}, _data={self._data is not None}, _glfw_viewer_running={self._glfw_viewer_running}")
        
        if self._model is None or self._data is None:
            print("[SimuInterface] Cannot start viewer: model or data not initialized")
            return False
        
        if self._glfw_viewer_running:
            print("[SimuInterface] GLFW viewer already running, skipping")
            return True
        
        try:
            print("[SimuInterface] Starting GLFW viewer initialization...")
            
            if self._glfw_window is not None:
                print("[SimuInterface] Closing existing GLFW window...")
                self.close_glfw_viewer()
            
            if not SimuInterface._glfw_initialized:
                print("[SimuInterface] Initializing GLFW...")
                if not glfw.init():
                    print("[SimuInterface] Failed to initialize GLFW")
                    return False
                SimuInterface._glfw_initialized = True
            
            self._glfw_window = glfw.create_window(width, height, title, None, None)
            
            if not self._glfw_window:
                print("[SimuInterface] Failed to create GLFW window")
                return False
            
            print("[SimuInterface] GLFW window created successfully")
            glfw.make_context_current(self._glfw_window)
            glfw.swap_interval(1)
            
            framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self._glfw_window)
            print(f"[SimuInterface] Framebuffer size: {framebuffer_width}x{framebuffer_height}")
            
            print("[SimuInterface] Creating MuJoCo rendering context...")
            self._glfw_ctx = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
            
            print("[SimuInterface] Creating MuJoCo scene...")
            self._glfw_scn = mujoco.MjvScene(self._model, maxgeom=10000)
            self._glfw_pert = mujoco.MjvPerturb()
            self._glfw_cam = mujoco.MjvCamera()
            self._glfw_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self._glfw_cam.azimuth = 170
            self._glfw_cam.elevation = -20
            self._glfw_cam.distance = 2.5
            self._glfw_cam.lookat = np.array([0.3, 0.0, 0.3])
            
            self._glfw_opt = mujoco.MjvOption()
            self._glfw_opt.geomgroup[0] = 1
            self._glfw_opt.geomgroup[1] = 1
            self._glfw_opt.geomgroup[2] = 1
            
            self._glfw_viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
            
            self._glfw_viewer_running = True
            
            # QTimer 必须在 Qt 主线程创建，否则 timeout 信号不会触发
            from PyQt5.QtCore import QTimer, QThread
            in_main_thread = (not QT_AVAILABLE or 
                              QApplication.instance() is None or 
                              QThread.currentThread() == QApplication.instance().thread())
            if not in_main_thread:
                # 从后台线程调用：通过信号回到主线程创建 QTimer
                QTimer.singleShot(0, self._start_glfw_timer_qt)
            else:
                self._start_glfw_timer_qt()
            
            print(f"[SimuInterface] GLFW viewer started: {width}x{height}")
            return True
            
        except Exception as e:
            print(f"[SimuInterface] Failed to start GLFW viewer: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_glfw_timer_qt(self):
        """在 Qt 主线程中创建并启动 GLFW 渲染定时器"""
        from PyQt5.QtCore import QTimer
        print("[SimuInterface] _start_glfw_timer_qt called")
        
        if hasattr(self, '_glfw_timer') and self._glfw_timer is not None:
            print("[SimuInterface] Stopping existing timer...")
            try:
                self._glfw_timer.stop()
            except Exception:
                pass
            try:
                self._glfw_timer.deleteLater()
            except Exception:
                pass
            self._glfw_timer = None
        
        print("[SimuInterface] Creating new QTimer...")
        self._glfw_timer = QTimer()
        self._glfw_timer.timeout.connect(self._glfw_render_frame)
        self._glfw_timer.start(16)
        print(f"[SimuInterface] QTimer started with interval=16ms, active={self._glfw_timer.isActive()}")
    
    def _glfw_render_frame(self):
        """渲染一帧 (由 QTimer 调用) - 多视口布局"""
        if not self._glfw_viewer_running:
            return
        
        if self._glfw_window is None:
            return
        
        try:
            if glfw.window_should_close(self._glfw_window):
                self.close_glfw_viewer()
                return
            
            glfw.make_context_current(self._glfw_window)
            
            with self._lock:
                mujoco.mj_forward(self._model, self._data)
                
                viewport_width, viewport_height = glfw.get_framebuffer_size(self._glfw_window)
                
                overlay_w = int(viewport_width * 0.22)
                overlay_h = int(viewport_height * 0.22)
                
                main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
                
                mujoco.mjv_updateScene(
                    self._model, self._data, self._glfw_opt, self._glfw_pert, 
                    self._glfw_cam, mujoco.mjtCatBit.mjCAT_ALL, self._glfw_scn
                )
                mujoco.mjr_render(main_viewport, self._glfw_scn, self._glfw_ctx)
                
                cam_names = ['agentview', 'side', 'robot0_eye_in_hand']
                positions = [
                    (10, viewport_height - overlay_h - 10),
                    (viewport_width - overlay_w - 10, viewport_height - overlay_h - 10),
                    (viewport_width - overlay_w - 10, 10)
                ]
                
                for cam_name, (x, y) in zip(cam_names, positions):
                    try:
                        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                        if cam_id >= 0:
                            overlay_viewport = mujoco.MjrRect(x, y, overlay_w, overlay_h)
                            
                            overlay_cam = mujoco.MjvCamera()
                            overlay_cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                            overlay_cam.fixedcamid = cam_id
                            
                            mujoco.mjv_updateScene(
                                self._model, self._data, self._glfw_opt, self._glfw_pert,
                                overlay_cam, mujoco.mjtCatBit.mjCAT_ALL, self._glfw_scn
                            )
                            mujoco.mjr_render(overlay_viewport, self._glfw_scn, self._glfw_ctx)
                            
                            border = mujoco.MjrRect(x, y, overlay_w, 3)
                            mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100,
                                            mujoco.mjtGridPos.mjGRID_TOPLEFT, border,
                                            "", "", self._glfw_ctx)
                    except Exception as e:
                        pass
                
                import time as _time
                current_time = _time.time()
                if not hasattr(self, '_start_time'):
                    self._start_time = current_time
                
                sim_time = self._data.time if self._data else 0
                wall_time = current_time - self._start_time
                tick = getattr(self, '_render_tick', 0)
                self._render_tick = tick + 1
                
                info_lines = [
                    f"tick:        {tick:>6}",
                    f"sim time:    {sim_time:>8.2f}sec",
                    f"wall time:   {wall_time:>8.2f}sec",
                ]
                
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                                 mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                 main_viewport,
                                 "\n".join(info_lines), "", self._glfw_ctx)
                
                glfw.swap_buffers(self._glfw_window)
            
            glfw.poll_events()
            
        except Exception as e:
            print(f"[SimuInterface] GLFW render frame error: {e}")
            import traceback
            traceback.print_exc()
    
    def close_glfw_viewer(self):
        """关闭 GLFW 查看器"""
        print(f"[SimuInterface] close_glfw_viewer called, _glfw_viewer_running={self._glfw_viewer_running}")
        self._glfw_viewer_running = False
        
        if hasattr(self, '_glfw_timer') and self._glfw_timer:
            # QTimer 必须在创建它的线程（Qt主线程）中停止
            try:
                from PyQt5.QtCore import QThread
                if QT_AVAILABLE and QApplication.instance() is not None and \
                   QThread.currentThread() != QApplication.instance().thread():
                    # 从后台线程：通过 deleteLater 让主线程安全清理
                    self._glfw_timer.deleteLater()
                else:
                    self._glfw_timer.stop()
            except Exception:
                try:
                    self._glfw_timer.stop()
                except Exception:
                    pass
            self._glfw_timer = None
        
        # 清理 GLFW 渲染上下文（必须在销毁窗口前）
        self._glfw_ctx = None
        self._glfw_scn = None
        self._glfw_cam = None
        self._glfw_opt = None
        self._glfw_pert = None
        self._glfw_viewport = None
        
        if hasattr(self, '_glfw_window') and self._glfw_window:
            try:
                glfw.destroy_window(self._glfw_window)
            except Exception:
                pass
            self._glfw_window = None
        
        print("[SimuInterface] GLFW viewer closed")
    
    def sync_viewer(self):
        """同步更新查看器 (被动模式)"""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _get_ctrl_idx_for_joint(self, joint_idx: int) -> int:
        joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
        for i in range(self._model.nu):
            actuator_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name and joint_name and joint_name in actuator_name:
                return i
        return -1
    
    def print_actuator_mapping(self):
        """打印执行器映射信息"""
        print(f"[SimuInterface] Total actuators: {self._model.nu}")
        for i in range(self._model.nu):
            actuator_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"[SimuInterface] Actuator[{i}]: {actuator_name}")
        
        print(f"[SimuInterface] Joint to actuator mapping:")
        for i, joint_idx in enumerate(self._joint_indices):
            joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
            ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
            print(f"[SimuInterface] Joint {joint_name} (idx={joint_idx}) -> Actuator idx={ctrl_idx}")

    def get_joint_state(self) -> np.ndarray:
        if self._model is None or self._data is None:
            return np.zeros(6)
        with self._lock:
            try:
                joint_positions = []
                for joint_idx in self._joint_indices:
                    qpos_idx = self._model.jnt_qposadr[joint_idx]
                    joint_positions.append(self._data.qpos[qpos_idx])
                result = np.array(joint_positions)
                # print(f"[SimuInterface] get_joint_state: {np.rad2deg(result)}")
                return result
            except Exception as e:
                print(f"[SimuInterface] get_joint_state error: {e}")
                return np.zeros(6)

    def set_joint_target(self, positions: np.ndarray) -> bool:
        """设置关节目标位置"""
        if self._model is None or self._data is None:
            return False
        with self._lock:
            try:
                # 将角度归一化到 [-180, 180] 范围
                positions_normalized = np.mod(positions + 180, 360) - 180
                positions_rad = np.deg2rad(positions_normalized)  # 转换为弧度
                for i, joint_idx in enumerate(self._joint_indices):
                    if i < len(positions_rad):
                        ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
                        if ctrl_idx >= 0:
                            # 获取执行器的 ctrlrange
                            ctrl_range = self._model.actuator_ctrlrange[ctrl_idx]
                            # 裁剪到范围内
                            clipped_val = np.clip(positions_rad[i], ctrl_range[0], ctrl_range[1])
                            self._data.ctrl[ctrl_idx] = clipped_val
                        else:
                            print(f"[SimuInterface] Warning: No actuator found for joint {i} (idx={joint_idx})")
                return True
            except Exception as e:
                print(f"[SimuInterface] set_joint_target error: {e}")
                return False

    def get_gripper_state(self) -> float:
        if self._model is None or self._data is None:
            return 0.0
        with self._lock:
            try:
                # 读取 RIGHT_BOTTOM 的位置并映射回 0-1
                if len(self._gripper_indices) > 0:
                    qpos_idx = self._model.jnt_qposadr[self._gripper_indices[0]]
                    joint_pos = self._data.qpos[qpos_idx]
                    # RIGHT_BOTTOM: 0.8 (张开) -> 0 (闭合)
                    # 映射回 0-1: position = 1 - joint_pos / 0.8
                    return np.clip(1 - joint_pos / 0.8, 0.0, 1.0)
                return 0.0
            except Exception:
                return 0.0

    def set_gripper(self, position: float) -> bool:
        """设置夹爪位置"""
        if self._model is None or self._data is None:
            return False
        with self._lock:
            try:
                # position: 0 = 张开, 1 = 闭合
                # RIGHT_BOTTOM: ctrlrange [-0.2, 0.8]
                #   张开 (0) -> 0.8, 闭合 (1) -> 0
                # LEFT_BOTTOM: ctrlrange [-0.8, 0.2]
                #   张开 (0) -> -0.8, 闭合 (1) -> 0
                for joint_idx in self._gripper_indices:
                    ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
                    if ctrl_idx >= 0:
                        joint_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_idx)
                        if 'RIGHT' in joint_name:
                            # RIGHT_BOTTOM: 0.8 (张开) -> 0 (闭合)
                            self._data.ctrl[ctrl_idx] = 0.8 * (1 - position)
                        else:
                            # LEFT_BOTTOM: -0.8 (张开) -> 0 (闭合)
                            self._data.ctrl[ctrl_idx] = -0.8 * (1 - position)
                return True
            except Exception:
                return False

    def sync_control_to_current_state(self) -> bool:
        """将当前关节/夹爪位置同步为控制目标（绝对控制保持当前姿态）"""
        if self._model is None or self._data is None:
            return False

        with self._lock:
            try:
                all_joint_indices = list(self._joint_indices) + list(self._gripper_indices)
                for joint_idx in all_joint_indices:
                    ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
                    if ctrl_idx < 0:
                        continue
                    qpos_idx = self._model.jnt_qposadr[joint_idx]
                    joint_pos = float(self._data.qpos[qpos_idx])
                    ctrl_range = self._model.actuator_ctrlrange[ctrl_idx]
                    self._data.ctrl[ctrl_idx] = np.clip(joint_pos, ctrl_range[0], ctrl_range[1])
                return True
            except Exception as e:
                print(f"[SimuInterface] sync_control_to_current_state error: {e}")
                return False


    def get_object_position(self, object_name: str) -> np.ndarray:
        if self._model is None or self._data is None:
            return np.zeros(3)
        with self._lock:
            try:
                body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, object_name)
                if body_idx < 0:
                    return np.zeros(3)
                return self._data.xpos[body_idx].copy()
            except Exception:
                return np.zeros(3)

    def set_object_position(self, object_name: str, position: np.ndarray, reset_z: bool = False) -> bool:
        """设置物块位置。只设置 x 和 y，z 保持当前值（除非 reset_z=True）。
        如果 _object_physics_enabled 为 True，则只重置到初始位置，之后让物理引擎控制。"""
        if self._model is None or self._data is None:
            return False
        with self._lock:
            try:
                body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, object_name)
                if body_idx < 0:
                    return False
                joint_adr = self._model.body_jntadr[body_idx]
                if joint_adr >= 0:
                    qpos_adr = self._model.jnt_qposadr[joint_adr]
                    # 先设置 x/y，并统一姿态，z 在 reset_z 模式下按“底部高度”自动对齐
                    self._data.qpos[qpos_adr] = position[0]  # x
                    self._data.qpos[qpos_adr+1] = position[1]  # y
                    # 重置四元数为单位四元数 (w=1, x=0, y=0, z=0)
                    self._data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

                    # 重置速度
                    qvel_adr = self._model.jnt_dofadr[joint_adr]
                    # freejoint: 3线速度 + 3角速度 = 6个qvel
                    self._data.qvel[qvel_adr:qvel_adr+6] = 0

                    if reset_z:
                        # 将传入 z 解释为“物体底部目标高度”
                        # 优先使用对象里的 bottom site（若存在），其次再退化到 geom 包围球估计。
                        mujoco.mj_forward(self._model, self._data)

                        def _is_descendant_body(child_body_id: int, root_body_id: int) -> bool:
                            b = int(child_body_id)
                            while b >= 0:
                                if b == root_body_id:
                                    return True
                                parent_b = int(self._model.body_parentid[b])
                                if parent_b == b:
                                    break
                                b = parent_b
                            return False

                        bottom_site_z_list = []
                        for s in range(self._model.nsite):
                            site_body_id = int(self._model.site_bodyid[s])
                            if not _is_descendant_body(site_body_id, body_idx):
                                continue
                            site_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, s) or ""
                            if "bottom" in site_name.lower():
                                bottom_site_z_list.append(float(self._data.site_xpos[s][2]))

                        if len(bottom_site_z_list) > 0:
                            current_bottom_z = min(bottom_site_z_list)
                            dz = float(position[2]) - current_bottom_z
                            self._data.qpos[qpos_adr+2] += dz
                            print(f"[SimuInterface] set_object_position bottom-align(site): target_bottom_z={position[2]:.4f}, current_bottom_z={current_bottom_z:.4f}, dz={dz:.4f}")
                        else:
                            subtree_geom_ids = []
                            for g in range(self._model.ngeom):
                                geom_body_id = int(self._model.geom_bodyid[g])
                                if _is_descendant_body(geom_body_id, body_idx):
                                    subtree_geom_ids.append(g)

                            if len(subtree_geom_ids) > 0:
                                current_bottom_z = min(
                                    float(self._data.geom_xpos[g][2] - self._model.geom_rbound[g])
                                    for g in subtree_geom_ids
                                )
                                dz = float(position[2]) - current_bottom_z
                                self._data.qpos[qpos_adr+2] += dz
                                print(f"[SimuInterface] set_object_position bottom-align(geom): target_bottom_z={position[2]:.4f}, current_bottom_z={current_bottom_z:.4f}, dz={dz:.4f}")
                            else:
                                # 兜底：没有site/geom时直接设置 z
                                self._data.qpos[qpos_adr+2] = position[2]


                    
                    mujoco.mj_forward(self._model, self._data)


                    # 获取实际设置的位置（用于渲染同步）
                    actual_pos = self._data.qpos[qpos_adr:qpos_adr+3].copy()
                    # 同步更新渲染进程中的物块位置（只在初始化时）
                    if self._use_process_renderer and self._render_process:
                        self._render_process.update_object_position(actual_pos)

                return True
            except Exception as e:
                print(f"[SimuInterface] set_object_position error: {e}")
                return False

    def enable_object_physics(self, enabled: bool = True):
        """启用或禁用物块物理控制。启用后，物块由物理引擎控制，不再强制设置位置。"""
        self._object_physics_enabled = enabled
        print(f"[SimuInterface] Object physics enabled: {enabled}")

    def step(self, n_steps: int = 1000):
        if self._model is None or self._data is None:
            return

        with self._lock:
            # 在 step 之前同步 TIP 关节位置
            self._sync_gripper_tips()

            for _ in range(n_steps):
                mujoco.mj_step(self._model, self._data)
                # 同步更新查看器
                self.sync_viewer()

    def _sync_gripper_tips(self):
        """同步夹爪指尖关节位置（TIP 始终保持为 0）"""
        if len(self._gripper_tip_indices) >= 2:
            try:
                # RIGHT_TIP 始终为 0
                tip_qpos_idx = self._model.jnt_qposadr[self._gripper_tip_indices[0]]
                self._data.qpos[tip_qpos_idx] = 0.0

                # LEFT_TIP 始终为 0
                tip_qpos_idx = self._model.jnt_qposadr[self._gripper_tip_indices[1]]
                self._data.qpos[tip_qpos_idx] = 0.0
            except:
                pass

    def _ensure_renderer_for_current_thread(self):
        current_thread_id = threading.get_ident()
        current_thread_name = threading.current_thread().name

        if self._renderer is not None and self._renderer_thread_id != current_thread_id:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
            print(
                f"[SimuInterface] Renderer recreated for thread switch: "
                f"{self._renderer_thread_name} -> {current_thread_name}"
            )

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, height=self._render_height, width=self._render_width)
            self._renderer_thread_id = current_thread_id
            self._renderer_thread_name = current_thread_name

    def render(self, camera_name: Optional[str] = None) -> np.ndarray:

        if self._model is None or self._data is None:
            return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        
        with self._lock:
            try:
                mujoco.mj_forward(self._model, self._data)
                self._ensure_renderer_for_current_thread()
                
                if camera_name:

                    cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                    if cam_id >= 0:
                        self._renderer.update_scene(self._data, camera=cam_id)
                    else:
                        self._renderer.update_scene(self._data)
                else:
                    self._renderer.update_scene(self._data)
                
                # 关键：返回独立拷贝，避免不同相机/线程读到同一底层缓冲区导致串帧
                return np.ascontiguousarray(self._renderer.render()).copy()

                
            except Exception as e:
                print(f"[SimuInterface] Render error: {e}")
                return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)

    def get_camera_images(self, camera_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        cam_names = camera_names or self._camera_names

        if self._use_process_renderer and self._render_process:
            # 串行化渲染进程访问，避免 GUI 刷新与数据采集并发导致串帧/错帧
            with self._process_render_lock:
                # 每次渲染前同步关节/夹爪/物体状态，避免首任务未加载和帧错位
                with self._lock:
                    try:
                        joints_rad = self.get_joint_state()
                        self._render_process.update_joints(np.rad2deg(joints_rad))
                        self._render_process.update_gripper(self.get_gripper_state())
                        obj_pos = self.get_object_position(self._active_object_body_name)
                        self._render_process.update_object_position(obj_pos)
                    except Exception as e:
                        print(f"[SimuInterface] process-render sync error: {e}")

                proc_images = self._render_process.get_images()
                # 仅返回请求相机，防止字典残留/错配
                return {
                    name: np.ascontiguousarray(proc_images.get(name, np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8))).copy()
                    for name in cam_names
                }


        


        images = {}


        # 一次加锁完成多相机渲染，避免 GUI 线程与采集线程交错导致相机串帧
        with self._lock:
            try:
                if self._model is None or self._data is None:
                    return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

                mujoco.mj_forward(self._model, self._data)
                self._ensure_renderer_for_current_thread()

                for name in cam_names:

                    cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                    if cam_id >= 0:
                        self._renderer.update_scene(self._data, camera=cam_id)
                    else:
                        self._renderer.update_scene(self._data)
                    images[name] = np.ascontiguousarray(self._renderer.render()).copy()

            except Exception as e:
                print(f"[SimuInterface] get_camera_images error: {e}")
                return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

        return images


    def set_display_cameras(self, camera_names: List[str]):
        self._camera_names = camera_names

    def start_process_renderer(self, camera_names: Optional[List[str]] = None):
        from scripts.simu_render_process import SimuRenderProcess
        cam_names = camera_names or self._camera_names
        self._render_process = SimuRenderProcess(
            self._xml_path,
            cam_names,
            object_body_name=self._active_object_body_name,
        )
        self._render_process.start()
        self._use_process_renderer = True
        print(f"[SimuInterface] Render process started (object_body={self._active_object_body_name})")


    def start_render_process(self, camera_names: Optional[List[str]] = None):
        self.start_process_renderer(camera_names)

    def stop_process_renderer(self):
        if self._render_process:
            self._render_process.stop()
            self._render_process = None
        self._use_process_renderer = False

    def update_from_real(self, joint_positions: np.ndarray, gripper_position: float):
        """实机同步模式: 接收真实机器人数据并更新仿真"""
        self.set_joint_target(joint_positions)
        self.set_gripper(gripper_position)
        self.step(1000)
    
    # ========== 纯仿真模式: IK 控制方法 ==========
    
    def move_to_cartesian(self, position: np.ndarray, 
                          orientation: Optional[np.ndarray] = None,
                          duration: float = 2.0,
                          steps: int = 100,
                          step_callback: Optional[Callable[[], None]] = None) -> bool:

        """
        纯仿真模式: 移动末端执行器到目标笛卡尔坐标
        使用渐进式 IK 解算，每步执行一次 IK 迭代然后 step 更新
        
        Args:
            position: 目标位置 [x, y, z] (米)
            orientation: 目标姿态 (3x3旋转矩阵), 可选
            duration: 运动持续时间 (秒)
            steps: 插值步数
            
        Returns:
            是否成功
        """
        if not self._use_ik or self._ik_solver is None:
            print("[SimuInterface] IK not available, cannot move to cartesian position")
            return False
        
        print(f"[SimuInterface] Starting move_to_cartesian to {position}")
        
        try:
            # 获取当前关节角度和位置
            with self._lock:
                current_q = self.get_joint_state()
                current_pos = self._ik_solver.forward_kinematics(current_q)
            print(f"[SimuInterface] Current joints: {np.rad2deg(current_q)}")
            print(f"[SimuInterface] Current position: {current_pos}")
            
            target_pos = np.array(position)
            target_ori = np.array(orientation, dtype=float) if orientation is not None else None
            print(f"[SimuInterface] Target position: {target_pos}")
            if target_ori is not None:
                print(f"[SimuInterface] Target orientation matrix enabled")
            
            # 生成轨迹中间点 (笛卡尔空间插值)

            trajectory_points = []
            for i in range(steps + 1):
                alpha = i / steps
                interp_pos = current_pos + alpha * (target_pos - current_pos)
                trajectory_points.append(interp_pos)
            
            print(f"[SimuInterface] Generated {len(trajectory_points)} trajectory points")
            
            # 逐点执行：对每个中间点求解 IK 并执行一步仿真
            q = current_q.copy()
            
            for step_idx, traj_pos in enumerate(trajectory_points):
                if target_ori is not None:
                    try:
                        q_ik = self._ik_solver.inverse_kinematics(
                            traj_pos,
                            target_ori,
                            initial_guess=q,
                        )
                        solve_info = self._ik_solver.get_last_solve_info() if hasattr(self._ik_solver, 'get_last_solve_info') else {}
                        ik_success = bool(solve_info.get('success', False))
                        ik_pos_err = float(solve_info.get('position_error', 1e9))

                        # 注意：inverse_kinematics 总会返回 q，不会返回 None
                        # 若姿态约束不可达，退化为位置IK，避免路径长时间偏离目标
                        if ik_success or ik_pos_err < 0.02:
                            q = np.array(q_ik, dtype=float)
                            for i in range(len(q)):
                                q[i] = np.clip(q[i], self._ik_solver.joint_ranges[i][0], self._ik_solver.joint_ranges[i][1])
                        else:
                            current_pos = self._ik_solver.forward_kinematics(q)
                            pos_error = traj_pos - current_pos
                            J = self._ik_solver.jacobian(q)
                            J_pos = J[:3, :]
                            damping = 0.1
                            JtJ = J_pos.T @ J_pos
                            JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                            delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)
                            q = q + delta_q * 0.5
                    except Exception:
                        current_pos = self._ik_solver.forward_kinematics(q)
                        pos_error = traj_pos - current_pos
                        J = self._ik_solver.jacobian(q)
                        J_pos = J[:3, :]
                        damping = 0.1
                        JtJ = J_pos.T @ J_pos
                        JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                        delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)
                        q = q + delta_q * 0.5

                else:
                    # IK求解是纯计算，不访问MuJoCo资源，不需要锁
                    for ik_iter in range(10):
                        current_pos = self._ik_solver.forward_kinematics(q)
                        pos_error = traj_pos - current_pos
                        error_norm = np.linalg.norm(pos_error)
                        
                        if error_norm < 1e-3:
                            break
                        
                        J = self._ik_solver.jacobian(q)
                        J_pos = J[:3, :]
                        
                        damping = 0.1
                        JtJ = J_pos.T @ J_pos
                        JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                        delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)
                        
                        max_delta = 0.2
                        if np.linalg.norm(delta_q) > max_delta:
                            delta_q = delta_q / np.linalg.norm(delta_q) * max_delta
                        
                        q = q + delta_q * 0.5
                        
                        for i in range(len(q)):
                            q[i] = np.clip(q[i], self._ik_solver.joint_ranges[i][0], 
                                          self._ik_solver.joint_ranges[i][1])

                
                # 只在访问MuJoCo资源时加锁
                with self._lock:
                    for j, joint_idx in enumerate(self._joint_indices):
                        if j < len(q):
                            ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
                            if ctrl_idx >= 0:
                                ctrl_range = self._model.actuator_ctrlrange[ctrl_idx]
                                clipped_val = np.clip(q[j], ctrl_range[0], ctrl_range[1])
                                self._data.ctrl[ctrl_idx] = clipped_val
                    
                    mujoco.mj_step(self._model, self._data)
                    
                    if step_idx % 10 == 0:
                        self.sync_viewer()
                
                if step_callback is not None:
                    try:
                        step_callback()
                    except Exception as cb_e:
                        print(f"[SimuInterface] step_callback error: {cb_e}")

                if step_idx % 20 == 0:
                    final_pos = self._ik_solver.forward_kinematics(q)
                    final_error = np.linalg.norm(target_pos - final_pos)
                    print(f"[SimuInterface] Step {step_idx}/{steps}, target: {traj_pos}, current: {final_pos}, error to final: {final_error:.4f}m")

            
            print(f"[SimuInterface] Moved to position: {position}")
            return True
            
        except Exception as e:
            print(f"[SimuInterface] move_to_cartesian error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def move_to_cartesian_no_ik(self, position: np.ndarray) -> bool:
        """
        临时测试版本：不使用 IK，直接设置关节角度
        用于调试卡死问题
        """
        print(f"[SimuInterface] move_to_cartesian_no_ik to {position}")
        
        try:
            with self._lock:
                current_q = self.get_joint_state()
                print(f"[SimuInterface] Current joints: {np.rad2deg(current_q)}")
                
                current_pos = self._ik_solver.forward_kinematics(current_q)
                print(f"[SimuInterface] Current position: {current_pos}")
                
                target_pos = np.array(position)
                print(f"[SimuInterface] Target position: {target_pos}")
                
                diff = target_pos - current_pos
                print(f"[SimuInterface] Position diff: {diff}")
                
                print("[SimuInterface] IK disabled for testing, skipping IK solving")
                print("[SimuInterface] Test mode: returning True immediately without moving")
                
                return True
                
        except Exception as e:
            print(f"[SimuInterface] move_to_cartesian_no_ik error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_tcp_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """获取末端执行器 (TCP) 的位置和旋转矩阵"""
        if self._model is None or self._data is None:
            return None, None

        with self._lock:
            try:
                site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, self._tcp_site_name)
                if site_id < 0:
                    return None, None
                mujoco.mj_forward(self._model, self._data)
                pos = self._data.site_xpos[site_id].copy()
                rot = self._data.site_xmat[site_id].reshape(3, 3).copy()
                return pos, rot
            except Exception as e:
                print(f"[SimuInterface] get_tcp_pose error: {e}")
                return None, None

    def get_tcp_position(self) -> Optional[np.ndarray]:
        """获取末端执行器 (TCP) 的当前位置"""
        pos, _ = self.get_tcp_pose()
        return pos

    
    def joint_to_cartesian(self, joint_angles: np.ndarray) -> Optional[np.ndarray]:
        """
        将关节角度转换为笛卡尔坐标 (正运动学)
        
        Args:
            joint_angles: 关节角度 (弧度)
            
        Returns:
            末端执行器位置 [x, y, z]
        """
        if self._ik_solver is None:
            print("[SimuInterface] IK solver not available")
            return None
        
        with self._lock:
            try:
                return self._ik_solver.forward_kinematics(joint_angles)
            except Exception as e:
                print(f"[SimuInterface] joint_to_cartesian error: {e}")
                return None
    
    def cartesian_to_joint(self, position: np.ndarray, 
                           orientation: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        将笛卡尔坐标转换为关节角度 (逆运动学)
        
        Args:
            position: 目标位置 [x, y, z]
            orientation: 目标姿态 (可选)
            
        Returns:
            (joint_angles, success) - 关节角度和是否成功
        """
        if self._ik_solver is None:
            print("[SimuInterface] IK solver not available")
            return None, False
        
        with self._lock:
            try:
                current_q = self.get_joint_state()
                joint_angles = self._ik_solver.inverse_kinematics(
                    position, orientation, initial_guess=current_q
                )
                is_valid, _ = self._ik_solver.check_joint_limits(joint_angles)
                solve_info = self._ik_solver.get_last_solve_info() if hasattr(self._ik_solver, 'get_last_solve_info') else {}
                return joint_angles, (is_valid and solve_info.get('success', True))

            except Exception as e:
                print(f"[SimuInterface] cartesian_to_joint error: {e}")
                return None, False
    
    def is_ik_available(self) -> bool:
        """检查 IK 是否可用"""
        return self._use_ik and self._ik_solver is not None

    def disconnect(self):
        self.stop_process_renderer()
        if self._renderer:
            self._renderer.close()
            self._renderer = None
        self._renderer_thread_id = None
        self._renderer_thread_name = None
        self._model = None
        self._data = None



class MockSimuInterface:
    """模拟仿真接口 - 用于测试"""
    
    def __init__(self, xml_path: Optional[str] = None, camera_names: Optional[List[str]] = None):
        self._xml_path = xml_path
        self._joint_state = np.zeros(6)
        self._gripper_state = 0.0
        self._camera_names = camera_names or ['agentview', 'top']
        self._connected = False

    def initialize(self, xml_path: Optional[str] = None) -> bool:
        self._connected = True
        print("[MockSimuInterface] Initialized")
        return True

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def set_joint_target(self, positions: np.ndarray) -> bool:
        self._joint_state = np.array(positions[:6])
        return True

    def get_gripper_state(self) -> float:
        return self._gripper_state

    def set_gripper(self, position: float) -> bool:
        self._gripper_state = np.clip(position, 0.0, 1.0)
        return True

    def get_object_position(self, object_name: str) -> np.ndarray:
        return np.array([0.215, -0.614, 0.17])

    def set_object_position(self, object_name: str, position: np.ndarray) -> bool:
        return True

    def step(self, n_steps: int = 1000):
        pass

    def render(self, camera_name: Optional[str] = None) -> np.ndarray:
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def get_camera_images(self, camera_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        return {name: np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) 
                for name in (camera_names or self._camera_names)}

    def set_display_cameras(self, camera_names: List[str]):
        self._camera_names = camera_names

    def start_process_renderer(self, camera_names: Optional[List[str]] = None):
        pass

    def start_render_process(self, camera_names: Optional[List[str]] = None):
        self.start_process_renderer(camera_names)

    def stop_process_renderer(self):
        pass

    def update_from_real(self, joint_positions: np.ndarray, gripper_position: float):
        self._joint_state = np.array(joint_positions[:6])
        self._gripper_state = gripper_position

    def disconnect(self):
        self._connected = False
