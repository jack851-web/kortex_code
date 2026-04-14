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
import gc
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
        # 数据采集状态标志：采集期间暂停 GLFW viewer 渲染，避免 OpenGL context 竞争
        self._collecting_active = False
        # 每线程渲染器缓存（避免线程切换时反复创建/销毁导致 OpenGL OOM）
        self._thread_renderers: Dict[int, mujoco.Renderer] = {}

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

        init_thread_name = threading.current_thread().name
        print(f"[SimuInterface] === initialize() starting on thread [{init_thread_name}] ===", flush=True)

        try:
            # ============================================================
            # 阶段0：关闭 GLFW 查看器（必须在替换 model/data 之前）
            # ============================================================
            print(f"[SimuInterface] >>> PHASE-0: closing GLFW viewer...", flush=True)
            self.close_glfw_viewer()
            print(f"[SimuInterface] >>> PHASE-0: GLFW viewer closed", flush=True)

            # 清理旧的 IK 求解器和模拟控制器（必须在加载新模型前，因为它们持有旧 MjData 引用）
            self._ik_solver = None
            self._sim_controller = None

            # ============================================================
            # 阶段1：安全释放旧的模型/渲染资源（全部在锁内完成）
            # ============================================================
            print(f"[SimuInterface] >>> PHASE-1: acquiring lock for resource cleanup...", flush=True)
            with self._lock:
                print(f"[SimuInterface] >>> PHASE-1: lock acquired", flush=True)
                
                # 显式释放旧的 MuJoCo 模型和数据
                old_model = self._model
                old_data = self._data
                self._model = None
                self._data = None
                del old_model
                del old_data

                # 清理全局 renderer（不跨线程销毁 GLFW 窗口，只释放 MjrContext）
                if self._renderer is not None:
                    self._safe_close_renderer(self._renderer, destroy_window=False)
                    self._renderer = None
                self._renderer_thread_id = None
                self._renderer_thread_name = None

                # 清理缓存的线程渲染器
                for tid, renderer in list(self._thread_renderers.items()):
                    self._safe_close_renderer(renderer, destroy_window=False)
                self._thread_renderers.clear()

                # 【关键修复】不调用 _viewer.close()！
                # launch_passive() 在创建线程上创建内部 GLFW 窗口，
                # 跨线程调用 .close() 会触发跨线程 glfw 操作 → Windows C 层 segfault。
                # 只清空引用，让 Python GC 安全回收（GLFW 窗口会随进程退出自动关闭）。
                if self._viewer is not None:
                    print(f"[SimuInterface] WARNING: Dropping old passive viewer reference "
                          f"(was created on another thread, cannot safely close here)", flush=True)
                    self._viewer = None

            print(f"[SimuInterface] >>> PHASE-1: resource cleanup done, lock released", flush=True)

            # 短暂等待 + 强制 GC，确保 OpenGL 和 MuJoCo 旧资源完全释放
            import time as _time
            _time.sleep(0.15)
            gc.collect()

            # ============================================================
            # 阶段2：加载新模型
            # ============================================================
            print(f"[SimuInterface] >>> PHASE-2: loading model from {self._xml_path}", flush=True)
            self._model = mujoco.MjModel.from_xml_path(self._xml_path)
            self._data = mujoco.MjData(self._model)
            print(f"[SimuInterface] >>> PHASE-2: model loaded OK", flush=True)
            
            # 获取关节索引
            self._joint_indices = []
            for name in self._joint_names:
                idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if idx >= 0:
                    self._joint_indices.append(idx)
                    qpos_idx = self._model.jnt_qposadr[idx]
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
            print(f"[SimuInterface] Found cameras: {camera_names}", flush=True)
            
            # ============================================================
            # 阶段3：mj_forward + IK 初始化
            # ============================================================
            print("[SimuInterface] >>> PHASE-3: mj_forward start", flush=True)
            mujoco.mj_forward(self._model, self._data)
            print("[SimuInterface] >>> PHASE-3: mj_forward done", flush=True)
            
            print(f"[SimuInterface] >>> PHASE-3: IK init, _use_ik={self._use_ik}, IK_AVAILABLE={IK_AVAILABLE}", flush=True)
            if self._use_ik and IK_AVAILABLE:
                try:
                    print("[SimuInterface] >>> PHASE-3: Creating MuJoCoIK...", flush=True)
                    self._ik_solver = MuJoCoIK(self._model, self._data, self._tcp_site_name)
                    print(f"[SimuInterface] IK solver initialized", flush=True)
                    print(f"[SimuInterface] IK joints: {self._ik_solver.joint_names}", flush=True)
                except Exception as e:
                    print(f"[SimuInterface] Failed to initialize IK: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    self._use_ik = False
            else:
                print("[SimuInterface] >>> PHASE-3: SKIPPED (IK disabled)", flush=True)
            print("[SimuInterface] >>> PHASE-3: IK init done", flush=True)
            
            # ============================================================
            # 阶段4：启动查看器（仅在明确请求时）
            # ============================================================
            print(f"[SimuInterface] >>> PHASE-4: start_viewer, show_viewer={show_viewer}", flush=True)
            if show_viewer:
                print("[SimuInterface] >>> PHASE-4: Calling start_viewer()...", flush=True)
                self.start_viewer()
                print("[SimuInterface] >>> PHASE-4: start_viewer() returned", flush=True)
            else:
                print("[SimuInterface] >>> PHASE-4: SKIPPED (show_viewer=False)", flush=True)
            
            print(f"[SimuInterface] Model initialized successfully", flush=True)
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
            print(f"[SimuInterface] Error initializing model: {e}", flush=True)
            import traceback
            traceback.print_exc()
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
        
        使用独立线程运行 GLFW 渲染循环，避免与 Qt 主线程冲突
        """
        print(f"[SimuInterface] start_glfw_viewer called: width={width}, height={height}, title={title}")
        print(f"[SimuInterface] _model={self._model is not None}, _data={self._data is not None}, _glfw_viewer_running={self._glfw_viewer_running}")
        
        if self._model is None or self._data is None:
            print("[SimuInterface] Cannot start viewer: model or data not initialized")
            return False
        
        if self._glfw_viewer_running:
            print("[SimuInterface] GLFW viewer already running, skipping")
            return True
        
        # 保存参数供线程使用
        self._glfw_width = width
        self._glfw_height = height
        self._glfw_title = title
        self._glfw_stop_event = threading.Event()
        
        # 启动独立的 GLFW 渲染线程
        self._glfw_thread = threading.Thread(
            target=self._glfw_viewer_loop,
            daemon=True,
            name="GLFW-Viewer-Thread"
        )
        self._glfw_thread.start()
        
        # 等待窗口创建完成（最多等待 5 秒）
        import time
        timeout = 5.0
        start_wait = time.time()
        while not self._glfw_viewer_running and (time.time() - start_wait) < timeout:
            time.sleep(0.05)
        
        if self._glfw_viewer_running:
            print(f"[SimuInterface] GLFW viewer started in separate thread: {width}x{height}")
            return True
        else:
            print("[SimuInterface] GLFW viewer failed to start within timeout")
            return False
    
    def _glfw_viewer_loop(self):
        """GLFW 查看器的主循环（在独立线程中运行）"""
        try:
            print("[SimuInterface] [_glfw_viewer_loop] Starting...")
            
            # 初始化 GLFW（必须在创建窗口的线程中）
            # 注意：disconnect() 可能调用了 glfw.terminate()，需要重新 init
            if not SimuInterface._glfw_initialized:
                if not glfw.init():
                    print("[SimuInterface] Failed to initialize GLFW")
                    return
                SimuInterface._glfw_initialized = True
                print("[SimuInterface] GLFW initialized")
            
            # 设置窗口提示（在创建窗口前）
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)  # 确保窗口可见
            glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
            
            # 创建窗口
            self._glfw_window = glfw.create_window(
                self._glfw_width, self._glfw_height, self._glfw_title, None, None
            )
            
            if not self._glfw_window:
                print("[SimuInterface] Failed to create GLFW window")
                return
            
            print("[SimuInterface] GLFW window created")
            glfw.make_context_current(self._glfw_window)
            glfw.swap_interval(1)
            
            framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self._glfw_window)
            print(f"[SimuInterface] Framebuffer: {framebuffer_width}x{framebuffer_height}")
            
            # 创建 MuJoCo 渲染资源（必须在 GL 上下文所在的线程）
            self._glfw_ctx = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
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
            
            # 显式显示窗口
            glfw.show_window(self._glfw_window)
            print("[SimuInterface] GLFW window shown")
            
            print("[SimuInterface] GLFW viewer running, entering render loop...")
            
            # 渲染循环
            import time as _time
            self._start_time = _time.time()
            self._render_frame_count = 0
            
            while not self._glfw_stop_event.is_set() and self._glfw_window:
                if glfw.window_should_close(self._glfw_window):
                    break
                
                self._glfw_render_frame()
                self._render_frame_count += 1
                
                # 每60帧打印一次状态
                if self._render_frame_count % 60 == 0:
                    print(f"[SimuInterface] Rendered {self._render_frame_count} frames")
                
                # 控制渲染频率 (~30 FPS，减少资源占用)
                _time.sleep(0.033)
            
            print(f"[SimuInterface] Exiting GLFW render loop, total frames: {self._render_frame_count}")
            
        except Exception as e:
            print(f"[SimuInterface] _glfw_viewer_loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._glfw_viewer_running = False
            # 显式释放 MuJoCo GPU 资源（内部已调用 gc.collect()）
            self._free_glfw_mujoco_resources()
            
            if self._glfw_window:
                try:
                    glfw.destroy_window(self._glfw_window)
                except Exception:
                    pass
                self._glfw_window = None
            
            print("[SimuInterface] GLFW viewer thread exited")

    def _glfw_render_frame(self):
        """渲染一帧（在线程渲染循环中调用）- 多视口布局
        
        使用非阻塞锁：如果锁被 IK/step 等操作持有，跳过本帧而非阻塞等待。
        这避免了 move_to_cartesian（IK迭代）和 step(300) 长时间持锁导致窗口白屏无响应。
        
        注意：GLFW 查看器使用独立的 GL 上下文和 MjrContext（自己的窗口），
        与数据采集线程的离屏 Renderer（另一个 GL 上下文/窗口）完全隔离，
        不受 _collecting_active 标志影响。
        """
        # 注意：移除了 _collecting_active 检查，因为 GLFW 查看器与采集 Renderer
        # 使用独立的 OpenGL 上下文，互不影响。之前因 start_recording() 在 
        # start_glfw_viewer() 之前执行，导致 _collecting_active=True 使窗口永远不渲染。
        
        if not self._glfw_viewer_running:
            return
        
        if self._glfw_window is None:
            return
        
        try:
            if glfw.window_should_close(self._glfw_window):
                # 只设置停止标志，让线程循环退出（不调用 close_glfw_viewer 避免死锁）
                if hasattr(self, '_glfw_stop_event'):
                    self._glfw_stop_event.set()
                return
            
            glfw.make_context_current(self._glfw_window)
            
            # 非阻塞锁：IK/step 持锁期间直接跳过帧，不阻塞等待
            if not self._lock.acquire(blocking=False):
                # 锁被占用，跳过此帧（只 poll 保持窗口响应，不 swap 旧画面避免闪烁）
                glfw.poll_events()
                return
            
            # === 诊断日志：前3帧打印每步状态 ===
            tick = getattr(self, '_render_tick', 0)
            do_diag = tick < 5

            try:
                # 检查模型和数据是否有效
                if self._model is None or self._data is None:
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: model/data is None, SKIP")
                    return

                try:
                    mujoco.mj_forward(self._model, self._data)
                except Exception as e:
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: mj_forward FAILED: {e}")
                    return  # 模型可能已被重新加载，跳过此帧
                
                if do_diag:
                    print(f"[GLFW-RENDER] tick={tick}: mj_forward OK")

                viewport_width, viewport_height = glfw.get_framebuffer_size(self._glfw_window)
                if viewport_width <= 0 or viewport_height <= 0:
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: bad viewport {viewport_width}x{viewport_height}, SKIP")
                    return
                
                if do_diag:
                    print(f"[GLFW-RENDER] tick={tick}: viewport={viewport_width}x{viewport_height}")

                overlay_w = int(viewport_width * 0.22)
                overlay_h = int(viewport_height * 0.22)

                main_viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

                try:
                    mujoco.mjv_updateScene(
                        self._model, self._data, self._glfw_opt, self._glfw_pert,
                        self._glfw_cam, mujoco.mjtCatBit.mjCAT_ALL, self._glfw_scn
                    )
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: mjv_updateScene OK (ngeom={self._glfw_scn.ngeom})")
                    
                    mujoco.mjr_render(main_viewport, self._glfw_scn, self._glfw_ctx)
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: mjr_render OK (main)")
                except Exception as e:
                    if do_diag:
                        print(f"[GLFW-RENDER] tick={tick}: render FAILED: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
                    return  # 渲染失败，跳过此帧

                # 使用 set_display_cameras() 设置的相机名称，分布于四角
                display_cams = getattr(self, '_camera_names', [])
                margin = 10
                corner_positions = [
                    (margin, viewport_height - overlay_h - margin),                     # 左上
                    (viewport_width - overlay_w - margin, viewport_height - overlay_h - margin),  # 右上
                    (margin, margin),                                                    # 左下
                    (viewport_width - overlay_w - margin, margin),                        # 右下
                ]
                for i, cam_name in enumerate(display_cams):
                    if i >= len(corner_positions):
                        break
                    x, y = corner_positions[i]
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
                    except Exception:
                        pass

                import time as _time
                current_time = _time.time()
                if not hasattr(self, '_start_time'):
                    self._start_time = current_time

                sim_time = self._data.time if self._data else 0
                wall_time = current_time - self._start_time
                self._render_tick = tick + 1

                info_lines = [
                    f"tick:        {tick:>6}",
                    f"sim time:    {sim_time:>8.2f}sec",
                    f"wall time:   {wall_time:>8.2f}sec",
                ]
                
                try:
                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150,
                                     mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                     main_viewport,
                                     "\n".join(info_lines), "", self._glfw_ctx)
                except Exception:
                    pass

                if do_diag:
                    print(f"[GLFW-RENDER] tick={tick}: calling swap_buffers...", flush=True)
                glfw.swap_buffers(self._glfw_window)
                if do_diag:
                    print(f"[GLFW-RENDER] tick={tick}: DONE", flush=True)
            finally:
                self._lock.release()

            glfw.poll_events()

        except Exception as e:
            # 不打印堆栈跟踪，避免日志刷屏
            if not hasattr(self, '_last_render_error_time') or \
               (hasattr(self, '_last_render_error_time') and 
                (time.time() - self._last_render_error_time) > 1.0):
                print(f"[SimuInterface] GLFW render frame error: {e}")
                import traceback
                traceback.print_exc()
                self._last_render_error_time = time.time()

    def _free_glfw_mujoco_resources(self):
        """显式释放 GLFW 相关的 MuJoCo GPU 资源
        
        MjrContext 持有 OpenGL 帧缓冲区、纹理等 GPU 资源，
        仅设为 None 依赖 GC 回收会导致 GPU 资源累积（GC 不保证及时执行）。
        必须通过 del + gc.collect() 强制立即释放。
        """
        # MjrContext: 持有 OpenGL 帧缓冲区/纹理，必须立即释放
        if self._glfw_ctx is not None:
            try:
                del self._glfw_ctx
            except Exception:
                pass
            self._glfw_ctx = None
        
        # MjvScene: 持有渲染缓冲区
        if self._glfw_scn is not None:
            try:
                del self._glfw_scn
            except Exception:
                pass
            self._glfw_scn = None
        
        # 其他 Mjv* 对象体积较小，直接置空即可
        self._glfw_cam = None
        self._glfw_opt = None
        self._glfw_pert = None
        self._glfw_viewport = None
        
        # 强制 GC 确保上述 del 触发的 __del__ 被执行
        gc.collect()

    def close_glfw_viewer(self):
        """关闭 GLFW 查看器
        
        注意：GLFW 窗口必须在创建它的线程中销毁。
        这里只设置停止标志，让渲染线程自行清理资源。
        """
        print(f"[SimuInterface] close_glfw_viewer called, _glfw_viewer_running={self._glfw_viewer_running}")
        
        if not self._glfw_viewer_running:
            # 即使查看器未运行，也要确保残留资源被清理
            self._free_glfw_mujoco_resources()
            print("[SimuInterface] GLFW viewer already stopped, resources cleaned")
            return
        
        # 通知线程停止（线程会在 finally 块中清理所有资源）
        if hasattr(self, '_glfw_stop_event') and self._glfw_stop_event:
            self._glfw_stop_event.set()
        
        # 等待线程结束（最多 5 秒）
        # 线程会在 finally 块中清理所有 GLFW 资源
        if hasattr(self, '_glfw_thread') and self._glfw_thread and self._glfw_thread.is_alive():
            self._glfw_thread.join(timeout=5.0)
            if self._glfw_thread.is_alive():
                print("[SimuInterface] Warning: GLFW thread did not stop within timeout")
                # 线程超时未退出，强制清理 MuJoCo 资源（可能导致渲染线程崩溃，但避免资源泄漏）
                self._free_glfw_mujoco_resources()
        
        self._glfw_viewer_running = False
        
        # 确保资源已释放（可能已由线程 finally 块释放，这里是兜底）
        self._free_glfw_mujoco_resources()
        
        self._glfw_window = None
        self._glfw_stop_event = None
        self._glfw_thread = None
        
        print("[SimuInterface] GLFW viewer closed")
    
    def sync_viewer(self):
        """同步更新查看器 (被动模式)"""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def _get_ctrl_idx_for_joint(self, joint_idx: int) -> int:
        if self._model is None:
            return -1
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return np.zeros(6)
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return False
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return 0.0
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return False
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
                # 锁内再次检查，防止竞态
                if self._model is None or self._data is None:
                    return False
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return np.zeros(3)
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return False
            try:
                body_idx = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, object_name)
                if body_idx < 0:
                    return False
                joint_adr = self._model.body_jntadr[body_idx]
                if joint_adr >= 0:
                    qpos_adr = self._model.jnt_qposadr[joint_adr]
                    # 先设置 x/y，并统一姿态，z 在 reset_z 模式下按"底部高度"自动对齐
                    self._data.qpos[qpos_adr] = position[0]  # x
                    self._data.qpos[qpos_adr+1] = position[1]  # y
                    # 重置四元数为单位四元数 (w=1, x=0, y=0, z=0)
                    self._data.qpos[qpos_adr+3:qpos_adr+7] = [1, 0, 0, 0]

                    # 重置速度
                    qvel_adr = self._model.jnt_dofadr[joint_adr]
                    # freejoint: 3线速度 + 3角速度 = 6个qvel
                    self._data.qvel[qvel_adr:qvel_adr+6] = 0

                    if reset_z:
                        # 将传入 z 解释为"物体底部目标高度"
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

        # 分段执行：每 _lock_chunk 步释放一次锁，让 GLFW viewer 和
        # SimuPublisher 有机会获取锁更新画面/采集数据
        _lock_chunk = 20  # 每次持锁执行的步数
        steps_done = 0

        while steps_done < n_steps:
            chunk = min(_lock_chunk, n_steps - steps_done)
            with self._lock:
                if steps_done == 0:
                    self._sync_gripper_tips()
                for _ in range(chunk):
                    mujoco.mj_step(self._model, self._data)
                self.sync_viewer()
            steps_done += chunk
            # 每个 chunk 之间让出，给 GLFW viewer 和 SimuPublisher 获取锁的机会
            if steps_done < n_steps:
                time.sleep(0.001)

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

    @staticmethod
    def _safe_close_renderer(renderer, destroy_window=False):
        """安全关闭 Renderer，防止 __del__ 二次调用导致 AttributeError
        
        Args:
            renderer: mujoco.Renderer 实例
            destroy_window: 是否同时销毁 GLFW offscreen 窗口。
                           在 disconnect() 中应为 True（最终清理，会 gc.collect + glfw.terminate）。
                           在 initialize()/线程切换时应为 False（避免跨线程 GLFW 操作导致 C 层崩溃）。
        """
        if renderer is None:
            return
        try:
            # 释放 MjrContext（GPU 帧缓冲区/纹理等资源，这是最占 GPU 内存的）
            if hasattr(renderer, '_mjr_context') and renderer._mjr_context is not None:
                try:
                    renderer._mjr_context.free()
                except Exception:
                    pass
                renderer._mjr_context = None
            # 释放 MjvScene（渲染缓冲区）
            if hasattr(renderer, '_scene') and renderer._scene is not None:
                renderer._scene = None
            if destroy_window:
                # 最终清理场景：主动销毁 GLFW 窗口 + 置空，防止 __del__ 二次调用
                if hasattr(renderer, '_gl_context') and renderer._gl_context is not None:
                    try:
                        renderer._gl_context.free()
                    except Exception:
                        pass
                    renderer._gl_context = None
            else:
                # 线程切换/initialize 场景：不能跨线程调用 glfw.destroy_window()
                # 必须阻止 GC 在错误线程触发 GLContext.__del__ -> free() -> glfw.destroy_window()
                # 解决方案：先将 GLContext 内部的 _context（GLFW 窗口指针）置空，
                # 使其 __del__ 调用 free() 时跳过 glfw.destroy_window()
                if hasattr(renderer, '_gl_context') and renderer._gl_context is not None:
                    gl_ctx = renderer._gl_context
                    if hasattr(gl_ctx, '_context') and gl_ctx._context is not None:
                        gl_ctx._context = None
                    renderer._gl_context = None
        except Exception:
            pass

    def _ensure_renderer_for_current_thread(self):
        """确保当前线程有可用的 Renderer
        
        采用单 Renderer 策略：所有线程共享同一个 Renderer 实例。
        mujoco.Renderer.render() 内部会调用 _gl_context.make_current()，
        因此同一个 Renderer 可以在不同线程间安全使用（通过锁串行化）。
        这避免了每线程缓存导致的 renderer 反复创建/销毁问题。
        """
        # 如果已有有效的 renderer，直接复用
        if self._renderer is not None:
            if hasattr(self._renderer, '_mjr_context') and self._renderer._mjr_context is not None:
                return
            # renderer 已失效（_mjr_context 被关闭），需要重建
            self._safe_close_renderer(self._renderer)
            self._renderer = None

        # 清理所有缓存的线程 renderer（如果有的话）
        for tid, rnd in list(self._thread_renderers.items()):
            self._safe_close_renderer(rnd)
        self._thread_renderers.clear()

        # 创建新的全局 renderer
        print(f"[SimuInterface] Creating new Renderer (thread: {threading.current_thread().name})")
        try:
            self._renderer = mujoco.Renderer(self._model, height=self._render_height, width=self._render_width)
        except Exception as e:
            print(f"[SimuInterface] Failed to create renderer: {e}")
            self._renderer = None
            return
        self._renderer_thread_id = threading.get_ident()
        self._renderer_thread_name = threading.current_thread().name

    def render(self, camera_name: Optional[str] = None) -> np.ndarray:
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
            try:
                mujoco.mj_forward(self._model, self._data)
                self._ensure_renderer_for_current_thread()
                
                if self._renderer is None:
                    return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
                
                if camera_name:

                    cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
                    if cam_id >= 0:
                        self._renderer.update_scene(self._data, camera=cam_id)
                    else:
                        self._renderer.update_scene(self._data)
                else:
                    self._renderer.update_scene(self._data)
                
                # 返回独立拷贝，避免渲染缓冲区被覆盖（单次 copy 即可）
                return np.copy(self._renderer.render())

                
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
                    name: np.copy(proc_images.get(name, np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)))
                    for name in cam_names
                }

        # === 诊断日志：前5次调用打印详情 ===
        diag_cnt = getattr(self, '_get_cam_diag_cnt', 0)
        do_diag = diag_cnt < 5

        images = {}

        # 采集期间：非采集线程的渲染请求直接返回空帧，避免 GL context 跨线程竞争
        if self._collecting_active and threading.current_thread() != getattr(self, '_collector_thread_id', None):
            if do_diag:
                print(f"[GET-CAM] #{diag_cnt}: collecting active + non-collector -> empty frames", flush=True)
            self._get_cam_diag_cnt = diag_cnt + 1
            return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

        # 一次加锁完成多相机渲染，避免 GUI 线程与采集线程交错导致相机串帧
        with self._lock:
            try:
                if self._model is None or self._data is None:
                    if do_diag:
                        print(f"[GET-CAM] #{diag_cnt}: model/data None -> empty", flush=True)
                    self._get_cam_diag_cnt = diag_cnt + 1
                    return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

                mujoco.mj_forward(self._model, self._data)

                if do_diag:
                    print(f"[GET-CAM] #{diag_cnt}: mj_forward OK, ensure_renderer...", flush=True)

                self._ensure_renderer_for_current_thread()

                if do_diag:
                    print(f"[GET-CAM] #{diag_cnt}: renderer={'OK' if self._renderer else 'NONE'}", flush=True)

                if self._renderer is None:
                    if do_diag:
                        print(f"[GET-CAM] #{diag_cnt}: renderer NONE -> empty", flush=True)
                    self._get_cam_diag_cnt = diag_cnt + 1
                    return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

                for name in cam_names:

                    cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                    if cam_id >= 0:
                        self._renderer.update_scene(self._data, camera=cam_id)
                    else:
                        self._renderer.update_scene(self._data)
                    img = self._renderer.render()

                    if do_diag:
                        img_min, img_max = img.min(), img.max()
                        print(f"[GET-CAM] #{diag_cnt}: cam={name} shape={img.shape} min={img_min} max={img_max} mean={img.mean():.1f}", flush=True)

                    images[name] = np.copy(img)

            except Exception as e:
                print(f"[SimuInterface] get_camera_images error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                self._get_cam_diag_cnt = diag_cnt + 1
                return {name: np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8) for name in cam_names}

        self._get_cam_diag_cnt = diag_cnt + 1
        return images


    def set_display_cameras(self, camera_names: List[str]):
        self._camera_names = camera_names

    def set_collecting_active(self, active: bool):
        """设置数据采集状态标志。
        
        采集期间（active=True）：
          - GLFW viewer 渲染循环暂停，避免与 Renderer 的 offscreen GL context 竞争
          - Qt GUI 的 get_camera_images() 返回空帧，不干扰采集线程的渲染
          
        非采集期间（active=False）：
          - GLFW viewer 恢复正常渲染
          - Qt GUI 可以正常获取预览图像
        """
        self._collecting_active = active

    def is_collecting_active(self) -> bool:
        return self._collecting_active

    def start_process_renderer(self, camera_names: Optional[List[str]] = None):
        from scripts.simu.render_process import SimuRenderProcess
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
            print(f"  _use_ik={self._use_ik}, _ik_solver={'None' if self._ik_solver is None else 'initialized'}, IK_AVAILABLE={IK_AVAILABLE}")
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
            # 优化：减少轨迹点数量以提高性能
            trajectory_points = []
            # 使用更少的插值点（从 steps+1 减少到约 20-30 个）
            n_interp = min(steps + 1, 50)  # 增加路径点数量，提高轨迹跟踪精度
            for i in range(n_interp):
                alpha = i / (n_interp - 1)  # 归一化到 [0, 1]
                interp_pos = current_pos + alpha * (target_pos - current_pos)
                trajectory_points.append(interp_pos)
            
            print(f"[SimuInterface] Generated {len(trajectory_points)} trajectory points (optimized)")
            
            # 逐点执行：对每个中间点求解 IK 并执行一步仿真
            q = current_q.copy()
            
            for step_idx, traj_pos in enumerate(trajectory_points):
                ik_start_time = time.time()  # IK 计时开始
                
                if target_ori is not None:
                    try:
                        # 增加 IK 迭代次数，提高精度
                        q_ik = self._ik_solver.inverse_kinematics(
                            traj_pos,
                            target_ori,
                            initial_guess=q,
                            max_iterations=50,  # 增加迭代次数
                            tolerance=5e-4,    # 收紧容差到 0.5mm
                        )
                        solve_info = self._ik_solver.get_last_solve_info() if hasattr(self._ik_solver, 'get_last_solve_info') else {}
                        ik_success = bool(solve_info.get('success', False))
                        ik_pos_err = float(solve_info.get('position_error', 1e9))
                        ik_iter = int(solve_info.get('iterations', 0))

                        ik_time = time.time() - ik_start_time
                        # 每步都打印 IK 信息（调试）
                        print(f"[SimuInterface] Step {step_idx}: IK time={ik_time*1000:.1f}ms, iter={ik_iter}, err={ik_pos_err:.4f}m")

                        # 注意：inverse_kinematics 总会返回 q，不会返回 None
                        # 若姿态约束不可达，退化为位置IK，避免路径长时间偏离目标
                        # 减小接受误差阈值，提高精度
                        if ik_success or ik_pos_err < 0.005:  # 误差小于 5mm 才接受
                            q = np.array(q_ik, dtype=float)
                            for i in range(len(q)):
                                q[i] = np.clip(q[i], self._ik_solver.joint_ranges[i][0], self._ik_solver.joint_ranges[i][1])
                        else:
                            # 使用快速雅可比迭代（单步）
                            current_pos = self._ik_solver.forward_kinematics(q)
                            pos_error = traj_pos - current_pos
                            J = self._ik_solver.jacobian(q, skip_forward=True)  # 跳过重复 forward
                            J_pos = J[:3, :]
                            damping = 0.1
                            JtJ = J_pos.T @ J_pos
                            JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                            delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)
                            q = q + delta_q * 0.5
                    except Exception:
                        current_pos = self._ik_solver.forward_kinematics(q)
                        pos_error = traj_pos - current_pos
                        J = self._ik_solver.jacobian(q, skip_forward=True)  # 跳过重复 forward
                        J_pos = J[:3, :]
                        damping = 0.1
                        JtJ = J_pos.T @ J_pos
                        JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                        delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)
                        q = q + delta_q * 0.5

                else:
                    # IK求解是纯计算，不访问MuJoCo资源，不需要锁
                    # 增加迭代次数，提高 IK 精度
                    for ik_iter in range(30):  # 增加到 30 次迭代
                        current_pos = self._ik_solver.forward_kinematics(q)
                        pos_error = traj_pos - current_pos
                        error_norm = np.linalg.norm(pos_error)

                        if error_norm < 5e-4:  # 收紧收敛阈值到 0.5mm
                            break

                        J = self._ik_solver.jacobian(q, skip_forward=True)  # 跳过重复 forward
                        J_pos = J[:3, :]

                        damping = 0.03  # 进一步减小阻尼，加快收敛
                        JtJ = J_pos.T @ J_pos
                        JtJ_damped = JtJ + damping**2 * np.eye(JtJ.shape[0])
                        delta_q = np.linalg.solve(JtJ_damped, J_pos.T @ pos_error)

                        max_delta = 0.2  # 减小步长限制，避免跳过最优解
                        if np.linalg.norm(delta_q) > max_delta:
                            delta_q = delta_q / np.linalg.norm(delta_q) * max_delta
                        
                        q = q + delta_q * 0.7  # 增大步长缩放因子
                        
                        for i in range(len(q)):
                            q[i] = np.clip(q[i], self._ik_solver.joint_ranges[i][0], 
                                          self._ik_solver.joint_ranges[i][1])
                    
                    ik_time = time.time() - ik_start_time
                    if step_idx % 5 == 0:
                        print(f"[SimuInterface] Step {step_idx}: IK time={ik_time*1000:.1f}ms, iter={ik_iter+1}, err={error_norm:.4f}m")

                
                # 只在访问MuJoCo资源时加锁
                # 每步执行多个 mj_step 以加速收敛，同时步间释放锁让 GLFW/SimuPublisher 更新
                with self._lock:
                    for j, joint_idx in enumerate(self._joint_indices):
                        if j < len(q):
                            ctrl_idx = self._get_ctrl_idx_for_joint(joint_idx)
                            if ctrl_idx >= 0:
                                ctrl_range = self._model.actuator_ctrlrange[ctrl_idx]
                                clipped_val = np.clip(q[j], ctrl_range[0], ctrl_range[1])
                                self._data.ctrl[ctrl_idx] = clipped_val
                    
                    for _ in range(10):  # 每个轨迹点执行 10 步仿真
                        mujoco.mj_step(self._model, self._data)
                    self.sync_viewer()
                
                # 每个轨迹点后短暂让出，给 GLFW viewer 和 SimuPublisher 获取锁的机会
                time.sleep(0.001)
                
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
        with self._lock:
            # 在锁内检查，避免并发问题
            if self._model is None or self._data is None:
                return None, None
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
        # 先关闭 GLFW 查看器（必须在清理其他资源前，它会释放 MjrContext GPU 资源）
        self.close_glfw_viewer()

        self.stop_process_renderer()

        # 加锁保护，防止其他线程在 _model/_data 置空后仍访问
        with self._lock:
            self._ik_solver = None
            self._sim_controller = None

            # 清理所有缓存的线程渲染器（防止 GPU/CPU 内存泄漏）
            # 注意：使用 destroy_window=False 避免跨线程销毁 GLFW 窗口
            for tid, renderer in list(self._thread_renderers.items()):
                self._safe_close_renderer(renderer, destroy_window=False)
            self._thread_renderers.clear()

            if self._renderer:
                self._safe_close_renderer(self._renderer, destroy_window=False)
                self._renderer = None
            self._renderer_thread_id = None
            self._renderer_thread_name = None
            
            # 不调用 _viewer.close()，原因同 initialize()：可能跨线程

            # 显式释放 MuJoCo 模型和数据
            self._model = None
            self._data = None

        # 强制 GC 回收所有已释放的 MuJoCo/OpenGL 资源
        gc.collect()

        # 注意：不调用 glfw.terminate()！
        # 原因：disconnect 后可能立即创建新的 SimuInterface 并需要 GLFW，
        # 而且 GC 可能延迟回收旧 renderer 的 GLContext，terminate 后 GLContext
        # 析构函数调用 glfw.destroy_window → 段错误/崩溃。
        # GLFW 全局状态在整个进程生命周期保持，程序退出时自动清理。

    def __del__(self):
        """析构函数：确保异常退出时也清理 GPU 资源"""
        try:
            # 显式释放 MjrContext GPU 资源
            if hasattr(self, '_glfw_ctx') and self._glfw_ctx is not None:
                try:
                    del self._glfw_ctx
                except Exception:
                    pass
                self._glfw_ctx = None
            
            if hasattr(self, '_thread_renderers') and self._thread_renderers:
                # __del__ 可能在任意线程被 GC 触发，不能用 destroy_window=True
                for tid, renderer in list(self._thread_renderers.items()):
                    self._safe_close_renderer(renderer, destroy_window=False)
                self._thread_renderers.clear()
            if hasattr(self, '_renderer') and self._renderer:
                # 同上，不跨线程销毁 GLFW 窗口
                self._safe_close_renderer(self._renderer, destroy_window=False)
            # 注意：不在 __del__ 中调用 glfw.terminate()
            # 原因同 disconnect()：可能导致延迟 GC 的 GLContext 析构崩溃
        except Exception:
            pass


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

    def set_collecting_active(self, active: bool):
        """设置数据采集状态标志。
        
        采集期间（active=True）：
          - GLFW viewer 渲染循环暂停，避免与 Renderer 的 offscreen GL context 竞争
          - Qt GUI 的 get_camera_images() 返回空帧，不干扰采集线程的渲染
          
        非采集期间（active=False）：
          - GLFW viewer 恢复正常渲染
          - Qt GUI 可以正常获取预览图像
        """
        self._collecting_active = active

    def is_collecting_active(self) -> bool:
        return self._collecting_active

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
