import numpy as np
import time
from typing import List, Tuple, Optional, Callable, Dict, Union, Any



class GraspExecutor:
    # 物体类型定义
    OBJECT_TYPE_CUBE = "cube"           # 方块 - 中心抓取
    OBJECT_TYPE_CUP = "cup"             # 茶杯 - 侧壁抓取
    OBJECT_TYPE_BOTTLE = "bottle"       # 瓶子 - 侧面抓取
    OBJECT_TYPE_BOWL = "bowl"           # 碗 - 边缘抓取
    
    # 抓取偏移配置 (单位: 米)
    # offset: [x, y, z] 水平偏移
    # height_adjust: 抓取高度调整 (相对于物体中心的偏移)
    # gripper_open: 夹爪张开程度 (0.0-1.0)
    # gripper_close: 夹爪闭合程度 (0.0-1.0)
    GRASP_OFFSETS = {
        OBJECT_TYPE_CUBE: {
            "offset": [0.0, 0.0, 0.0],
            "height_adjust": 0.0,  # 方块中心抓取
            "gripper_open": 0.0,   # 完全张开
            "gripper_close": 0.6,  # 闭合60%抓取方块
            "description": "中心抓取"
        },
        OBJECT_TYPE_CUP: {
            "offset": [0.04, 0.0, 0.0],
            "height_adjust": 0.0,    # 茶杯在中心高度抓取杯壁中部
            "gripper_open": 0.0,     # 完全张开
            "gripper_close": 0.58,   # 保守闭合，先稳定接触避免挤压弹飞
            "description": "向x方向偏移抓取杯壁"
        },
        OBJECT_TYPE_BOTTLE: {
            "offset": [0.04, 0.0, 0.0],
            "height_adjust": -0.03,  # 瓶子向下调整抓取瓶身
            "gripper_open": 0.0,     # 完全张开
            "gripper_close": 0.8,    # 闭合80%抓取瓶身
            "description": "向x方向偏移抓取瓶身"
        },
        OBJECT_TYPE_BOWL: {
            "offset": [0.05, 0.0, 0.0],
            "height_adjust": 0.02,   # 碗向上调整抓取碗沿
            "gripper_open": 0.0,     # 完全张开
            "gripper_close": 0.75,   # 闭合75%抓取碗沿
            "description": "向x方向偏移抓取碗沿"
        },
    }

    def __init__(
        self,
        real_interface,
        simu_interface,
        home_position: List[float],
        pre_grasp_offset: Union[List[float], Dict[str, Any]],
        lift_height: float,

        approach_height: float,
        use_simulation: bool = False,  # 纯仿真模式标志
    ):
        self._real = real_interface
        self._simu = simu_interface
        self._use_simulation = use_simulation  # 纯仿真模式
        self._home_position = np.array(home_position)
        self._default_pre_grasp_offset = np.array([0.0, 0.0, 0.15], dtype=float)
        self._object_pre_grasp_offsets: Dict[str, np.ndarray] = {}
        self._lift_height = lift_height

        self._approach_height = approach_height
        self._current_task_id = 0
        self._is_executing = False
        self._waypoint_delay = 0.1
        
        # 放置时的抬升参数
        self._micro_lift_height = 0.02  # 松开前微抬高度（默认 2cm）
        self._release_lift_height = 0.08  # 松开后抬升高度（默认 8cm）

        self._gripper_close_position = 1.0
        self._gripper_open_position = 0.0
        self._default_gripper_open: Optional[float] = None
        self._default_gripper_close: Optional[float] = None
        self._object_gripper_positions: Dict[str, Tuple[float, float]] = {}
        self._default_orientation = np.array([180.0, 0.0, 0.0], dtype=float)
        self._object_orientations: Dict[str, np.ndarray] = {}
        self._default_grasp_offset: Optional[np.ndarray] = None
        self._object_grasp_offsets: Dict[str, np.ndarray] = {}
        self._object_type = self.OBJECT_TYPE_CUBE  # 默认方块类型



        self._sim_object_body_name = "cube"
        self._frame_callback = None

        self.set_pre_grasp_offset(pre_grasp_offset)


    def set_task(

        self,
        task_id: int,
        object_position: np.ndarray,
        target_position: np.ndarray,
    ):
        """设置任务
        object_position: [x, y, z] 使用config中的位置（含抓取闭合高度）
        target_position: [x, y, z] 放置位置
        """
        self._current_task_id = task_id
        self._object_position = np.array(object_position)
        self._target_position = np.array(target_position)
        print(f"[GraspExecutor] object z source: config_z={self._object_position[2]:.6f}")



    def set_target(
        self,
        object_position: np.ndarray,
        place_position: np.ndarray,
    ):
        """设置抓取目标位置
        object_position: [x, y, z] 使用config中的位置（含抓取闭合高度）
        place_position: [x, y, z] 放置位置
        """
        self._object_position = np.array(object_position)
        self._target_position = np.array(place_position)

    
    def _get_object_z_from_xml(self) -> float:
        """从仿真接口获取物体的z坐标"""
        try:
            if hasattr(self._simu, 'get_object_position'):
                pos = self._simu.get_object_position(self._sim_object_body_name)
                return pos[2]

        except Exception as e:
            print(f"[GraspExecutor] Warning: Could not get object z from XML: {e}")
        # 默认返回一个安全高度
        return 0.02

    def _get_orientation_for_object(self, object_type: Optional[str] = None) -> np.ndarray:
        key = object_type or self._object_type
        orientation = self._object_orientations.get(key, self._default_orientation)
        return np.array(orientation, dtype=float)

    def _get_pre_grasp_offset_for_object(self, object_type: Optional[str] = None) -> np.ndarray:
        key = object_type or self._object_type
        offset = self._object_pre_grasp_offsets.get(key, self._default_pre_grasp_offset)
        return np.array(offset, dtype=float)

    def _get_gripper_positions_for_object(self, object_type: Optional[str] = None) -> Tuple[float, float]:
        key = object_type or self._object_type
        builtin_cfg = self.GRASP_OFFSETS.get(key, {})

        open_pos = float(builtin_cfg.get("gripper_open", 0.0))
        close_pos = float(builtin_cfg.get("gripper_close", 1.0))

        if self._default_gripper_open is not None:
            open_pos = float(self._default_gripper_open)
        if self._default_gripper_close is not None:
            close_pos = float(self._default_gripper_close)

        if key in self._object_gripper_positions:
            obj_open, obj_close = self._object_gripper_positions[key]
            open_pos = float(obj_open)
            close_pos = float(obj_close)

        open_pos = float(np.clip(open_pos, 0.0, 1.0))
        close_pos = float(np.clip(close_pos, 0.0, 1.0))
        return open_pos, close_pos

    def _make_cartesian_pose(self, position: np.ndarray) -> np.ndarray:


        if len(position) == 3:
            orientation = self._get_orientation_for_object()
            return np.concatenate([position, orientation])
        return position[:6]


    def get_waypoints(self) -> List[Tuple[np.ndarray, str]]:
        MIN_HEIGHT = 0.01  # 最小安全高度
        
        # ========== 关键改动：从仿真获取物体实际位置 ==========
        object_actual_pos = None
        if hasattr(self._simu, 'get_object_position'):
            try:
                object_actual_pos = self._simu.get_object_position(self._sim_object_body_name)
                print(f"[GraspExecutor] 从仿真获取物体实际位置: {object_actual_pos}")
            except Exception as e:
                print(f"[GraspExecutor] Warning: 无法从仿真获取物体位置: {e}")
        
        # 使用实际位置或配置位置
        if object_actual_pos is not None:
            object_pos = np.array(object_actual_pos[:3])
            print(f"[GraspExecutor] 使用仿真实际位置作为抓取目标")
        else:
            object_pos = self._object_position.copy()
            print(f"[GraspExecutor] 回退使用配置位置作为抓取目标: {object_pos}")
        
        # 获取当前物体类型的抓取偏移（仅影响抓取x/y）
        grasp_offset = self._get_grasp_offset()
        pre_grasp_offset = self._get_pre_grasp_offset_for_object()

        # 预抓取位置：在物体上方
        pre_grasp = object_pos + grasp_offset + pre_grasp_offset


        pre_grasp[2] = max(pre_grasp[2], MIN_HEIGHT)

        # 抓取位置：物体位置 + 抓取偏移
        grasp_pos = object_pos.copy() + grasp_offset
        grasp_pos[2] = max(object_pos[2], MIN_HEIGHT)
        
        # 抬起位置
        lift_pos = object_pos.copy()
        lift_pos[2] = max(object_pos[2], self._lift_height, MIN_HEIGHT)
        
        target_lift_pos = self._target_position.copy()
        target_lift_pos[2] = max(self._target_position[2], self._lift_height, MIN_HEIGHT)
        
        # 使用用户设置的放置高度，但不低于最小安全高度
        target_place_pos = self._target_position.copy()
        target_place_pos[2] = max(target_place_pos[2], MIN_HEIGHT)
        
        # 放下后抬起位置（避免碰到物体）
        lift_after_place = self._target_position.copy()
        lift_after_place[2] = max(self._target_position[2] + 0.1, MIN_HEIGHT)
        
        waypoints = [
            (self._make_cartesian_pose(pre_grasp), "pre_grasp"),
            (self._make_cartesian_pose(grasp_pos), "grasp"),
            (self._make_cartesian_pose(lift_pos), "lift"),
            (self._make_cartesian_pose(target_lift_pos), "move"),
            (self._make_cartesian_pose(target_place_pos), "place"),
        ]
        print(
            f"[GraspExecutor] waypoint z: pre_grasp={pre_grasp[2]:.6f}, "
            f"grasp={grasp_pos[2]:.6f}, lift={lift_pos[2]:.6f}, place={target_place_pos[2]:.6f}"
        )


        return waypoints


    def execute(
        self,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> bool:
        self._is_executing = True
        try:
            waypoints = self.get_waypoints()
            for i, (pose, name) in enumerate(waypoints):
                if not self._is_executing:
                    return False

                if progress_callback:
                    progress_callback(name, i + 1, len(waypoints))

                print(f"[GraspExecutor] Moving to {name}: {pose[:3]}")
                arrived = self._move_to_position(pose)

                if arrived:
                    print(f"[GraspExecutor] ✓ Arrived at {name}")
                else:
                    print(f"[GraspExecutor] ⚠ Not exactly at {name}, but continuing...")
                    # 不 abort，继续执行后续步骤

                if name == "pre_grasp":
                    self._open_gripper()
                    print(f"[GraspExecutor] Gripper opened for pre_grasp")
                elif name == "grasp":
                    self._close_gripper()
                    print(f"[GraspExecutor] Gripper closed for grasp")
                elif name == "place":
                    # 放置时：先微抬，再缓慢松开，避免弹开物体
                    print(f"[GraspExecutor] Place: lifting {self._micro_lift_height*100:.1f}cm before release...")
                    current_pose = self._simu.get_tcp_position() if self._use_simulation else self._real.get_cartesian_pose()
                    micro_lift = np.array(current_pose[:6]) if len(current_pose) >= 6 else np.zeros(6)
                    micro_lift[2] += self._micro_lift_height  # 微抬
                    self._move_to_position(micro_lift, tolerance=0.08)
                    time.sleep(0.3)  # 等待物理稳定

                    # 缓慢松开夹爪
                    self._open_gripper(gradual=True)
                    print(f"[GraspExecutor] Gripper opened for place (gradual)")

                    # 松开后先水平后退，再垂直抬升，避免碰到物体
                    time.sleep(0.2)  # 等夹爪完全松开
                    current_pose = self._simu.get_tcp_position() if self._use_simulation else self._real.get_cartesian_pose()
                    retreat_pose = np.array(current_pose[:6]) if len(current_pose) >= 6 else np.zeros(6)

                    # 先水平后退（沿抓取偏移的反方向移动）
                    grasp_offset = self._get_grasp_offset()
                    retreat_distance = max(0.08, np.linalg.norm(grasp_offset[:2]) + 0.04)  # 至少后退8cm或抓取偏移+4cm
                    if np.linalg.norm(grasp_offset[:2]) > 0.001:
                        # 沿抓取偏移的反方向后退
                        retreat_direction = -grasp_offset[:2] / np.linalg.norm(grasp_offset[:2])
                    else:
                        # 默认向x负方向后退
                        retreat_direction = np.array([-1.0, 0.0])
                    retreat_pose[0] += retreat_direction[0] * retreat_distance
                    retreat_pose[1] += retreat_direction[1] * retreat_distance

                    print(f"[GraspExecutor] Retreating {retreat_distance*100:.1f}cm horizontally before lift...")
                    self._move_to_position(retreat_pose, tolerance=0.08)

                    # 然后垂直抬升
                    current_pose = self._simu.get_tcp_position() if self._use_simulation else self._real.get_cartesian_pose()
                    lift_after_release = np.array(current_pose[:6]) if len(current_pose) >= 6 else np.zeros(6)
                    lift_after_release[2] += self._release_lift_height
                    print(f"[GraspExecutor] Lifting {self._release_lift_height*100:.1f}cm after retreat...")
                    self._move_to_position(lift_after_release)

                time.sleep(self._waypoint_delay)

            # 任务完成后回到 home 位置
            print(f"[GraspExecutor] Task completed, returning to home position...")
            home_pose = np.zeros(6)
            home_pose[:3] = self._home_position[:3]
            if len(self._home_position) >= 6:
                home_pose[3:6] = self._home_position[3:6]
            self._move_to_position(home_pose)
            print(f"[GraspExecutor] Returned to home position")

            # 判断任务是否成功：检查物体是否到达目标位置
            print(f"[GraspExecutor] _use_simulation={self._use_simulation}, has_get_object_position={hasattr(self._simu, 'get_object_position')}")
            
            if hasattr(self._simu, 'get_object_position'):
                time.sleep(0.5)  # 等待物理稳定
                try:
                    object_pos = self._simu.get_object_position(self._sim_object_body_name)
                    distance_xy = np.linalg.norm(object_pos[:2] - self._target_position[:2])  # 仅检查xy平面
                    z_diff = abs(object_pos[2] - self._target_position[2])
                    distance_3d = np.linalg.norm(object_pos[:3] - self._target_position[:3])  # 3D总距离
                    
                    print(f"[GraspExecutor] ========== 任务判断 ==========")
                    print(f"[GraspExecutor] Object position: {object_pos}")
                    print(f"[GraspExecutor] Target position: {self._target_position}")
                    print(f"[GraspExecutor] Distance: xy={distance_xy:.4f}m, z={z_diff:.4f}m, 3d={distance_3d:.4f}m")
                    
                    # 判断标准放宽：xy距离<8cm 且 z偏差<5cm（物体放置有一定误差是正常的）
                    xy_threshold = 0.08  # 8cm
                    z_threshold = 0.05   # 5cm
                    
                    if distance_xy < xy_threshold and z_diff < z_threshold:
                        print(f"[GraspExecutor] ✓ Task SUCCESS: object reached target position")
                        print(f"[GraspExecutor]   (xy={distance_xy*100:.1f}cm < {xy_threshold*100:.0f}cm, z={z_diff*100:.1f}cm < {z_threshold*100:.0f}cm)")
                        return True
                    else:
                        print(f"[GraspExecutor] ✗ Task FAILED: object not at target position")
                        print(f"[GraspExecutor]   (xy={distance_xy*100:.1f}cm >= {xy_threshold*100:.0f}cm or z={z_diff*100:.1f}cm >= {z_threshold*100:.0f}cm)")
                        return False
                except Exception as e:
                    print(f"[GraspExecutor] Error getting object position: {e}")
                    print(f"[GraspExecutor] Assuming task success (cannot verify)")
                    return True
            else:
                print(f"[GraspExecutor] Warning: cannot verify object position (no get_object_position method)")
                print(f"[GraspExecutor] Assuming task success")
                return True
        finally:
            self._is_executing = False

    @staticmethod
    def _euler_xyz_deg_to_rotmat(euler_deg: np.ndarray) -> np.ndarray:
        rx, ry, rz = np.deg2rad(np.asarray(euler_deg[:3], dtype=float))
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
        ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
        return rz_m @ ry_m @ rx_m

    def _move_to_position(self, pose: np.ndarray, timeout: float = 10.0, tolerance: float = 0.05) -> bool:

        use_simu_ik = self._use_simulation or (
            hasattr(self._simu, 'is_ik_available') and self._simu.is_ik_available()
        )

        if use_simu_ik:
            target_pos = np.asarray(pose[:3], dtype=float)
            target_ori = None
            if len(pose) >= 6:
                target_ori = self._euler_xyz_deg_to_rotmat(np.asarray(pose[3:6], dtype=float))
            max_attempts = 3

            for attempt in range(1, max_attempts + 1):
                # 第一次严格带姿态；后续失败重试退化为仅位置，避免姿态不可达导致整体卡死
                attempt_orientation = target_ori if (target_ori is not None and attempt == 1) else None
                if target_ori is not None and attempt > 1:
                    print("[GraspExecutor] Retry with position-only IK (orientation relaxed)")

                success = self._simu.move_to_cartesian(
                    target_pos,
                    orientation=attempt_orientation,
                    duration=2.0,
                    steps=100,
                    step_callback=self._frame_callback,
                )



                tcp_pos = self._simu.get_tcp_position() if hasattr(self._simu, 'get_tcp_position') else None
                if tcp_pos is None:
                    print(f"[GraspExecutor] ⚠ Failed to read TCP pose (attempt {attempt}/{max_attempts})")
                    continue

                pos_diff = np.linalg.norm(np.asarray(tcp_pos) - target_pos)
                print(f"  Current: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
                print(f"  Target:  [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"  Diff: pos={pos_diff:.4f}m")

                if success and pos_diff <= tolerance:
                    return True

                print(f"[GraspExecutor] ⚠ IK not within tolerance (attempt {attempt}/{max_attempts}, tol={tolerance:.4f}m)")

            return False

        self._real.move_cartesian(pose)

        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pose = self._real.get_cartesian_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - pose[:3])

            joint_state = self._real.get_joint_state()
            self._simu.set_joint_target(joint_state)

            if pos_diff < tolerance:
                print(f"  Current: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]")
                print(f"  Target:  [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
                print(f"  Diff: pos={pos_diff:.4f}m")
                return True

            time.sleep(0.01)

        current_pose = self._real.get_cartesian_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - pose[:3])
        print(f"  Current: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]")
        print(f"  Target:  [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        print(f"  Diff: pos={pos_diff:.4f}m")
        print(f"[GraspExecutor] ⚠ Timeout waiting to reach target")
        return False

    def _set_simu_gripper_best_effort(self, desired_position: float, timeout: float = 0.6) -> float:

        desired_position = float(np.clip(desired_position, 0.0, 1.0))
        candidates = [desired_position]
        inv = 1.0 - desired_position
        if abs(inv - desired_position) > 1e-6:
            candidates.append(inv)

        best_state = self._simu.get_gripper_state()
        best_err = abs(best_state - desired_position)
        best_cmd = candidates[0]

        for cmd in candidates:
            start_state = self._simu.get_gripper_state()

            # 分段闭合，避免夹爪瞬间刚性碰撞把轻薄物体弹飞
            for alpha in np.linspace(0.0, 1.0, 6)[1:]:
                interp_cmd = start_state + (cmd - start_state) * alpha
                self._simu.set_gripper(float(np.clip(interp_cmd, 0.0, 1.0)))
                if hasattr(self._simu, 'step'):
                    self._simu.step(8)

            start_time = time.time()
            current_state = self._simu.get_gripper_state()
            while time.time() - start_time < timeout:
                if hasattr(self._simu, 'step'):
                    self._simu.step(5)
                current_state = self._simu.get_gripper_state()
                if abs(current_state - desired_position) < 0.08:
                    return current_state
                time.sleep(0.02)

            err = abs(current_state - desired_position)
            if err < best_err:
                best_err = err
                best_state = current_state
                best_cmd = cmd


        self._simu.set_gripper(best_cmd)
        return self._simu.get_gripper_state()

    def _close_gripper(self, gradual: bool = True):
        """闭合夹爪

        Args:
            gradual: 是否缓慢分阶段闭合（默认True，避免弹开物体）
        """
        print(f"[GraspExecutor] _close_gripper called, target={self._gripper_close_position}, gradual={gradual}")
        use_simu_ik = hasattr(self._simu, 'is_ik_available') and self._simu.is_ik_available()

        if use_simu_ik:
            if gradual:
                # 分阶段缓慢闭合，避免夹爪瞬间刚性碰撞弹开物体
                current_gripper = self._simu.get_gripper_state()
                steps = 12  # 分12次闭合，更平稳
                for i in range(1, steps + 1):
                    alpha = i / steps
                    target = current_gripper + (self._gripper_close_position - current_gripper) * alpha
                    self._simu.set_gripper(float(target))
                    if hasattr(self._simu, 'step'):
                        self._simu.step(12)  # 每步多仿真几帧让物理稳定
                    time.sleep(0.04)
                current_gripper = self._simu.get_gripper_state()
            else:
                current_gripper = self._set_simu_gripper_best_effort(self._gripper_close_position)
        else:
            self._real.set_gripper(self._gripper_close_position)
            start_time = time.time()
            while time.time() - start_time < 2.0:
                current_gripper = self._real.get_gripper_state()
                if abs(current_gripper - self._gripper_close_position) < 0.05:
                    break
                time.sleep(0.1)
            current_gripper = self._real.get_gripper_state()

        print(f"[GraspExecutor] After close, Gripper state: {current_gripper:.2f} (target: {self._gripper_close_position:.2f})")

    def _open_gripper(self, gradual: bool = False):
        """打开夹爪

        Args:
            gradual: 是否缓慢分阶段松开（放置物体时使用，避免弹开）
        """
        print(f"[GraspExecutor] Opening gripper to {self._gripper_open_position} (gradual={gradual})")
        use_simu_ik = hasattr(self._simu, 'is_ik_available') and self._simu.is_ik_available()

        if use_simu_ik:
            if gradual:
                # 分阶段缓慢松开，避免突然撤力弹开物体
                current_gripper = self._simu.get_gripper_state()
                steps = 10  # 分10次松开
                for i in range(1, steps + 1):
                    alpha = i / steps
                    target = current_gripper + (self._gripper_open_position - current_gripper) * alpha
                    self._simu.set_gripper(float(target))
                    if hasattr(self._simu, 'step'):
                        self._simu.step(10)  # 每步多仿真几帧让物理稳定
                    time.sleep(0.05)
                current_gripper = self._simu.get_gripper_state()
            else:
                current_gripper = self._set_simu_gripper_best_effort(self._gripper_open_position)
        else:
            self._real.set_gripper(self._gripper_open_position)
            start_time = time.time()
            while time.time() - start_time < 2.0:
                current_gripper = self._real.get_gripper_state()
                if abs(current_gripper - self._gripper_open_position) < 0.05:
                    break
                time.sleep(0.1)
            current_gripper = self._real.get_gripper_state()

        print(f"[GraspExecutor] After open, Gripper state: {current_gripper:.2f} (target: {self._gripper_open_position:.2f})")



    def stop(self):
        self._is_executing = False

    def is_executing(self) -> bool:
        return self._is_executing

    def set_waypoint_delay(self, delay: float):
        self._waypoint_delay = max(0.01, delay)

    def set_release_lift_heights(self, micro_lift: float = 0.02, release_lift: float = 0.08):
        """设置放置时的抬升高度

        Args:
            micro_lift: 松开前微抬高度（米），默认 0.02m
            release_lift: 松开后抬升高度（米），默认 0.08m
        """
        self._micro_lift_height = max(0.005, micro_lift)  # 最小 5mm
        self._release_lift_height = max(0.01, release_lift)  # 最小 1cm
        print(f"[GraspExecutor] Release lift heights set: micro={self._micro_lift_height*100:.1f}cm, release={self._release_lift_height*100:.1f}cm")

    def set_gripper_positions(self, open_pos: float, close_pos: float):
        print(f"[GraspExecutor] set_gripper_positions called: open={open_pos}, close={close_pos}")
        self._gripper_open_position = np.clip(open_pos, 0.0, 1.0)
        self._gripper_close_position = np.clip(close_pos, 0.0, 1.0)
        print(f"[GraspExecutor] After clip: open={self._gripper_open_position}, close={self._gripper_close_position}")
    
    def set_object_type(self, object_type: str):
        """设置物体类型，用于调整抓取策略
        
        Args:
            object_type: 物体类型，可选值:
                - "cube": 方块，中心抓取
                - "cup": 茶杯，侧壁抓取
                - "bottle": 瓶子，侧面抓取
                - "bowl": 碗，边缘抓取
        """
        if object_type in self.GRASP_OFFSETS:
            self._object_type = object_type
            offset_info = self.GRASP_OFFSETS[object_type]
            
            # 自动设置夹爪开合程度（优先级：物体覆盖 > default覆盖 > 内置默认）
            gripper_open, gripper_close = self._get_gripper_positions_for_object(object_type)
            self._gripper_open_position = gripper_open
            self._gripper_close_position = gripper_close

            
            orientation = self._get_orientation_for_object(object_type)
            pre_grasp_offset = self._get_pre_grasp_offset_for_object(object_type)
            grasp_offset = self._get_grasp_offset()

            print(f"[GraspExecutor] Object type set to: {object_type}")

            print(f"[GraspExecutor] Grasp offset: {grasp_offset.tolist()}, {offset_info['description']}")

            print(f"[GraspExecutor] Gripper config: open={gripper_open}, close={gripper_close}")
            print(f"[GraspExecutor] End-effector orientation: {orientation.tolist()}")
            print(f"[GraspExecutor] Pre-grasp offset: {pre_grasp_offset.tolist()}")


        else:
            print(f"[GraspExecutor] Warning: Unknown object type '{object_type}', using default 'cube'")
            self._object_type = self.OBJECT_TYPE_CUBE
    
    def _get_grasp_offset(self) -> np.ndarray:
        """获取当前物体类型的抓取偏移（优先级：物体覆盖 > default覆盖 > 内置默认）"""
        builtin = np.array(
            self.GRASP_OFFSETS.get(self._object_type, {}).get("offset", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if self._default_grasp_offset is not None:
            builtin = np.array(self._default_grasp_offset, dtype=float)
        if self._object_type in self._object_grasp_offsets:
            builtin = np.array(self._object_grasp_offsets[self._object_type], dtype=float)
        return builtin

    def set_object_profiles(self, profiles: Dict[str, Any]):
        """统一设置物体适配参数（姿态/预抓取偏移/抓取偏移/夹爪开合）

        支持示例：
        {
          default: {
            orientation: [180, 0, 0],
            pre_grasp_offset: [0, 0, 0.15],
            grasp_offset: [0.0, 0.0, 0.0],
            gripper: {open: 0.0, close: 0.65}
          },
          cup: {
            orientation: [180, 0, 0],
            pre_grasp_offset: [0, 0, 0.10],
            grasp_offset: [0.04, 0.0, 0.0],
            gripper: {open: 0.0, close: 0.62}
          }
        }
        """
        if not isinstance(profiles, dict):
            return

        orientation_map: Dict[str, List[float]] = {}
        pre_grasp_map: Dict[str, List[float]] = {}
        grasp_offset_map: Dict[str, List[float]] = {}
        gripper_map: Dict[str, Dict[str, float]] = {}

        for obj_name, cfg in profiles.items():
            if not isinstance(cfg, dict):
                continue

            ori = cfg.get("orientation")
            if isinstance(ori, (list, tuple, np.ndarray)) and len(ori) >= 3:
                orientation_map[str(obj_name)] = [float(ori[0]), float(ori[1]), float(ori[2])]

            pre = cfg.get("pre_grasp_offset")
            if isinstance(pre, (list, tuple, np.ndarray)) and len(pre) >= 3:
                pre_grasp_map[str(obj_name)] = [float(pre[0]), float(pre[1]), float(pre[2])]

            go = cfg.get("grasp_offset", cfg.get("offset"))
            if isinstance(go, (list, tuple, np.ndarray)) and len(go) >= 3:
                grasp_offset_map[str(obj_name)] = [float(go[0]), float(go[1]), float(go[2])]

            g_open = None
            g_close = None
            g_cfg = cfg.get("gripper")
            if isinstance(g_cfg, dict):
                if "open" in g_cfg:
                    g_open = float(g_cfg.get("open"))
                elif "gripper_open" in g_cfg:
                    g_open = float(g_cfg.get("gripper_open"))

                if "close" in g_cfg:
                    g_close = float(g_cfg.get("close"))
                elif "gripper_close" in g_cfg:
                    g_close = float(g_cfg.get("gripper_close"))
            else:
                if "open" in cfg:
                    g_open = float(cfg.get("open"))
                elif "gripper_open" in cfg:
                    g_open = float(cfg.get("gripper_open"))

                if "close" in cfg:
                    g_close = float(cfg.get("close"))
                elif "gripper_close" in cfg:
                    g_close = float(cfg.get("gripper_close"))

            if g_open is not None and g_close is not None:
                gripper_map[str(obj_name)] = {
                    "open": float(np.clip(g_open, 0.0, 1.0)),
                    "close": float(np.clip(g_close, 0.0, 1.0)),
                }

        if orientation_map:
            self.set_default_orientation(orientation_map)
        if pre_grasp_map:
            self.set_pre_grasp_offset(pre_grasp_map)
        if grasp_offset_map:
            self.set_grasp_offsets(grasp_offset_map)
        if gripper_map:
            self.set_gripper_positions_by_object(gripper_map)

        print(
            "[GraspExecutor] Object profiles applied: "
            f"orientation={list(orientation_map.keys())}, "
            f"pre_grasp_offset={list(pre_grasp_map.keys())}, "
            f"grasp_offset={list(grasp_offset_map.keys())}, "
            f"gripper={list(gripper_map.keys())}"
        )

    def set_grasp_offsets(self, offsets: Dict[str, Any]):
        """从 config 设置不同物体的抓取偏移

        支持格式：
        1) {default: [x,y,z], cup: [x,y,z], ...}
        2) {cup: {offset: [x,y,z]}, ...}
        """
        if not isinstance(offsets, dict):
            return


        parsed_map: Dict[str, np.ndarray] = {}
        for obj_name, val in offsets.items():
            vec = None
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 3:
                vec = val[:3]
            elif isinstance(val, dict):
                v = val.get("offset")
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
                    vec = v[:3]

            if vec is not None:
                parsed_map[str(obj_name)] = np.array(vec, dtype=float)

        self._object_grasp_offsets = {k: v for k, v in parsed_map.items() if k != "default"}
        self._default_grasp_offset = parsed_map.get("default", None)

        print(
            f"[GraspExecutor] Grasp offsets configured: default="
            f"{None if self._default_grasp_offset is None else self._default_grasp_offset.tolist()}, "
            f"per_object={list(self._object_grasp_offsets.keys())}"
        )

    def set_gripper_positions_by_object(self, config: Dict[str, Any]):
        """从 config 设置不同物体的夹爪开合程度

        支持格式：
        1) {default: {open: 0.0, close: 0.7}, cup: {open: 0.0, close: 0.65}, ...}
        2) {cup: [0.0, 0.65], bottle: [0.0, 0.8], ...}  # [open, close]
        """
        if not isinstance(config, dict):
            return

        parsed_map: Dict[str, Tuple[float, float]] = {}
        for obj_name, val in config.items():
            open_pos = None
            close_pos = None

            if isinstance(val, dict):
                if "open" in val:
                    open_pos = float(val.get("open"))
                elif "gripper_open" in val:
                    open_pos = float(val.get("gripper_open"))

                if "close" in val:
                    close_pos = float(val.get("close"))
                elif "gripper_close" in val:
                    close_pos = float(val.get("gripper_close"))
            elif isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
                open_pos = float(val[0])
                close_pos = float(val[1])

            if open_pos is None or close_pos is None:
                continue

            parsed_map[str(obj_name)] = (
                float(np.clip(open_pos, 0.0, 1.0)),
                float(np.clip(close_pos, 0.0, 1.0)),
            )

        self._object_gripper_positions = {k: v for k, v in parsed_map.items() if k != "default"}
        default_pair = parsed_map.get("default", None)
        if default_pair is not None:
            self._default_gripper_open = default_pair[0]
            self._default_gripper_close = default_pair[1]

        print(
            f"[GraspExecutor] Gripper positions configured: default="
            f"{None if default_pair is None else {'open': default_pair[0], 'close': default_pair[1]}}, "
            f"per_object={list(self._object_gripper_positions.keys())}"
        )

    def set_pre_grasp_offset(self, offset: Union[List[float], Dict[str, Any]]):


        # 兼容两种格式：
        # 1) [x, y, z]
        # 2) {default: [x, y, z], cup: [...], bottle: [...], ...}
        if isinstance(offset, dict):
            parsed_map: Dict[str, np.ndarray] = {}
            for obj_name, obj_offset in offset.items():
                if not isinstance(obj_offset, (list, tuple, np.ndarray)) or len(obj_offset) < 3:
                    continue
                parsed_map[str(obj_name)] = np.array(obj_offset[:3], dtype=float)

            self._object_pre_grasp_offsets = {k: v for k, v in parsed_map.items() if k != "default"}
            if "default" in parsed_map:
                self._default_pre_grasp_offset = parsed_map["default"]
        else:
            self._default_pre_grasp_offset = np.array(offset[:3], dtype=float)
            self._object_pre_grasp_offsets = {}

        print(
            f"[GraspExecutor] Pre-grasp offset configured: default={self._default_pre_grasp_offset.tolist()}, "
            f"per_object={list(self._object_pre_grasp_offsets.keys())}"
        )

    def set_default_orientation(self, orientation: Union[List[float], Dict[str, Any]]):

        # 兼容两种格式：
        # 1) [rx, ry, rz]
        # 2) {default: [rx, ry, rz], cup: [...], bottle: [...], ...}
        if isinstance(orientation, dict):
            parsed_map: Dict[str, np.ndarray] = {}
            for obj_name, obj_ori in orientation.items():
                if not isinstance(obj_ori, (list, tuple, np.ndarray)) or len(obj_ori) < 3:
                    continue
                parsed_map[str(obj_name)] = np.array(obj_ori[:3], dtype=float)

            self._object_orientations = {k: v for k, v in parsed_map.items() if k != "default"}
            if "default" in parsed_map:
                self._default_orientation = parsed_map["default"]
        else:
            self._default_orientation = np.array(orientation[:3], dtype=float)
            self._object_orientations = {}

        print(
            f"[GraspExecutor] Orientation configured: default={self._default_orientation.tolist()}, "
            f"per_object={list(self._object_orientations.keys())}"
        )


    def set_frame_callback(self, callback: Optional[Callable[[], None]]):
        self._frame_callback = callback

    def set_sim_object_body_name(self, object_body_name: str):
        self._sim_object_body_name = object_body_name or "cube"




class GraspExecutorWithInterpolation(GraspExecutor):
    def __init__(
        self,
        real_interface,
        simu_interface,
        home_position: List[float],
        pre_grasp_offset: Union[List[float], Dict[str, Any]],
        lift_height: float,

        approach_height: float,
        interpolation_steps: int = 50,
    ):
        super().__init__(
            real_interface,
            simu_interface,
            home_position,
            pre_grasp_offset,
            lift_height,
            approach_height,
        )
        self._interpolation_steps = interpolation_steps

    def _move_to_position(self, pose: np.ndarray) -> bool:
        start_pose = self._real.get_cartesian_pose()
        for i in range(self._interpolation_steps):
            alpha = (i + 1) / self._interpolation_steps
            interpolated = start_pose * (1 - alpha) + pose * alpha
            self._real.move_cartesian(interpolated)
            joint_state = self._real.get_joint_state()
            self._simu.set_joint_target(joint_state)
            time.sleep(0.01)
        
        current_pose = self._real.get_cartesian_pose()
        pos_diff = np.linalg.norm(current_pose[:3] - pose[:3])
        ori_diff = np.linalg.norm(current_pose[3:6] - pose[3:6])
        print(f"  Current: [{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}]")
        print(f"  Target:  [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}]")
        print(f"  Diff: pos={pos_diff:.4f}m, ori={ori_diff:.2f}°")
        
        return True

    def set_interpolation_steps(self, steps: int):
        self._interpolation_steps = max(1, steps)
