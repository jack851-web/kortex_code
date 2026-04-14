import threading
import time
import numpy as np
from typing import Optional, Callable


class SyncController:
    def __init__(self, real, simu, on_sync_callback: Callable = None):
        self._real = real
        self._simu = simu
        self._sync_thread = None
        self._stop_event = threading.Event()
        self._is_syncing = False
        self._sync_interval = 0.01
        self._last_real_joints = np.zeros(6)
        self._last_simu_joints = np.zeros(6)
        self._sync_count = 0
        self._on_sync_callback = on_sync_callback

        self._adhesion_enabled = False
        self._adhesion_attached = False
        self._adhesion_object_body_name = "cube"
        self._adhesion_object_pos = np.zeros(3, dtype=float)
        self._adhesion_plate_pos = np.zeros(3, dtype=float)
        self._attach_distance = 0.06
        self._detach_distance = 0.05
        self._gripper_close_threshold = 0.3
        self._last_adhesion_state = False
        self._adhesion_log_cooldown = 0
        self._detach_check_counter = 0

    def start_sync(self):
        if self._is_syncing:
            return
        self._stop_event.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        self._is_syncing = True

    def stop_sync(self):
        if not self._is_syncing:
            return
        self._stop_event.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=2.0)
        self._is_syncing = False

    def _sync_loop(self):
        debug_counter = 0
        while not self._stop_event.is_set():
            try:
                real_joints = self._real.get_joint_state()
                self._simu.set_joint_target(real_joints)
                
                real_gripper = self._real.get_gripper_state()
                self._simu.set_gripper(real_gripper)
                
                self._simu.step(n_steps=1000)
                
                self._update_adhesion(real_gripper)

                if hasattr(self._simu, '_render_process') and self._simu._render_process is not None:
                    self._simu._render_process.update_joints(real_joints)
                    self._simu._render_process.update_gripper(real_gripper)
                    active_body = self._simu.get_active_object_body_name() if hasattr(self._simu, 'get_active_object_body_name') else self._adhesion_object_body_name
                    obj_pos = self._simu.get_object_position(active_body)
                    self._simu._render_process.update_object_position(obj_pos)

                
                self._last_real_joints = real_joints.copy()
                simu_joints = self._simu.get_joint_state()
                self._last_simu_joints = simu_joints.copy()
                self._sync_count += 1
                
                debug_counter += 1
                if debug_counter % 50 == 0:
                    print(f"[SyncController] === Debug {debug_counter} ===")
                    print(f"[SyncController] Real joints: {real_joints}")
                    print(f"[SyncController] Simu joints (rad): {simu_joints}")
                    print(f"[SyncController] Simu joints (deg): {np.rad2deg(simu_joints)}")
                    print(f"[SyncController] Ctrl values: {self._simu._data.ctrl[:6]}")
                    print(f"[SyncController] Diff (deg): {np.abs(real_joints - np.rad2deg(simu_joints))}")
                    print(f"[SyncController] Adhesion: enabled={self._adhesion_enabled}, attached={self._adhesion_attached}")
                
                if self._on_sync_callback is not None:
                    try:
                        self._on_sync_callback()
                    except Exception as e:
                        print(f"[SyncController] Callback error: {e}")
                        
            except Exception as e:
                print(f"[SyncController] Sync error: {e}")
            time.sleep(self._sync_interval)

    def is_synced(self) -> bool:
        return self._is_syncing

    def get_sync_count(self) -> int:
        return self._sync_count

    def get_last_real_joints(self) -> np.ndarray:
        return self._last_real_joints.copy()

    def get_last_simu_joints(self) -> np.ndarray:
        return self._last_simu_joints.copy()

    def set_sync_interval(self, interval: float):
        self._sync_interval = max(0.001, interval)

    def set_callback(self, callback: Callable):
        self._on_sync_callback = callback

    def set_adhesion_targets(
        self,
        object_body_name: str,
        object_pos: np.ndarray,
        plate_pos: np.ndarray,
        attach_distance: float = 0.06,
        detach_distance: float = 0.05,
        gripper_close_threshold: float = 0.3,
    ):
        self._adhesion_object_body_name = object_body_name or "cube"
        self._adhesion_object_pos = np.asarray(object_pos[:3], dtype=float)
        self._adhesion_plate_pos = np.asarray(plate_pos[:3], dtype=float)
        self._attach_distance = float(max(0.005, attach_distance))
        self._detach_distance = float(max(0.005, detach_distance))
        self._gripper_close_threshold = float(np.clip(gripper_close_threshold, 0.0, 1.0))
        self._adhesion_enabled = True
        self._adhesion_attached = False
        self._last_adhesion_state = False
        print(f"[SyncController] 吸附目标已设置: object={object_body_name}, attach_dist={self._attach_distance}m, detach_dist={self._detach_distance}m")

    def clear_adhesion_targets(self):
        self._adhesion_enabled = False
        self._adhesion_attached = False
        self._last_adhesion_state = False
        print("[SyncController] 吸附目标已清除")

    def is_adhesion_attached(self) -> bool:
        return self._adhesion_attached

    def _update_adhesion(self, gripper_state: float):
        if not self._adhesion_enabled:
            return
        if not hasattr(self._simu, 'get_tcp_position'):
            return

        tcp_pos = self._simu.get_tcp_position()
        if tcp_pos is None:
            return
        tcp_pos = np.asarray(tcp_pos[:3], dtype=float)

        body_name = self._adhesion_object_body_name
        obj_pos = np.asarray(self._simu.get_object_position(body_name), dtype=float)

        gripper_is_closed = gripper_state > self._gripper_close_threshold

        if not self._adhesion_attached:
            dist_to_object = np.linalg.norm(tcp_pos - obj_pos)
            if dist_to_object <= self._attach_distance and gripper_is_closed:
                self._adhesion_attached = True
                self._last_adhesion_state = True
                print(f"[SyncController] 物体已吸附: body={body_name}, dist={dist_to_object:.4f}m, gripper={gripper_state:.3f}")

        if self._adhesion_attached:
            if gripper_is_closed:
                self._simu.set_object_position(body_name, tcp_pos, reset_z=False)
            else:
                self._detach_check_counter += 1
                if self._detach_check_counter >= 10:
                    self._detach_check_counter = 0
                    dist_to_plate = np.linalg.norm(obj_pos - self._adhesion_plate_pos)
                    if dist_to_plate <= self._detach_distance:
                        self._adhesion_attached = False
                        self._last_adhesion_state = False
                        self._simu.set_object_position(body_name, self._adhesion_plate_pos, reset_z=True)
                        print(f"[SyncController] 物体已脱附并放置: body={body_name}, dist_to_plate={dist_to_plate:.4f}m")
                    else:
                        self._adhesion_attached = False
                        self._last_adhesion_state = False
                        print(f"[SyncController] 物体已脱附(夹爪松开): body={body_name}, gripper={gripper_state:.3f}")

        self._adhesion_log_cooldown = max(0, self._adhesion_log_cooldown - 1)


class SyncControllerWithInterpolation(SyncController):

    def __init__(self, real, simu, interpolation_steps: int = 10, on_sync_callback: Callable = None):
        super().__init__(real, simu, on_sync_callback)
        self._interpolation_steps = interpolation_steps
        self._target_joints = np.zeros(6)
        self._current_joints = np.zeros(6)
        self._interpolation_count = 0

    def _sync_loop(self):
        while not self._stop_event.is_set():
            try:
                real_joints = self._real.get_joint_state()
                self._target_joints = real_joints.copy()
                for i in range(self._interpolation_steps):
                    if self._stop_event.is_set():
                        break
                    alpha = i / self._interpolation_steps
                    interpolated = self._current_joints * (1 - alpha) + self._target_joints * alpha
                    self._simu.set_joint_target(interpolated)
                    self._simu.step()
                    self._simu.update_viewer()
                    self._current_joints = interpolated.copy()
                    time.sleep(self._sync_interval / self._interpolation_steps)
                
                self._last_real_joints = real_joints.copy()
                simu_joints = self._simu.get_joint_state()
                self._last_simu_joints = simu_joints.copy()
                self._sync_count += 1
                
                if self._on_sync_callback is not None:
                    try:
                        self._on_sync_callback()
                    except Exception as e:
                        print(f"[SyncController] Callback error: {e}")
                        
            except Exception as e:
                print(f"[SyncController] Sync error: {e}")

    def set_interpolation_steps(self, steps: int):
        self._interpolation_steps = max(1, steps)
