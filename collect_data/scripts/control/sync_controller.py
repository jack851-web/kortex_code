import threading
import time
import logging
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


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
        consecutive_errors = 0
        max_consecutive_errors = 50
        while not self._stop_event.is_set():
            try:
                real_joints = self._real.get_joint_state()
                self._simu.set_joint_target(real_joints)

                real_gripper = self._real.get_gripper_state()
                self._simu.set_gripper(real_gripper)

                self._simu.step(n_steps=1)

                active_body = self._simu.get_active_object_body_name() if hasattr(self._simu, 'get_active_object_body_name') else "cube"
                obj_pos = self._simu.get_object_position(active_body)
                self._simu.update_render_state(real_joints, real_gripper, obj_pos)

                self._last_real_joints = real_joints.copy()
                simu_joints = self._simu.get_joint_state()
                self._last_simu_joints = simu_joints.copy()
                self._sync_count += 1
                consecutive_errors = 0

                debug_counter += 1
                if debug_counter % 50 == 0:
                    logger.info(f"=== Debug {debug_counter} ===")
                    logger.info(f"Real joints (deg): {real_joints}")
                    logger.info(f"Simu joints (deg): {simu_joints}")
                    logger.info(f"Diff (deg): {np.abs(real_joints - simu_joints)}")

                if self._on_sync_callback is not None:
                    try:
                        self._on_sync_callback()
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Sync error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"连续 {max_consecutive_errors} 次同步错误，暂停同步")
                    self._is_syncing = False
                    break
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
                    self._simu.sync_viewer()
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
                        logger.error(f"Callback error: {e}")

            except Exception as e:
                logger.error(f"Sync error: {e}")

    def set_interpolation_steps(self, steps: int):
        self._interpolation_steps = max(1, steps)
