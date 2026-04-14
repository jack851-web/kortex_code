"""
真实机器人数据发布者

在独立线程中运行，以固定频率采集真实机器人状态并发布到 MessageBroker。
真实相机返回 BGR 格式（OpenCV 原生），在 topic 消息中标注为 BGR。
"""
import threading
import time
import numpy as np
from typing import Optional, Dict

from scripts.core.message_bus import MessageBroker
from scripts.core.topic_defs import (
    REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER, REAL_STATUS,
    ALL_REAL_TOPICS,
)


class RealPublisher:
    """真实机器人数据发布者

    在独立线程中运行，以固定频率采集真实机器人状态并发布到 MessageBroker。
    """

    def __init__(self, real_interface, broker: MessageBroker, fps: int = 20):
        self._real = real_interface
        self._broker = broker
        self._fps = fps
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 注册所有真实机器人话题
        for topic_name in ALL_REAL_TOPICS:
            self._broker.create_topic(topic_name)

        # 统计
        self._publish_count = 0
        self._error_count = 0

    def start(self):
        """启动发布线程"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._publish_loop, daemon=True, name="RealPublisher")
        self._thread.start()
        print(f"[RealPublisher] Started (fps={self._fps})")

    def stop(self):
        """停止发布线程"""
        if not self._running:
            return
        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                print("[RealPublisher] WARNING: thread did not stop within timeout")
            self._thread = None
        print("[RealPublisher] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def _publish_loop(self):
        """发布循环"""
        interval = 1.0 / self._fps if self._fps > 0 else 0.05

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
                self._broker.publish(REAL_STATUS, f"error: {e}")
                if self._error_count <= 3:
                    print(f"[RealPublisher] Error in publish loop: {e}")
                self._stop_event.wait(0.1)

    def _publish_once(self):
        """采集一帧真实机器人数据并发布"""
        real = self._real
        if real is None:
            self._broker.publish(REAL_STATUS, "disconnected")
            return

        try:
            # 真实机器人各接口已经自带线程安全，不需要加锁

            # 相机图像（BGR 格式）
            images = real.get_camera_images()
            self._broker.publish(REAL_IMAGES, images)  # BGR 格式

            # 关节状态（度）
            try:
                joints = real.get_joint_state()
                self._broker.publish(REAL_JOINTS, joints)
            except Exception:
                pass

            # 笛卡尔位姿
            try:
                cartesian = real.get_cartesian_pose()
                self._broker.publish(REAL_CARTESIAN, cartesian)
            except Exception:
                pass

            # 夹爪状态
            try:
                gripper = real.get_gripper_state()
                self._broker.publish(REAL_GRIPPER, gripper)
            except Exception:
                pass

            self._broker.publish(REAL_STATUS, "connected")

        except Exception as e:
            self._broker.publish(REAL_STATUS, f"error: {e}")
            raise
