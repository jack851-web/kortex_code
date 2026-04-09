from __future__ import annotations

# Apply protobuf compatibility fix at the very beginning
import collections
if not hasattr(collections, 'MutableSequence'):
    import collections.abc
    collections.MutableSequence = collections.abc.MutableSequence

# Fix other potential missing attributes for complete compatibility
missing_attrs = [
    'MutableMapping', 'MutableSet', 'MutableSequence',
    'Mapping', 'Set', 'Sequence', 'ByteString'
]

for attr in missing_attrs:
    if not hasattr(collections, attr):
        import collections.abc
        setattr(collections, attr, getattr(collections.abc, attr))

import collections
if not hasattr(collections, 'MutableSequence'):
    import collections.abc
    collections.MutableSequence = collections.abc.MutableSequence

missing_attrs = [
    'MutableMapping', 'MutableSet', 'MutableSequence',
    'Mapping', 'Set', 'Sequence', 'ByteString'
]

for attr in missing_attrs:
    if not hasattr(collections, attr):
        import collections.abc
        setattr(collections, attr, getattr(collections.abc, attr))

import atexit
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from omegaconf import DictConfig
from torch import Tensor

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.robots.robot import Robot
from lerobot.processor import RobotAction, RobotObservation
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.Exceptions.KServerException import KServerException

from .config_gen3_lite import Gen3LiteConfig

JOINT_LIMITS = {
    "joint_1": {"min": 0.0, "max": 360.0},
    "joint_2": {"min": 0.0, "max": 360.0},
    "joint_3": {"min": 0.0, "max": 360.0},
    "joint_4": {"min": 0.0, "max": 360.0},
    "joint_5": {"min": 0.0, "max": 360.0},
    "joint_6": {"min": 0.0, "max": 360.0},
}

GRIPPER_LIMITS = {"min": 0.0, "max": 1.0}

HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class Gen3Lite(Robot):
    """Gen3Lite机械臂实现"""
    
    config_class = Gen3LiteConfig
    name = "gen3lite"
    
    def __init__(self, config: Gen3LiteConfig):
        super().__init__(config)
        self.config = config
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        self.base_cyclic = None
        self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_names = [
            "joint_1", "joint_2", "joint_3", 
            "joint_4", "joint_5", "joint_6"
        ]
        self.action_timeout = 20  # 动作超时时间（秒）
    
    @property
    def observation_features(self) -> Dict[str, Any]:
        features = {f"{joint}.pos": float for joint in self.joint_names}
        features.update({f"{joint}.vel": float for joint in self.joint_names})
        features.update({
            "ee.x": float, "ee.y": float, "ee.z": float,
            "ee.wx": float, "ee.wy": float, "ee.wz": float
        })
        if self.config.gripper_enabled:
            # 2-Finger 夹爪：两个独立手指 + 兼容的平均值
            features["gripper.pos"] = float
            features["gripper.finger_1.pos"] = float
            features["gripper.finger_2.pos"] = float
        # 添加相机特征
        for cam in self.cameras:
            features[cam] = (self.config.cameras[cam].height, 
                           self.config.cameras[cam].width, 3)
        return features
    
    @property
    def action_features(self) -> Dict[str, Any]:
        if self.config.control_mode == "joint":
            features = {f"{joint}.pos": float for joint in self.joint_names}
            # 2-Finger 夹爪：支持两种模式
            if self.config.gripper_enabled:
                features["gripper.pos"] = float  # 兼容模式：同时控制两个手指
                features["gripper.finger_1.pos"] = float  # 独立控制手指1
                features["gripper.finger_2.pos"] = float  # 独立控制手指2
            return features
        else:  # cartesian
            features = {
                "ee.x": float, "ee.y": float, "ee.z": float,
                "ee.wx": float, "ee.wy": float, "ee.wz": float
            }
            if self.config.gripper_enabled:
                features["gripper.pos"] = float
                features["gripper.finger_1.pos"] = float
                features["gripper.finger_2.pos"] = float
            return features
    
    @property
    def is_connected(self) -> bool:
        return self.router is not None and all(cam.is_connected for cam in self.cameras.values())
    
    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        # 创建连接
        self.transport = TCPTransport()
        self.error_callback = lambda e: print('[--ERROR--] {}'.format(e))
        self.router = RouterClient(self.transport, self.error_callback)
        self.transport.connect(self.config.ip_address, 10000)

        # 创建会话
        self.session_info = Session_pb2.CreateSessionInfo()
        self.session_info.username = self.config.username
        self.session_info.password = self.config.password
        self.session_info.session_inactivity_timeout = 60000
        self.session_info.connection_inactivity_timeout = 2000
        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(self.session_info)

        # 创建服务
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)
        
        # 连接相机
        for cam in self.cameras.values():
            cam.connect()
        
        # 配置机器人
        self.configure()
        
        print(f"{self} connected.")
    
    @property
    def is_calibrated(self) -> bool:
        # Gen3Lite机械臂通常在出厂时已经校准，所以这里返回True
        return True
    
    def calibrate(self) -> None:
        # Gen3Lite机械臂通常不需要用户校准，所以这里是一个空操作
        pass
    
    def configure(self) -> None:
        # 设置控制模式为高级模式
        self.set_servoing_mode('high')
        
        # 可以在这里添加其他配置，如速度限制等
    
    def set_servoing_mode(self, mode='high'):
        """设置伺服模式
        
        Args:
            mode: 'high' 或 'low'，分别对应高级和低级伺服模式
        """
        servoing_mode_dict = {
            'high': Base_pb2.SINGLE_LEVEL_SERVOING,
            'low': Base_pb2.LOW_LEVEL_SERVOING
        }
        servoing_mode_info = Base_pb2.ServoingModeInformation()
        servoing_mode_info.servoing_mode = servoing_mode_dict[mode]
        self.base.SetServoingMode(servoing_mode_info)
    
    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs = {}
        
        # 获取关节状态
        try:
            joint_angles = self.base.GetMeasuredJointAngles()
            for i, joint_angle in enumerate(joint_angles.joint_angles):
                joint_name = self.joint_names[i]
                obs[f"{joint_name}.pos"] = joint_angle.value
        except KServerException as ex:
            print(f"Error getting joint angles: {ex}")
        
        # 获取笛卡尔空间状态
        try:
            feedback = self.base_cyclic.RefreshFeedback()
            obs["ee.x"] = feedback.base.tool_pose_x
            obs["ee.y"] = feedback.base.tool_pose_y
            obs["ee.z"] = feedback.base.tool_pose_z
            obs["ee.wx"] = feedback.base.tool_pose_theta_x
            obs["ee.wy"] = feedback.base.tool_pose_theta_y
            obs["ee.wz"] = feedback.base.tool_pose_theta_z
        except KServerException as ex:
            print(f"Error getting cartesian pose: {ex}")
        
        # 获取夹爪状态 (2-Finger 夹爪有两个手指)
        if self.config.gripper_enabled:
            try:
                grip_request = Base_pb2.GripperRequest()
                grip_request.mode = Base_pb2.GRIPPER_POSITION
                grip_measure = self.base.GetMeasuredGripperMovement(grip_request)
                if len(grip_measure.finger) >= 2:
                    # 两个手指的位置
                    obs["gripper.finger_1.pos"] = grip_measure.finger[0].value
                    obs["gripper.finger_2.pos"] = grip_measure.finger[1].value
                    # 兼容：同时也提供平均值
                    obs["gripper.pos"] = (grip_measure.finger[0].value + grip_measure.finger[1].value) / 2.0
                elif len(grip_measure.finger) == 1:
                    obs["gripper.pos"] = grip_measure.finger[0].value
                    obs["gripper.finger_1.pos"] = grip_measure.finger[0].value
                    obs["gripper.finger_2.pos"] = grip_measure.finger[0].value
            except KServerException as ex:
                print(f"Error getting gripper position: {ex}")
        
        # 获取相机图像
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        
        return obs
    
    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.config.control_mode == "joint":
            joint_angles = []
            for joint_name, value in action.items():
                if joint_name.endswith(".pos"):
                    joint_angles.append(value)
            
            if joint_angles:
                if not self._check_joint_limits(joint_angles):
                    raise ValueError("Joint angles exceed hardware limits")
                self.arm_move_angular(joint_angles)
        else:
            cartesian_pose = [
                action.get("ee.x", 0.0),
                action.get("ee.y", 0.0),
                action.get("ee.z", 0.0),
                action.get("ee.wx", 0.0),
                action.get("ee.wy", 0.0),
                action.get("ee.wz", 0.0)
            ]
            self.arm_move_cartesian(cartesian_pose)
        
        if self.config.gripper_enabled and "gripper.pos" in action:
            gripper_pos = action["gripper.pos"]
            if not self._check_gripper_limits(gripper_pos):
                raise ValueError("Gripper position exceeds hardware limits")
            self.gripper_move_position(gripper_pos)
        elif self.config.gripper_enabled and ("gripper.finger_1.pos" in action or "gripper.finger_2.pos" in action):
            # 支持分别控制两个手指
            finger1_pos = action.get("gripper.finger_1.pos", 0.0)
            finger2_pos = action.get("gripper.finger_2.pos", 0.0)
            if not (self._check_gripper_limits(finger1_pos) and self._check_gripper_limits(finger2_pos)):
                raise ValueError("Gripper position exceeds hardware limits")
            self.gripper_move_individual(finger1_pos, finger2_pos)
        
        return action
    
    def step(self, action: RobotAction) -> tuple[RobotObservation, dict]:
        """执行一步动作并返回观测和信息
        
        Args:
            action: 要执行的动作
            
        Returns:
            tuple: (observation, info) - 观测数据和信息字典
        """
        self.send_action(action)
        obs = self.get_observation()
        info = {"success": True}
        return obs, info
    
    def reset(self) -> RobotObservation:
        """重置到初始状态
        
        Returns:
            RobotObservation: 重置后的观测数据
        """
        self.arm_move_to_home()
        return self.get_observation()
    
    def arm_move_to_home(self) -> bool:
        """移动到home位置
        
        Returns:
            bool: 动作是否成功完成
        """
        return self.arm_move_angular(HOME_POSITION, "home")
    
    def arm_move_cartesian(self, pose, action_name='default'):
        """笛卡尔空间移动
        
        Args:
            pose: 笛卡尔空间位姿 [x, y, z, theta_x, theta_y, theta_z]
            action_name: 动作名称
        
        Returns:
            bool: 动作是否成功完成
        """
        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ''
        target_pose = action.reach_pose.target_pose
        target_pose.x = pose[0]
        target_pose.y = pose[1]
        target_pose.z = pose[2]
        target_pose.theta_x = pose[3]
        target_pose.theta_y = pose[4]
        target_pose.theta_z = pose[5]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self._action_notification_callback(e),
            Base_pb2.NotificationOptions()
        )

        try:
            self.base.ExecuteAction(action)
            finished = e.wait(self.action_timeout)
            if not finished:
                print(f"Warning: Action '{action_name}' timeout")
            return finished
        except KServerException as ex:
            print(f"Error executing cartesian action: {ex}")
            self.arm_stop()
            return False
        finally:
            self.base.Unsubscribe(notification_handle)

    def arm_move_angular(self, jointangles, action_name='default'):
        """关节空间移动
        
        Args:
            jointangles: 关节角度列表 [joint1, joint2, joint3, joint4, joint5, joint6]
            action_name: 动作名称
        
        Returns:
            bool: 动作是否成功完成
        """
        if not self._check_joint_limits(jointangles):
            raise ValueError("Joint angles exceed hardware limits")
        
        action = Base_pb2.Action()
        action.name = action_name
        action.application_data = ""

        actuator_count = self.base.GetActuatorCount()

        for joint_id in range(min(len(jointangles), actuator_count.count)):
            joint_angle = action.reach_joint_angles.joint_angles.joint_angles.add()
            joint_angle.joint_identifier = joint_id
            joint_angle.value = jointangles[joint_id]

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(
            self._action_notification_callback(e),
            Base_pb2.NotificationOptions()
        )
        
        try:
            self.base.ExecuteAction(action)
            finished = e.wait(self.action_timeout)
            if not finished:
                print(f"Warning: Action '{action_name}' timeout")
            return finished
        except KServerException as ex:
            print(f"Error executing action: {ex}")
            self.arm_stop()
            return False
        finally:
            self.base.Unsubscribe(notification_handle)
    
    def _check_joint_limits(self, jointangles: list) -> bool:
        """检查关节角度是否在安全限位内
        
        Args:
            jointangles: 关节角度列表
            
        Returns:
            bool: 是否在限位内
        """
        for i, angle in enumerate(jointangles):
            if i >= len(self.joint_names):
                break
            joint_name = self.joint_names[i]
            limits = JOINT_LIMITS.get(joint_name, {"min": 0.0, "max": 360.0})
            if not (limits["min"] <= angle <= limits["max"]):
                print(f"Warning: {joint_name} angle {angle:.4f} exceeds limits [{limits['min']}, {limits['max']}]")
                return False
        return True
    
    def _check_gripper_limits(self, position: float) -> bool:
        """检查夹爪位置是否在安全限位内
        
        Args:
            position: 夹爪位置
            
        Returns:
            bool: 是否在限位内
        """
        if not (GRIPPER_LIMITS["min"] <= position <= GRIPPER_LIMITS["max"]):
            print(f"Warning: Gripper position {position:.4f} exceeds limits [{GRIPPER_LIMITS['min']}, {GRIPPER_LIMITS['max']}]")
            return False
        return True
    
    def _action_notification_callback(self, event: threading.Event):
        """动作通知回调函数
        
        Args:
            event: 线程事件对象
        """
        def callback(notification):
            print(f"[--EVENT--] {Base_pb2.ActionEvent.Name(notification.action_event)}")
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                event.set()
        return callback

    def arm_move_angular_speed(self, speeds):
        """关节速度控制
        
        Args:
            speeds: 关节速度列表 [speed1, speed2, speed3, speed4, speed5, speed6]
        
        Returns:
            bool: 命令是否成功发送
        """
        joint_speeds = Base_pb2.JointSpeeds()

        for i, speed in enumerate(speeds):
            joint_speed = joint_speeds.joint_speeds.add()
            joint_speed.joint_identifier = i 
            joint_speed.value = speed
            joint_speed.duration = 0
        
        self.base.SendJointSpeedsCommand(joint_speeds)
        return True
    
    def arm_stop(self):
        """停止机械臂运动"""
        self.base.Stop()

    def clear_faults(self):
        """清除机器人故障状态"""
        try:
            self.base.Stop()
            time.sleep(0.5)
            self.base.ClearFaults()
            time.sleep(1.0)
            print("Faults cleared successfully")
            return True
        except Exception as e:
            print(f"Failed to clear faults: {e}")
            return False

    def gripper_move_position(self, position, finger_id: int = None):
        """控制夹爪位置 (2-Finger 夹爪)
        
        Args:
            position: 夹爪位置，范围通常为 0.0-1.0
            finger_id: 手指ID，1=手指1, 2=手指2, None=两个手指同时
        """
        grip_command = Base_pb2.GripperCommand()
        grip_command.mode = Base_pb2.GRIPPER_POSITION
        
        if finger_id is not None:
            # 控制单个手指
            finger = grip_command.gripper.finger.add()
            finger.finger_identifier = finger_id
            finger.value = position
        else:
            # 控制两个手指 (同时)
            for fid in [1, 2]:
                finger = grip_command.gripper.finger.add()
                finger.finger_identifier = fid
                finger.value = position
        
        self.base.SendGripperCommand(grip_command)
    
    def gripper_move_individual(self, finger1_pos: float, finger2_pos: float):
        """分别控制两个手指的位置
        
        Args:
            finger1_pos: 手指1位置 (0.0-1.0)
            finger2_pos: 手指2位置 (0.0-1.0)
        """
        grip_command = Base_pb2.GripperCommand()
        grip_command.mode = Base_pb2.GRIPPER_POSITION
        
        finger1 = grip_command.gripper.finger.add()
        finger1.finger_identifier = 1
        finger1.value = finger1_pos
        
        finger2 = grip_command.gripper.finger.add()
        finger2.finger_identifier = 2
        finger2.value = finger2_pos
        
        self.base.SendGripperCommand(grip_command)

    def gripper_move_speed(self, speed):
        """控制夹爪速度
        
        Args:
            speed: 夹爪速度
        """
        grip_command = Base_pb2.GripperCommand()
        finger = grip_command.gripper.finger.add()
        grip_command.mode = Base_pb2.GRIPPER_SPEED
        finger.finger_identifier = 1
        finger.value = speed
        self.base.SendGripperCommand(grip_command)
    
    @check_if_not_connected
    def disconnect(self) -> None:
        if self.session_manager:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000 
            self.session_manager.CloseSession(router_options)
        
        if self.transport:
            self.transport.disconnect()
        
        self.router = None
        self.base = None
        self.base_cyclic = None
        self.session_manager = None
        self.transport = None
        
        for cam in self.cameras.values():
            cam.disconnect()
        
        print(f"{self} disconnected.")