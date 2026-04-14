"""
话题名称定义 - 标准化的消息接口

遵循类 ROS2 命名规范:
  /命名空间/数据类型
"""
# ================================================================
# 仿真 → GUI / DataCollector (发布者: SimuPublisher)
# ================================================================
SIMU_IMAGES = "/simu/images"              # Dict[str, np.ndarray], RGB 格式
SIMU_JOINTS = "/simu/joints"              # np.ndarray, 6 joints (rad)
SIMU_GRIPPER = "/simu/gripper"            # float, 0~1
SIMU_TCP_POSE = "/simu/tcp_pose"          # Tuple[np.ndarray, np.ndarray]: (pos_xyz, rotmat_3x3)
SIMU_OBJECT_POS = "/simu/object_position"  # np.ndarray, xyz
SIMU_STATUS = "/simu/status"              # str: "idle" / "running" / "error:..."
SIMU_CARTESIAN = "/simu/cartesian"        # np.ndarray, 6: [x,y,z,wx,wy,wz]

# ================================================================
# 真实机器人 → GUI / DataCollector (发布者: RealPublisher)
# ================================================================
REAL_IMAGES = "/real/images"               # Dict[str, np.ndarray], BGR 格式
REAL_JOINTS = "/real/joints"              # np.ndarray, 6 joints (deg)
REAL_CARTESIAN = "/real/cartesian"        # np.ndarray, 6: [x,y,z,wx,wy,wz]
REAL_GRIPPER = "/real/gripper"            # float, 0~1
REAL_STATUS = "/real/status"              # str: "connected" / "disconnected" / "error:..."

# ================================================================
# GUI → 仿真 (发布者: Qt MainWindow)
# ================================================================
GUI_COMMAND = "/gui/command"              # str: "start_task" / "stop_task" / "pause" / "resume"
GUI_JOINT_TARGET = "/gui/joint_target"    # np.ndarray, 6 joints (deg)
GUI_GRIPPER_TARGET = "/gui/gripper_target"  # float, 0~1
GUI_CARTESIAN_TARGET = "/gui/cartesian_target"  # np.ndarray, pos+euler
GUI_SCENE_CONFIG = "/gui/scene_config"    # dict: {xml_path, object_xml, object_body_name, ...}

# ================================================================
# 数据采集 → GUI
# ================================================================
COLLECT_STATUS = "/collect/status"         # str: "recording" / "idle" / "paused"
COLLECT_TASK = "/collect/task"             # dict: {task_id, task_name, current, total}
COLLECT_LOG = "/collect/log"              # Tuple[str, str]: (message, level)
COLLECT_EPISODE = "/collect/episode"      # dict: {episode_id, frame_count, ...}

# ================================================================
# 所有话题列表（用于批量注册）
# ================================================================
ALL_SIMU_TOPICS = [
    SIMU_IMAGES, SIMU_JOINTS, SIMU_GRIPPER,
    SIMU_TCP_POSE, SIMU_OBJECT_POS, SIMU_STATUS, SIMU_CARTESIAN,
]

ALL_REAL_TOPICS = [
    REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER, REAL_STATUS,
]

ALL_COLLECT_TOPICS = [
    COLLECT_STATUS, COLLECT_TASK, COLLECT_LOG, COLLECT_EPISODE,
]
