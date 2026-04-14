"""
scripts 模块 - 数据收集系统核心组件

模块结构:
- core:      核心基础设施（消息总线、话题定义）
- real:      真实机器人相关模块
- simu:      仿真相关模块
- control:   控制器模块
- _legacy:   已弃用模块（向后兼容）
"""

# 核心模块
from scripts.core import (
    MessageBroker, Topic,
    SIMU_IMAGES, SIMU_JOINTS, SIMU_GRIPPER, SIMU_TCP_POSE,
    SIMU_OBJECT_POS, SIMU_STATUS, SIMU_CARTESIAN, ALL_SIMU_TOPICS,
    REAL_IMAGES, REAL_JOINTS, REAL_CARTESIAN, REAL_GRIPPER, REAL_STATUS,
    ALL_REAL_TOPICS,
    COLLECT_STATUS, COLLECT_TASK, COLLECT_LOG, COLLECT_EPISODE,
    ALL_COLLECT_TOPICS,
)

# 真实机器人模块
from scripts.real import (
    RealInterface, MockRealInterface, RobotNotConnectedError,
    RealPublisher, RealDataCollector,
    SimpleCamera, CameraManager,
)

# 仿真模块
from scripts.simu import (
    SimuInterface, MockSimuInterface,
    SimuPublisher, SimuDataCollector,
    SimuManager, SimuRenderProcess,
)

# 控制模块
from scripts.control import (
    SyncController, SyncControllerWithInterpolation,
    GraspExecutor, GraspExecutorWithInterpolation,
)

# 旧版兼容
from scripts._legacy import DataCollector

__all__ = [
    # Core
    'MessageBroker', 'Topic',
    'SIMU_IMAGES', 'SIMU_JOINTS', 'SIMU_GRIPPER', 'SIMU_TCP_POSE',
    'SIMU_OBJECT_POS', 'SIMU_STATUS', 'SIMU_CARTESIAN', 'ALL_SIMU_TOPICS',
    'REAL_IMAGES', 'REAL_JOINTS', 'REAL_CARTESIAN', 'REAL_GRIPPER', 'REAL_STATUS',
    'ALL_REAL_TOPICS',
    'COLLECT_STATUS', 'COLLECT_TASK', 'COLLECT_LOG', 'COLLECT_EPISODE',
    'ALL_COLLECT_TOPICS',
    # Real
    'RealInterface', 'MockRealInterface', 'RobotNotConnectedError',
    'RealPublisher', 'RealDataCollector',
    'SimpleCamera', 'CameraManager',
    # Simu
    'SimuInterface', 'MockSimuInterface',
    'SimuPublisher', 'SimuDataCollector',
    'SimuManager', 'SimuRenderProcess',
    # Control
    'SyncController', 'SyncControllerWithInterpolation',
    'GraspExecutor', 'GraspExecutorWithInterpolation',
    # Legacy
    'DataCollector',
]
