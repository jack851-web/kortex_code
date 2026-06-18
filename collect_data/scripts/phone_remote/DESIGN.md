# 手机遥操作系统 - 设计文档

## 1. 系统概述

### 1.1 目标

在现有定点抓取数据收集系统基础上，增加手机遥操作模式。通过 Android 手机（基于 ARCore）获取用户手持位姿，以增量方式控制机械臂末端，实现灵活的示教式数据采集。

### 1.2 核心特性

- **增量控制**：手机移动量直接映射为机械臂末端位移增量，减少累积误差
- **ARCore 位姿追踪**：6DoF 位姿（位置 + 姿态），精度高、无漂移
- **双模式支持**：实机模式和仿真模式共用同一套遥操作逻辑
- **交互式方向标定**：运行时通过 6 方向移动校准手机坐标系与机械臂坐标系的映射
- **任务生命周期复用**：完全复用现有 `DataCollectionSystem` 的 episode 管理
- **音效提示**：Episode 保存完成后自动播放提示音，告知用户可开始下一个 episode

### 1.3 与现有代码的关系

```
现有架构：
  DataCollectionSystem → GraspExecutor → SimuInterface/RealInterface
                         ↓ 自动执行 waypoint
                      DataCollector (记录数据)

新增遥操作架构：
  DataCollectionSystem → TeleopController → SimuInterface/RealInterface
                         ↑ 接收 ARCore 增量输入
                      DataCollector (记录数据，不变)
                    ↗ WebSocketServer ← Flutter App (ARCore + UI)
```

**不改变的部分**：`DataCollectionSystem` 的任务管理、`DataCollector` 的数据采集与存储、`SimuManager` 的仿真生命周期。

**新增/替换的部分**：用 `TeleopController` 替代 `GraspExecutor` 作为执行器；新增 `WebSocketServer` 作为通信层。

---

## 2. PC 端实现（Python）

### 2.1 文件结构

```
scripts/phone_remote/
  ├── __init__.py              # 模块导出
  ├── protocol.py              # 通信协议定义（消息类型、序列化）
  ├── websocket_server.py      # WebSocket 服务端
  └── controller.py            # TeleopController 遥操作控制器
```

### 2.2 通信协议 (`protocol.py`)

#### 2.2.1 消息类型定义

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class MsgType(str, Enum):
    # === 手机 → PC ===
    POSE_DELTA = "pose_delta"           # ARCore 增量位姿 (高频 ~30-60Hz)
    GRIPPER = "gripper"                 # 夹爪控制 (0.0~1.0)
    ALIGN_DIRECTION = "align_direction" # 方向标定: {"axis": "y+", "phone_axis": "x+"}
    START_TASK = "start_task"           # 开始任务
    START_EPISODE = "start_episode"     # 开始新 episode
    END_EPISODE = "end_episode"         # 结束当前 episode
    END_TASK = "end_task"               # 结束当前任务
    EMERGENCY_STOP = "emergency_stop"   # 急停
    PING = "ping"                       # 心跳

    # === PC → 手机 ===
    MODE = "mode"                       # 当前模式: real / simu
    STATUS = "status"                   # 状态更新 (collecting, episode_id 等)
    EPISODE_SAVED = "episode_saved"     # Episode 已保存（触发音效）
    TASK_COMPLETE = "task_complete"     # 所有任务完成
    TCP_POSE = "tcp_pose"              # PC 端回传的实际末端位姿
    ERROR = "error"                     # 错误信息
    PONG = "pong"                       # 心跳响应


@dataclass
class PoseDeltaMessage:
    """ARCore 增量位姿消息"""
    type: str = MsgType.POSE_DELTA.value
    dx: float = 0.0        # X 轴增量 (m)
    dy: float = 0.0        # Y 轴增量 (m)
    dz: float = 0.0        # Z 轴增量 (m)
    droll: float = 0.0     # Roll 增量 (deg)
    dpitch: float = 0.0    # Pitch 增量 (deg)
    dyaw: float = 0.0      # Yaw 增量 (deg)


@dataclass
class GripperMessage:
    """夹爪控制消息"""
    type: str = MsgType.GRIPPER.value
    value: float = 0.5     # 夹爪开合度 0.0(闭合) ~ 1.0(张开)
```

#### 2.2.2 序列化工具

```python
import json


def encode_message(msg) -> bytes:
    """将消息对象编码为 JSON 字节串"""
    if hasattr(msg, "__dataclass_fields__"):
        d = {k: v for k, v in msg.__dict__.items() if v is not None}
    elif isinstance(msg, dict):
        d = msg
    else:
        raise ValueError(f"Unsupported message type: {type(msg)}")
    return json.dumps(d).encode("utf-8")


def decode_message(data: bytes) -> dict:
    """解码 JSON 字节串为字典"""
    return json.loads(data.decode("utf-8"))
```

### 2.3 WebSocket 服务端 (`websocket_server.py`)

#### 2.3.1 设计要点

- 使用 `websockets` 库（异步）
- 单连接设计：同一时间只允许一个手机连接
- 回调驱动：收到消息后调用注册的回调函数
- 心跳检测：定期 ping，超时断开

#### 2.3.2 接口设计

```python
class WebSocketServer:
    """WebSocket 服务端 - 接收手机遥操作数据"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        on_pose_delta=None,       # Callable[[PoseDeltaMessage], None]
        on_gripper=None,          # Callable[[float], None]
        on_command=None,          # Callable[[dict], None]
        on_connect=None,           # Callable[[], None]
        on_disconnect=None,        # Callable[[], None]
        heartbeat_interval=5.0,   # 心跳间隔(秒)
        heartbeat_timeout=15.0,    # 心跳超时(秒)
    ):
        """
        Args:
            on_pose_delta: 收到 ARCore 增量位姿时的回调
            on_gripper: 收到夹爪指令时的回调 (value: 0.0~1.0)
            on_command: 收到控制指令时的回调 (start_task/start_episode 等)
            on_connect: 客户端连接成功回调
            on_disconnect: 客户端断开回调
        """

    def start(self):
        """启动服务（非阻塞，在后台线程中运行）"""

    def stop(self):
        """停止服务"""

    def send(self, msg) -> bool:
        """向已连接的手机发送消息。返回是否发送成功。"""

    @property
    def is_connected(self) -> bool:
        """是否有手机已连接"""
```

#### 2.3.3 核心逻辑伪代码

```python
async def _handle_client(self, websocket, path):
    """处理单个客户端连接"""
    self._websocket = websocket
    self._connected = True
    if self._on_connect:
        self._on_connect()

    try:
        async for raw_msg in websocket:
            try:
                msg = decode_message(raw_msg)
                msg_type = msg.get("type", "")

                if msg_type == MsgType.POSE_DELTA.value:
                    delta = PoseDeltaMessage(**msg)
                    if self._on_pose_delta:
                        self._on_pose_delta(delta)

                elif msg_type == MsgType.GRIPPER.value:
                    value = msg.get("value", 0.5)
                    if self._on_gripper:
                        self._on_gripper(value)

                elif msg_type in (
                    MsgType.START_TASK.value,
                    MsgType.START_EPISODE.value,
                    MsgType.END_EPISODE.value,
                    MsgType.END_TASK.value,
                    MsgType.EMERGENCY_STOP.value,
                ):
                    if self._on_command:
                        self._on_command({"type": msg_type})

                elif msg_type == MsgType.ALIGN_DIRECTION.value:
                    if self._on_command:
                        self._on_command({
                            "type": "align_direction",
                            "axis": msg.get("axis"),
                            "phone_axis": msg.get("phone_axis"),
                        })

                elif msg_type == MsgType.PING.value:
                    await self.send({"type": MsgType.PONG.value})

            except Exception as e:
                await self.send({
                    "type": MsgType.ERROR.value,
                    "message": f"消息解析失败: {e}"
                })
    except Exception:
        pass
    finally:
        self._connected = False
        self._websocket = None
        if self._on_disconnect:
            self._on_disconnect()
```

### 2.4 遥操作控制器 (`controller.py`)

#### 2.4.1 设计要点

- 继承或替代 `GraspExecutor` 的执行角色
- 维护对齐状态和目标位姿
- 支持动态轴映射（来自手机端的方向标定结果）
- 安全限制：速度限幅、工作空间边界检查
- 仅在 `_is_recording` 为 True 时才发送机械臂指令
- 每个 episode 结束后自动回到初始位

#### 2.4.2 动态轴映射

与之前硬编码映射不同，这里使用手机端标定得到的映射关系：

```python
# 标定数据结构（由手机端发来或从配置加载）
_axis_mapping = {
    "x+": "phone_x+",   # 机械臂 X+ 对应手机哪个轴
    "x-": "phone_x-",
    "y+": "phone_y+",
    "y-": "phone_y-",
    "z+": "phone_z+",
    "z-": "phone_z-",
}
```

#### 2.4.3 接口设计

```python
class TeleopController:
    """遥操作控制器 - 将 ARCore 增量位姿映射到机械臂末端"""

    def __init__(
        self,
        robot_interface=None,      # RealInterface 或 None (仿真模式)
        simu_interface=None,      # SimuInterface 或 None
        position_scale: float = 1.0,       # 位置缩放因子
        orientation_scale: float = 0.5,    # 姿态缩放因子
        max_linear_speed: float = 0.5,      # 最大线速度 (m/s)
        max_angular_speed: float = 30.0,    # 最大角速度 (deg/s)
        workspace_limits=None,             # 工作空间限制 [xmin,xmax,ymin,ymax,zmin,zmax]
        axis_mapping=None,                 # 轴映射字典 (默认使用标准映射)
    ):
        """
        Args:
            position_scale: 手机移动距离与机械臂移动距离的比例。
                           1.0 表示手机移动 10cm，机械臂也移动 10cm。
            axis_mapping: 6 方向轴映射字典。如果为 None，使用默认标准映射。
        """

    @property
    def is_aligned(self) -> bool:
        """是否已完成对齐（有初始位姿基准）"""

    @property
    def current_target(self) -> np.ndarray:
        """当前目标位姿 [x,y,z,rx,ry,rz]"""

    def align(self) -> bool:
        """开始对齐：读取当前机械臂末端位姿作为基准点
        
        Returns:
            是否对齐成功
        """

    def set_axis_mapping(self, mapping: dict):
        """设置轴映射（来自手机端标定结果）"""

    def apply_delta(self, delta: PoseDeltaMessage):
        """应用 ARCore 增量位姿
        
        将增量根据轴映射叠加到当前目标位姿，然后发送给机械臂。
        
        Args:
            delta: ARCore 增量位姿 (手机坐标系原始值)
        """

    def move_to_initial(self):
        """回到初始位姿（每个 episode 结束后调用）"""

    def set_collecting_active(self, active: bool):
        """设置数据采集状态。仅 active=True 时才发送机械臂指令。"""

    def emergency_stop(self):
        """急停：立即停止所有运动"""
```

#### 2.4.4 核心逻辑 - 增量应用（带动态轴映射）

```python
def apply_delta(self, delta: PoseDeltaMessage):
    if not self._aligned or not self._collecting_active:
        return

    # 使用轴映射表将手机轴向转换为机械臂轴向
    # delta 中的 dx/dy/dz 是手机 ARCore 原始坐标系的值
    # 需要根据 _axis_mapping 决定它们对应机械臂的哪个轴
    
    robot_deltas = [0.0] * 6  # [dx, dy, dz, droll, dpitch, dyaw] in robot frame
    
    phone_deltas = {
        "x+": delta.dx, "x-": -delta.dx,
        "y+": delta.dy, "y-": -delta.dy,
        "z+": delta.dz, "z-": -delta.dz,
    }
    
    for robot_axis, phone_axis in self._axis_mapping.items():
        # robot_axis: "x+", "x-", "y+", ...
        idx = {"x": 0, "y": 1, "z": 2}[robot_axis[0]]
        sign = 1 if robot_axis[1] == "+" else -1
        robot_deltas[idx] += sign * phone_deltas.get(phone_axis, 0.0)
    
    # 应用缩放并更新目标位姿
    for i in range(3):
        scale = self._position_scale if i < 3 else self._orientation_scale
        self._target[i] += robot_deltas[i] * scale
    
    # 姿态增量
    self._target[3] += delta.droll * self._orientation_scale
    self._target[4] += delta.dpitch * self._orientation_scale
    self._target[5] += delta.dyaw * self._orientation_scale

    # 工作空间限制
    if self._workspace_limits is not None:
        for i in range(3):
            lo, hi = self._workspace_limits[i*2], self._workspace_limits[i*2+1]
            self._target[i] = np.clip(self._target[i], lo, hi)

    # 发送到机械臂
    self._send_target()

def _send_target(self):
    if self._simu is not None:
        # 仿真模式：IK 控制
        target_pos = self._target[:3]
        orientation = self._euler_to_rotmat(self._target[3:])
        self._simu.move_to_cartesian(
            target_pos,
            orientation=orientation,
            duration=0.05,
            steps=5,
            step_callback=self._frame_callback,
        )
    elif self._robot is not None:
        # 实机模式：笛卡尔控制
        self._robot.move_cartesian(self._target.copy())
```

### 2.5 与 DataCollectionSystem 的集成

在 `main_qt.py` 中添加遥操作模式的初始化：

```python
# 在 initialize() 方法中，仿真/实机初始化完成后：

if self._teleop_mode:  # 新增命令行参数 --teleop
    from scripts.phone_remote import WebSocketServer, TeleopController
    
    # 创建遥操作控制器
    self._teleop_controller = TeleopController(
        robot_interface=self._real,
        simu_interface=self._simu,
        position_scale=self._config.get('teleop', {}).get('position_scale', 1.0),
        workspace_limits=self._config.get('workspace', {}).get('limits'),
    )
    
    # 创建 WebSocket 服务
    ws_config = self._config.get('teleop', {}).get('websocket', {})
    self._ws_server = WebSocketServer(
        host=ws_config.get('host', '0.0.0.0'),
        port=ws_config.get('port', 8765),
        on_pose_delta=lambda delta: self._on_teleop_pose_delta(delta),
        on_gripper=lambda value: self._on_teleop_gripper(value),
        on_command=lambda cmd: self._on_teleop_command(cmd),
        on_connect=lambda: self._log("手机已连接", "SUCCESS"),
        on_disconnect=lambda: self._log("手机已断开", "WARNING"),
    )
    self._ws_server.start()
    
    mode = "simu" if self._simu is not None else "real"
    self._ws_server.send({"type": "mode", "value": mode})
    self._log(f"遥操作模式已启动 ({mode})，等待手机连接...", "INFO")

def _on_teleop_command(self, cmd: dict):
    """处理手机端控制指令"""
    cmd_type = cmd["type"]
    
    if cmd_type == "start_task":
        self.start_collection()       # 或对应方法
    elif cmd_type == "start_episode":
        self.execute_next_task()      # 开始新 episode
    elif cmd_type == "end_episode":
        self.finish_current_task()    # 结束当前 episode
        # 回到初始位
        if self._teleop_controller:
            self._teleop_controller.move_to_initial()
        # 通知手机 episode 已保存（触发音效）
        if self._ws_server:
            self._ws_server.send({
                "type": "episode_saved",
                "id": self._current_episode,
                "success": True,
            })
    elif cmd_type == "end_task":
        self.stop()
    elif cmd_type == "emergency_stop":
        self.stop()  # 急停 = stop + 丢弃数据
    elif cmd_type == "align_direction":
        # 记录标定结果，稍后在全部完成后一次性设置
        pass  # 需要缓存 6 个方向的映射
```

---

## 3. Flutter App 实现（基于参考 UI 设计）

> **重要规则：UI 中严禁使用任何 emoji，所有图标必须使用 SVG 手绘实现。**

### 3.1 参考界面总览

App 共三个页面，通过页面跳转切换：

| 页面 | 功能 | 入口 |
|------|------|------|
| 设置页 | IP/端口配置、方向标定状态、关于 | 首次启动默认页 / 主控页返回箭头 |
| 主控页 | 连接管理、AR 可视化开关、TCP 位姿显示、夹爪控制、任务控制、急停 | 设置页「保存并连接」后进入 |
| 方向校准页 | 6 方向交互式标定 | 设置页「重新标定」按钮 |

### 3.2 技术选型

| 功能 | 方案 | 说明 |
|------|------|------|
| ARCore 追踪 | `ar_core_flutter_plugin` | Google 官方插件 |
| WebSocket | `web_socket_channel` | Dart 官方库 |
| 音效提示 | `audioplayers` | 跨平台音频播放 |
| 状态管理 | Riverpod (`flutter_riverpod`) | 响应式状态管理 |
| 本地存储 | `shared_preferences` | 存储 IP/端口/校准数据 |
| SVG 图标 | `flutter_svg` | 所有图标手绘 SVG |

### 3.3 项目结构

```
phone_app/                          # Flutter 项目根目录
├── lib/
│   ├── main.dart                  # 入口
│   ├── app.dart                   # MaterialApp 配置 + 路由
│   │
│   ├── core/
│   │   ├── constants.dart         # 常量（端口、默认值等）
│   │   ├── theme.dart             # 主题色、字体、圆角
│   │   └── storage_service.dart   # SharedPreferences 封装
│   │
│   ├── models/
│   │   ├── calibration_data.dart  # 校准数据模型（6方向映射）
│   │   └── tcp_pose.dart          # TCP 位姿数据模型
│   │
│   ├── services/
│   │   ├── arcore_service.dart    # ARCore 追踪服务
│   │   ├── websocket_service.dart # WebSocket 通信服务
│   │   └── sound_service.dart     # 音效播放服务
│   │
│   ├── providers/
│   │   ├── connection_provider.dart    # 连接状态 (IP/端口/是否已连接)
│   │   ├── mode_provider.dart          # 模式状态 (real/simu)
│   │   ├── pose_provider.dart          # TCP 位姿实时数据
│   │   ├── gripper_provider.dart       # 夹爪开合度 (0~1)
│   │   ├── task_provider.dart          # 任务状态 (采集中/episode编号)
│   │   └── calibration_provider.dart  # 校准状态与数据
│   │
│   ├── screens/
│   │   ├── settings_screen.dart        # 设置页（第1屏）
│   │   ├── main_control_screen.dart    # 主控页（第2屏）
│   │   └── calibration_screen.dart     # 方向校准页（第3屏）
│   │
│   └── widgets/
│       ├── connection_status_bar.dart  # 连接状态栏组件
│       ├── tcp_pose_display.dart       # TCP 位姿数值显示
│       ├── gripper_button.dart         # 夹爪张开/闭合大按钮
│       ├── task_action_buttons.dart    # 开始任务/Ep/结束任务按钮组
│       ├── emergency_stop_button.dart  # 急停按钮
│       ├── direction_item.dart         # 校准页单个方向条目
│       └── svg_icon.dart              # SVG 图标工具类
│
├── assets/
│   ├── sounds/
│   │   ├── episode_saved.mp3      # Episode 保存完成提示音（核心！）
│   │   ├── connected.mp3          # 连接成功音效
│   │   └── emergency.mp3          # 急停音效
│   └── icons/
│       ├── plus_circle.svg        # 张开图标（圆形内加号）
│       ├── minus_circle.svg       # 闭合图标（圆形内减号）
│       ├── arrow_left.svg         # 返回箭头
│       └── check.svg              # 对勾图标
│
├── pubspec.yaml
└── android/app/src/main/
    ├── AndroidManifest.xml        # ARCore 权限
    └── build.gradle               # minSdkVersion 24+
```

### 3.4 依赖配置 (`pubspec.yaml`)

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # ARCore
  ar_core_flutter_plugin: ^0.7.0
  
  # 网络
  web_socket_channel: ^3.0.0
  connectivity_plus: ^6.0.0
  
  # 音频
  audioplayers: ^6.0.0
  
  # 状态管理
  flutter_riverpod: ^2.5.0
  
  # 本地存储
  shared_preferences: ^2.2.0
  
  # SVG 图标（所有图标必须手绘 SVG，禁止使用 emoji）
  flutter_svg: ^2.0.9
  
  # 工具
  vector_math: ^2.1.4
  permission_handler: ^11.3.0

dev_dependencies:
  flutter_test:
    sdk: flutter
  flutter_lints: ^3.0.0
```

### 3.5 页面详细设计

#### 3.5.1 设置页 (`settings_screen.dart`) — 第1屏

**参考截图布局**：

```
┌─────────────────────────────────────┐
│  ← 设置                              │  ← 标题栏，左侧 SVG 返回箭头
├─────────────────────────────────────┤
│                                     │
│  PG 连接配置                         │  ← Section 标题（黑色粗体）
│  保存后下次自动连接                    │  ← Section 副标题（灰色小字）
│                                     │
│  PC IP 地址                          │  ← 输入框标签
│  ┌─────────────────────────────┐    │
│  │ 192.168.1.100               │    │  ← 文本输入框（圆角边框）
│  └─────────────────────────────┘    │
│                                     │
│  端口                                │  ← 输入框标签
│  ┌─────────────────────────────┐    │
│  │ 8765                        │    │  ← 数字输入框
│  └─────────────────────────────┘    │
│                                     │
│  ┌─────────────────────────────┐    │
│  │      保存并连接               │    │  ← 黑色填充按钮，白色文字，圆角
│  └─────────────────────────────┘    │
│                                     │
│  方向标定                            │  ← Section 标题
│                                     │
│  ✓ 已标定 · 上次标定 2024-05-10      │  ← 已标定行（绿色 SVG 对勾 + 文字）
│  ┌─────────────────────────────┐    │
│  │      重新标定                 │    │  ← 白色边框按钮（未标定时为黑色填充）
│  └─────────────────────────────┘    │
│                                     │
│  关于                                │  ← Section 标题
│  Kortex 遥操作控制 v0.1.0            │  ← 应用名+版本号
│  连接真实机械臂进行数据采集            │  ← 描述文字（灰色小字）
│                                     │
└─────────────────────────────────────┘
```

**交互逻辑**：
- 用户输入 PC IP 和端口号
- 点击「保存并连接」→ 保存到 SharedPreferences → 建立 WebSocket 连接 → 成功后 push 到主控页
- 如果已做过方向标定，显示上次标定时间和绿色对勾 SVG 图标
- 点击「重新标定」push 到方向校准页
- 未标定时，「重新标定」按钮样式变为黑色填充（更突出）

**关键代码结构**：

```dart
class SettingsScreen extends ConsumerStatefulWidget {
  @override
  ConsumerState<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends ConsumerState<SettingsScreen> {
  final _ipController = TextEditingController(text: '192.168.1.100');
  final _portController = TextEditingController(text: '8765');
  
  Future<void> _saveAndConnect() async {
    final ip = _ipController.text.trim();
    final port = int.tryParse(_portController.text) ?? 8765;
    
    // 1. 保存到本地存储
    await ref.read(storageServiceProvider).saveConnection(ip, port);
    
    // 2. 建立 WebSocket 连接
    final success = await ref.read(webSocketProvider.notifier).connect(ip, port);
    if (!success) {
      // 显示错误提示
      return;
    }
    
    // 3. 播放连接音效
    SoundService().playConnected();
    
    // 4. 跳转到主控页
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const MainControlScreen()),
    );
  }
  
  @override
  Widget build(BuildContext context) {
    final calData = ref.watch(calibrationProvider);
    final isCalibrated = calData != null && calData.isComplete;
    
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            // 标题栏
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Row(
                children: [
                  GestureDetector(
                    onTap: () => /* 可选：退出确认 */,
                    child: SvgPicture.asset('assets/icons/arrow_left.svg', width: 24),
                  ),
                  SizedBox(width: 12),
                  Text('设置', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w600)),
                ],
              ),
            ),
            
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // === PG 连接配置 ===
                    Text('PG 连接配置', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                    SizedBox(height: 4),
                    Text('保存后下次自动连接', style: TextStyle(fontSize: 12, color: Colors.grey)),
                    SizedBox(height: 16),
                    
                    TextField(labelText: 'PC IP 地址', controller: _ipController),
                    SizedBox(height: 12),
                    TextField(
                      labelText: '端口',
                      controller: _portController,
                      keyboardType: TextInputType.number,
                    ),
                    SizedBox(height: 20),
                    
                    SizedBox(
                      width: double.infinity,
                      child: FilledButton(
                        onPressed: _saveAndConnect,
                        style: FilledButton.styleFrom(backgroundColor: Color(0xFF212121)),
                        child: Text('保存并连接'),
                      ),
                    ),
                    
                    SizedBox(height: 32),
                    
                    // === 方向标定 ===
                    Text('方向标定', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                    SizedBox(height: 16),
                    
                    if (isCalibrated) ...[
                      Row(
                        children: [
                          SvgPicture.asset('assets/icons/check.svg', width: 18, color: Colors.green),
                          SizedBox(width: 8),
                          Text('已标定 · 上次标定 ${_formatDate(calData.calibratedAt)}',
                              style: TextStyle(fontSize: 13)),
                        ],
                      ),
                      SizedBox(height: 12),
                      OutlinedButton(
                        onPressed: () => _goToCalibration(),
                        child: Text('重新标定'),
                      ),
                    ] else ...[
                      SizedBox(
                        width: double.infinity,
                        child: FilledButton(
                          onPressed: () => _goToCalibration(),
                          child: Text('开始标定'),
                        ),
                      ),
                    ],
                    
                    SizedBox(height: 32),
                    
                    // === 关于 ===
                    Text('关于', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                    SizedBox(height: 8),
                    Text('Kortex 遥操作控制 v0.1.0', style: TextStyle(fontSize: 14)),
                    SizedBox(height: 4),
                    Text('连接真实机械臂进行数据采集', style: TextStyle(fontSize: 12, color: Colors.grey)),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  void _goToCalibration() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const CalibrationScreen()),
    );
  }
}
```

#### 3.5.2 主控页 (`main_control_screen.dart`) — 第2屏

**参考截图布局**：

```
┌─────────────────────────────────────┐
│  ←                                   │  ← 左上角 SVG 返回箭头（回设置页）
│                                     │
│  ● 未连接                    [连接]  │  ← 状态行：红点SVG + "未连接" + 黑色按钮
│  (或: ● 已连接 (仿真))                 │  ← 已连接时显示绿点 + 模式名
│                                     │
│  AR 可视化：关闭           ( ○ )    │  ← Toggle 开关行
│  点击按钮切换以显示基元                │  ← 灰色说明小字
│                                     │
│  TCP 位姿                             │  ← 数据区标题
│  58 Hz                               │  ← 刷新频率（左下绿色小字）
│  ┌──────────┬──────────┬──────────┐  │
│  │ X +0.342 │ Y -0.128 │ Z +0.517 │  │  ← 位置三列（等宽）
│  ├──────────┼──────────┼──────────┤  │
│  │Rx  1.24° │Ry -0.87° │Rz  2.21° │  │  ← 姿态三列
│  └──────────┴──────────┴──────────┘  │
│                                     │
│  夹爪控制                             │  ← Section 标题
│                              0.70   │  ← 当前值（右对齐）
│  ┌────────────────┐ ┌────────────────┐│
│  │      ⊕         │ │      ⊖         ││  ← 两个大方形按钮（等宽）
│  │     张开        │ │     闭合        ││  ← 绿底黑字 / 黑底白字
│  │   前往继续      │ │   前往继续      ││  ← 副标签（灰色小字）
│  └────────────────┘ └────────────────┘│
│         0.65 ± 0.05 额定值           │  ← 范围说明（居中灰色小字）
│                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │  ← 三按钮行（等宽）
│  │  开始任务  │ │开始/结束Ep│ │结束任务 ││  ← 绿边框 / 黄橙边框 / 紫边框
│  └──────────┘ └──────────┘ └──────┘ │
│                                     │
│  ┌─────────────────────────────────┐│
│  │       急 停 STOP                 ││  ← 红色填充大按钮（全宽）
│  └─────────────────────────────────┘│
│                                     │
└─────────────────────────────────────┘
```

**各区域详细说明**：

**(1) 连接状态栏**

- 未连接：红色实心圆点 SVG + "未连接" + 黑色「连接」按钮
- 已连接：绿色实心圆点 SVG + "已连接" + 模式标识("仿真"/"实机")
- 点击「连接」触发 WebSocket 连接

**(2) AR 可视化开关**

- Toggle 开关控制 AR 相机预览渲染
- 关闭时仅追踪不渲染（省电）
- 开启时屏幕叠加 AR 视觉标记

**(3) TCP 位姿显示区**

- 6 个数值分两行三列展示
- 第一行：X/Y/Z 位置（m），正负号 + 3 位小数
- 第二行：Rx/Ry/Rz 姿态（deg），带角度符号 + 2 位小数
- 左下角绿色字体显示刷新频率（Hz）
- 数值每帧实时更新

**(4) 夹爪控制**

- 「张开」：绿色背景(#4CAF50)，圆形加号 SVG 图标，主文字"张开"，副文字"前往继续"
- 「闭合」：黑色背景(#212121)，圆形减号 SVG 图标，主文字"闭合"，副文字"前往继续"
- 右上角显示当前夹爪值（0.00 ~ 1.00）
- 底部显示额定范围
- 点击发送 gripper 指令给 PC 端

**(5) 任务操作按钮组**

- 「开始任务」：绿色边框 → 发送 `start_task`
- 「开始/结束Ep」：黄橙色边框 → 切换功能（未采集→start_episode / 采集中→end_episode）
- 「结束任务」：紫色边框 → 发送 `end_task`
- 三按钮等宽排列

**(6) 急停按钮**

- 红色填充(#F44336)，白色粗体文字
- 全宽，高度约 56px
- 点击发送 `emergency_stop` + 触觉反馈

**关键代码结构**：

```dart
class MainControlScreen extends ConsumerStatefulWidget {
  @override
  ConsumerState<MainControlScreen> createState() => _MainControlScreenState();
}

class _MainControlScreenState extends ConsumerState<MainControlScreen> {
  @override
  Widget build(BuildContext context) {
    final connState = ref.watch(connectionProvider);
    final isConnected = connState.isConnected;
    final mode = ref.watch(modeProvider);              // 'real' or 'simu'
    final pose = ref.watch(poseProvider);               // TCPPose 数据
    final gripperValue = ref.watch(gripperProvider);    // 0.0 ~ 1.0
    final taskState = ref.watch(taskProvider);
    final isCollecting = taskState.isCollecting;
    
    return Scaffold(
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 顶部导航
            Padding(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: GestureDetector(
                onTap: () => Navigator.pop(context),  // 回设置页
                child: SvgPicture.asset('assets/icons/arrow_left.svg', width: 24),
              ),
            ),
            
            // 连接状态栏
            ConnectionStatusBar(isConnected: isConnected, mode: mode),
            
            SizedBox(height: 8),
            
            // AR 可视化开关
            ArToggleSwitch(),
            
            SizedBox(height: 16),
            
            Expanded(
              child: SingleChildScrollView(
                padding: EdgeInsets.symmetric(horizontal: 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // TCP 位姿显示
                    TcpPoseDisplay(pose: pose, frequency: pose?.frequency ?? 0),
                    
                    SizedBox(height: 20),
                    
                    // 夹爪控制
                    GripperControlPanel(value: gripperValue),
                    
                    SizedBox(height: 16),
                    
                    // 任务按钮组
                    TaskActionButtons(isCollecting: isCollecting),
                    
                    SizedBox(height: 12),
                    
                    // 急停按钮
                    EmergencyStopButton(),
                    SizedBox(height: 20),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

#### 3.5.3 方向校准页 (`calibration_screen.dart`) — 第3屏

**参考截图布局**：

```
┌─────────────────────────────────────┐
│  ← 方向校准                           │  ← 标题栏 + SVG 返回箭头
│  0/6                                 │  ← 进度指示器（已完成数/总数）
├─────────────────────────────────────┤
│                                     │
│  依次向 6 个方向移动手机              │  ← 操作指引（黑色粗体）
│  屏幕朝前，音量键朝上                 │  ← 握持姿势提醒（灰色）
│  移动后保持机械臂可见移动方向           │  ← 操作要点（灰色小字）
│                                     │
│  手机移动 → 机械臂响应                │  ← 映射说明标题
│                                     │
│  ┌───────────────────────────────┐  │
│  │  手机向上          Y+ 方向  [选择]│  ← 方向 1/6
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  手机向下          Y- 方向  [选择]│  ← 方向 2/6
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  手机向左          X- 方向  [选择]│  ← 方向 3/6
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  手机向右          X+ 方向  [选择]│  ← 方向 4/6
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  手机向前          Z+ 方向  [选择]│  ← 方向 5/6
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │  手机向后          Z- 方向  [选择]│  ← 方向 6/6
│  └───────────────────────────────┘  │
│                                     │
│  完成后自动保存，下次连接直接使用       │  ← 底部说明（灰色小字）
│                                     │
│  ┌───────────────────────────────┐  │
│  │         完成校准                │  ← 全宽按钮（全部选中后激活）
│  └───────────────────────────────┘  │
│                                     │
└─────────────────────────────────────┘
```

**交互流程详解**：

```
步骤 1: 用户进入校准页
步骤 2: 按照握持要求拿好手机（屏幕朝前，音量键朝上）
步骤 3: 点击第一个方向「手机向上」右侧的「选择」按钮
步骤 4: 将手机沿"上"方向移动一段距离（约 10cm）并保持稳定
步骤 5: App 自动检测位移方向 → 关联到 Y+ 轴
步骤 6: 该条目变为已选中状态（绿色高亮 + "已完成"）
步骤 7: 重复步骤 3-6，完成其余 5 个方向
步骤 8: 全部 6 个方向选中后，「完成校准」按钮激活
步骤 9: 点击「完成校准」，保存 CalibrationData 到本地存储
步骤 10: 自动 pop 回设置页，标定状态更新为"已标定"
```

**校准数据模型**：

```dart
/// 校准数据 - 6方向的轴映射关系
///
/// key: 机械臂基坐标系轴向 ('x+', 'x-', 'y+', 'y-', 'z+', 'z-')
/// value: 手机 ARCore 坐标系轴向 ('x+', 'x-', 'y+', 'y-', 'z+', 'z-')
/// 
/// 不同手机的 ARCore 坐标系可能与标准定义不同，
/// 因此需要运行时通过实际移动来标定映射关系。

class CalibrationData {
  final Map<String, String> axisMapping;  // 6个方向的映射
  final DateTime calibratedAt;
  
  bool get isComplete => axisMapping.length == 6;
  
  Map<String, dynamic> toJson() => {
    'axis_mapping': axisMapping,
    'calibrated_at': calibratedAt.toIso8601String(),
  };
  
  factory CalibrationData.fromJson(Map<String, dynamic> json) {
    return CalibrationData(
      axisMapping: Map<String, String>.from(json['axis_mapping'] ?? {}),
      calibratedAt: DateTime.parse(json['calibrated_at']),
    );
  }
  
  factory CalibrationData.empty() => CalibrationData(
    axisMapping: {},
    calibratedAt: DateTime.now(),
  );
}
```

**单个方向条目组件**：

```dart
class DirectionItem extends StatelessWidget {
  final String label;           // 显示名称："手机向上"
  final String axisLabel;       // 轴向标签："Y+ 方向"
  final String robotAxis;       // 对应的机械臂轴："y+"
  final bool isSelected;        // 是否已完成此方向
  final VoidCallback onSelect;  // 点击「选择」回调
  
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 6),
      padding: EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(8),
        border: Border.all(
          color: isSelected ? Color(0xFF4CAF50) : Color(0xFFE0E0E0),
        ),
        color: isSelected ? Color(0xFFE8F5E9) : Colors.white,
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: TextStyle(fontWeight: FontWeight.w500, fontSize: 14)),
                SizedBox(height: 2),
                Text(axisLabel, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
              ],
            ),
          ),
          SizedBox(
            height: 36,
            child: OutlinedButton(
              onPressed: isSelected ? null : onSelect,
              style: OutlinedButton.styleFrom(
                foregroundColor: isSelected ? Colors.green : null,
                side: BorderSide(color: isSelected ? Colors.green : Colors.grey[400]),
              ),
              child: Text(isSelected ? '已完成' : '选择', style: TextStyle(fontSize: 13)),
            ),
          ),
        ],
      ),
    );
  }
}
```

**标定核心逻辑**：

```dart
/// 在用户点击某个方向的「选择」按钮时调用
///
/// 流程：
/// 1. 记录当前帧的手机绝对位姿作为起点
/// 2. 提示用户沿该方向移动手机
/// 3. 在接下来的约 1 秒内持续采样
/// 4. 找到位移变化最大的轴方向
/// 5. 将该手机轴关联到此条目对应的机械臂轴
Future<void> _onDirectionSelected(String robotAxis) async {
  // 1. 记录起始位姿
  final startPose = await _arcoreService.getCurrentPose();
  if (startPose == null) return;
  
  // 2-3. 等待移动并采样（最多 3 秒，或检测到足够大的位移）
  Pose? endPose;
  await Future.delayed(Duration(milliseconds: 1500));
  endPose = await _arcoreService.getCurrentPose();
  
  if (endPose == null || startPose == endPose) {
    // 未检测到移动，提示用户重试
    return;
  }
  
  // 4. 计算位移向量，找出最大变化轴
  final dx = endPose.x - startPose.x;
  final dy = endPose.y - startPose.y;
  final dz = endPose.z - startPose.z;
  
  final absDeltas = [(dx, 'x'), (dy, 'y'), (dz, 'z')];
  absDeltas.sort((a, b) => b.$1.abs().compareTo(a.$1.abs()));
  final dominantAxis = absDeltas.first.$2;
  final sign = absDeltas.first.$1 >= 0 ? '+' : '-';
  final phoneAxis = '$dominantAxis$sign';
  
  // 5. 保存映射
  setState(() {
    _calibrationData.axisMapping[robotAxis] = phoneAxis;
    _selectedDirections.add(robotAxis);
  });
}
```

### 3.6 核心服务模块

#### 3.6.1 ARCore 服务 (`arcore_service.dart`)

```dart
/// ARCore 位姿追踪服务
///
/// 职责：
/// 1. 初始化 ARCore Session（请求相机权限）
/// 2. 以固定频率（最高 60Hz）获取设备 6DoF 位姿
/// 3. 计算相对于上一帧的增量位姿
/// 4. 通过 Stream 广播绝对位姿和增量位姿
/// 5. 支持校准模式下的单次位姿采样

class ArCoreService {
  // 绝对位姿流 (~30-60 Hz)
  final _poseController = StreamController<TcpPose>.broadcast();
  Stream<TcpPose> get poseStream => _poseController.stream;
  
  // 增量位姿流（用于遥操作控制）
  final _deltaController = StreamController<PoseDelta>.broadcast();
  Stream<PoseDelta> get deltaStream => _deltaController.stream;
  
  Pose? _previousPose;
  DateTime? _previousTimestamp;
  int _frameCount = 0;
  DateTime? _lastFreqCalcTime;
  
  static const double noiseThresholdPosition = 0.001;  // 1mm
  static const double noiseThresholdRotation = 0.5;    // 0.5 deg
  
  /// 初始化 ARCore
  Future<bool> init() async {
    // 1. 请求 CAMERA 权限
    // 2. 创建 ArCoreSession
    // 3. 配置 TrackingMode = ROTATION_AND_POSITION
    // 4. 返回初始化是否成功
  }
  
  /// 开始追踪
  void startTracking({int targetFps = 60}) {
    // 启动 Timer.periodic，按 targetFps 采样 ArFrame
  }
  
  /// 停止追踪
  void stopTracking() {
    // 停止 Timer
    // pause session
  }
  
  /// 获取当前单次位姿（用于校准时同步采样）
  Future<Pose?> getCurrentPose() async {
    // 同步读取最新一帧的 camera pose
  }
  
  /// 重置参考帧（新 episode 开始时调用，增量为零点）
  void resetReference() {
    _previousPose = null;
    _previousTimestamp = null;
  }
  
  /// 内部帧处理
  void _processFrame(ArFrame frame) {
    _frameCount++;
    final now = DateTime.now();
    
    // 计算频率（每秒更新一次）
    if (_lastFreqCalcTime != null &&
        now.difference(_lastFreqCalcTime!).inMilliseconds >= 1000) {
      final elapsed = now.difference(_lastFreqCalcTime!).inMicroseconds / 1e6;
      _currentFrequency = _frameCount / elapsed;
      _frameCount = 0;
      _lastFreqCalcTime = now;
    } else if (_lastFreqCalcTime == null) {
      _lastFreqCalcTime = now;
      _frameCount = 0;
    }
    
    final currentPose = _extractPose(frame);
    
    // 广播绝对位姿
    _poseController.add(TcpPose.fromArPose(currentPose, frequency: _currentFrequency));
    
    // 计算增量
    if (_previousPose != null && _previousTimestamp != null) {
      final delta = _computeDelta(_previousPose!, currentPose);
      
      // 噪声过滤
      if (_isAboveThreshold(delta)) {
        _deltaController.add(PoseDelta(
          dx: delta.x, dy: delta.y, dz: delta.z,
          droll: delta.roll, dpitch: delta.pitch, dyaw: delta.yaw,
          timestamp: now,
        ));
      }
    }
    
    _previousPose = currentPose;
    _previousTimestamp = now;
  }
  
  double _currentFrequency = 0.0;
  
  void dispose() {
    stopTracking();
    _poseController.close();
    _deltaController.close();
  }
}
```

#### 3.6.2 WebSocket 服务 (`websocket_service.dart`)

```dart
/// WebSocket 通信服务
///
/// 协议消息格式（JSON）：
///
/// 手机 -> PC:
///   {"type": "pose_delta", "dx": 0.01, "dy": -0.02, "dz": 0.0, ...}  高频位姿增量
///   {"type": "gripper", "value": 0.8}                                  夹爪控制
///   {"type": "start_task"}                                             开始任务
///   {"type": "start_episode"}                                          开始 episode
///   {"type": "end_episode"}                                            结束 episode
///   {"type": "end_task"}                                               结束任务
///   {"type": "emergency_stop"}                                         急停
///   {"type": "ping"}                                                   心跳
///
/// PC -> 手机:
///   {"type": "mode", "value": "real|simu"}                             模式通知
///   {"type": "status", "collecting": true/false}                       采集状态
///   {"type": "episode_saved", "id": 5, "success": true}               Episode 保存(触发音效!)
///   {"type": "task_complete"}                                          任务完成
///   {"type": "tcp_pose", "x": ..., "y": ..., "z": ..., ...}           PC端回传实际末端位姿
///   {"type": "error", "message": "..."}                                 错误信息
///   {"type": "pong"}                                                   心跳响应

class WebSocketService {
  WebSocketChannel? _channel;
  bool _isConnected = false;
  Timer? _heartbeatTimer;
  
  // 回调
  Function(String mode)? onModeReceived;
  Function(bool collecting)? onStatusChanged;
  Function(int id, bool success)? onEpisodeSaved;
  Function()? onTaskComplete;
  Function(String message)? onError;
  Function(TcpPose)? onTcpPoseReceived;
  
  /// 连接服务器
  Future<bool> connect(String ip, int port) async {
    try {
      _channel = WebSocketChannel.connect(Uri.parse('ws://$ip:$port'));
      _isConnected = true;
      _listen();
      _startHeartbeat();
      return true;
    } catch (e) {
      _isConnected = false;
      onError?.call('连接失败: $e');
      return false;
    }
  }
  
  /// 断开连接
  void disconnect() {
    _stopHeartbeat();
    _channel?.sink.close();
    _channel = null;
    _isConnected = false;
  }
  
  /// 发送位姿增量（高频调用，需高效）
  void sendPoseDelta(PoseDelta delta) {
    _send({'type': 'pose_delta', ...delta.toJson()});
  }
  
  /// 发送夹爪指令
  void sendGripper(double value) {
    _send({'type': 'gripper', 'value': value});
  }
  
  /// 发送控制指令
  void sendCommand(String type) {
    _send({'type': type});
  }
  
  /// 消息监听与分发
  void _listen() {
    _channel?.stream.listen((message) {
      final data = jsonDecode(message as String);
      switch (data['type']) {
        case 'mode':
          onModeReceived?.call(data['value']);
          break;
        case 'status':
          onStatusChanged?.call(data['collecting'] ?? false);
          break;
        case 'episode_saved':
          onEpisodeSaved?.call(data['id'] ?? 0, data['success'] ?? false);
          break;
        case 'task_complete':
          onTaskComplete?.call();
          break;
        case 'tcp_pose':
          onTcpPoseReceived?.call(TcpPose.fromJson(data));
          break;
        case 'error':
          onError?.call(data['message'] ?? '');
          break;
      }
    }, onError: (e) {
      _isConnected = false;
      onError?.call('连接断开: $e');
    }, onDone: () {
      _isConnected = false;
    });
  }
  
  void _send(Map<String, dynamic> msg) {
    if (_isConnected && _channel != null) {
      _channel!.sink.add(jsonEncode(msg));
    }
  }
  
  // 心跳: 每 5s 发 ping，15s 无 pong 则判定断连
  void _startHeartbeat() {
    _heartbeatTimer = Timer.periodic(Duration(seconds: 5), (_) {
      _send({'type': 'ping'});
    });
    // TODO: pong 超时检测
  }
  
  void _stopHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
  }
}
```

#### 3.6.3 音效服务 (`sound_service.dart`)

```dart
/// 音效播放服务（单例）
///
/// 使用场景：
/// 1. WebSocket 连接成功 → playConnected()
/// 2. Episode 保存完成（核心提示！） → playEpisodeSaved()
///    用户听到这个声音后知道可以开始下一个 episode
/// 3. 急停 → playEmergency()

class SoundService {
  static final SoundService _instance = SoundService._internal();
  factory SoundService() => _instance;
  SoundService._internal();
  
  AudioPlayer? _player;
  
  /// 预加载音效资源（App 启动时调用一次）
  Future<void> preloadAll() async {
    _player = AudioPlayer();
    // 预加载但不播放，确保后续低延迟
  }
  
  /// 连接成功
  void playConnected() => _playAsset('sounds/connected.mp3');
  
  /// Episode 保存完成 —— 核心提示音！
  void playEpisodeSaved() => _playAsset('sounds/episode_saved.mp3');
  
  /// 急停
  void playEmergency() => _playAsset('sounds/emergency.mp3');
  
  void _playAsset(String path) {
    _player?.stop();
    _player?.setSource(AssetSource(path));
    _player?.resume();
  }
  
  void dispose() {
    _player?.dispose();
    _player = null;
  }
}
```

### 3.7 完整操作流程

#### 3.7.1 首次使用流程

```
[打开 App]
    ↓
[设置页] — 显示默认 IP(192.168.1.100)/端口(8765)
    ↓
[首次使用，未标定] — 「开始标定」按钮黑色填充（突出）
    ↓
[点击「开始标定」] → push 进入方向校准页
    ↓
[方向校准页] — 按照 6 方向逐一标定（上/下/左/右/前/后）
    ↓
[点击「完成校准」] — 保存 CalibrationData 到 SharedPreferences
    ↓
[自动 pop 回设置页] — 显示"已标定 · 上次标定时间"
    ↓
[点击「保存并连接」] — 建立 WebSocket 连接
    ↓
[连接成功] — 播放 connected 音效 → push 到主控页
    ↓
[主控页收到 PC 端 mode 消息] — 显示"已连接 (仿真)"或"已连接 (实机)"
```

#### 3.7.2 日常使用流程（已标定）

```
[打开 App]
    ↓
[设置页] — 自动填入上次保存的 IP/端口
    ↓
[点击「保存并连接」]
    ↓
[主控页] — 显示"已连接"，收到 PC 端 mode(real/simu)
    ↓
[点击「开始任务」] — PC 端创建仿真场景/初始化实机
    ↓
[点击「开始/结束Ep」(此时功能=开始)] — PC 端开始采集 + 机械臂回初始位
    ↓
[持手机操作] — ARCore 追踪 → 增量位姿 → WebSocket(~60Hz) → PC → 机械臂移动
    ↓              同时数据持续采集写入 LeRobot dataset
[操作完成]
    ↓
[点击「开始/结束Ep」(此时功能=结束)] — PC 端保存 episode + 机械臂回初始位
    ↓
[收到 episode_saved 消息] — 播放 episode_saved 提示音！
    ↓ ("可以开始下一个 episode 了")
[重复上述 Ep 循环...]
    ↓
[所有任务完成 或 点击「结束任务」]
    ↓
[或点击「急停 STOP」— 立即停止并丢弃当前数据]
```

### 3.8 主题与视觉规范

```dart
// theme.dart
class AppTheme {
  // 主色调（严格匹配参考截图）
  static const Color primaryGreen = Color(0xFF4CAF50);     // 张开按钮 / 开始任务边框
  static const Color primaryBlack = Color(0xFF212121);     // 闭合按钮 / 主要填充按钮
  static const Color primaryRed = Color(0xFFF44336);       // 急停按钮
  static const Color accentYellowOrange = Color(0xFFFFB300); // 开始/结束 Ep 边框
  static const Color accentPurple = Color(0xFF9C27B0);     // 结束任务边框
  
  // 状态颜色
  static const Color dotGreen = Color(0xFF4CAF50);         // 已连接绿点
  static const Color dotRed = Color(0xFFE57373);           // 未连接红点
  
  // 背景色
  static const Color background = Color(0xFFFAFAFA);       // 页面背景（接近白色）
  static const Color cardBackground = Colors.white;         // 卡片/输入框背景
  
  // 字体大小
  static const double fontSizeTitle = 18.0;     // 页面标题
  static const double fontSizeSection = 16.0;   // Section 标题
  static const double fontSizeBody = 14.0;      // 正文
  static const double fontSizeCaption = 12.0;   // 副标题/说明
  static const double fontSizeValue = 15.0;     // TCP 数值
  
  // 圆角
  static const double radiusCard = 8.0;         // 卡片/输入框
  static const double radiusButton = 8.0;       // 按钮
  
  // 间距
  static const double pagePaddingH = 20.0;      // 页面水平内边距
  static const double sectionGap = 28.0;        // Section 之间间距
  static const double itemGapV = 8.0;           // 条目垂直间距
}
```

### 3.9 SVG 图标清单

所有图标必须手绘 SVG，禁止使用 emoji、icon font 或图片。

| 文件名 | 用途 | 尺寸 | 描述 |
|--------|------|------|------|
| `arrow_left.svg` | 导航返回 | 24x24 | 向左箭头，线条粗细 2px |
| `plus_circle.svg` | 夹爪张开 | 48x48 | 圆圈内加号，线条粗细 2.5px |
| `minus_circle.svg` | 夹爪闭合 | 48x48 | 圆圈内减号，线条粗细 2.5px |
| `check.svg` | 已标定/已完成 | 18x18 | 对勾标记，填充样式 |
| `dot_filled.svg` | 状态指示点 | 12x12 | 实心圆（绿色/红色两版，通过 color tint） |

### 3.10 Android 配置

```xml
<!-- android/app/src/main/AndroidManifest.xml -->
<manifest>
    <uses-permission android:name="android.permission.CAMERA" />
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
    <uses-permission android:name="android.permission.VIBRATE" />  <!-- 触觉反馈 -->

    <application ...>
        <!-- ARCore 要求 -->
        <meta-data
            android:name="com.google.ar.core"
            android:value="required" />
    </application>

    <!-- OpenGL ES 3.0+ (ARCore 要求) -->
    <uses-feature android:glEsVersion="0x30002" android:required="true" />
</manifest>
```

```gradle
// android/app/build.gradle
android {
    compileSdkVersion 34
    defaultConfig {
        minSdkVersion 24      // ARCore 最低要求
        targetSdkVersion 34
    }
}
```

---

## 4. 配置扩展