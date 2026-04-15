# 模型评估系统设计文档

## 1. 概述

### 1.1 目标
评估使用收集数据训练出的策略模型在仿真环境中的表现，支持：
- **无头模式**：在租用服务器上运行，无显示器
- **脚本化批量评估**：一次运行多个模型/多个任务
- **结果自动保存**：JSON/CSV 格式，便于后续分析

### 1.2 评估指标
| 指标 | 说明 |
|------|------|
| 任务成功率 | 成功完成任务的 episode 比例 |
| 平均步数 | 完成任务所需的平均步数 |
| 推理延迟 | 每步策略推理时间（GPU/CPU） |
| 轨迹相似度 | 与专家演示的 DTW 距离 |

### 1.3 运行环境
- **服务器环境**：无显示器，使用 MuJoCo 离屏渲染
- **GPU 支持**：CUDA 加速策略推理
- **批量评估**：支持评估多个 checkpoint 或多个任务配置

## 2. 数据格式与模型格式

### 2.1 数据格式 (LeRobot v3.0)
```json
{
  "codebase_version": "v3.0",
  "features": {
    "observation.state": {"dtype": "float32", "shape": [7]},  // 6 joints + gripper
    "action": {"dtype": "float32", "shape": [7]},              // delta joints + gripper
    "observation.images.top": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.images.side": {"dtype": "video", "shape": [480, 640, 3]},
    "observation.images.hand": {"dtype": "video", "shape": [480, 640, 3]}
  }
}
```

### 2.2 支持的策略类型
| 策略 | 特点 | 推荐场景 |
|------|------|----------|
| ACT | 动作分块，Transformer架构 | 多步骤任务 |
| Diffusion | 扩散模型，高质量动作 | 精细操作 |
| VQBeT | 离散化动作空间 | 长序列任务 |
| TDMPC | 模型预测控制 | 动态环境 |

### 2.3 模型检查点格式
- **Diffusion Policy**: `.ckpt` 文件，包含 `cfg`, `model`, `ema_model`
- **LeRobot Policy**: HuggingFace 格式，`config.json` + `model.safetensors`

## 3. 代码架构

### 3.1 目录结构
```
d:\VLA\kortex_code\eval\
├── __init__.py
├── eval.py                    # 主入口：CLI 命令
├── batch_eval.py              # 批量评估脚本
├── config/
│   ├── __init__.py
│   ├── eval_config.py         # 评估配置类
│   ├── eval_config.yaml       # 默认配置文件
│   └── batch_config.yaml      # 批量评估配置
├── envs/
│   ├── __init__.py
│   └── simu_env.py            # 仿真环境封装（无头模式）
├── policies/
│   ├── __init__.py
│   ├── policy_wrapper.py      # 策略封装基类
│   ├── diffusion_policy.py    # Diffusion Policy 封装
│   └── lerobot_policy.py      # LeRobot Policy 封装
├── metrics/
│   ├── __init__.py
│   ├── success_checker.py     # 任务成功判定
│   └── report.py              # 评估报告生成
└── utils/
    ├── __init__.py
    └── logger.py              # 日志工具
```

### 3.2 核心类设计

#### 3.2.1 配置类 (config/eval_config.py)
```python
@dataclass
class EvalConfig:
    # 策略配置
    policy_path: str                    # 模型路径
    policy_type: str = "diffusion"      # diffusion / act / vqbet
    
    # 评估配置
    n_episodes: int = 10                # 评估轮数
    max_steps_per_episode: int = 500    # 每轮最大步数
    control_frequency: float = 10.0     # 控制频率 (Hz)
    
    # 环境配置
    simu_xml_path: str = ""             # 仿真场景XML
    headless: bool = True               # 无头模式（服务器必须为True）
    
    # 任务配置
    task_config_path: str = ""          # 任务配置文件
    initial_joint_angles: List[float]   # 初始关节角度
    
    # 输出配置
    output_dir: str = "outputs/eval"
    save_video: bool = True             # 保存视频（用于离线查看）
    save_trajectory: bool = True        # 保存轨迹数据
    
    # 设备配置
    device: str = "cuda"                # cuda / cpu
    seed: int = 42                      # 随机种子（可复现）
```

#### 3.2.2 批量评估配置 (config/batch_config.yaml)
```yaml
# 批量评估配置：一次评估多个模型
experiments:
  - name: "diffusion_policy_ep100"
    policy_path: "outputs/diffusion_policy/checkpoints/epoch_100.ckpt"
    policy_type: "diffusion"
    
  - name: "diffusion_policy_ep200"
    policy_path: "outputs/diffusion_policy/checkpoints/epoch_200.ckpt"
    policy_type: "diffusion"
    
  - name: "act_model"
    policy_path: "outputs/act_model"
    policy_type: "act"

# 通用评估配置
eval:
  n_episodes: 20
  max_steps_per_episode: 500
  control_frequency: 10.0
  seed: 42
  device: "cuda"

# 环境配置
env:
  simu_xml: "kortex_simu/simu/env/task_pick_place.xml"
  headless: true

# 任务配置
tasks:
  - object_position: [0.35, 0.15, 0.44]
    plate_position: [0.35, -0.15, 0.44]
    description: "task1_right"
  - object_position: [0.35, -0.15, 0.44]
    plate_position: [0.35, 0.15, 0.44]
    description: "task2_left"
  - object_position: "random"
    plate_position: "random"
    description: "task_random"

# 输出配置
output:
  dir: "outputs/batch_eval"
  save_video: true
  save_trajectory: true
  report_format: "json"  # json / csv
```

#### 3.2.3 仿真环境 (envs/simu_env.py)
```python
class SimuEvalEnv:
    """MuJoCo 仿真评估环境（无头模式）
    
    专为服务器环境设计：
    - 使用 MuJoCo 离屏渲染，无需显示器
    - 支持多线程安全（批量评估）
    - 复用 collect_data/scripts/simu_interface.py
    """
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self._step_count = 0
        self._current_task = None
        
        # 初始化 MuJoCo（无头模式）
        self._model = mujoco.MjModel.from_xml_path(config.simu_xml_path)
        self._data = mujoco.MjData(self._model)
        
        # 离屏渲染器
        self._renderer = mujoco.Renderer(
            self._model,
            height=480,
            width=640
        )
        
        # 相机配置
        self._camera_names = ['top', 'side', 'hand']
        self._cameras = {}
        for cam_name in self._camera_names:
            cam_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            )
            self._cameras[cam_name] = cam_id
        
        # 关节和夹爪索引
        self._joint_indices = self._get_joint_indices()
        self._gripper_indices = self._get_gripper_indices()
    
    def reset(self, task_config: dict) -> dict:
        """重置环境"""
        mujoco.mj_resetData(self._model, self._data)
        
        # 设置初始关节角度
        if self.config.initial_joint_angles:
            for i, angle in enumerate(self.config.initial_joint_angles):
                self._data.qpos[self._joint_indices[i]] = angle
        
        # 设置物体位置
        object_pos = np.array(task_config['object_position'])
        plate_pos = np.array(task_config['plate_position'])
        self._set_object_position('object', object_pos)
        self._set_object_position('plate', plate_pos)
        
        # 前向运动学
        mujoco.mj_forward(self._model, self._data)
        
        self._step_count = 0
        self._current_task = task_config
        
        return self.get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        """执行动作
        
        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper] 或关节增量
        """
        # 应用动作到仿真
        self._apply_action(action)
        
        # 步进仿真
        for _ in range(10):  # 10 substeps per control step
            mujoco.mj_step(self._model, self._data)
        
        self._step_count += 1
        
        # 获取观测
        obs = self.get_observation()
        
        # 检查任务完成
        success = self.check_task_success()
        done = success or self._step_count >= self.config.max_steps_per_episode
        
        # 计算奖励（可选）
        reward = 1.0 if success else 0.0
        
        info = {
            'success': success,
            'step': self._step_count,
            'object_pos': self._get_object_position('object'),
            'target_pos': np.array(self._current_task['plate_position'])
        }
        
        return obs, reward, done, info
    
    def get_observation(self) -> dict:
        """获取符合 LeRobot 格式的观测"""
        # 关节状态
        joint_state = np.array([
            self._data.qpos[i] for i in self._joint_indices
        ], dtype=np.float32)
        
        # 夹爪状态
        gripper_state = np.mean([
            self._data.qpos[i] for i in self._gripper_indices
        ])
        gripper_state = np.clip(gripper_state, 0.0, 1.0)
        
        # 图像（离屏渲染）
        images = {}
        for cam_name in self._camera_names:
            self._renderer.update_scene(self._data, camera=self._cameras[cam_name])
            images[f'observation.images.{cam_name}'] = self._renderer.render()
        
        return {
            'observation.state': np.concatenate([joint_state, [gripper_state]]).astype(np.float32),
            **images
        }
    
    def check_task_success(self) -> bool:
        """检查物体是否到达目标位置"""
        if self._current_task is None:
            return False
        
        object_pos = self._get_object_position('object')
        target_pos = np.array(self._current_task['plate_position'])
        
        xy_distance = np.linalg.norm(object_pos[:2] - target_pos[:2])
        z_diff = abs(object_pos[2] - target_pos[2])
        
        return xy_distance < 0.08 and z_diff < 0.05
    
    def render_video_frame(self, camera_name: str = 'top') -> np.ndarray:
        """渲染单帧图像（用于保存视频）"""
        self._renderer.update_scene(self._data, camera=self._cameras[camera_name])
        return self._renderer.render()
    
    def close(self):
        """清理资源"""
        if self._renderer:
            self._renderer.close()
```

#### 3.2.4 策略封装 (policies/policy_wrapper.py)
```python
class PolicyWrapper(ABC):
    """策略封装基类"""
    
    @abstractmethod
    def load(self, checkpoint_path: str, device: str = 'cuda'):
        """加载模型检查点"""
        pass
    
    @abstractmethod
    def reset(self):
        """重置策略内部状态"""
        pass
    
    @abstractmethod
    def predict(self, observation: dict) -> np.ndarray:
        """根据观测预测动作"""
        pass


class DiffusionPolicyWrapper(PolicyWrapper):
    """Diffusion Policy 封装"""
    
    def __init__(self):
        self.device = 'cuda'
        self._policy = None
        self._cfg = None
        self._n_obs_steps = 2
        self._obs_buffer = []
    
    def load(self, checkpoint_path: str, device: str = 'cuda'):
        import dill
        self.device = device
        
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill, map_location=device)
        self._cfg = payload['cfg']
        
        # 重建 workspace
        from diffusion_policy.workspace.base_workspace import BaseWorkspace
        cls = hydra.utils.get_class(self._cfg._target_)
        workspace = cls(self._cfg)
        workspace.load_payload(payload)
        
        # 获取策略
        self._policy = workspace.model
        if self._cfg.training.use_ema:
            self._policy = workspace.ema_model
        
        self._policy.to(device)
        self._policy.eval()
        
        # 配置推理参数
        self._policy.num_inference_steps = 16
        self._n_obs_steps = self._cfg.n_obs_steps
    
    def reset(self):
        self._obs_buffer = []
        if hasattr(self._policy, 'reset'):
            self._policy.reset()
    
    def predict(self, observation: dict) -> np.ndarray:
        # 更新观测缓冲
        self._obs_buffer.append(observation)
        if len(self._obs_buffer) > self._n_obs_steps:
            self._obs_buffer.pop(0)
        
        # 填充观测
        while len(self._obs_buffer) < self._n_obs_steps:
            self._obs_buffer.insert(0, observation)
        
        # 构建 tensor
        obs_dict = self._build_obs_tensor()
        
        with torch.no_grad():
            result = self._policy.predict_action(obs_dict)
            action = result['action'][0].cpu().numpy()
        
        return action
    
    def _build_obs_tensor(self) -> dict:
        """构建策略输入张量"""
        # 实现细节...
        pass


class LeRobotPolicyWrapper(PolicyWrapper):
    """LeRobot 策略封装"""
    
    def __init__(self, policy_type: str = 'act'):
        self.policy_type = policy_type
        self.device = 'cuda'
        self._policy = None
    
    def load(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        from lerobot.policies.factory import get_policy_class
        policy_cls = get_policy_class(self.policy_type)
        self._policy = policy_cls.from_pretrained(checkpoint_path)
        self._policy.to(device)
        self._policy.eval()
    
    def reset(self):
        pass
    
    def predict(self, observation: dict) -> np.ndarray:
        obs_tensor = self._convert_observation(observation)
        with torch.no_grad():
            action = self._policy.select_action(obs_tensor)
        return action.cpu().numpy()
```

#### 3.2.5 评估运行器
```python
class EvalRunner:
    """单模型评估运行器"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.env = None
        self.policy = None
        self.video_writer = None
    
    def run(self) -> dict:
        """运行评估"""
        # 初始化环境和策略
        self.env = SimuEvalEnv(self.config)
        self.policy = self._create_policy()
        
        results = []
        
        for episode_idx in range(self.config.n_episodes):
            print(f"[Eval] Episode {episode_idx + 1}/{self.config.n_episodes}")
            
            # 重置
            task_config = self._get_task_config(episode_idx)
            obs = self.env.reset(task_config)
            self.policy.reset()
            
            # 视频录制
            if self.config.save_video:
                self._init_video_writer(episode_idx)
            
            # 轨迹记录
            trajectory = {'observations': [], 'actions': [], 'timestamps': []}
            inference_times = []
            
            episode_start = time.time()
            done = False
            step = 0
            
            while not done and step < self.config.max_steps_per_episode:
                # 策略推理
                infer_start = time.time()
                action = self.policy.predict(obs)
                inference_times.append(time.time() - infer_start)
                
                # 执行
                next_obs, reward, done, info = self.env.step(action)
                
                # 记录
                trajectory['observations'].append(obs)
                trajectory['actions'].append(action)
                trajectory['timestamps'].append(time.time() - episode_start)
                
                # 保存视频帧
                if self.config.save_video:
                    frame = self.env.render_video_frame('top')
                    self._write_video_frame(frame)
                
                obs = next_obs
                step += 1
            
            # 关闭视频
            if self.video_writer:
                self._close_video_writer()
            
            # 记录结果
            results.append({
                'episode_idx': episode_idx,
                'task': task_config.get('description', f'task_{episode_idx}'),
                'success': info['success'],
                'steps': step,
                'duration': time.time() - episode_start,
                'avg_inference_time': np.mean(inference_times),
                'trajectory': trajectory if self.config.save_trajectory else None
            })
            
            print(f"  Success: {info['success']}, Steps: {step}")
        
        # 清理
        self.env.close()
        
        return self._generate_report(results)
    
    def _create_policy(self) -> PolicyWrapper:
        if self.config.policy_type == 'diffusion':
            wrapper = DiffusionPolicyWrapper()
        else:
            wrapper = LeRobotPolicyWrapper(self.config.policy_type)
        wrapper.load(self.config.policy_path, self.config.device)
        return wrapper
    
    def _generate_report(self, results: List[dict]) -> dict:
        success_count = sum(1 for r in results if r['success'])
        return {
            'policy_path': self.config.policy_path,
            'policy_type': self.config.policy_type,
            'n_episodes': self.config.n_episodes,
            'success_rate': success_count / len(results),
            'avg_steps': np.mean([r['steps'] for r in results]),
            'avg_duration': np.mean([r['duration'] for r in results]),
            'avg_inference_time_ms': np.mean([r['avg_inference_time'] for r in results]) * 1000,
            'episode_results': [
                {k: v for k, v in r.items() if k != 'trajectory'}
                for r in results
            ]
        }
```

#### 3.2.6 批量评估脚本 (batch_eval.py)
```python
#!/usr/bin/env python
"""
批量评估脚本 - 用于服务器无头环境

Usage:
    # 单模型评估
    python batch_eval.py --config config/eval_config.yaml
    
    # 批量评估多个模型
    python batch_eval.py --batch_config config/batch_config.yaml
    
    # 命令行快速评估
    python batch_eval.py \
        --policy_path outputs/model.ckpt \
        --policy_type diffusion \
        --n_episodes 20 \
        --output_dir outputs/eval
"""

import argparse
import json
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def run_single_eval(config: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个模型评估"""
    from eval_config import EvalConfig
    from eval import EvalRunner
    
    cfg = EvalConfig(**config)
    runner = EvalRunner(cfg)
    return runner.run()


def run_batch_eval(batch_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """批量评估多个模型"""
    results = []
    
    output_dir = Path(batch_config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 通用配置
    common_config = {
        'n_episodes': batch_config['eval']['n_episodes'],
        'max_steps_per_episode': batch_config['eval']['max_steps_per_episode'],
        'control_frequency': batch_config['eval']['control_frequency'],
        'seed': batch_config['eval']['seed'],
        'device': batch_config['eval']['device'],
        'simu_xml_path': batch_config['env']['simu_xml'],
        'headless': batch_config['env']['headless'],
        'output_dir': str(output_dir),
        'save_video': batch_config['output']['save_video'],
        'save_trajectory': batch_config['output']['save_trajectory'],
    }
    
    # 遍历所有实验
    for exp in batch_config['experiments']:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp['name']}")
        print(f"Policy: {exp['policy_path']}")
        print(f"{'='*60}")
        
        config = {
            **common_config,
            'policy_path': exp['policy_path'],
            'policy_type': exp['policy_type'],
        }
        
        start_time = time.time()
        report = run_single_eval(config)
        elapsed = time.time() - start_time
        
        report['experiment_name'] = exp['name']
        report['elapsed_time'] = elapsed
        
        results.append(report)
        
        # 打印结果
        print(f"\nResults for {exp['name']}:")
        print(f"  Success Rate: {report['success_rate']*100:.1f}%")
        print(f"  Avg Steps: {report['avg_steps']:.1f}")
        print(f"  Avg Inference: {report['avg_inference_time_ms']:.2f}ms")
        print(f"  Elapsed: {elapsed:.1f}s")
    
    return results


def save_results(results: List[Dict], output_dir: Path, format: str = 'json'):
    """保存评估结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == 'json':
        output_file = output_dir / f"eval_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'csv':
        import pandas as pd
        output_file = output_dir / f"eval_results_{timestamp}.csv"
        df = pd.DataFrame([{
            'experiment': r['experiment_name'],
            'success_rate': r['success_rate'],
            'avg_steps': r['avg_steps'],
            'avg_duration': r['avg_duration'],
            'avg_inference_ms': r['avg_inference_time_ms'],
        } for r in results])
        df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument('--config', type=str, help="单模型评估配置文件")
    parser.add_argument('--batch_config', type=str, help="批量评估配置文件")
    
    # 命令行快速评估参数
    parser.add_argument('--policy_path', type=str, help="模型路径")
    parser.add_argument('--policy_type', type=str, default='diffusion')
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=500)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='outputs/eval')
    
    args = parser.parse_args()
    
    if args.batch_config:
        # 批量评估模式
        with open(args.batch_config) as f:
            batch_config = yaml.safe_load(f)
        
        results = run_batch_eval(batch_config)
        save_results(
            results, 
            Path(batch_config['output']['dir']),
            batch_config['output'].get('report_format', 'json')
        )
        
    elif args.config:
        # 配置文件模式
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        result = run_single_eval(config)
        save_results([result], Path(args.output_dir))
        
    elif args.policy_path:
        # 命令行模式
        config = {
            'policy_path': args.policy_path,
            'policy_type': args.policy_type,
            'n_episodes': args.n_episodes,
            'max_steps_per_episode': args.max_steps,
            'device': args.device,
            'output_dir': args.output_dir,
        }
        result = run_single_eval(config)
        save_results([result], Path(args.output_dir))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
```

### 3.3 任务成功判定 (metrics/success_checker.py)
```python
class TaskSuccessChecker:
    """任务成功判定器"""
    
    def __init__(self, config: dict = None):
        self.thresholds = {
            'xy_distance': 0.08,  # 8cm
            'z_distance': 0.05,   # 5cm
            'gripper_closed': 0.3
        }
        if config:
            self.thresholds.update(config)
    
    def check_pick_place(
        self,
        object_pos: np.ndarray,
        target_pos: np.ndarray,
        gripper_state: float
    ) -> Tuple[bool, dict]:
        """检查抓取放置任务是否成功"""
        xy_distance = np.linalg.norm(object_pos[:2] - target_pos[:2])
        z_distance = abs(object_pos[2] - target_pos[2])
        
        success = (
            xy_distance < self.thresholds['xy_distance'] and
            z_distance < self.thresholds['z_distance']
        )
        
        info = {
            'xy_distance': xy_distance,
            'z_distance': z_distance,
            'success': success
        }
        
        return success, info
```

## 4. 使用流程

### 4.1 单模型评估
```bash
# 方式1: 使用配置文件
python batch_eval.py --config config/eval_config.yaml

# 方式2: 命令行参数
python batch_eval.py \
    --policy_path outputs/diffusion_policy/checkpoints/latest.ckpt \
    --policy_type diffusion \
    --n_episodes 20 \
    --device cuda \
    --output_dir outputs/eval/$(date +%Y%m%d)
```

### 4.2 批量评估多个模型
```bash
# 评估多个 checkpoint，比较不同训练阶段的表现
python batch_eval.py --batch_config config/batch_config.yaml
```

### 4.3 服务器后台运行
```bash
# 使用 nohup 后台运行
nohup python batch_eval.py --batch_config config/batch_config.yaml > eval.log 2>&1 &

# 或使用 screen
screen -S eval
python batch_eval.py --batch_config config/batch_config.yaml
# Ctrl+A, D 分离

# 使用 tmux
tmux new -s eval
python batch_eval.py --batch_config config/batch_config.yaml
# Ctrl+B, D 分离
```

### 4.4 输出文件结构
```
outputs/eval/
├── eval_results_20240115_143022.json    # 评估结果汇总
├── eval_results_20240115_143022.csv     # CSV格式（可选）
├── episode_0.mp4                         # 单集视频
├── episode_1.mp4
├── ...
└── trajectories/
    ├── episode_0.pkl                     # 轨迹数据
    ├── episode_1.pkl
    └── ...
```

### 4.5 评估结果格式 (JSON)
```json
[
  {
    "experiment_name": "diffusion_policy_ep100",
    "policy_path": "outputs/diffusion_policy/checkpoints/epoch_100.ckpt",
    "policy_type": "diffusion",
    "n_episodes": 20,
    "success_rate": 0.85,
    "avg_steps": 245.3,
    "avg_duration": 24.5,
    "avg_inference_time_ms": 15.2,
    "elapsed_time": 512.3,
    "episode_results": [
      {"episode_idx": 0, "success": true, "steps": 230, "task": "task1"},
      {"episode_idx": 1, "success": true, "steps": 251, "task": "task2"},
      ...
    ]
  },
  ...
]
```

## 5. 评估指标

### 5.1 主要指标
| 指标 | 说明 | 计算方式 |
|------|------|----------|
| 成功率 | 任务成功比例 | `success_count / n_episodes` |
| 平均步数 | 完成任务的平均步数 | `mean(steps)` |
| 平均时间 | 完成任务的平均时间（秒） | `mean(duration)` |
| 推理延迟 | 每步策略推理时间（毫秒） | `mean(inference_times) * 1000` |

### 5.2 任务成功判定
```python
# Pick-Place 任务
success = (
    xy_distance < 0.08 and  # 物体与目标位置 xy 距离 < 8cm
    z_distance < 0.05       # z 方向偏差 < 5cm
)
```

### 5.3 结果对比示例
```
Experiment                    Success Rate    Avg Steps    Inference (ms)
------------------------------------------------------------------------
diffusion_policy_ep100           75.0%          245.3         15.2
diffusion_policy_ep200           85.0%          232.1         15.4
act_model                        70.0%          268.5          8.3
```

## 7. 与现有代码的集成

### 7.1 复用模块
| 模块 | 来源 | 用途 |
|------|------|------|
| SimuInterface | `collect_data/scripts/simu_interface.py` | 参考 MuJoCo 初始化 |
| 配置系统 | `collect_data/config/` | 任务配置格式 |
| 坐标变换 | `collect_data/main_qt.py` | `_transform_position` |

### 7.2 数据格式兼容
- 评估观测格式与训练数据格式一致（LeRobot v3.0）
- 支持多相机：top, side, hand
- 状态维度：6 joints + 1 gripper

## 8. 实现计划

### Phase 1: 基础框架 (1天)
- [x] 设计文档
- [ ] 目录结构创建
- [ ] 配置类实现
- [ ] 日志工具

### Phase 2: 仿真环境 (1-2天)
- [ ] SimuEvalEnv 实现（无头模式）
- [ ] MuJoCo 离屏渲染
- [ ] 视频保存功能

### Phase 3: 策略封装 (1天)
- [ ] Diffusion Policy 封装
- [ ] LeRobot Policy 封装
- [ ] 观测格式转换

### Phase 4: 批量评估 (1天)
- [ ] EvalRunner 实现
- [ ] batch_eval.py 脚本
- [ ] 结果保存（JSON/CSV）

### Phase 5: 测试验证 (1天)
- [ ] 单元测试
- [ ] 端到端测试
- [ ] 服务器环境测试

## 9. 服务器环境注意事项

### 9.1 无头渲染
```bash
# 确保环境变量正确
export MUJOCO_GL=egl  # 或 osmesa

# 检查 EGL 支持
python -c "import mujoco; print(mujoco.mj_version_string())"
```

### 9.2 GPU 使用
```bash
# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python batch_eval.py --batch_config ...

# CPU 回退
python batch_eval.py --device cpu --batch_config ...
```

### 9.3 资源管理
- 每个 episode 后清理 MuJoCo 资源
- 大批量评估时分批加载模型
- 视频保存使用 H.264 压缩减少存储

### 9.4 错误处理
- 自动重试失败的 episode
- 记录错误日志
- 中断后可从上次进度继续

## 10. 扩展功能（可选）

### 10.1 WandB 集成
```python
import wandb

# 初始化
wandb.init(project="policy-eval", config=config)

# 记录指标
wandb.log({
    "success_rate": report['success_rate'],
    "avg_steps": report['avg_steps'],
})

# 结束
wandb.finish()
```

### 10.2 分布式评估
```python
# 使用 Ray 并行评估
import ray

@ray.remote(num_gpus=0.25)
def eval_checkpoint(policy_path):
    # 评估逻辑
    pass

# 并行运行
results = ray.get([
    eval_checkpoint.remote(path) 
    for path in policy_paths
])
```
