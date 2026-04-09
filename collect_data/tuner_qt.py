"""
物体参数标定独立入口（与 main_qt.py 分离）
"""
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from PyQt5.QtWidgets import QApplication
from gui import ObjectProfileTunerWindow
from main_qt import DataCollectionSystem


def main():
    script_dir = Path(__file__).parent
    config_path = str(script_dir / "config" / "tuner_minimal.yaml")
    use_real = False

    print("启动模式: 模拟模式（配置已写死）")
    print(f"配置文件: {config_path}")

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    system = DataCollectionSystem(config_path, use_real=use_real, show_simu_viewer=False)
    if not system.initialize():
        print("[ERROR] 系统初始化失败")
        return 1

    tuner_window = ObjectProfileTunerWindow(
        data_system=system,
        config_path=config_path,
        default_task_id=None,
        default_object_name=None,
    )
    tuner_window.show()

    app.aboutToQuit.connect(system.cleanup)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
