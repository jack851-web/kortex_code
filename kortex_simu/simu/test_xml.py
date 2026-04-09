#!/usr/bin/env python
"""
MuJoCo XML 模型加载和可视化测试脚本
用于加载 gen3_lite_gen3_lite_2f.xml 文件并通过界面手动控制关节
"""

import mujoco
import mujoco.viewer
import numpy as np

def main():
    # XML 文件路径
    xml_path = r"D:\VLA\kortex_lerobot\kortex_simu\simu\robot\gen3_lite_gen3_lite_2f.xml"
    
    print("正在加载 MuJoCo 模型...")
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 打印模型基本信息
    print(f"\n关节数量: {model.njnt}")
    print(f"执行器数量: {model.nu}")
    print(f"传感器数量: {model.nsensor}")
    
    # 打印关节信息
    print("\n关节列表:")
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  J{i}: {jnt_name}")
    
    # 打印传感器信息
    if model.nsensor > 0:
        print("\n传感器列表:")
        for i in range(model.nsensor):
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            print(f"  Sensor {i}: {sensor_name}")
    
    print("\n启动 MuJoCo 查看器...")
    print("提示: 在查看器界面中，可以通过拖动滑块手动控制各个关节")
    
    # 启动交互式查看器
    # 用户可以通过界面手动控制关节
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()
