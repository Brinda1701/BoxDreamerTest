# -*- coding: utf-8 -*-
"""
process_dji_model.py
====================
【步骤 1 基础工具：模型几何规范化与 3D 角点提取】

核心目的：
1. 尺度检测与统一：若开源 CAD 模型单位为毫米 (mm)，自动缩放到计算机视觉算法通用的标准单位“米 (m)”；
2. 几何中心归零：将网格包围盒中心严格平移对齐到局部坐标系原点 (0, 0, 0)，避免后续旋转与位姿估计产生几何偏差；
3. 提取 3D Bounding Box 8 角点：提取外接包围盒的 8 个顶点局部三维坐标 (8x3) 并导出为 dji_bbox_corners.npy；
4. 导出居中后的 OBJ 模型：供后续 BlenderProc 渲染、纹理烘焙以及坐标变换加载。
"""

import os
import sys
import numpy as np
import trimesh

# 保证在 Windows 控制台环境下中文输出正常
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. 寻找并加载原始 STL 模型
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# 优先在当前目录查找，若不存在则在 step-1 原始资产子目录下查找
stl_path = os.path.join(script_dir, "Osmo_Action_4.stl")
if not os.path.exists(stl_path):
    stl_path_step1 = os.path.join(script_dir, "step-1", "Osmo_Action_4.stl")
    if os.path.exists(stl_path_step1):
        stl_path = stl_path_step1
    else:
        raise FileNotFoundError(f"未找到原始 STL 文件: {stl_path} 或 {stl_path_step1}")

print(f"正在加载 CAD 模型: {stl_path}")
mesh = trimesh.load(stl_path, force="mesh")
print("原始模型尺寸 (长宽高范围):", mesh.extents)

# ---------------------------------------------------------------------------
# 2. 检查并校正几何单位 (CAD 通常以毫米 mm 为单位)
# ---------------------------------------------------------------------------
# 深度学习位姿估计与相机投影坐标系中，主流标准单位通常是米 (m)。
# DJI Action 4 实际外形尺寸约为 70.5mm x 32.4mm x 44.5mm。
# 如果最大边长 > 1.0 (如 70.5)，判定当前为毫米单位，需乘以 0.001 缩放为米。
if np.max(mesh.extents) > 1.0:
    print(">> 检测到模型单位为毫米 (mm)，正在自动缩放到标准单位米 (m)...")
    mesh.apply_scale(0.001)

# ---------------------------------------------------------------------------
# 3. 将模型几何中心平移对齐到局部坐标系原点 (0, 0, 0)
# ---------------------------------------------------------------------------
# 让物体的几何对称中心与旋转中心重合，消除任意位姿采样时的刚体偏心距离
print(f"原始几何中心: {mesh.centroid}")
mesh.apply_translation(-mesh.centroid)
print(f"平移归零后几何中心: {mesh.centroid} (接近 [0, 0, 0])")

# ---------------------------------------------------------------------------
# 4. 获取紧致轴对齐外接包围盒 (3D Bounding Box) 与 8 个角点
# ---------------------------------------------------------------------------
# bbox.vertices 返回形状为 (8, 3) 的三维局部坐标数组
bbox = mesh.bounding_box
bbox_corners = np.asarray(bbox.vertices, dtype=np.float32)  # shape: (8, 3)，单位：米

# ---------------------------------------------------------------------------
# 5. 保存 8 个 3D 角点坐标真值
# ---------------------------------------------------------------------------
# 该文件是关键点检测（步骤 3）与位姿解算（步骤 5）的核心真值对照表：
# 后续在已知预测 2D 像素坐标的情况下，配合这 8 个局部 3D 坐标即可调用 OpenCV PnP 恢复物体 6D 位姿。
corners_save_path = os.path.join(script_dir, "dji_bbox_corners.npy")
np.save(corners_save_path, bbox_corners)

# ---------------------------------------------------------------------------
# 6. 保存规范化后的居中 OBJ 模型
# ---------------------------------------------------------------------------
obj_save_path = os.path.join(script_dir, "dji_action4_centered.obj")
mesh.export(obj_save_path)

print("\n=================== 预处理完成 ===================")
print("校正后模型实际尺寸 (米):", mesh.extents)
print("模型最终中心坐标:", mesh.centroid)
print("8 个 3D 角点局部基准坐标 (shape: 8x3, 单位: 米):\n", bbox_corners)
print(f"已生成文件 1: {obj_save_path}")
print(f"已生成文件 2: {corners_save_path}")
print("==================================================")