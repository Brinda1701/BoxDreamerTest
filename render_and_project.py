# -*- coding: utf-8 -*-
"""
render_and_project.py
=====================
【步骤 2/3 核心验证单元：3D 目标渲染与 8 角点 2D 像素投影】

核心目的：
1. 演示并验证“3D 几何坐标 -> 相机坐标系 -> 2D 像素坐标”的标准针孔投影数学模型；
2. 验证 dji_bbox_corners.npy 中的 8 个 3D 角点经相机内参 K 投影后，能否与渲染图中的物体外轮廓像素严格对齐；
3. 输出带 3D 绿色立体包围盒连线的验证图片 (render_0000_bbox.png)，供肉眼直接检验几何真值的绝对准确性；
4. 此文件中的投影算法是后续步骤 3/4 中“生成 8 通道高斯 Heatmap 训练标签”的直接代码模板！

运行方式 (必须在 BlenderProc 环境下运行):
    blenderproc run render_and_project.py
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import blenderproc as bproc
import bpy

# 保证控制台正常输出中文
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 1. 初始化 BlenderProc 物理渲染管线
# ---------------------------------------------------------------------------
bproc.init()

# ---------------------------------------------------------------------------
# 2. 导入大疆相机规范模型并设置材质
# ---------------------------------------------------------------------------
# 优先加载包含精细贴图的模型，若无则回退至居中基础模型
refined_obj = SCRIPT_DIR / "dji_action4_refined.obj"
default_obj = SCRIPT_DIR / "dji_action4_centered.obj"
obj_path = refined_obj if refined_obj.exists() else default_obj
if not obj_path.exists():
    raise FileNotFoundError(f"找不到模型文件: {obj_path}，请先准备模型文件")

print(f"正在加载相机模型: {obj_path}")
obj = bproc.loader.load_obj(str(obj_path))[0]
obj.set_shading_mode("auto")

# 若模型本身无内嵌材质槽，才赋予深灰/哑光黑色兜底材质
if not obj.get_materials():
    dji_mat = bproc.material.create("dji_body")
    dji_mat.set_principled_shader_value("Base Color", [0.05, 0.05, 0.05, 1.0])
    dji_mat.set_principled_shader_value("Roughness", 0.45)
    obj.replace_materials(dji_mat)
else:
    print(f"保留模型自带材质槽: {[m.get_name() for m in obj.get_materials()]}")

# ---------------------------------------------------------------------------
# 3. 创建背景衬衫/桌面平板并赋予布料贴图
# ---------------------------------------------------------------------------
# 创建一个 1m x 1m 的平面作为衬托桌面，放置在相机下方 5cm 处
bg_plane = bproc.object.create_primitive("PLANE", scale=[1.0, 1.0, 1.0])
bg_plane.set_location([0.0, 0.0, -0.05])
bg_mat = bproc.material.create("bg_cloth")

cloth_tex_path = SCRIPT_DIR / "resources" / "textures" / "denmin_fabric_02_diff_1k.jpg"
if cloth_tex_path.exists():
    cloth_img = bpy.data.images.load(str(cloth_tex_path), check_existing=True)
    bg_mat.set_principled_shader_value("Base Color", cloth_img)
bg_plane.replace_materials(bg_mat)

# ---------------------------------------------------------------------------
# 4. 配置环境光照 (HDRI 或备用点光源)
# ---------------------------------------------------------------------------
hdr_path = SCRIPT_DIR / "resources" / "hdris" / "small_empty_room_3_1k.hdr"
if hdr_path.exists():
    bproc.lighting.set_world_background_hdr_img(str(hdr_path))
else:
    # 若无 HDRI，创建一盏斜上方点光源作为兜底照明
    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([0.3, -0.3, 0.6])
    light.set_energy(150)

# ---------------------------------------------------------------------------
# 5. 配置相机参数与位姿 (Camera Setup)
# ---------------------------------------------------------------------------
# 图像分辨率
H, W = 480, 640
bproc.camera.set_resolution(W, H)

# 相机内参矩阵 K (3x3):
# [[fx,  0, cx],
#  [ 0, fy, cy],
#  [ 0,  0,  1]]
K = np.array([
    [550.0,   0.0, 320.0],
    [  0.0, 550.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float64)
bproc.camera.set_intrinsics_from_K_matrix(K, W, H)

# 设定相机空间位置：位于物体斜上方 (x=0.1m, y=-0.35m, z=0.3m)
cam_pos = np.array([0.1, -0.35, 0.3], dtype=np.float64)
# 光轴朝向：从相机位置指向坐标原点 (物体中心)
forward_dir = -cam_pos / np.linalg.norm(cam_pos)
cam_rot = bproc.camera.rotation_from_forward_vec(forward_dir)
# 构造 4x4 外参位姿矩阵 (Camera-to-World, 即世界系下的相机位置与旋转)
cam_pose = bproc.math.build_transformation_mat(cam_pos, cam_rot)
bproc.camera.add_camera_pose(cam_pose)

# ---------------------------------------------------------------------------
# 6. 计算 8 个角点的 2D 像素投影坐标 (核心数学公式推导)
# ---------------------------------------------------------------------------
corners_3d_path = SCRIPT_DIR / "dji_bbox_corners.npy"
if not corners_3d_path.exists():
    raise FileNotFoundError(f"找不到角点文件: {corners_3d_path}，请先运行 process_dji_model.py")

corners_3d = np.load(str(corners_3d_path))  # shape: (8, 3)，单位：米

# -----------------------------------------------------------------------
# 【核心投影步骤 A】: 世界/物体坐标系 -> 相机观察坐标系 (World2Cam)
# -----------------------------------------------------------------------
# cam_pose 是 Cam2World (从相机系到世界系)，其逆矩阵即为 World2Cam (从世界系到相机系)
world2cam = np.linalg.inv(cam_pose)

# 将 (8, 3) 扩充为齐次坐标 (8, 4)
corners_3d_homo = np.hstack([corners_3d, np.ones((8, 1), dtype=np.float64)])

# 刚体坐标变换: P_cam = T_w2c * P_world
corners_cam = (world2cam @ corners_3d_homo.T).T[:, :3]

# -----------------------------------------------------------------------
# 【核心投影步骤 B】: 针孔相机透视投影 (Pinhole Projection)
# -----------------------------------------------------------------------
# 根据相机成像模型:
#   s * [u, v, 1]^T = K * [X_cam, Y_cam, Z_cam]^T
# 其中 s = Z_cam (深度)，消去 s 得到像素坐标:
#   u = fx * (X_cam / Z_cam) + cx
#   v = fy * (Y_cam / Z_cam) + cy
corners_2d = []
for pt in corners_cam:
    p_proj = K @ pt
    u = p_proj[0] / p_proj[2]
    v = p_proj[1] / p_proj[2]
    corners_2d.append([u, v])
corners_2d = np.array(corners_2d, dtype=np.float32)

# ---------------------------------------------------------------------------
# 7. 物理光线追踪渲染并保存输出
# ---------------------------------------------------------------------------
output_dir = SCRIPT_DIR / "archive" / "render_project_test"
output_dir.mkdir(parents=True, exist_ok=True)

print("正在调用 Cycles 渲染器生成图像...")
data = bproc.renderer.render()
rgb_img = data["colors"][0]  # shape: (480, 640, 3)
bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

# 保存纯净的渲染原图
raw_render_path = output_dir / "render_0000.png"
cv2.imwrite(str(raw_render_path), bgr_img)

# 保存计算出的 8 个 2D 角点真值坐标 (8, 2)
corners_2d_path = output_dir / "corners_2d_0000.npy"
np.save(str(corners_2d_path), corners_2d)

# ---------------------------------------------------------------------------
# 8. 绘制 3D 立体包围盒 (Wireframe Bounding Box) 进行视觉对齐核验
# ---------------------------------------------------------------------------
# 将 8 个 2D 投影点绘制在图像上，并连成 12 条立方体棱边。
# 如果投影数学公式完全正确，绿色的立方体线框将严丝合缝地贴合在大疆相机的机身外边缘！
vis_img = bgr_img.copy()

# 绘制 8 个角点 (实心红圈)
for idx, (cu, cv) in enumerate(corners_2d):
    pt = (int(round(cu)), int(round(cv)))
    cv2.circle(vis_img, pt, radius=4, color=(0, 0, 255), thickness=-1)
    cv2.putText(vis_img, str(idx), (pt[0] + 5, pt[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

# 立方体 12 条棱边的顶点索引连接对 (基于轴对齐包围盒顶点的拓扑连接)
# trimesh.bounding_box 导出的 8 个顶点顺序构成的 12 条棱
from scipy.spatial import distance_matrix
# 自动按 3D 几何距离最小的边构建 12 条棱边
dist_mat = distance_matrix(corners_3d, corners_3d)
np.fill_diagonal(dist_mat, np.inf)
edges = set()
for i in range(8):
    # 每个顶点连接距离最近的 3 个正交相邻顶点
    nearest_3 = np.argsort(dist_mat[i])[:3]
    for j in nearest_3:
        edge = tuple(sorted((i, j)))
        edges.add(edge)

# 在图像上绘制绿色立体线框 (厚度 2 像素)
for i, j in edges:
    p1 = (int(round(corners_2d[i][0])), int(round(corners_2d[i][1])))
    p2 = (int(round(corners_2d[j][0])), int(round(corners_2d[j][1])))
    cv2.line(vis_img, p1, p2, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

bbox_vis_path = output_dir / "render_0000_bbox.png"
cv2.imwrite(str(bbox_vis_path), vis_img)

print("\n=================== 渲染与真值投影验证完成 ===================")
print(f"1. 纯净渲染图像:   {raw_render_path}")
print(f"2. 2D 角点坐标文件: {corners_2d_path}")
print(f"3. 3D包围盒验证图: {bbox_vis_path} (打开此图可肉眼确认 8 角点贴合精度)")
print("\n8 个投影到图像上的 2D 像素坐标 (u, v):\n", corners_2d)
print("============================================================")

