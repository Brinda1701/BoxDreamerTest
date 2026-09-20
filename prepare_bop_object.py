#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_bop_object.py
=====================
【步骤 1 规范化工具：构建国际标准 BOP 目标模型目录】

核心目的：
将预处理或烘焙好的 3D 模型转换为 BOP (Benchmark for 6D Object Pose) 国际标准数据结构：

    bop_datasets/dji/
    |-- camera.json            # 统一的标准相机内参规范文件
    |-- models/
        |-- models_info.json   # 目标 3D 尺寸、外接包围盒极值以及 3D 直径 (Diameter)
        |-- obj_000001.ply     # 统一为毫米 (mm) 单位且中心居中的模型 PLY

关键知识点说明：
1. 【BOP 尺度协议】：
   BOP 标准数据集约定 3D CAD 模型以及平移向量 t 统一使用【毫米 (mm)】为物理单位；
   而前面几何处理阶段为了方便 OpenCV/PyTorch 通常用【米 (m)】，故此脚本负责进行 1000x 等比转换。
2. 【3D 模型直径 (Diameter)】：
   三维模型表面任意两点间的最大欧氏距离。
   该数值是 6D 位姿估计核心评价指标 ADD / ADD-S (例如 0.1d 精度阈值) 的归一化基准！
3. 【models_info.json】：
   供位姿估计网络、评估工具 (bop_toolkit) 以及 BlenderProc 渲染器解析物体几何边界。
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

# 保证控制台正常输出中文
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 默认相机内参 (与实测头戴/胸前相机 640x480 分辨率保持一致)
# depth_scale = 0.1 表示深度图每个灰度值对应 0.1 毫米 (即 10 对应 1 毫米)
# ---------------------------------------------------------------------------
CAMERA_JSON_DEFAULTS = {
    "cx": 325.2611083984375,
    "cy": 242.04899588216654,
    "depth_scale": 0.1,
    "fx": 572.411363389757,
    "fy": 573.5704328585578,
    "height": 480,
    "width": 640,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="生成符合 BOP 标准的物体模型文件夹与配置文件。"
    )
    parser.add_argument(
        "--input_obj",
        type=str,
        default=os.path.join(SCRIPT_DIR, "dji_action4_centered.obj"),
        help="输入三维模型路径 (可传入米制 OBJ 或烘焙完成的 PLY)。",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "bop_datasets", "dji"),
        help="输出 BOP 数据集根目录 (默认为 bop_datasets/dji)。",
    )
    return parser.parse_args()


def load_and_normalize_mesh(obj_path):
    """加载模型并规范化缩放到 BOP 毫米 (mm) 单位，中心对齐原点。

    处理逻辑：
    - 若输入模型为米单位 (max_extent < 1.0，如 0.0705m)，自动等比缩放 1000 倍为毫米；
    - 若已经是毫米单位 (max_extent 处于 1~200mm 之间)，保持不变；
    - 重新校准模型几何中心至 (0, 0, 0)，确保 BlenderProc 相机球面采样基准稳定。
    """
    if not os.path.isfile(obj_path):
        raise FileNotFoundError("找不到输入模型: %s" % obj_path)

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if mesh is None or len(mesh.vertices) == 0:
        raise ValueError("模型文件无有效三角面网格: %s" % obj_path)

    extents_m = np.asarray(mesh.extents)
    if extents_m.max() < 1.0:
        # 米 -> 毫米
        mesh.apply_scale(1000.0)
        print("检测到输入单位为米 (m)，已等比缩放到 BOP 毫米 (mm) 单位 (x1000)。")
    elif extents_m.max() <= 200.0:
        print("检测到输入模型已为毫米 (mm) 单位，无需尺度缩放。")
    else:
        raise ValueError(
            "异常的模型尺度: 最大边长 = %.2f。预期应为米 (~0.07) 或毫米 (~70)。" % extents_m.max()
        )

    # 保证 CAD 中心处于世界原点
    mesh.apply_translation(-np.asarray(mesh.centroid))
    return mesh


def estimate_diameter(mesh, max_samples=8192, block_size=512, seed=0):
    """精确计算物体的 3D 直径 (模型表面任意两点间的最大欧氏距离)。

    算法原理：
    1. 快速随机采样 8192 个点进行分块距离矩阵粗算；
    2. 结合最远点采样 (Farthest-point refinement) 迭代迭代逼近全网格顶点的全局最大直径；
    3. 该直径是 6D 位姿评估中 ADD-S 误差度量的重要基准阈值 (如 0.1 * diameter)。
    """
    pts = np.asarray(mesh.vertices, dtype=np.float64)
    num_pts = len(pts)
    if num_pts < 2:
        raise ValueError("网格至少需要 2 个顶点才能计算直径。")

    rng = np.random.default_rng(seed)
    if num_pts > max_samples:
        sample_idx = rng.choice(num_pts, size=max_samples, replace=False)
        sample = pts[sample_idx]
    else:
        sample = pts

    sample32 = sample.astype(np.float32)
    best = 0.0
    for start in range(0, len(sample32), block_size):
        block = sample32[start:start + block_size]
        diff = block[:, None, :] - sample32[None, :, :]
        dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
        best = max(best, float(np.sqrt(dist_sq).max()))

    # 最远点迭代优化 (迭代 8 次，逼近真实 CAD 直径)
    anchor = pts[int(np.argmax(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))]
    for _ in range(8):
        dist_sq = np.einsum("ij,ij->i", pts - anchor, pts - anchor)
        best = max(best, float(np.sqrt(dist_sq).max()))
        anchor = pts[int(np.argmax(dist_sq))]
    return best


def build_models_info(mesh):
    """计算生成 BOP 标准的 models_info.json 字典条目 (object id = 1)。

    包含：
    - diameter: 物体三维直径 (mm)
    - min_x, min_y, min_z: 局部外接包围盒最小角点坐标 (mm)
    - size_x, size_y, size_z: 三个轴向的几何长宽高尺寸 (mm)
    """
    bounds = np.asarray(mesh.bounds)  # (min_xyz, max_xyz)，单位 mm
    size = bounds[1] - bounds[0]
    diameter = estimate_diameter(mesh)

    info = {
        "1": {
            "diameter": round(float(diameter), 4),
            "min_x": round(float(bounds[0, 0]), 4),
            "min_y": round(float(bounds[0, 1]), 4),
            "min_z": round(float(bounds[0, 2]), 4),
            "size_x": round(float(size[0]), 4),
            "size_y": round(float(size[1]), 4),
            "size_z": round(float(size[2]), 4),
        }
    }

    # 合理性自检：三维直径绝不可能小于物体在任一单轴上的最大尺寸
    if info["1"]["diameter"] < 0.999 * float(size.max()):
        raise RuntimeError(
            "估计的直径 (%.4f) 小于单轴最大尺寸 (%.4f)，计算异常。"
            % (info["1"]["diameter"], size.max())
        )
    return info


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("Wrote %s" % path)


def main():
    args = parse_args()

    obj_path = os.path.abspath(args.input_obj)
    dataset_dir = os.path.abspath(args.dataset_dir)
    models_dir = os.path.join(dataset_dir, "models")

    print("Loading %s ..." % obj_path)
    mesh_mm = load_and_normalize_mesh(obj_path)

    extents_mm = np.asarray(mesh_mm.extents)
    print(
        "Scaled extents (mm): x=%.3f y=%.3f z=%.3f"
        % (extents_mm[0], extents_mm[1], extents_mm[2])
    )

    ply_path = os.path.join(models_dir, "obj_000001.ply")
    os.makedirs(models_dir, exist_ok=True)
    mesh_mm.export(ply_path)
    print("Exported %s" % ply_path)

    info = build_models_info(mesh_mm)
    write_json(os.path.join(models_dir, "models_info.json"), info)
    write_json(os.path.join(dataset_dir, "camera.json"), CAMERA_JSON_DEFAULTS)

    # Round-trip validation of the exported PLY.
    ply_mesh = trimesh.load(ply_path, force="mesh", process=False)
    ply_extents = np.asarray(ply_mesh.extents)
    if np.abs(ply_extents - extents_mm).max() > 0.01:
        raise RuntimeError("PLY export changed the model extents unexpectedly.")

    print("\n--- BOP preparation finished ---")
    print("Dataset root : %s" % dataset_dir)
    print("Model        : %s" % ply_path)
    print(
        "BBox (mm)    : min=[%.3f, %.3f, %.3f] size=[%.3f, %.3f, %.3f]"
        % (
            info["1"]["min_x"],
            info["1"]["min_y"],
            info["1"]["min_z"],
            info["1"]["size_x"],
            info["1"]["size_y"],
            info["1"]["size_z"],
        )
    )
    print("Diameter (mm): %.4f" % info["1"]["diameter"])


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass
    main()
