#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""texture_a2_bake_front.py

路线 A 第 2 步: 用正视角照片(主屏面 4.png、镜头面 5.png)烘焙顶点颜色。

原理:
  1) CAD 的 X=宽(70.5mm)、Y=深(32.4mm)、Z=高(44.5mm);
     自动分析顶点法线后判断: 镜头面在 -Y 侧, 主屏面在 +Y 侧。
  2) 用户点选的 4 个“假想直角”作为照片里机身外轮廓矩形;
     它们对应 CAD 该侧平面的 4 个角点, 用最小二乘仿射变换
     把 CAD 的 (X, Z) 坐标映射到照片像素 (u, v)。
  3) 每个顶点按“离镜头面/主屏面的距离”混合两张照片的颜色,
     侧面与过渡区平滑过渡到程序化深灰, 避免出现明显断层。

输出: 带每顶点 RGB 颜色的 PLY, 可直接被 run_official_pbr.py 加载渲染。

用法:
  python texture_a2_bake_front.py \
      --ply bop_datasets/dji/models/obj_000001.ply \
      --corners work/corners.json \
      --images work/DJI_alpha/5.png work/DJI_alpha/4.png \
      --out work/obj_000001_colored.ply
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh


def parse_args():
    parser = argparse.ArgumentParser(description="把正面照片烘焙成模型顶点颜色。")
    parser.add_argument("--ply", type=str, required=True, help="BOP 模型 PLY(mm)。")
    parser.add_argument("--corners", type=str, required=True, help="角点 JSON。")
    parser.add_argument(
        "--images",
        nargs=2,
        required=True,
        help="依次给镜头面照片与主屏面照片(透明 PNG)。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="work/obj_000001_colored.ply",
        help="输出带顶点颜色的 PLY。",
    )
    parser.add_argument(
        "--lens_mirror",
        action="store_true",
        help="若预览发现镜头面左右镜像, 加上该参数重跑。",
    )
    parser.add_argument(
        "--screen_mirror",
        action="store_true",
        help="若预览发现主屏面左右镜像, 加上该参数重跑。",
    )
    return parser.parse_args()


def smoothstep(edge0, edge1, x):
    """0~1 平滑过渡函数, 用于侧面颜色淡出。"""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_matte_albedo(rgb):
    """把产品图压成哑光反照率。

    产品渲染图里黑色机身常带大面积白色摄影棚反光, 直接烘进贴图会让模型
    变成银白色。这里用 18 + 0.35*color 的曲线压暗: 纯黑保持近黑(约18),
    白色高光最多降到约107, 保留细节但不再刺眼。
    """
    return 18.0 + 0.5 * rgb


def load_corner_map(corners_path, image_name):
    """读取某个文件名的 4 个角点, 顺序固定为左上右上右下左下。"""
    with open(corners_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = data.get(image_name)
    if rec is None:
        raise KeyError("角点 JSON 中没有 %s" % image_name)
    return np.asarray([[c["x"], c["y"]] for c in rec["corners"]], dtype=np.float64)


def estimate_affine(plane_corners_xz, image_corners_uv):
    """最小二乘求 (X,Z)->(u,v) 的 2x3 仿射矩阵。"""
    affine, _ = cv2.estimateAffine2D(
        plane_corners_xz.astype(np.float32),
        image_corners_uv.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=8.0,
    )
    if affine is None:
        # 四点共面不可能失败, 失败则退化为三点精确仿射。
        affine = cv2.getAffineTransform(
            plane_corners_xz[:3].astype(np.float32),
            image_corners_uv[:3].astype(np.float32),
        )
    return affine.astype(np.float64)


def sample_image(img_rgba, plane_xz, affine):
    """把 (X,Z) 顶点按仿射矩阵映射到图像并双线性采样。"""
    ones = np.ones((len(plane_xz), 1), dtype=np.float64)
    uv = np.hstack([plane_xz, ones]) @ affine.T  # (N,2)
    u = uv[:, 0]
    v = uv[:, 1]
    h, w = img_rgba.shape[:2]

    valid = (u >= 0.5) & (u < w - 0.5) & (v >= 0.5) & (v < h - 0.5)
    rgb = np.full((len(plane_xz), 3), 90.0, dtype=np.float64)
    alpha = np.zeros(len(plane_xz), dtype=np.float64)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return rgb, alpha

    uu = u[idx]
    vv = v[idx]
    x0 = np.floor(uu).astype(int)
    y0 = np.floor(vv).astype(int)
    fx = uu - x0
    fy = vv - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    def gather(r, c, ch):
        return img_rgba[r, c, ch].astype(np.float64)

    for ch in range(3):
        top = gather(y0, x0, ch) * (1.0 - fx) + gather(y0, x1, ch) * fx
        bottom = gather(y1, x0, ch) * (1.0 - fx) + gather(y1, x1, ch) * fx
        rgb[idx, ch] = top * (1.0 - fy) + bottom * fy

    a_top = gather(y0, x0, 3) * (1.0 - fx) + gather(y0, x1, 3) * fx
    a_bottom = gather(y1, x0, 3) * (1.0 - fx) + gather(y1, x1, 3) * fx
    alpha[idx] = a_top * (1.0 - fy) + a_bottom * fy
    return rgb, alpha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    mesh = trimesh.load(args.ply, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
    zmin, zmax = verts[:, 2].min(), verts[:, 2].max()
    x_half = (xmax - xmin) / 2.0
    z_half = (zmax - zmin) / 2.0
    y_lens = float(verts[:, 1].min())   # 镜头面侧(-Y)
    y_screen = float(verts[:, 1].max()) # 主屏面侧(+Y)

    img_lens = cv2.imread(args.images[0], cv2.IMREAD_UNCHANGED)
    img_screen = cv2.imread(args.images[1], cv2.IMREAD_UNCHANGED)
    if img_lens is None or img_screen is None:
        raise FileNotFoundError("请先运行 texture_a0_prep_photos.py 生成透明 PNG。")
    if img_lens.shape[2] == 3:
        img_lens = np.dstack([img_lens, np.full(img_lens.shape[:2], 255, np.uint8)])
    if img_screen.shape[2] == 3:
        img_screen = np.dstack([img_screen, np.full(img_screen.shape[:2], 255, np.uint8)])
    # BGR -> RGB(交换第 0/2 通道)。
    img_lens[:, :, [0, 2]] = img_lens[:, :, [2, 0]]
    img_screen[:, :, [0, 2]] = img_screen[:, :, [2, 0]]

    lens_corners_img = load_corner_map(args.corners, "5.png")
    screen_corners_img = load_corner_map(args.corners, "4.png")

    # 平面矩形角点顺序: 左上、右上、右下、左下。
    x = np.array([-x_half, x_half, x_half, -x_half])
    z = np.array([z_half, z_half, -z_half, -z_half])
    if args.lens_mirror:
        x = -x
    lens_plane = np.column_stack([x, z])

    x = np.array([x_half, -x_half, -x_half, x_half])
    z = np.array([z_half, z_half, -z_half, -z_half])
    if args.screen_mirror:
        x = -x
    screen_plane = np.column_stack([x, z])

    h_lens = estimate_affine(lens_plane, lens_corners_img)
    h_screen = estimate_affine(screen_plane, screen_corners_img)
    print("镜头面仿射矩阵:\n%s" % np.round(h_lens, 4))
    print("主屏面仿射矩阵:\n%s" % np.round(h_screen, 4))

    plane_xz = verts[:, [0, 2]]
    rgb_lens, a_lens = sample_image(img_lens, plane_xz, h_lens)
    rgb_screen, a_screen = sample_image(img_screen, plane_xz, h_screen)
    rgb_lens = to_matte_albedo(rgb_lens)
    rgb_screen = to_matte_albedo(rgb_screen)

    # 离两个面的距离越远, 照片权重越低, 平滑过渡到深灰。
    d_lens = np.abs(verts[:, 1] - y_lens)
    d_screen = np.abs(verts[:, 1] - y_screen)
    w_lens = 1.0 - smoothstep(1.0, 7.0, d_lens)
    w_screen = 1.0 - smoothstep(1.0, 7.0, d_screen)
    a_lens *= w_lens
    a_screen *= w_screen
    # alpha 范围是 0~255, 先归一化到 0~1 再当权重使用。
    a_lens /= 255.0
    a_screen /= 255.0

    # 侧面/顶底暂用接近机身黑的深色(约 28/255)填充;
    # 若后续用 1-3 号斜视角补侧面, 这里会被照片颜色替代。
    gray = np.array([20.0, 20.0, 20.0])
    color = np.tile(gray, (len(verts), 1))
    weight = np.zeros(len(verts))
    color += a_lens[:, None] * (rgb_lens - gray)
    weight += a_lens
    color += a_screen[:, None] * (rgb_screen - gray)
    weight += a_screen
    # 透明照片区域/距离过远时自然回落为灰色。
    colors = np.clip(color, 0, 255).astype(np.uint8)

    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    # 统计: 有多少顶点拿到主要照片颜色(权重 > 0.4)。
    covered = int((weight > 0.4).sum())
    print("输出: %s" % out_path)
    print("顶点总数: %d, 被照片覆盖(权重>0.4): %d (%.1f%%)"
          % (len(verts), covered, 100.0 * covered / len(verts)))
    print("若预览中文字/屏幕左右镜像, 重跑时加 --lens_mirror / --screen_mirror。")


if __name__ == "__main__":
    main()
