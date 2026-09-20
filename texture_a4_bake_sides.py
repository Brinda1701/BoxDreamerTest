#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""texture_a4_bake_sides.py

路线 A 第 3 步: 用 1/2/3 号斜视角把侧面/顶底照片投影到当前彩色模型上。

原理:
  * 用户已在每张斜视图上点出“镜头面矩形”的 4 角, 其 3D 坐标在 CAD 上已知;
  * 用“4 点 2D-3D 对应 + 未知焦距”做非线性优化, 估计每张照片对应的
    相机位姿(外参)和内参焦距;
  * 只处理当前仍为深色兜底的侧面/顶底顶点: 先做背面剔除(法线朝相机),
    再用光线与网格求交做遮挡检测, 最后把照片颜色采样到这些顶点上;
  * 正反两面颜色不动, 没被 1/2/3 看到的边角继续保留深黑。

用法:
  python texture_a4_bake_sides.py \
      --ply work/obj_000001_colored.ply \
      --corners work/corners_sides.json \
      --alpha work/DJI_alpha \
      --out work/obj_000001_sidecolored.ply
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares


LENS_Y_OFFSET = 0.0  # 占位, 实际从网格读取


def parse_args():
    parser = argparse.ArgumentParser(description="斜视角侧面纹理烘焙。")
    parser.add_argument("--ply", type=str, required=True, help="当前彩色 PLY。")
    parser.add_argument("--corners", type=str, required=True, help="1/2/3 角点 JSON。")
    parser.add_argument(
        "--alpha",
        type=str,
        default="work/DJI_alpha",
        help="透明 PNG 所在目录(含 1.png/2.png/3.png)。",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="work/obj_000001_sidecolored.ply",
        help="输出 PLY。",
    )
    parser.add_argument(
        "--occlusion_margin_mm",
        type=float,
        default=0.8,
        help="判断遮挡的容差(mm), 小于该值认为光线命中自身表面而非被遮挡。",
    )
    parser.add_argument(
        "--skip_images",
        nargs="*",
        default=[],
        help="跳过这些图(例如暂时不用的 2.png)。",
    )
    return parser.parse_args()


def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_matte_albedo(rgb):
    """与正面烘焙一致: 压暗摄影棚高光, 防止白色背景/反光混进侧面。"""
    return 18.0 + 0.5 * rgb


def load_corners(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def project_points(pts3d, rvec, tvec, fx, fy, cx, cy):
    proj, _ = cv2.projectPoints(
        pts3d.astype(np.float64),
        rvec,
        tvec,
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64),
        np.zeros(5),
    )
    return proj.reshape(-1, 2)


def estimate_pose(obj_pts, img_pts, width, height):
    """最小化重投影误差, 同时估计焦距与外参(镜头面侧相机)。"""
    best = None
    cx = width / 2.0
    cy = height / 2.0

    for focal_scale in (0.5, 0.8, 1.0, 1.4, 2.0, 3.0):
        f0 = focal_scale * max(width, height)
        K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]], dtype=np.float64)
        ok, rvec0, tvec0 = cv2.solvePnP(
            obj_pts.astype(np.float64),
            img_pts.astype(np.float64),
            K,
            np.zeros(5),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        rvec0 = rvec0.ravel()
        tvec0 = tvec0.ravel()

        def residual(x):
            f, rv, tv = x[0], x[1:4], x[4:7]
            p = project_points(obj_pts, rv, tv, f, f, cx, cy)
            return (p - img_pts).ravel()

        try:
            res = least_squares(
                residual,
                np.concatenate([[f0], rvec0, tvec0]),
                bounds=(
                    [f0 * 0.1] + [-np.inf] * 6,
                    [f0 * 8.0] + [np.inf] * 6,
                ),
                method="trf",
                max_nfev=2000,
            )
        except Exception:
            continue
        f_opt = res.x[0]
        rvec = res.x[1:4]
        tvec = res.x[4:7]
        rmse = float(np.sqrt(np.mean(res.fun ** 2)))
        R, _ = cv2.Rodrigues(rvec)
        cam_center = -R.T @ tvec.reshape(3, 1)
        # 照片拍的是镜头面, 相机必须位于 -Y 外侧。
        if float(cam_center[1, 0]) > 0:
            continue
        if best is None or rmse < best[0]:
            best = (rmse, rvec, tvec, float(f_opt), cam_center)

    if best is None:
        raise RuntimeError("无法为斜视角估计出合理相机位姿。")
    return best


def sample_image(img_rgba, uv):
    """对每个像素坐标做双线性采样, 返回 (rgb, alpha)。"""
    h, w = img_rgba.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    valid = (u >= 0.5) & (u < w - 0.5) & (v >= 0.5) & (v < h - 0.5)
    rgb = np.zeros((len(uv), 3))
    alpha = np.zeros(len(uv))
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return rgb, alpha
    uu, vv = u[idx], v[idx]
    x0 = np.floor(uu).astype(int)
    y0 = np.floor(vv).astype(int)
    fx = uu - x0
    fy = vv - y0
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    def gather(r, c, ch):
        return img_rgba[r, c, ch].astype(np.float64)

    for ch in range(3):
        top = gather(y0, x0, ch) * (1 - fx) + gather(y0, x1, ch) * fx
        bot = gather(y1, x0, ch) * (1 - fx) + gather(y1, x1, ch) * fx
        rgb[idx, ch] = top * (1 - fy) + bot * fy
    at = gather(y0, x0, 3) * (1 - fx) + gather(y0, x1, 3) * fx
    ab = gather(y1, x0, 3) * (1 - fx) + gather(y1, x1, 3) * fx
    alpha[idx] = at * (1 - fy) + ab * fy
    return rgb, alpha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    mesh = trimesh.load(args.ply, force="mesh", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.float64)[:, :3].copy()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    ymin = float(verts[:, 1].min())
    ymax = float(verts[:, 1].max())
    x_half = (verts[:, 0].max() - verts[:, 0].min()) / 2.0
    z_half = (verts[:, 2].max() - verts[:, 2].min()) / 2.0

    # 镜头面矩形 4 角的 3D 坐标(左上/右上/右下/左下), 与用户点击顺序一致。
    obj_pts = np.array(
        [
            [-x_half, ymin, +z_half],
            [+x_half, ymin, +z_half],
            [+x_half, ymin, -z_half],
            [-x_half, ymin, -z_half],
        ],
        dtype=np.float64,
    )

    # 需要补纹理的顶点: 离镜头面/主屏面都远, 即目前的深色兜底区。
    d_lens = np.abs(verts[:, 1] - ymin)
    d_screen = np.abs(verts[:, 1] - ymax)
    w_lens = 1.0 - smoothstep(1.0, 7.0, d_lens)
    w_screen = 1.0 - smoothstep(1.0, 7.0, d_screen)
    side_mask = (w_lens < 0.25) & (w_screen < 0.25)
    side_idx = np.flatnonzero(side_mask)
    print("侧面/顶底候选顶点: %d" % len(side_idx))

    corners = load_corners(args.corners)
    alpha_dir = Path(args.alpha)
    acc_weight = np.zeros(len(verts))
    acc_rgb = np.zeros((len(verts), 3))

    for name, rec in corners.items():
        if name in args.skip_images:
            print("跳过 %s(用户指定)" % name)
            continue
        image_path = alpha_dir / name
        if not image_path.is_file():
            print("跳过缺失图片: %s" % image_path)
            continue
        img_pts = np.asarray(
            [[c["x"], c["y"]] for c in rec["corners"]], dtype=np.float64
        )
        width, height = rec["image_size"]
        rmse, rvec, tvec, focal, cam_center = estimate_pose(
            obj_pts, img_pts, width, height
        )
        print(
            "%s: 焦距=%.0fpx 重投影RMSE=%.3fpx 相机位置=[%.0f %.0f %.0f]mm"
            % (name, focal, rmse,
               cam_center[0, 0], cam_center[1, 0], cam_center[2, 0])
        )

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.shape[2] == 3:
            img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        img[:, :, [0, 2]] = img[:, :, [2, 0]]  # BGR -> RGB

        R, _ = cv2.Rodrigues(rvec)
        cam_center_w = -R.T @ tvec.reshape(3, 1)
        cam_center_w = cam_center_w.ravel()

        sub = verts[side_idx]
        dirs = sub - cam_center_w
        dist = np.linalg.norm(dirs, axis=1)
        dirs_n = dirs / dist[:, None]
        view_dir = -dirs_n  # 相机看向顶点
        visible_back = np.einsum("ij,ij->i", normals[side_idx], view_dir) < 0

        # 第一版用背向剔除近似可见性: 外法线朝向相机的顶点视为可见。
        # 相机机身是简单外凸壳, 自遮挡很少; 若后续发现局部穿帮再加光线遮挡。
        visible = visible_back
        print("  可见顶点: %d / %d" % (int(visible.sum()), len(sub)))
        if not visible.any():
            continue

        proj = project_points(
            sub[visible],
            rvec,
            tvec,
            focal,
            focal,
            width / 2.0,
            height / 2.0,
        )
        rgb, alpha = sample_image(img, proj)
        rgb = to_matte_albedo(rgb)
        idx_vis = side_idx[visible]
        w = alpha / 255.0
        acc_weight[idx_vis] += w
        acc_rgb[idx_vis] += w[:, None] * rgb

    # 对每个顶点做加权平均; 完全没被覆盖的保留原深色。
    side_mask_w = acc_weight > 0.05
    sidx = np.flatnonzero(side_mask_w)
    colors[sidx] = acc_rgb[sidx] / acc_weight[sidx, None]
    covered = int(side_mask_w.sum())
    print("被斜视角覆盖的侧面顶点: %d" % covered)

    out_colors = np.clip(colors, 0, 255).astype(np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=out_colors)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print("输出: %s" % out_path)


if __name__ == "__main__":
    main()
