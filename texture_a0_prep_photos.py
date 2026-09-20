#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""texture_a0_prep_photos.py

路线 A 的第 0 步：把 DJI 产品图转成带 alpha 的透明 PNG。

原理：产品图背景通常是纯色/接近纯色，我们从图片四条边取样作为背景色，
按颜色距离切出前景（相机本体），取最大连通域并做羽化边缘，保存为 RGBA PNG。

用法：
    python texture_a0_prep_photos.py \
        --src resources/DJI \
        --out work/DJI_alpha \
        --images 4.jpg 5.webp
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="产品图去背景并转透明 PNG。")
    parser.add_argument("--src", type=str, default="resources/DJI")
    parser.add_argument("--out", type=str, default="work/DJI_alpha")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="只处理指定的图片名; 缺省处理目录下全部图片。",
    )
    parser.add_argument(
        "--color_threshold",
        type=int,
        default=35,
        help="背景判定阈值(0~255)。越大越容易把浅色机身一起切掉。",
    )
    return parser.parse_args()


def border_background_color(image_bgr, border_width=6):
    """用四边像素的中位数近似估计背景色。"""
    h, w = image_bgr.shape[:2]
    borders = np.concatenate(
        [
            image_bgr[:border_width, :].reshape(-1, 3),
            image_bgr[-border_width:, :].reshape(-1, 3),
            image_bgr[:, :border_width].reshape(-1, 3),
            image_bgr[:, -border_width:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(borders, axis=0)


def remove_background(image_bgr, color_threshold=35, min_radius=3):
    """返回 (前景BGR, 0~255 的 alpha 蒙版)。"""
    bg = border_background_color(image_bgr)
    # 与边框背景色差异较大的像素判为前景。
    diff = np.abs(image_bgr.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    fg_mask = (diff > color_threshold).astype(np.uint8) * 255

    # 形态学清理: 去掉背景噪点并合并前景碎片。
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 只保留最大连通域(相机本体)。
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    if n <= 1:
        raise RuntimeError("未能从前景中检测到任何连通域。")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg_mask = (labels == largest).astype(np.uint8) * 255

    # 内部空洞填充(经典的“反色 + 从背景种子洪水填充”方法)。
    h, w = fg_mask.shape
    seed = None
    for probe in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0)]:
        if fg_mask[probe[1], probe[0]] == 0:
            seed = probe
            break
    if seed is not None:
        fg_inv = 255 - fg_mask
        flood = fg_inv.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, seed, 128)
        # 外边界被改成 128, 只剩前景内部的洞保持 255。
        holes = (flood == 255).astype(np.uint8) * 255
        fg_mask = cv2.bitwise_or(fg_mask, holes)
    else:
        # 相机贴边时退化为形态学闭运算补洞。
        kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_big)

    # 边缘羽化, 避免后续贴图出现硬边。
    alpha = cv2.GaussianBlur(fg_mask, (0, 0), sigmaX=1.2)
    return image_bgr, alpha


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if not src_dir.is_dir():
        raise FileNotFoundError("找不到目录: %s" % src_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.images:
        files = [src_dir / name for name in args.images]
    else:
        files = sorted(
            p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError("源目录下没有可用图片: %s" % src_dir)

    for image_path in files:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print("跳过无法读取的图片: %s" % image_path)
            continue
        fg, alpha = remove_background(img, color_threshold=args.color_threshold)
        rgba = np.dstack([fg, alpha])
        out_path = out_dir / (image_path.stem + ".png")
        cv2.imwrite(str(out_path), rgba)
        fg_ratio = float((alpha > 128).mean())
        print(
            "%-12s -> %s (前景占比 %.1f%%)"
            % (image_path.name, out_path.name, fg_ratio * 100.0)
        )
    print("完成。输出目录: %s" % out_dir)


if __name__ == "__main__":
    main()
