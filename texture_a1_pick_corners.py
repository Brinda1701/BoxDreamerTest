#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""texture_a1_pick_corners.py

路线 A 的第 1 步：在每张透明前景图上点选相机本体外轮廓的 4 个角点。

点击顺序固定为:
    1 = 左上, 2 = 右上, 3 = 右下, 4 = 左下
(以图片里“相机屏幕/镜头朝向你”时看到的矩形外轮廓为准, 点圆角最外侧即可)

用法:
    python texture_a1_pick_corners.py \
        --src work/DJI_alpha \
        --images 4.png 5.png \
        --out work/corners.json

完成后脚本会输出 work/corners.json, 供下一步做仿射对齐与纹理烘焙。
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CORNER_NAMES = ["左上", "右上", "右下", "左下"]


def parse_args():
    parser = argparse.ArgumentParser(description="交互式点选相机外轮廓四角。")
    parser.add_argument("--src", type=str, default="work/DJI_alpha")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="要标注的透明 PNG, 例如 4.png 5.png。",
    )
    parser.add_argument("--out", type=str, default="work/corners.json")
    return parser.parse_args()


def ask_face_label(image_name):
    """让用户说明这张图对应机身的哪个面, 用于后续烘焙时选择 3D 平面对应关系。"""
    while True:
        label = input(
            "%s 显示的是相机哪个面? "
            "[lens=镜头面 / screen=主屏面 / side=侧面 / top=顶面 / bottom=底面] > "
            % image_name
        ).strip().lower()
        if label in ("lens", "screen", "side", "top", "bottom"):
            return label
        print("无法识别, 请重新输入 lens/screen/side/top/bottom。")


def pick_corners(image_path, face_label):
    """用 OpenCV 交互窗口依次收集 4 个角点(点击顺序见窗口标题)。"""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError("无法读取: %s" % image_path)
    height, width = img.shape[:2]
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        # 灰色半透明背景上展示, 便于看清透明区域。
        checker = np.full_like(bgr, 200)
        display = np.where(alpha[:, :, None] > 64, bgr, checker)
    else:
        display = img.copy()

    window_name = "pick_corners"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, max(720, width), max(720, height))
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((float(x), float(y)))
            print("  已点击第 %d 点: %s (%.1f, %.1f)"
                  % (len(clicks), CORNER_NAMES[len(clicks) - 1], x, y))

    cv2.setMouseCallback(window_name, on_mouse)
    while len(clicks) < 4:
        canvas = display.copy()
        for idx, (x, y) in enumerate(clicks):
            cv2.circle(canvas, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(canvas, CORNER_NAMES[idx], (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        hint = "点目标主面(镜头面/主屏面)矩形四角: 1左上 2右上 3右下 4左下 | q=放弃"
        cv2.putText(canvas, hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 0), 3)
        cv2.putText(canvas, hint, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 1)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(window_name)
            raise RuntimeError("用户放弃标注 %s。" % image_path.name)
    cv2.destroyWindow(window_name)
    return [
        {"label": CORNER_NAMES[i], "x": clicks[i][0], "y": clicks[i][1]}
        for i in range(4)
    ]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    src_dir = Path(args.src)
    out_path = Path(args.out)
    if not src_dir.is_dir():
        raise FileNotFoundError("找不到目录: %s" % src_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = {}
    for name in args.images:
        image_path = src_dir / name
        if not image_path.is_file():
            raise FileNotFoundError("找不到图片: %s" % image_path)
        face = ask_face_label(name)
        corners = pick_corners(image_path, face)
        records[name] = {
            "image_size": list(cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED).shape[:2][::-1]),
            "face": face,
            "corners": corners,
        }
        print("已记录 %s: %s" % (name, face))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("完成, 已保存: %s" % out_path)


if __name__ == "__main__":
    main()
