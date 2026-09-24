#!/usr/bin/env python3
# verify_sam2_tracking.py - 独立质检：单看 SAM2 的 2D 分割与跟踪效果
#
# 目的：脱离 BoxDreamer，把 SAM2 生成的 mask（彩色半透明覆盖）和 bbox 绘制到原视频上，
# 一眼验证：
# 1. SAM2 是否把手、桌面、衣服阴影误识别进相机了？
# 2. 弱光、遮挡时 mask 是否散架、漂移或跟丢？
# 3. 两台相机是否有身份混淆？

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

CAM_LEFT_ID = 1
CAM_RIGHT_ID = 2

def mask_to_bbox(mask_bool):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def enhance_image_for_detection(img_bgr, mode="boost_5x", gamma=1.8, clip_limit=4.0):
    if mode == "none":
        return img_bgr
    out = img_bgr.copy()
    if "boost" in mode:
        factor = 5.0 if "5x" in mode else 3.5
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if "gamma" in mode or mode in ("clahe_gamma", "boost_5x"):
        inv_gamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)
    if "clahe" in mode or mode in ("clahe_gamma", "boost_5x"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
    return out

def tighten_to_camera_body(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    ideal_max_h = w * 0.65
    if h > ideal_max_h:
        y2 = y1 + ideal_max_h
    margin_x = w * 0.03
    x1 += margin_x
    x2 -= margin_x
    return [float(x1), float(y1), float(x2), float(y2)]

def main():
    parser = argparse.ArgumentParser(description="SAM2 独立追踪质检工具")
    parser.add_argument("--video", type=str, default="/mnt/data/home/zhoujiayan/BoxDreamer/test_video/head_left_rgb_raw.mp4")
    parser.add_argument("--output", type=str, default="/mnt/data/home/zhoujiayan/BoxDreamer/test_video/vis_sam2_inspection.mp4")
    parser.add_argument("--yolo_weights", type=str, default="/mnt/data/home/zhoujiayan/BoxDreamer/runs/detect/train/weights/best.pt")
    parser.add_argument("--sam2_checkpoint", type=str, default="/mnt/data/home/zhoujiayan/sam2/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--max_frames", type=int, default=600, help="默认质检前 600 帧（涵盖最容易出问题的手势遮挡与暗光段，极速出片）")
    args = parser.parse_args()

    from ultralytics import YOLO
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"--> [SAM2 独立质检] 设备: {device}")
    print(f"--> 视频路径: {args.video}")
    print(f"--> 输出路径: {args.output} (质检前 {args.max_frames} 帧)")
    print("=" * 60)

    # 1. YOLO 捕获初始帧
    yolo = YOLO(args.yolo_weights)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_init = None
    right_init = None

    for fi in range(min(200, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        det_frame = enhance_image_for_detection(frame, mode="boost_5x")
        res = yolo.predict(det_frame, imgsz=1280, conf=0.15, iou=0.45, verbose=False)[0]
        raw = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        valid = [(b, c) for b, c in zip(raw, confs) if 60 < (b[2]-b[0]) < 1200 and 60 < (b[3]-b[1]) < 1200]

        if len(valid) >= 2:
            sorted_v = sorted(valid, key=lambda x: (x[0][0] + x[0][2]) / 2.0)
            left_init = (fi, tighten_to_camera_body(sorted_v[0][0]))
            right_init = (fi, tighten_to_camera_body(sorted_v[-1][0]))
            print(f"--> [第 {fi} 帧成功初始化两台相机（1.6:1纯净机身，表带已剔除）]")
            print(f"    Left:  {[round(x) for x in left_init[1]]}")
            print(f"    Right: {[round(x) for x in right_init[1]]}")
            break
        elif len(valid) == 1:
            b, c = valid[0]
            cx = (b[0] + b[2]) / 2.0
            clean_b = tighten_to_camera_body(b)
            if cx < orig_w / 2.0 and left_init is None:
                left_init = (fi, clean_b)
            elif cx >= orig_w / 2.0 and right_init is None:
                right_init = (fi, clean_b)
            if left_init and right_init:
                print(f"--> [累积捕获两台相机（表带已剔除）: Left帧{left_init[0]}, Right帧{right_init[0]}]")
                break

    cap.release()

    # 2. SAM2 初始化
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=args.video, offload_video_to_cpu=True, offload_state_to_cpu=True)
    predictor.reset_state(inference_state)

    for oid, (f_idx, box) in [(CAM_LEFT_ID, left_init), (CAM_RIGHT_ID, right_init)]:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=f_idx,
            obj_id=oid,
            box=np.array(box, dtype=np.float32)
        )

    # 3. 逐帧运行 SAM2 并直接把 Mask + BBox 画在视频帧上导出
    cap = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))

    colors = {
        CAM_LEFT_ID: (0, 255, 128),   # 绿色
        CAM_RIGHT_ID: (255, 160, 0)   # 橙色
    }

    print(f"--> 正在实时生成带有 SAM2 分割彩色遮罩的质检视频...")
    curr_frame_idx = 0
    max_f = min(args.max_frames, total_frames)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            if curr_frame_idx >= max_f:
                break

            ret, frame = cap.read()
            if not ret:
                break

            vis = frame.copy()
            # 用于半透明混合的图层
            overlay = vis.copy()

            for i, oid in enumerate(out_obj_ids):
                oid = int(oid)
                mask = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                color = colors.get(oid, (255, 255, 255))
                bbox = mask_to_bbox(mask)

                # 1. 涂上 SAM2 预测的像素级轮廓 (高亮显示)
                overlay[mask] = color

                # 2. 绘制外接边界框与状态标签
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 3)
                    name = "Cam #1 (Left)" if oid == CAM_LEFT_ID else "Cam #2 (Right)"
                    cv2.putText(vis, f"{name} [Area: {int(mask.sum())}px]", (bx1, max(30, by1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(vis, f"Cam #{oid} [LOST/遮挡]", (50, 50 + oid * 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # 半透明混合：0.65 原图 + 0.35 预测掩码，能清清楚楚看清是否把手包进去了
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            writer.write(vis)

            curr_frame_idx += 1
            if curr_frame_idx % 50 == 0 or curr_frame_idx == max_f:
                print(f"\r  [质检进度] {curr_frame_idx}/{max_f} 帧", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n\n🎉 [质检视频完成] -> {args.output}")

if __name__ == "__main__":
    main()

