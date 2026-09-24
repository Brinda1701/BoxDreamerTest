#!/usr/bin/env python3
# infer_video_sam2.py - YOLO + SAM2 联合追踪 + BoxDreamer 3D 线框渲染
#
# 两阶段离线管线：
#   Phase 1 (服务器 GPU): YOLO 初始化 → SAM2 全程追踪 → 输出 bbox 缓存 JSON
#   Phase 2 (本地/服务器): 读取缓存 → BoxDreamer 角点回归 → 渲染输出视频
#
# 用法：
#   # 仅运行 Phase 1（服务器）
#   python infer_video_sam2.py --phase 1
#
#   # 仅运行 Phase 2（本地）
#   python infer_video_sam2.py --phase 2
#
#   # 一次性运行两阶段（服务器全流程）
#   python infer_video_sam2.py --phase all

import os
import sys
import json
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# 12 条棱边连接关系
EDGES_12 = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# 两台相机在 SAM2 中的 obj_id
CAM_LEFT_ID  = 1  # 画面左侧相机
CAM_RIGHT_ID = 2  # 画面右侧相机


# ============================================================
# 几何工具函数
# ============================================================

def segments_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return (ccw(p1,p3,p4)!=ccw(p2,p3,p4)) and (ccw(p1,p2,p3)!=ccw(p1,p2,p4))

def untangle_cuboid_2d(pts):
    p = pts.copy()
    face_checks = [
        (0,1,2,3,1,3),(4,5,6,7,5,7),(0,1,4,5,1,5),
        (2,3,6,7,3,7),(0,2,4,6,2,6),(1,3,5,7,3,7),
    ]
    for a,b,c,d,s1,s2 in face_checks:
        if segments_intersect(p[a],p[b],p[c],p[d]):
            p[s1],p[s2] = p[s2].copy(),p[s1].copy()
    return p

def mask_to_bbox(mask_bool):
    """将二值 mask 转为 [x1, y1, x2, y2] bbox（若 mask 为空返回 None）"""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def adaptive_crop(frame, bbox, target_size=224):
    """
    与 infer_video.py 完全一致的自适应外扩视口裁切
    bbox: [bx1, by1, bx2, by2]
    返回 (crop_bgr_resized, rx1, ry1, crop_size)
    """
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bw = bx2 - bx1
    bh = by2 - by1
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    max_dim = max(bw, bh)

    if max_dim < 250:
        ratio = 1.90
    elif max_dim < 380:
        ratio = 1.60
    else:
        ratio = 1.35

    crop_size = int(max(max_dim * ratio, 300))
    half = crop_size // 2
    rx1, ry1 = int(cx - half), int(cy - half)
    rx2, ry2 = rx1 + crop_size, ry1 + crop_size

    pl = max(0, -rx1); pt = max(0, -ry1)
    pr = max(0, rx2-w); pb = max(0, ry2-h)

    if pl or pt or pr or pb:
        pad = cv2.copyMakeBorder(frame, pt, pb, pl, pr, cv2.BORDER_REFLECT)
        crop = pad[ry1+pt:ry2+pt, rx1+pl:rx2+pl]
    else:
        crop = frame[ry1:ry2, rx1:rx2]

    return cv2.resize(crop, (target_size, target_size)), rx1, ry1, crop_size

def draw_3d_wireframe(vis, corners_2d, color=(0,255,128), label=None, label_pos=None):
    p = np.asarray(corners_2d, dtype=np.int32)
    for i1, i2 in EDGES_12:
        cv2.line(vis, tuple(p[i1]), tuple(p[i2]), color, 3, cv2.LINE_AA)
    for pt in p:
        cv2.circle(vis, tuple(pt), 6, (0,0,255), -1, cv2.LINE_AA)
    if label and label_pos:
        cv2.putText(vis, label, (label_pos[0], max(35, label_pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
def tighten_to_camera_body(box):
    """
    针对手持/手腕穿戴场景的物理先验校准：
    DJI Action 4 物理尺寸约为 70.5mm x 44.2mm (真实宽高比约为 1.6 : 1)。
    原始 YOLO 经常把下方的快拆底座、卡扣、手腕绑带一起包进去变成 1:1 的大正方形大框。
    本函数以机身顶部为锚点，将下边界强制截断，只保留纯净的机身矩形！
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # 真实机身高度约为宽度的 0.62 ~ 0.65 倍，严控最大高度不超过 0.65 倍
    ideal_max_h = w * 0.65
    if h > ideal_max_h:
        # 下方多出的 35%~40% 全是手腕带与底座杂质，直接物理切除！
        y2 = y1 + ideal_max_h

    # 左右边缘微收 3%，防止吸入手指侧面杂边
    margin_x = w * 0.03
    x1 += margin_x
    x2 -= margin_x

    return [float(x1), float(y1), float(x2), float(y2)]


def enhance_image_for_detection(img_bgr, mode="boost_5x", gamma=1.8, clip_limit=4.0):

    """
    专门为 YOLO 目标检测生成的高动态对比度/暗部提亮图像。
    强力拉开深色物体（黑色相机机身）与深色背景（黑色衣服/阴影）之间的边界反差。
    支持 5x 强力线性增益提亮，彻底消除暗光死角。
    """
    if mode == "none":
        return img_bgr

    out = img_bgr.copy()

    # 1. 强力亮度提亮 (5倍或3.5倍线性增益)
    if "boost" in mode:
        factor = 5.0 if "5x" in mode else 3.5
        # 转换至浮点数进行无损倍数放大，并截断在 255
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # 2. 伽马非线性暗区拉伸
    if "gamma" in mode or mode in ("clahe_gamma", "boost_5x"):
        inv_gamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)

    # 3. 局部对比度自适应均衡化 (CLAHE)
    if "clahe" in mode or mode in ("clahe_gamma", "boost_5x"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    return out


# ============================================================
# Phase 1：YOLO 初始化 + SAM2 全程追踪 → 缓存 bbox JSON
# ============================================================

def run_phase1(args):
    print("=" * 60)
    print("Phase 1: YOLO 初始化 + SAM2 全段视频追踪 (带 CLAHE+Gamma 暗光增强)")
    print("=" * 60)

    from ultralytics import YOLO
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError:
        print("[ERROR] 未找到 sam2 模块！请先安装：")
        print("  pip install git+https://github.com/facebookresearch/sam2.git")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> 设备: {device}")

    # ---- 加载 YOLO ----
    print(f"--> 加载 YOLO: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)

    # ---- 加载 SAM2 ----
    print(f"--> 加载 SAM2: {args.sam2_checkpoint}")
    predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=device)

    # ---- 在首段视频中寻找两台相机的初始提示框 ----
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"--> 视频: {args.video}  分辨率: {orig_w}x{orig_h}  总帧数: {total_frames}  FPS: {fps:.1f}")

    scan_limit = min(300, total_frames)
    print(f"--> 扫描前 {scan_limit} 帧寻找两台相机（CLAHE+Gamma增强 + conf=0.15）...")

    left_init_candidate = None   # (frame_idx, box, score)
    right_init_candidate = None  # (frame_idx, box, score)

    for fi in range(scan_limit):
        ret, frame = cap.read()
        if not ret:
            break

        # 核心：使用暗光增强帧进行检测
        det_frame = enhance_image_for_detection(frame, mode="clahe_gamma", gamma=1.6, clip_limit=3.5)
        results = yolo.predict(det_frame, imgsz=1280, conf=0.15, iou=0.45, verbose=False)[0]
        raw = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        valid_boxes = []
        for b, c in zip(raw, confs):
            bw = b[2] - b[0]
            bh = b[3] - b[1]
            if 60 < bw < 1200 and 60 < bh < 1200:
                valid_boxes.append((b, c))

        if fi % 15 == 0:
            print(f"    第 {fi:3d} 帧: 检测到 {len(valid_boxes)} 个相机候选")

        if len(valid_boxes) >= 2:
            # 排序后一次性提取左右两台相机，并用物理先验强制砍掉底部表带与底座
            sorted_v = sorted(valid_boxes, key=lambda item: (item[0][0] + item[0][2]) / 2.0)
            l_box, l_conf = sorted_v[0]
            r_box, r_conf = sorted_v[-1]
            left_init_candidate = (fi, tighten_to_camera_body(l_box), float(l_conf))
            right_init_candidate = (fi, tighten_to_camera_body(r_box), float(r_conf))
            print(f"--> [成功在第 {fi} 帧捕获两台相机（已按 1.6:1 纯净机身物理比例剔除表带）！]")
            print(f"    Left  #{CAM_LEFT_ID}: {[round(x) for x in left_init_candidate[1]]} (conf: {l_conf:.2f})")
            print(f"    Right #{CAM_RIGHT_ID}: {[round(x) for x in right_init_candidate[1]]} (conf: {r_conf:.2f})")
            break
        elif len(valid_boxes) == 1:
            b, c = valid_boxes[0]
            cx = (b[0] + b[2]) / 2.0
            clean_b = tighten_to_camera_body(b)
            if cx < orig_w / 2.0:
                if left_init_candidate is None or c > left_init_candidate[2]:
                    left_init_candidate = (fi, clean_b, float(c))
            else:
                if right_init_candidate is None or c > right_init_candidate[2]:
                    right_init_candidate = (fi, clean_b, float(c))

            if left_init_candidate is not None and right_init_candidate is not None:
                print(f"--> [在多帧累积中捕获两台纯净机身！]")
                print(f"    Left  #{CAM_LEFT_ID} (帧 {left_init_candidate[0]}): {[round(x) for x in left_init_candidate[1]]}")
                print(f"    Right #{CAM_RIGHT_ID} (帧 {right_init_candidate[0]}): {[round(x) for x in right_init_candidate[1]]}")
                break

    cap.release()


    if left_init_candidate is None or right_init_candidate is None:
        print("[ERROR] 未能在前 300 帧中稳定捕获两台相机！")
        print(f"    当前记录状态: Left={left_init_candidate is not None}, Right={right_init_candidate is not None}")
        sys.exit(1)

    # ---- SAM2 初始化 inference_state ----
    print("--> 初始化 SAM2 inference_state (开启 CPU 显存卸载，防止 2850 帧显存溢出)...")
    with torch.inference_mode():
        # offload_video_to_cpu=True: 将 2850 帧未压缩原图保存在内存，仅在计算当前帧时移入 GPU (显存从 34G 降至 2G)
        try:
            inference_state = predictor.init_state(
                video_path=args.video,
                offload_video_to_cpu=True,
                offload_state_to_cpu=True
            )
        except TypeError:
            # 兼容老版本 SAM2 参数名
            inference_state = predictor.init_state(
                video_path=args.video,
                offload_video_to_cpu=True
            )
        predictor.reset_state(inference_state)


        # 针对两台相机分别注入提示框（即使不是同一帧，SAM2 也天然支持在不同帧分别添加提示！）
        init_prompts = [
            (CAM_LEFT_ID, left_init_candidate[0], left_init_candidate[1]),
            (CAM_RIGHT_ID, right_init_candidate[0], right_init_candidate[1])
        ]

        for obj_id, f_idx, box in init_prompts:
            box_arr = np.array(box, dtype=np.float32)
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=f_idx,
                obj_id=obj_id,
                box=box_arr,
            )

        # ---- SAM2 全程传播 (开启 bfloat16 Tensor Core 硬件级 3~4倍极限加速) ----
        print("--> 开始 SAM2 全程高速传播 (bfloat16 加速 + 实时写盘断点保护)...")
        all_bboxes = {}   # {frame_idx: {obj_id: [x1,y1,x2,y2] or None}}

        with torch.autocast("cuda", dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                frame_result = {}
                for i, oid in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).squeeze().cpu().numpy()
                    bbox = mask_to_bbox(mask)
                    frame_result[int(oid)] = bbox
                all_bboxes[int(out_frame_idx)] = frame_result

                # 每 100 帧实时增量同步写入磁盘，彻底杜绝数据丢失！
                if (out_frame_idx + 1) % 100 == 0 or (out_frame_idx + 1) == total_frames:
                    tracked_left  = sum(1 for f in all_bboxes.values() if f.get(CAM_LEFT_ID) is not None)
                    tracked_right = sum(1 for f in all_bboxes.values() if f.get(CAM_RIGHT_ID) is not None)
                    print(f"\r  [⚡极速追踪] {out_frame_idx+1}/{total_frames} 帧 | Left: {tracked_left}, Right: {tracked_right}", end="", flush=True)

                    # 实时写入临时缓存
                    temp_cache = {
                        "video": str(args.video),
                        "total_frames": total_frames,
                        "fps": fps,
                        "init_left_frame": left_init_candidate[0],
                        "init_right_frame": right_init_candidate[0],
                        "cam_left_id": CAM_LEFT_ID,
                        "cam_right_id": CAM_RIGHT_ID,
                        "bboxes": {str(k): v for k, v in all_bboxes.items()},
                    }
                    with open(args.bbox_cache, "w") as f:
                        json.dump(temp_cache, f)


    print(f"\n--> SAM2 传播完成！共处理 {len(all_bboxes)} 帧")

    # 统计追踪覆盖率
    n = len(all_bboxes)
    left_ok  = sum(1 for f in all_bboxes.values() if f.get(CAM_LEFT_ID) is not None)
    right_ok = sum(1 for f in all_bboxes.values() if f.get(CAM_RIGHT_ID) is not None)
    print(f"    Left  追踪率: {left_ok}/{n} ({100*left_ok/n:.1f}%)")
    print(f"    Right 追踪率: {right_ok}/{n} ({100*right_ok/n:.1f}%)")

    # 保存 bbox 缓存
    cache = {
        "video": str(args.video),
        "total_frames": total_frames,
        "fps": fps,
        "init_left_frame": left_init_candidate[0],
        "init_right_frame": right_init_candidate[0],
        "cam_left_id": CAM_LEFT_ID,
        "cam_right_id": CAM_RIGHT_ID,
        "bboxes": {str(k): v for k, v in all_bboxes.items()},
    }
    with open(args.bbox_cache, "w") as f:
        json.dump(cache, f)
    print(f"\n--> SAM2 bbox 缓存已保存至: {args.bbox_cache}")



# ============================================================
# Phase 2：BoxDreamer 推理 + 3D 线框渲染
# ============================================================

def run_phase2(args):
    print("=" * 60)
    print("Phase 2: BoxDreamer 角点回归 + 3D 线框渲染")
    print("=" * 60)

    from models import BoxDreamerModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> 设备: {device}")

    # ---- 加载 bbox 缓存 ----
    print(f"--> 读取 SAM2 bbox 缓存: {args.bbox_cache}")
    with open(args.bbox_cache, "r") as f:
        cache = json.load(f)
    all_bboxes = {int(k): v for k, v in cache["bboxes"].items()}
    total_frames = cache["total_frames"]
    fps = cache["fps"]
    print(f"    总帧数: {total_frames}, FPS: {fps:.1f}")

    # ---- 加载 BoxDreamer ----
    print(f"--> 加载 BoxDreamer: {args.boxer_weights}")
    boxdreamer = BoxDreamerModel(device=device)
    boxdreamer.load_state_dict(torch.load(args.boxer_weights, map_location=device))
    boxdreamer.to(device)
    boxdreamer.eval()


    # ---- 时序平滑状态 ----
    last_corners = {CAM_LEFT_ID: None, CAM_RIGHT_ID: None}
    last_bbox    = {CAM_LEFT_ID: None, CAM_RIGHT_ID: None}
    lost_count   = {CAM_LEFT_ID: 0,    CAM_RIGHT_ID: 0}
    alpha = 0.70   # 时序平滑权重

    # ---- 视频读写 ----
    cap = cv2.VideoCapture(args.video)
    start = args.start_frame
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    n = min(args.max_frames, total_frames - start) if args.max_frames > 0 else (total_frames - start)

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (orig_w, orig_h))

    print(f"--> 输出: {args.output}  起始帧: {start}  处理帧数: {n}")

    cam_colors = {CAM_LEFT_ID: (0, 255, 128), CAM_RIGHT_ID: (255, 180, 0)}
    cam_names  = {CAM_LEFT_ID: "Camera #1 (Left - SAM2)", CAM_RIGHT_ID: "Camera #2 (Right - SAM2)"}

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = start + i
        vis = frame.copy()

        frame_bboxes = all_bboxes.get(frame_idx, {})

        for cam_id in [CAM_LEFT_ID, CAM_RIGHT_ID]:
            bbox = frame_bboxes.get(cam_id) or frame_bboxes.get(str(cam_id))

            if bbox is not None:
                # 以 SAM2 mask bbox 更新追踪状态（指数滑动平均平滑）
                bbox_arr = np.array(bbox, dtype=np.float64)
                if last_bbox[cam_id] is None:
                    last_bbox[cam_id] = bbox_arr
                else:
                    last_bbox[cam_id] = 0.6 * last_bbox[cam_id] + 0.4 * bbox_arr
                lost_count[cam_id] = 0
            else:
                # SAM2 丢失该目标
                lost_count[cam_id] += 1
                if last_bbox[cam_id] is None or lost_count[cam_id] > 30:
                    continue  # 太久没有追踪到，跳过本帧

            curr_box = last_bbox[cam_id]
            crop_rgb, rx1, ry1, crop_size = adaptive_crop(frame, curr_box.astype(int))
            tensor = (torch.from_numpy(crop_rgb[:,:,::-1].copy()).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0)

            with torch.no_grad():
                _, pred_224 = boxdreamer(tensor, return_coords=True)
            p = pred_224[0].cpu().numpy()

            pts_orig = np.zeros_like(p, dtype=np.float64)
            pts_orig[:, 0] = rx1 + p[:, 0] * (crop_size / 224.0)
            pts_orig[:, 1] = ry1 + p[:, 1] * (crop_size / 224.0)

            pts_clean = untangle_cuboid_2d(pts_orig)

            # 时序平滑
            if last_corners[cam_id] is None:
                last_corners[cam_id] = pts_clean.copy()
            else:
                last_corners[cam_id] = alpha * last_corners[cam_id] + (1 - alpha) * pts_clean

            color = cam_colors[cam_id]
            name = cam_names[cam_id]
            if lost_count[cam_id] > 0:
                name += f" [推测中 +{lost_count[cam_id]}帧]"
            draw_3d_wireframe(vis, last_corners[cam_id], color=color,
                              label=name, label_pos=(int(curr_box[0]), int(curr_box[1])-15))

        writer.write(vis)
        if (i+1) % 50 == 0 or (i+1) == n:
            print(f"\r  [渲染] {i+1}/{n} 帧 ({100*(i+1)/n:.1f}%)", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n\n🎉 视频生成完毕: {args.output}")


# ============================================================
# 入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="YOLO + SAM2 + BoxDreamer 3D 追踪器")

    p.add_argument("--phase", choices=["1","2","all"], default="all",
                   help="运行阶段: 1=SAM2预计算, 2=BoxDreamer渲染, all=两阶段全流程")
    p.add_argument("--video", type=str,
                   default=str(PARENT_DIR / "test_video" / "head_left_rgb_raw.mp4"))
    p.add_argument("--output", type=str,
                   default=str(PARENT_DIR / "test_video" / "output_sam2_final.mp4"))
    p.add_argument("--bbox_cache", type=str,
                   default=str(PARENT_DIR / "test_video" / "sam2_bbox_cache.json"),
                   help="SAM2 bbox 预计算缓存路径（Phase1 写入，Phase2 读取）")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0, help="0=全段视频")

    # YOLO 与 SAM2 路径
    p.add_argument("--yolo_weights", type=str,
                   default=str(PARENT_DIR / "runs" / "detect" / "train" / "weights" / "best.pt"))
    p.add_argument("--boxer_weights", type=str,
                   default=str(SCRIPT_DIR / "best_boxdreamer.pth"))
    p.add_argument("--sam2_checkpoint", type=str,
                   default="/mnt/data/home/zhoujiayan/sam2/checkpoints/sam2.1_hiera_large.pt",
                   help="SAM2 模型 checkpoint 路径")
    p.add_argument("--sam2_config", type=str,
                   default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="SAM2 配置文件（相对于 SAM2 repo 根目录）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.phase in ("1", "all"):
        run_phase1(args)
    if args.phase in ("2", "all"):
        run_phase2(args)

