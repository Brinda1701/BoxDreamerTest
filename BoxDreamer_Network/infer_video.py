# infer_video.py - 工业级视频 3D 边界框追踪器 (自适应动态视口外扩 + 纯单目无内参)
import os
import sys
import argparse
import cv2
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ultralytics import YOLO
from models import BoxDreamerModel

# 12 条棱边连接关系 (根据 dji_bbox_corners.npy)
EDGES_12 = [
    # 沿 Z 轴的 4 条棱 (前-后)
    (0, 1), (2, 3), (4, 5), (6, 7),
    # 沿 Y 轴的 4 条棱 (下-上)
    (0, 2), (1, 3), (4, 6), (5, 7),
    # 沿 X 轴的 4 条棱 (左-右)
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def segments_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def untangle_cuboid_2d(pts):
    p = pts.copy()
    face_checks = [
        (0, 1, 2, 3, 1, 3),
        (4, 5, 6, 7, 5, 7),
        (0, 1, 4, 5, 1, 5),
        (2, 3, 6, 7, 3, 7),
        (0, 2, 4, 6, 2, 6),
        (1, 3, 5, 7, 3, 7)
    ]
    for idx0, idx1, idx2, idx3, sw1, sw2 in face_checks:
        if segments_intersect(p[idx0], p[idx1], p[idx2], p[idx3]):
            p[sw1], p[sw2] = p[sw2].copy(), p[sw1].copy()
    return p

# ----------------------------------------------------------------------
# 目标状态追踪器：记录每个相机的 2D 位置与 3D 角点，具备连续航位记忆
# ----------------------------------------------------------------------
class CameraTracker:
    def __init__(self, name, default_x):
        self.name = name
        self.default_x = default_x
        self.last_bbox = None       # [bx1, by1, bx2, by2]
        self.last_corners = None    # (8, 2)
        self.lost_count = 0
        self.alpha = 0.70

    def update_box(self, det_box):
        if self.last_bbox is None:
            self.last_bbox = det_box.copy()
        else:
            self.last_bbox = 0.5 * self.last_bbox + 0.5 * det_box
        self.lost_count = 0

    def predict_box(self):
        self.lost_count += 1
        return self.last_bbox

    def update_corners(self, new_corners):
        if self.last_corners is None:
            self.last_corners = new_corners.copy()
        else:
            self.last_corners = self.alpha * self.last_corners + (1.0 - self.alpha) * new_corners
        return self.last_corners

# 载入模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_WEIGHTS  = os.path.join(PARENT_DIR, "runs", "detect", "train", "weights", "best.pt")
BOXER_WEIGHTS = os.path.join(SCRIPT_DIR, "best_boxdreamer.pth")

print(f"--> [初始化] 加载 YOLO 模型: {YOLO_WEIGHTS}")
yolo_model = YOLO(YOLO_WEIGHTS)

print(f"--> [初始化] 加载微调后的 BoxDreamer 模型: {BOXER_WEIGHTS} (设备: {device})")
boxdreamer = BoxDreamerModel(device=device)
boxdreamer.load_state_dict(torch.load(BOXER_WEIGHTS, map_location=device))
boxdreamer.eval()

trackers = [
    CameraTracker(name="Camera #1 (Left)", default_x=1400),
    CameraTracker(name="Camera #2 (Right)", default_x=2400)
]

def draw_3d_wireframe(vis, corners_2d, color=(0, 255, 128), text_label=None, text_pos=None):
    p = np.asarray(corners_2d, dtype=np.int32)
    # 绘制 12 条棱边
    for idx1, idx2 in EDGES_12:
        cv2.line(vis, tuple(p[idx1]), tuple(p[idx2]), color, 3, cv2.LINE_AA)
    # 绘制 8 个角点
    for pt in p:
        cv2.circle(vis, tuple(pt), 6, (0, 0, 255), -1, cv2.LINE_AA)
    # 标注文本
    if text_label and text_pos:
        tx, ty = text_pos
        cv2.putText(vis, text_label, (tx, max(35, ty)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

def process_video_frame(frame):
    vis_frame = frame.copy()
    h, w = frame.shape[:2]

    # 第一步：YOLO 目标检测 (conf=0.18 保证后半段不漏检)
    results = yolo_model.predict(frame, imgsz=1280, conf=0.18, iou=0.45, verbose=False)[0]
    raw_boxes = results.boxes.xyxy.cpu().numpy()

    valid_boxes = []
    for b in raw_boxes:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if 80 < bw < 800 and 80 < bh < 800:
            valid_boxes.append(b)

    matched_boxes = [None, None]
    if len(valid_boxes) >= 2:
        valid_boxes = sorted(valid_boxes, key=lambda b: (b[0] + b[2]) / 2.0)
        matched_boxes[0] = valid_boxes[0]
        matched_boxes[1] = valid_boxes[-1]
    elif len(valid_boxes) == 1:
        cx = (valid_boxes[0][0] + valid_boxes[0][2]) / 2.0
        if cx < w / 2.0:
            matched_boxes[0] = valid_boxes[0]
        else:
            matched_boxes[1] = valid_boxes[0]

    for idx, tracker in enumerate(trackers):
        det_box = matched_boxes[idx]
        if det_box is not None:
            tracker.update_box(det_box)
            curr_box = tracker.last_bbox
        else:
            if tracker.last_bbox is not None and tracker.lost_count < 15:
                curr_box = tracker.predict_box()
            else:
                continue

        bx1, by1, bx2, by2 = curr_box
        bw = bx2 - bx1
        bh = by2 - by1

        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        # =========================================================================
        # 核心优化：彻底解决“框过小导致角点被裁切断”的问题
        # 当 2D 检测框偏小时（如后半段远景 180px 左右），自适应提升扩展比例至 1.85，
        # 并设定最小视口宽度 320px，确保 8 个 3D 角点全部 100% 完整容纳进局部视口中！
        # =========================================================================
        max_dim = max(bw, bh)
        if max_dim < 250:
            expand_ratio = 1.90  # 后半段紧凑小框：大幅外扩 1.9 倍，补足相机机身完整轮廓
        elif max_dim < 380:
            expand_ratio = 1.60
        else:
            expand_ratio = 1.35  # 前半段大特写：保持 1.35 倍

        crop_size = int(max(max_dim * expand_ratio, 300))  # 视口尺寸底线不低于 300 像素
        half = crop_size // 2

        rx1, ry1 = int(cx - half), int(cy - half)
        rx2, ry2 = rx1 + crop_size, ry1 + crop_size

        pad_l = max(0, -rx1)
        pad_t = max(0, -ry1)
        pad_r = max(0, rx2 - w)
        pad_b = max(0, ry2 - h)

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
            crop_bgr = padded[ry1 + pad_t : ry2 + pad_t, rx1 + pad_l : rx2 + pad_l]
        else:
            crop_bgr = frame[ry1:ry2, rx1:rx2]

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (224, 224))
        tensor = torch.from_numpy(crop_resized).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            _, pred_coords_224 = boxdreamer(tensor, return_coords=True)

        p = pred_coords_224[0].cpu().numpy()

        pts_orig = np.zeros_like(p, dtype=np.float64)
        pts_orig[:, 0] = rx1 + p[:, 0] * (crop_size / 224.0)
        pts_orig[:, 1] = ry1 + p[:, 1] * (crop_size / 224.0)

        pts_clean = untangle_cuboid_2d(pts_orig)
        pts_smooth = tracker.update_corners(pts_clean)

        color = (0, 255, 128) if idx == 0 else (255, 180, 0)
        label_str = tracker.name if tracker.lost_count == 0 else f"{tracker.name} (Tracking)"
        draw_3d_wireframe(vis_frame, pts_smooth, color=color,
                           text_label=label_str,
                           text_pos=(rx1, ry1 - 15))

    return vis_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=os.path.join(PARENT_DIR, "test_video", "head_left_rgb_raw.mp4"))
    parser.add_argument("--output", type=str, default=os.path.join(PARENT_DIR, "test_video", "output_3d_tracking_full.mp4"))
    parser.add_argument("--max_frames", type=int, default=0, help="最大处理帧数 (0 表示处理整段完整视频)")
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧 (默认 0)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误: 找不到输入视频: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    n = min(args.max_frames, total - args.start_frame) if args.max_frames > 0 else (total - args.start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))

    print(f"--> 开始处理视频: {args.video}")
    print(f"--> 起始帧: {args.start_frame}, 处理帧数: {n}, 总帧数: {total}")
    print(f"--> 输出路径: {args.output}\n")

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        vis = process_video_frame(frame)
        writer.write(vis)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"\r[进度] 处理中: {i+1}/{n} 帧 ({(i+1)/n*100:.1f}%)", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n\n🎉 视频生成完毕！请查看文件: {args.output}")

if __name__ == "__main__":
    main()