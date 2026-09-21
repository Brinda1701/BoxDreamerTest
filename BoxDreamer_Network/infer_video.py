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
            # 尺寸骤缩防护：当暗光/同色衣服导致 YOLO 框面积暴跌超过 35% 时，降低新框权重
            last_w = max(1.0, self.last_bbox[2] - self.last_bbox[0])
            last_h = max(1.0, self.last_bbox[3] - self.last_bbox[1])
            new_w = max(1.0, det_box[2] - det_box[0])
            new_h = max(1.0, det_box[3] - det_box[1])
            area_ratio = (new_w * new_h) / (last_w * last_h)

            # 若框突变缩小，减弱吸收权重 (0.25)，防止视口被暗区吞噬的缩小框带偏
            w = 0.25 if area_ratio < 0.65 else 0.50
            self.last_bbox = (1.0 - w) * self.last_bbox + w * det_box
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

# ----------------------------------------------------------------------
# 图像增强模块：为 YOLO 目标检测生成高对比度/暗部提亮图像
# 突出暗色相机与深色衣服/阴影之间的边界轮廓
# ----------------------------------------------------------------------
def enhance_image_for_detection(img_bgr, mode="clahe_gamma", gamma=1.6, clip_limit=3.5):
    """
    专门为 YOLO 目标检测生成的高动态对比度/暗部提亮图像。
    强力拉开深色物体（黑色相机机身）与深色背景（黑色衣服/阴影）之间的边界反差。
    支持模式:
      - 'clahe_gamma': 伽马拉升暗部 + LAB-CLAHE 自适应局部反差增强 (推荐，轮廓最清晰)
      - 'clahe': 仅限制对比度自适应直方图均衡化
      - 'gamma': 仅非线性伽马提亮暗区
      - 'linear': 线性增益提亮 (I * alpha + beta)
      - 'none': 保持原图
    """
    if mode == "none":
        return img_bgr

    out = img_bgr.copy()

    # 1. 伽马暗区拉伸
    if mode in ("gamma", "clahe_gamma"):
        inv_gamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        out = cv2.LUT(out, table)

    # 2. 局部对比度自适应均衡化
    if mode in ("clahe", "clahe_gamma"):
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)

    elif mode == "linear":
        out = cv2.convertScaleAbs(out, alpha=1.35, beta=40)

    return out

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

def process_video_frame(frame, enhance_mode="clahe_gamma", gamma=1.6, clip_limit=3.5, crop_source="orig", debug_vis=True):
    vis_frame = frame.copy()
    h, w = frame.shape[:2]

    # 第一步：为 YOLO 生成对比度与暗部增强帧，凸显相机轮廓
    det_frame = enhance_image_for_detection(frame, mode=enhance_mode, gamma=gamma, clip_limit=clip_limit)

    # YOLO 目标检测 (使用增强图检测，conf=0.18)
    results = yolo_model.predict(det_frame, imgsz=1280, conf=0.18, iou=0.45, verbose=False)[0]
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

        # 视口裁剪：根据 crop_source 决定是从原图切还是从增强图切
        # 默认 'orig'：保持原图送入 BoxDreamer，彻底避免 ViT 域偏移
        source_frame = det_frame if crop_source == "enhanced" else frame

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(source_frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
            crop_bgr = padded[ry1 + pad_t : ry2 + pad_t, rx1 + pad_l : rx2 + pad_l]
        else:
            crop_bgr = source_frame[ry1:ry2, rx1:rx2]

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

        # 调试模式可视化：画出 YOLO 检测框 (黄框) 与 视口裁剪框 (白灰框)
        if debug_vis:
            if det_box is not None:
                cv2.rectangle(vis_frame, (int(det_box[0]), int(det_box[1])),
                              (int(det_box[2]), int(det_box[3])), (0, 230, 255), 2, cv2.LINE_AA)
                cv2.putText(vis_frame, "YOLO 2D", (int(det_box[0]), max(20, int(det_box[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.rectangle(vis_frame, (rx1, ry1), (rx2, ry2), (180, 180, 180), 1, cv2.LINE_AA)

        color = (0, 255, 128) if idx == 0 else (255, 180, 0)
        label_str = tracker.name if tracker.lost_count == 0 else f"{tracker.name} (Tracking)"
        draw_3d_wireframe(vis_frame, pts_smooth, color=color,
                           text_label=label_str,
                           text_pos=(rx1, ry1 - 15))

    if debug_vis:
        status_text = f"Mode: {enhance_mode} | Gamma: {gamma} | CLAHE: {clip_limit} | CropSrc: {crop_source}"
        cv2.putText(vis_frame, status_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return vis_frame

def main():
    parser = argparse.ArgumentParser(description="工业级 3D 边界框视频推理 (含暗部轮廓增强与解耦检测)")
    parser.add_argument("--video", type=str, default=os.path.join(PARENT_DIR, "test_video", "head_left_rgb_raw.mp4"))
    parser.add_argument("--output", type=str, default=os.path.join(PARENT_DIR, "test_video", "output_3d_tracking_full.mp4"))
    parser.add_argument("--max_frames", type=int, default=0, help="最大处理帧数 (0 表示处理整段完整视频)")
    parser.add_argument("--start_frame", type=int, default=0, help="起始帧 (默认 0)")
    
    # 增强参数配置
    parser.add_argument("--enhance_mode", type=str, default="clahe_gamma",
                        choices=["clahe_gamma", "clahe", "gamma", "linear", "none"],
                        help="YOLO 检测增强模式：clahe_gamma(伽马+局部对比度，推荐), clahe, gamma, linear, none")
    parser.add_argument("--gamma", type=float, default=1.6, help="伽马暗区拉伸系数 (>1.0 提亮暗部，默认 1.6)")
    parser.add_argument("--clip_limit", type=float, default=3.5, help="CLAHE 对比度限幅系数 (默认 3.5)")
    parser.add_argument("--crop_source", type=str, default="orig", choices=["orig", "enhanced"],
                        help="送入 BoxDreamer 的裁切来源：'orig'(原图，推荐无域漂移) 或 'enhanced'(增强图)")
    parser.add_argument("--no_debug_vis", action="store_true", help="关闭 YOLO 2D 框与视口白框的调试绘制")
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
    print(f"--> 图像增强模式: {args.enhance_mode} (Gamma={args.gamma}, CLAHE={args.clip_limit})")
    print(f"--> BoxDreamer 裁切来源: {args.crop_source}")
    print(f"--> 输出路径: {args.output}\n")

    debug_vis = not args.no_debug_vis

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        vis = process_video_frame(
            frame,
            enhance_mode=args.enhance_mode,
            gamma=args.gamma,
            clip_limit=args.clip_limit,
            crop_source=args.crop_source,
            debug_vis=debug_vis
        )
        writer.write(vis)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"\r[进度] 处理中: {i+1}/{n} 帧 ({(i+1)/n*100:.1f}%)", end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n\n🎉 视频生成完毕！请查看文件: {args.output}")

if __name__ == "__main__":
    main()