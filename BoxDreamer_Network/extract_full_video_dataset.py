# extract_full_video_dataset.py - 全视频 2850 帧真实姿态与抗遮挡真值挖掘引擎
import os
import sys
import json
import cv2
import numpy as np
import torch
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

# 相机内参 (针对 3248x2464 分辨率)
K = np.array([
    [2905.0, 0.0, 1650.7],
    [0.0, 2944.3, 1242.5],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

tag_size = 25.4  # mm
tag_3d = np.array([
    [-tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2,  tag_size/2, 0.0],
    [-tag_size/2,  tag_size/2, 0.0]
], dtype=np.float64)

# DJI Action 4 物理尺寸 (mm)
w_cam = 70.5
h_cam = 44.2
d_cam = 32.4
cx_tag = 22.5
cy_tag = 0.0

x_L = cx_tag + w_cam/2
x_R = cx_tag - w_cam/2
y_T = -h_cam/2
y_B = +h_cam/2
z_F = d_cam
z_R = 0.0

# 8 个 3D 角点 (与 BOP / dji_bbox_corners.npy 严丝合缝)
corners_bop_in_tag = np.array([
    [x_L, y_B, z_F],  # 0: 左前下
    [x_L, y_T, z_F],  # 1: 左前上
    [x_L, y_B, z_R],  # 2: 左后下
    [x_L, y_T, z_R],  # 3: 左后上
    [x_R, y_B, z_F],  # 4: 右前下
    [x_R, y_T, z_F],  # 5: 右前上
    [x_R, y_B, z_R],  # 6: 右后下
    [x_R, y_T, z_R],  # 7: 右后上
], dtype=np.float64)

EDGES_12 = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def adaptive_crop_and_map(frame, pts_2d, target_size=224, yolo_box=None):
    """
    与 infer_video.py 推理期保持 100% 一致的自适应外扩视口裁切规则。
    重要：裁切基准必须与推理期相同——使用 YOLO 检测框中心与尺寸，而非 3D 角点质心。
    若无 YOLO 检测框（未检测到目标），则退化为 3D 角点质心裁切（兜底）。
    """
    h_img, w_img = frame.shape[:2]

    if yolo_box is not None:
        # ==== 主路径：与 infer_video.py 完全一致的 YOLO 框驱动裁切 ====
        bx1, by1, bx2, by2 = yolo_box
        bw = bx2 - bx1
        bh = by2 - by1
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        max_dim = max(bw, bh)
    else:
        # ==== 兜底路径：无 YOLO 时用 3D 角点质心（仅极少数帧） ====
        min_xy = np.min(pts_2d, axis=0)
        max_xy = np.max(pts_2d, axis=0)
        bw = max_xy[0] - min_xy[0]
        bh = max_xy[1] - min_xy[1]
        cx = (min_xy[0] + max_xy[0]) / 2.0
        cy = (min_xy[1] + max_xy[1]) / 2.0
        max_dim = max(bw, bh)

    # 与 infer_video.py 完全相同的分档外扩比例
    if max_dim < 250:
        expand_ratio = 1.90
    elif max_dim < 380:
        expand_ratio = 1.60
    else:
        expand_ratio = 1.35

    crop_size = int(max(max_dim * expand_ratio, 300))
    half = crop_size // 2

    rx1, ry1 = int(cx - half), int(cy - half)
    rx2, ry2 = rx1 + crop_size, ry1 + crop_size

    pad_l = max(0, -rx1)
    pad_t = max(0, -ry1)
    pad_r = max(0, rx2 - w_img)
    pad_b = max(0, ry2 - h_img)

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        padded = cv2.copyMakeBorder(frame, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
        crop_bgr = padded[ry1 + pad_t : ry2 + pad_t, rx1 + pad_l : rx2 + pad_l]
    else:
        crop_bgr = frame[ry1:ry2, rx1:rx2]

    # 将 2D 角点映射到 224x224 局部视口空间
    scale_factor = float(target_size) / float(crop_size)
    pts_224 = (pts_2d - np.array([rx1, ry1])) * scale_factor
    crop_224 = cv2.resize(crop_bgr, (target_size, target_size))

    return crop_224, pts_224


def main():
    video_path = os.path.join(PROJECT_ROOT, "test_video", "head_left_rgb_raw.mp4")
    out_dir = os.path.join(PROJECT_ROOT, "real_dataset_full")
    img_dir = os.path.join(out_dir, "images")
    vis_dir = os.path.join(out_dir, "vis")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    yolo_weights = os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt")
    has_yolo = os.path.exists(yolo_weights)
    if has_yolo:
        print(f"--> [初始化] 加载 YOLO 检测器: {yolo_weights}")
        yolo_model = YOLO(yolo_weights)
    else:
        print("--> [提示] 未找到 YOLO 权重，仅使用 AprilTag PnP 挖掘")
        yolo_model = None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"--> 开始扫描整段视频: {video_path} (共 {total_frames} 帧)")

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    samples = []
    step = 2  # 隔帧采样：2850 帧取 1425 帧，覆盖全时段同时剔除静止帧冗余
    frame_idx = 0
    gt_count = 0

    # 记录时序轨迹用于插值平滑：cam_trajectories[cid] = {f_idx: (rvec, tvec, p2d)}
    cam_trajectories = {18: {}, 19: {}}

    print("--> [Phase 1] 扫描全视频并提取 AprilTag 毫米级精准 3D 真值...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                for i, cid in enumerate(ids.ravel()):
                    if cid in [18, 19]:
                        tag_pts = corners[i][0]
                        ok, rvec, tvec = cv2.solvePnP(tag_3d, tag_pts, K, dist_coeffs)
                        if ok:
                            proj_pts, _ = cv2.projectPoints(corners_bop_in_tag, rvec, tvec, K, dist_coeffs)
                            p2d = proj_pts.reshape(-1, 2)
                            cam_trajectories[cid][frame_idx] = (rvec, tvec, p2d)

        frame_idx += 1
        if frame_idx % 200 == 0 or frame_idx == total_frames:
            print(f"\r[进度] 已扫描 {frame_idx}/{total_frames} 帧 (Tag18: {len(cam_trajectories[18])}, Tag19: {len(cam_trajectories[19])})", end="", flush=True)

    print("\n\n--> [Phase 2] 时序姿态插值与自适应视口裁切导出...")
    # 对每台相机的轨迹进行短时缺失（< 25 帧）三次 Hermite / 线性插值，覆盖手部部分遮挡区间
    interpolated_trajectories = {18: {}, 19: {}}

    for cid in [18, 19]:
        known_frames = sorted(cam_trajectories[cid].keys())
        if not known_frames:
            continue

        for idx, f in enumerate(known_frames):
            interpolated_trajectories[cid][f] = cam_trajectories[cid][f]

            # 检查与下一帧之间的时间空隙
            if idx < len(known_frames) - 1:
                next_f = known_frames[idx + 1]
                gap = next_f - f
                # 若缺失在 2~20 帧以内（手快速遮挡掠过），进行平滑姿态插值
                if 2 < gap <= 20:
                    r1, t1, _ = cam_trajectories[cid][f]
                    r2, t2, _ = cam_trajectories[cid][next_f]
                    for mid_f in range(f + step, next_f, step):
                        alpha = float(mid_f - f) / float(gap)
                        mid_r = (1.0 - alpha) * r1 + alpha * r2
                        mid_t = (1.0 - alpha) * t1 + alpha * t2
                        proj_pts, _ = cv2.projectPoints(corners_bop_in_tag, mid_r, mid_t, K, dist_coeffs)
                        interpolated_trajectories[cid][mid_f] = (mid_r, mid_t, proj_pts.reshape(-1, 2))

    print(f"--> 姿态插值完成: Cam 18 共 {len(interpolated_trajectories[18])} 帧，Cam 19 共 {len(interpolated_trajectories[19])} 帧")

    # 重新读取视频导出切片
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    saved_count = 0
    yolo_used = 0
    yolo_fallback = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 用 YOLO 检测当前帧的两台相机，构建 cam_id -> yolo_box 映射
        # 这样训练时的裁切基准与推理完全一致
        yolo_boxes_this_frame = {}  # {cid: [bx1, by1, bx2, by2]}
        if yolo_model is not None and frame_idx % step == 0:
            yolo_results = yolo_model.predict(frame, imgsz=1280, conf=0.18, iou=0.45, verbose=False)[0]
            raw_boxes = yolo_results.boxes.xyxy.cpu().numpy()
            valid = [b for b in raw_boxes if 80 < (b[2]-b[0]) < 800 and 80 < (b[3]-b[1]) < 800]
            if len(valid) >= 2:
                valid_sorted = sorted(valid, key=lambda b: (b[0]+b[2])/2.0)
                yolo_boxes_this_frame[18] = valid_sorted[0]   # 最左侧检测框 -> Cam 18
                yolo_boxes_this_frame[19] = valid_sorted[-1]  # 最右侧检测框 -> Cam 19
            elif len(valid) == 1:
                cx_v = (valid[0][0]+valid[0][2])/2.0
                w_full = frame.shape[1]
                if cx_v < w_full / 2.0:
                    yolo_boxes_this_frame[18] = valid[0]
                else:
                    yolo_boxes_this_frame[19] = valid[0]

        for cid in [18, 19]:
            if frame_idx in interpolated_trajectories[cid]:
                _, _, p2d = interpolated_trajectories[cid][frame_idx]

                # 检查角点是否在合理视野范围内
                h_img, w_img = frame.shape[:2]
                if np.all(p2d[:, 0] > -100) and np.all(p2d[:, 0] < w_img + 100) and \
                   np.all(p2d[:, 1] > -100) and np.all(p2d[:, 1] < h_img + 100):

                    # 优先用 YOLO 检测框作为裁切基准，与推理端完全对齐
                    yolo_box = yolo_boxes_this_frame.get(cid, None)
                    if yolo_box is not None:
                        yolo_used += 1
                    else:
                        yolo_fallback += 1

                    crop_224, p_224 = adaptive_crop_and_map(frame, p2d, target_size=224, yolo_box=yolo_box)

                    img_name = f"frame_{frame_idx:04d}_cam{cid}.jpg"
                    img_path = os.path.join(img_dir, img_name)
                    cv2.imwrite(img_path, crop_224)

                    # 随机抽取保存可视化校验图
                    if saved_count % 20 == 0:
                        vis = crop_224.copy()
                        for i1, i2 in EDGES_12:
                            pt1 = (int(p_224[i1][0]), int(p_224[i1][1]))
                            pt2 = (int(p_224[i2][0]), int(p_224[i2][1]))
                            cv2.line(vis, pt1, pt2, (0, 255, 128), 2)
                        for pt in p_224:
                            cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                        cv2.imwrite(os.path.join(vis_dir, f"vis_{img_name}"), vis)

                    samples.append({
                        "image_file": img_name,
                        "frame_idx": frame_idx,
                        "camera_id": int(cid),
                        "corners_224": p_224.tolist()
                    })
                    saved_count += 1

        frame_idx += 1
        if frame_idx % 250 == 0 or frame_idx == total_frames:
            print(f"\r[导出切片] 进度: {frame_idx}/{total_frames} 帧 (已生成 {saved_count} 张样本 | YOLO对齐: {yolo_used}, 兜底: {yolo_fallback})", end="", flush=True)

    cap.release()


    labels_path = os.path.join(out_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"\n\n🎉 [数据挖掘成功] 共从原视频中提取出 {len(samples)} 张高质量真实切片！")
    print(f"--> 数据集路径: {out_dir}")
    print(f"--> 标签文件:   {labels_path}")
    print(f"--> 校验切片:   {vis_dir} (可打开抽检 3D 框贴合度)")

if __name__ == "__main__":
    main()

