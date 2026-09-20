# auto_track_folding_dataset.py - 严格双目标（Dual-Camera）成对约束生成器
# 铁律：每张训练图片必须且只能严格包含【左相机 + 右相机】2个精准框，绝不允许单框误导网络！
import os
import sys
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(PROJECT_ROOT, "test_video", "head_left_rgb_raw.mp4")

if not os.path.exists(video_path):
    print(f"❌ 找不到视频: {video_path}")
    sys.exit(1)

DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_full_scene_dataset")
vis_dir = os.path.join(DATASET_DIR, "vis_preview")

# 彻底清空旧数据集与旧预览图
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR, ignore_errors=True)

for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# 3248x2464 真实相机内参与 AprilTag 空间几何
K = np.array([
    [2905.0, 0.0, 1650.7],
    [0.0, 2944.3, 1242.5],
    [0.0, 0.0, 1.0]
], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

tag_size = 25.4 # mm
tag_3d = np.array([
    [-tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2, -tag_size/2, 0.0],
    [ tag_size/2,  tag_size/2, 0.0],
    [-tag_size/2,  tag_size/2, 0.0]
], dtype=np.float64)

# DJI Action 4 物理尺寸 (mm)
w_cam, h_cam, d_cam = 70.5, 44.2, 32.4
cx_tag, cy_tag = 22.5, 0.0
x_L = cx_tag + w_cam/2
x_R = cx_tag - w_cam/2
y_T = -h_cam/2
y_B = +h_cam/2
z_F = d_cam
z_R = 0.0

corners_bop_in_tag = np.array([
    [x_L, y_B, z_F], [x_L, y_T, z_F],
    [x_L, y_B, z_R], [x_L, y_T, z_R],
    [x_R, y_B, z_F], [x_R, y_T, z_F],
    [x_R, y_B, z_R], [x_R, y_T, z_R],
], dtype=np.float64)

cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

TRAIN_W = 1280
scale_factor = TRAIN_W / float(orig_w)
TRAIN_H = int(orig_h * scale_factor)

all_annotations = {} # f_idx -> list of exactly 2 boxes: [box_left, box_right]

# --------------------------------------------------------------------------
# Phase 1: 扫描前 110 帧，提取 AprilTag 正面双相机毫米级真值 (严格要求成对检出)
# --------------------------------------------------------------------------
print("--> [Phase 1/3] 扫描前 110 帧提取 AprilTag 双相机 100% 毫米级绝对真值...")
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(aruco_dict)

for f_idx in range(0, min(110, total_frames)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and len(ids) >= 2:
        detected_tags = {}
        for i, cid in enumerate(ids.ravel()):
            if cid in [18, 19]:
                detected_tags[cid] = corners[i][0]

        # 必须 18(左相机) 和 19(右相机) 同时完美检出
        if 18 in detected_tags and 19 in detected_tags:
            boxes_pair = []
            for cid in [18, 19]:
                pts = detected_tags[cid]
                ok, rvec, tvec = cv2.solvePnP(tag_3d, pts, K, dist_coeffs)
                if not ok: break

                proj_pts, _ = cv2.projectPoints(corners_bop_in_tag, rvec, tvec, K, dist_coeffs)
                p2d = proj_pts.reshape(-1, 2)

                min_x, max_x = np.min(p2d[:, 0]), np.max(p2d[:, 0])
                min_y, max_y = np.min(p2d[:, 1]), np.max(p2d[:, 1])

                cx = (min_x + max_x) / 2.0
                cy = (min_y + max_y) / 2.0
                bw = (max_x - min_x) * 0.90
                bh = (max_y - min_y) * 0.90

                bx1 = int(max(0, cx - bw/2.0))
                by1 = int(max(0, cy - bh/2.0))
                bx2 = int(min(orig_w, cx + bw/2.0))
                by2 = int(min(orig_h, cy + bh/2.0))
                boxes_pair.append((bx1, by1, bx2, by2))

            if len(boxes_pair) == 2 and f_idx % 3 == 0:
                # 按照横坐标左右排序确保 [左相机, 右相机]
                boxes_pair.sort(key=lambda b: b[0])
                all_annotations[f_idx] = boxes_pair

print(f"  -> Phase 1 成功提取双相机成对真值: {len(all_annotations)} 帧！")

# --------------------------------------------------------------------------
# Phase 2: 折衣服阶段 (帧 150~2000) - 手腕刚性几何空间双相机严格成对锁定
# --------------------------------------------------------------------------
print("\n--> [Phase 2/3] 折衣服阶段手腕刚性空间双相机严格配对 (必须双框齐全，杜绝单框)...")
pose_weights = os.path.join(PROJECT_ROOT, "yolov8n-pose.pt")
pose_model = YOLO(pose_weights)

mid_x = orig_w // 2
BOX_W, BOX_H = 190, 150 # 原图 3248x2464 尺度下 DJI Action 4 物理边界

folding_samples = range(150, min(2000, total_frames), 12)
for f_idx in folding_samples:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: continue

    results = pose_model.predict(frame, imgsz=1280, conf=0.05, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints) == 0:
        continue

    kpts_xy = results.keypoints.xy.cpu().numpy()
    kpts_conf = results.keypoints.conf.cpu().numpy()
    num_persons = len(kpts_xy)

    left_candidates = []
    right_candidates = []

    for p_idx in range(num_persons):
        for kp_id in [9, 10]:
            u, v = kpts_xy[p_idx, kp_id]
            c = kpts_conf[p_idx, kp_id]
            if c < 0.10 or v < 1100: continue

            if u < mid_x:
                left_candidates.append((u, v, c))
            else:
                right_candidates.append((u, v, c))

    if not left_candidates or not right_candidates:
        # 如果有一只手没检出，直接放弃该帧，绝不生成“单框”残缺图片！
        continue

    # 分别取左右置信度最高的手腕
    best_l = max(left_candidates, key=lambda x: x[1]*0.7 + x[2]*1000)
    best_r = max(right_candidates, key=lambda x: x[1]*0.7 + x[2]*1000)

    lw_u, lw_v, _ = best_l
    rw_u, rw_v, _ = best_r

    # 左右手腕横向间距必须大于 350 像素（杜绝左右混淆在同一只手上）
    if rw_u - lw_u < 350:
        continue

    # 左手相机：刚性位于左手腕内下侧
    cam_l_cx = lw_u + 130
    cam_l_cy = lw_v + 210
    bl_x1 = int(max(0, cam_l_cx - BOX_W / 2))
    bl_y1 = int(max(0, cam_l_cy - BOX_H / 2))
    bl_x2 = int(min(orig_w, cam_l_cx + BOX_W / 2))
    bl_y2 = int(min(orig_h, cam_l_cy + BOX_H / 2))

    # 右手相机：刚性位于右手腕内下侧
    cam_r_cx = rw_u - 170
    cam_r_cy = rw_v + 210
    br_x1 = int(max(0, cam_r_cx - BOX_W / 2))
    br_y1 = int(max(0, cam_r_cy - BOX_H / 2))
    br_x2 = int(min(orig_w, cam_r_cx + BOX_W / 2))
    br_y2 = int(min(orig_h, cam_r_cy + BOX_H / 2))

    # 左右框绝不能相交或倒置
    if br_x1 > bl_x2:
        all_annotations[f_idx] = [(bl_x1, bl_y1, bl_x2, bl_y2), (br_x1, br_y1, br_x2, br_y2)]

print(f"  -> Phase 2 成功提取双相机成对折衣服样本: {len(all_annotations)} 帧！")
print(f"--> [数据汇总] 全场景高质量训练集总帧数: {len(all_annotations)} 帧（每帧 100% 包含双相机）！")

# --------------------------------------------------------------------------
# Phase 3: 保存 1280 分辨率标准数据集及全新双框预览图
# --------------------------------------------------------------------------
print(f"\n--> [Phase 3/3] 正在保存 1280 分辨率标准数据集及全新双框预览图...")
np.random.seed(42)
all_keys = sorted(list(all_annotations.keys()))
val_set = set(np.random.choice(all_keys, size=max(1, int(len(all_keys) * 0.15)), replace=False))

for idx, f_idx in enumerate(all_keys):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: continue

    split = "val" if f_idx in val_set else "train"
    stem = f"scene_{f_idx:04d}"

    img_1280 = cv2.resize(frame, (TRAIN_W, TRAIN_H))
    cv2.imwrite(os.path.join(DATASET_DIR, "images", split, f"{stem}.jpg"), img_1280)

    vis = img_1280.copy()
    lbl_file = os.path.join(DATASET_DIR, "labels", split, f"{stem}.txt")

    with open(lbl_file, "w") as f_out:
        for (bx1, by1, bx2, by2) in all_annotations[f_idx]:
            x1_s = bx1 * scale_factor
            y1_s = by1 * scale_factor
            x2_s = bx2 * scale_factor
            y2_s = by2 * scale_factor

            cx = (x1_s + x2_s) / 2.0 / float(TRAIN_W)
            cy = (y1_s + y2_s) / 2.0 / float(TRAIN_H)
            w = (x2_s - x1_s) / float(TRAIN_W)
            h = (y2_s - y1_s) / float(TRAIN_H)

            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            w = float(np.clip(w, 0.01, 1.0))
            h = float(np.clip(h, 0.01, 1.0))

            f_out.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            rx1, ry1 = int((cx - w/2) * TRAIN_W), int((cy - h/2) * TRAIN_H)
            rx2, ry2 = int((cx + w/2) * TRAIN_W), int((cy + h/2) * TRAIN_H)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(vis, f"Camera ({rx2-rx1}x{ry2-ry1})", (rx1, max(15, ry1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 重点抽查预览保存 (每隔 3 张存一张预览)
    if idx % 3 == 0 or f_idx in [0, 30, 60, 90, 200, 500, 1000, 1500]:
        cv2.imwrite(os.path.join(vis_dir, f"vis_{stem}.jpg"), vis)

cap.release()

yaml_path = os.path.join(PROJECT_ROOT, "dji_full_scene_server.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""path: /mnt/data/home/zhoujiayan/BoxDreamer/yolo_full_scene_dataset
train: images/train
val: images/val
names:
  0: dji_camera
""")

print(f"\n🎉 100% 双框齐备数据集构建完毕！")
print(f"  - 数据集存储目录: {DATASET_DIR}")
print(f"  - 双框预览图目录: {vis_dir}")
