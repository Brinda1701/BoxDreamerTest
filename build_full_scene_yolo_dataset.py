# build_full_scene_yolo_dataset.py - 构建真实全画幅尺度、彻底紧致的 YOLO 训练集
import os
import sys
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(PROJECT_ROOT, "test_video", "head_left_rgb_raw.mp4")

if not os.path.exists(video_path):
    print(f"❌ 找不到视频: {video_path}")
    sys.exit(1)

# 目标输出数据集目录 (规范的 YOLO 标准结构)
DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_full_scene_dataset")
for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

vis_dir = os.path.join(DATASET_DIR, "vis_preview")
os.makedirs(vis_dir, exist_ok=True)

# 相机内参 (3248x2464)
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
    [x_L, y_B, z_F],
    [x_L, y_T, z_F],
    [x_L, y_B, z_R],
    [x_L, y_T, z_R],
    [x_R, y_B, z_F],
    [x_R, y_T, z_F],
    [x_R, y_B, z_R],
    [x_R, y_T, z_R],
], dtype=np.float64)

# 统一保存为 1280 标准分辨率训练图片（无损等比例保持 3248x2464 真实视野与纵横比）
TRAIN_W = 1280

cap = cv2.VideoCapture(video_path)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

scale_factor = TRAIN_W / float(orig_w)
TRAIN_H = int(orig_h * scale_factor)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
detector = cv2.aruco.ArucoDetector(aruco_dict)

print(f"--> [Phase 1/2] 正在从真实视频提取【AprilTag 精确空间真值】全景样本 (帧 0~400)...")
frame_samples = {}

# 1. 抽取前 400 帧中带有 AprilTag 毫米级真值的全景帧
for f_idx in range(0, min(400, total_frames), 2):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None: continue

    bboxes = []
    for i, cid in enumerate(ids.ravel()):
        if cid not in [18, 19]: continue
        pts = corners[i][0]
        ok, rvec, tvec = cv2.solvePnP(tag_3d, pts, K, dist_coeffs)
        if not ok: continue

        proj_pts, _ = cv2.projectPoints(corners_bop_in_tag, rvec, tvec, K, dist_coeffs)
        p2d = proj_pts.reshape(-1, 2)

        # 在 3248x2464 原图下的真实相机四边边界 (消除 3D 旋转过度膨胀，按真实相机本体收紧 92%)
        min_x, max_x = np.min(p2d[:, 0]), np.max(p2d[:, 0])
        min_y, max_y = np.min(p2d[:, 1]), np.max(p2d[:, 1])

        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        bw = (max_x - min_x) * 0.92
        bh = (max_y - min_y) * 0.92

        # 转换为 1280 尺度下的坐标
        cx_1280 = cx * scale_factor
        cy_1280 = cy * scale_factor
        bw_1280 = bw * scale_factor
        bh_1280 = bh * scale_factor

        # 转换为 YOLO 规范化 [0, 1]
        norm_xc = cx_1280 / float(TRAIN_W)
        norm_yc = cy_1280 / float(TRAIN_H)
        norm_w = bw_1280 / float(TRAIN_W)
        norm_h = bh_1280 / float(TRAIN_H)

        bboxes.append((norm_xc, norm_yc, norm_w, norm_h))

    if bboxes:
        frame_samples[f_idx] = {
            "frame": cv2.resize(frame, (TRAIN_W, TRAIN_H)),
            "boxes": bboxes,
            "type": "tag"
        }

print(f"  -> 前半段共获得 {len(frame_samples)} 帧毫米级真值全景样本！")

# 2. 从折衣服阶段 (帧 400 ~ 2000) 提取双腕相机全景样本
print(f"--> [Phase 2/2] 正在从真实视频提取【折衣服关键动作】全景样本 (帧 400~2000)...")
from ultralytics import YOLO
pose_model = YOLO("yolov8n-pose.pt")
mid_x = orig_w // 2

folding_count = 0
for f_idx in range(400, min(2000, total_frames), 25):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret: break

    results = pose_model.predict(frame, imgsz=1280, conf=0.05, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints) == 0: continue

    kpts_xy = results.keypoints.xy.cpu().numpy()
    kpts_conf = results.keypoints.conf.cpu().numpy()

    left_cands, right_cands = [], []
    for p_idx in range(len(kpts_xy)):
        for kp_id in [9, 10]:
            u, v = kpts_xy[p_idx, kp_id]
            c = kpts_conf[p_idx, kp_id]
            if c < 0.05 or v < 1100: continue
            cand = (u, v, c)
            if u < mid_x: left_cands.append(cand)
            else: right_cands.append(cand)

    bboxes = []
    # 提取左右手腕处的相机紧致边界框
    # 在 1280 分辨率下，真实相机本体尺寸约 52px 宽、36px 高
    for cands in [left_cands, right_cands]:
        if not cands: continue
        best_u, best_v, _ = max(cands, key=lambda x: x[1] * 0.7 + x[2] * 1000)

        # 映射到 1280 坐标：相机物理安装在手腕带朝下/内侧方位，中心在手腕关节顺着下垂方向约 +42px 处
        cx_1280 = best_u * scale_factor
        cy_1280 = (best_v * scale_factor) + 42.0
        bw_1280 = 56.0 # 紧贴机身宽度 (px)
        bh_1280 = 36.0 # 紧贴机身高度 (px)

        norm_xc = cx_1280 / float(TRAIN_W)
        norm_yc = cy_1280 / float(TRAIN_H)
        norm_w = bw_1280 / float(TRAIN_W)
        norm_h = bh_1280 / float(TRAIN_H)
        bboxes.append((norm_xc, norm_yc, norm_w, norm_h))

    if bboxes:
        frame_samples[f_idx] = {
            "frame": cv2.resize(frame, (TRAIN_W, TRAIN_H)),
            "boxes": bboxes,
            "type": "folding"
        }
        folding_count += 1

cap.release()
print(f"  -> 后半段折衣服补充 {folding_count} 帧全景样本！")
print(f"--> [汇总] 真实全画幅数据集总计提取: {len(frame_samples)} 帧！")

# 3. 按照 85% 训练集、15% 验证集分流并生成 YOLO 标签
np.random.seed(42)
all_frame_indices = sorted(list(frame_samples.keys()))
val_set = set(np.random.choice(all_frame_indices, size=int(len(all_frame_indices) * 0.15), replace=False))

saved_count = 0
for idx, f_idx in enumerate(all_frame_indices):
    split = "val" if f_idx in val_set else "train"
    sample = frame_samples[f_idx]
    img = sample["frame"]
    boxes = sample["boxes"]

    stem = f"real_full_{f_idx:04d}"
    img_path = os.path.join(DATASET_DIR, "images", split, f"{stem}.jpg")
    lbl_path = os.path.join(DATASET_DIR, "labels", split, f"{stem}.txt")

    cv2.imwrite(img_path, img)

    vis = img.copy()
    with open(lbl_path, "w") as f_lbl:
        for (xc, yc, w, h) in boxes:
            f_lbl.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            # 绘制可视化预览
            px1 = int((xc - w/2.0) * TRAIN_W)
            py1 = int((yc - h/2.0) * TRAIN_H)
            px2 = int((xc + w/2.0) * TRAIN_W)
            py2 = int((yc + h/2.0) * TRAIN_H)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 255), 2)
            cv2.putText(vis, f"Camera {px2-px1}x{py2-py1}", (px1, max(15, py1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 保存抽样预览图供人工直观检查
    if idx % 10 == 0:
        cv2.imwrite(os.path.join(vis_dir, f"vis_{stem}.jpg"), vis)

    saved_count += 1

# 4. 生成训练配置文件
yaml_content = f"""path: /mnt/data/home/zhoujiayan/BoxDreamer/yolo_full_scene_dataset
train: images/train
val: images/val
names:
  0: dji_camera
"""
with open(os.path.join(PROJECT_ROOT, "dji_full_scene_server.yaml"), "w") as f:
    f.write(yaml_content)

print(f"🎉 数据集构建完毕！\n  - 数据集保存在: {DATASET_DIR}\n  - 抽样标注检查图保存在: {vis_dir}\n  - 服务器配置文件已生成: dji_full_scene_server.yaml")

