# extract_real_dataset.py - Auto-extract Real Camera Training Dataset with Exact 3D Ground Truth

import os
import sys
import json
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Camera intrinsics for 3248x2464
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

# DJI Action 4 body dimensions in mm
w = 70.5
h = 44.2
d = 32.4
cx_tag = 22.5
cy_tag = 0.0

x_L = cx_tag + w/2
x_R = cx_tag - w/2
y_T = -h/2
y_B = +h/2
z_F = d
z_R = 0.0

# 8 corners corresponding exactly to BOP / dji_bbox_corners.npy:
# 0: Left, Front, Bottom
# 1: Left, Front, Top
# 2: Left, Rear, Bottom
# 3: Left, Rear, Top
# 4: Right, Front, Bottom
# 5: Right, Front, Top
# 6: Right, Rear, Bottom
# 7: Right, Rear, Top
corners_bop_in_tag = np.array([
    [x_L, y_B, z_F], # 0
    [x_L, y_T, z_F], # 1
    [x_L, y_B, z_R], # 2
    [x_L, y_T, z_R], # 3
    [x_R, y_B, z_F], # 4
    [x_R, y_T, z_F], # 5
    [x_R, y_B, z_R], # 6
    [x_R, y_T, z_R], # 7
], dtype=np.float64)

EDGES_12 = [
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7)
]

def safe_crop(img, cx, cy, crop_size):
    half = crop_size // 2
    h_img, w_img = img.shape[:2]
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_size, y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w_img)
    pad_bottom = max(0, y2 - h_img)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
        crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
    else:
        crop = img[y1:y2, x1:x2]

    return crop, (x1, y1)

def main():
    video_path = os.path.join(PROJECT_ROOT, 'test_video', 'head_left_rgb_raw.mp4')
    out_dir = os.path.join(PROJECT_ROOT, 'real_dataset')
    img_dir = os.path.join(out_dir, 'images')
    vis_dir = os.path.join(out_dir, 'vis')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(aruco_dict)

    samples = []
    max_frames = 110
    print(f'Extracting real dataset from frames 0 to {max_frames}...')

    for f_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        for i, cid in enumerate(ids.ravel()):
            if cid not in [18, 19]:
                continue

            pts = corners[i][0]
            ok, rvec, tvec = cv2.solvePnP(tag_3d, pts, K, dist_coeffs)
            if not ok:
                continue

            # Project 8 3D corners to 2D image coordinates
            proj_pts, _ = cv2.projectPoints(corners_bop_in_tag, rvec, tvec, K, dist_coeffs)
            p2d = proj_pts.reshape(-1, 2)

            # Center of the 8 corners in image
            cx = int(np.mean(p2d[:, 0]))
            cy = int(np.mean(p2d[:, 1]))

            # Crop size to comfortably contain the camera body and context
            crop_size = 640
            crop, (x1, y1) = safe_crop(frame, cx, cy, crop_size)

            # Map corners to crop coordinates
            p_crop = p2d - np.array([x1, y1])

            # Resize to 224x224
            crop_224 = cv2.resize(crop, (224, 224))
            scale_factor = 224.0 / float(crop_size)
            p_224 = p_crop * scale_factor

            # Save 224 image
            img_name = f'frame_{f_idx:04d}_cam{cid}.jpg'
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, crop_224)

            # Verification image
            vis = crop_224.copy()
            for idx1, idx2 in EDGES_12:
                pt1 = (int(p_224[idx1][0]), int(p_224[idx1][1]))
                pt2 = (int(p_224[idx2][0]), int(p_224[idx2][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            for pt in p_224:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(vis_dir, f'vis_{img_name}'), vis)

            samples.append({
                'image_file': img_name,
                'frame_idx': f_idx,
                'camera_id': int(cid),
                'corners_224': p_224.tolist()
            })

    cap.release()

    labels_path = os.path.join(out_dir, 'labels.json')
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)

    print(f'Extracted {len(samples)} high-quality real samples!')
    print(f'Labels saved to: {labels_path}')

if __name__ == '__main__':
    main()
