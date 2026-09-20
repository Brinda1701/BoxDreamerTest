import os, json, cv2, numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
bop_scene_dir = os.path.join(PROJECT_ROOT, "bop_datasets", "dji", "train_pbr", "000000")
corners_3d_path = os.path.join(PROJECT_ROOT, "dji_bbox_corners.npy")
corners_3d = np.load(corners_3d_path) * 1000.0

gt_path = os.path.join(bop_scene_dir, "scene_gt.json")
cam_path = os.path.join(bop_scene_dir, "scene_camera.json")
with open(gt_path) as f: scene_gt = json.load(f)
with open(cam_path) as f: scene_cam = json.load(f)

label_dir = os.path.join(bop_scene_dir, "labels")
os.makedirs(label_dir, exist_ok=True)

# 动态读取实际图像尺寸
sample_img_path = os.path.join(bop_scene_dir, "rgb", "000000.jpg")
if not os.path.exists(sample_img_path):
    sample_img_path = os.path.join(bop_scene_dir, "rgb", "000000.png")
_sample = cv2.imread(sample_img_path)
orig_h, orig_w = _sample.shape[:2]

for fid, gt_list in scene_gt.items():
    R = np.array(gt_list[0]["cam_R_m2c"]).reshape(3, 3)
    t = np.array(gt_list[0]["cam_t_m2c"]).reshape(3, 1)
    K = np.array(scene_cam[fid]["cam_K"]).reshape(3, 3)

    pts_cam = R @ corners_3d.T + t
    pts_homo = K @ pts_cam
    u = pts_homo[0] / pts_homo[2]
    v = pts_homo[1] / pts_homo[2]

    # 计算紧凑2D边界框：消除3D长方体倾斜投影时AABB包围盒虚胖放大的多余背景空间
    cx_px = (np.min(u) + np.max(u)) / 2.0
    cy_px = (np.min(v) + np.max(v)) / 2.0
    bw_px = (np.max(u) - np.min(u)) * 0.90
    bh_px = (np.max(v) - np.min(v)) * 0.90

    cx = np.clip(cx_px / orig_w, 0.0, 1.0)
    cy = np.clip(cy_px / orig_h, 0.0, 1.0)
    bw = np.clip(bw_px / orig_w, 0.001, 1.0)
    bh = np.clip(bh_px / orig_h, 0.001, 1.0)

    txt_name = f"{int(fid):06d}.txt"
    # 写入两处确保不同YOLO版本的寻址匹配
    with open(os.path.join(label_dir, txt_name), "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

# 为满足 YOLOv8 官方 'images' <-> 'labels' 自动路径替换规则，在 train_pbr/000000 建立标准目录结构
yolo_img_dir = os.path.join(bop_scene_dir, "images")
os.makedirs(yolo_img_dir, exist_ok=True)
rgb_dir = os.path.join(bop_scene_dir, "rgb")

# 建立无开销的硬链接或软引用，确保 images/ 与 labels/ 完美镜像对应
import shutil
for f in os.listdir(rgb_dir):
    if f.endswith(('.jpg', '.png')):
        src = os.path.join(rgb_dir, f)
        dst = os.path.join(yolo_img_dir, f)
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except Exception:
                shutil.copyfile(src, dst)

print(f"--> [YOLO Dataset] 成功构建 YOLO 官方标准 images/ 与 labels/ 镜像数据集 (共 {len(scene_gt)} 张)！")


# from ultralytics import YOLO

# # 载入预训练的轻量检测器
# model = YOLO('yolov8n.pt') 

# # 训练自定义相机检测（通常 20-30 个 epoch，十几分钟即可收敛）
# model.train(
#     data="dji_camera.yaml", # 包含 train/val 图像路径与 class 0: dji_camera
#     epochs=30,
#     imgsz=640,
#     batch=16
# )
# # 训练完成后会生成 runs/detect/train/weights/best.pt