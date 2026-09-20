# prepare_mixed_yolo_dataset.py - 自动将真实手腕数据(real_dataset)与合成BOP数据混合打包为YOLO训练集
import os
import sys
import json
import shutil
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
bop_scene_dir = os.path.join(PROJECT_ROOT, "bop_datasets", "dji", "train_pbr", "000000")
real_dataset_dir = os.path.join(PROJECT_ROOT, "real_dataset")

# 1. 目标混合训练集目录
mixed_dir = os.path.join(PROJECT_ROOT, "yolo_mixed_dataset")
mixed_img_train = os.path.join(mixed_dir, "images", "train")
mixed_lbl_train = os.path.join(mixed_dir, "labels", "train")
mixed_img_val = os.path.join(mixed_dir, "images", "val")
mixed_lbl_val = os.path.join(mixed_dir, "labels", "val")

for d in [mixed_img_train, mixed_lbl_train, mixed_img_val, mixed_lbl_val]:
    os.makedirs(d, exist_ok=True)

print("--> [Step 1/3] 正在转换 real_dataset 真实手腕图片为 YOLO 标签...")
real_labels_path = os.path.join(real_dataset_dir, "labels.json")
real_img_dir = os.path.join(real_dataset_dir, "images")

with open(real_labels_path, "r", encoding="utf-8") as f:
    real_samples = json.load(f)

# 将 80% 真实样本作为训练集，20% 作为验证集
np.random.seed(42)
indices = np.random.permutation(len(real_samples))
split_idx = int(len(real_samples) * 0.8)
train_indices = set(indices[:split_idx])

real_count = 0
for idx, item in enumerate(real_samples):
    img_name = item["image_file"]
    src_img = os.path.join(real_img_dir, img_name)
    if not os.path.exists(src_img):
        continue

    # 提取 8 角点并计算紧贴相机的 2D 外接框
    pts = np.array(item["corners_224"], dtype=np.float32)
    min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
    min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])

    cx = np.clip((min_x + max_x) / 2.0 / 224.0, 0.0, 1.0)
    cy = np.clip((min_y + max_y) / 2.0 / 224.0, 0.0, 1.0)
    bw = np.clip((max_x - min_x) / 224.0, 0.01, 1.0)
    bh = np.clip((max_y - min_y) / 224.0, 0.01, 1.0)

    # 分流到 train / val
    is_train = idx in train_indices
    dst_img_dir = mixed_img_train if is_train else mixed_img_val
    dst_lbl_dir = mixed_lbl_train if is_train else mixed_lbl_val

    dst_img = os.path.join(dst_img_dir, f"real_{img_name}")
    dst_lbl = os.path.join(dst_lbl_dir, f"real_{os.path.splitext(img_name)[0]}.txt")

    shutil.copyfile(src_img, dst_img)
    with open(dst_lbl, "w") as f_lbl:
        f_lbl.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    real_count += 1

print(f"--> [Step 1 成功] 共导入 {real_count} 张带有真实手腕/反光的切片图片！")

print("--> [Step 2/3] 正在混合 BOP 合成渲染样本...")
bop_rgb_dir = os.path.join(bop_scene_dir, "rgb")
bop_lbl_dir = os.path.join(bop_scene_dir, "labels")

bop_files = [f for f in os.listdir(bop_rgb_dir) if f.endswith(('.jpg', '.png'))]
bop_indices = np.random.permutation(len(bop_files))
bop_split = int(len(bop_files) * 0.8)
bop_train_set = set(bop_indices[:bop_split])

bop_count = 0
for idx, f in enumerate(bop_files):
    base_name = os.path.splitext(f)[0]
    src_img = os.path.join(bop_rgb_dir, f)
    src_lbl = os.path.join(bop_lbl_dir, f"{base_name}.txt")
    if not os.path.exists(src_lbl):
        continue

    is_train = idx in bop_train_set
    dst_img_dir = mixed_img_train if is_train else mixed_img_val
    dst_lbl_dir = mixed_lbl_train if is_train else mixed_lbl_val

    dst_img = os.path.join(dst_img_dir, f"bop_{f}")
    dst_lbl = os.path.join(dst_lbl_dir, f"bop_{base_name}.txt")

    shutil.copyfile(src_img, dst_img)
    shutil.copyfile(src_lbl, dst_lbl)
    bop_count += 1

print(f"--> [Step 2 成功] 共导入 {bop_count} 张 BOP 合成渲染样本！")
print(f"--> [汇总] 混合数据集总计: 训练集 {len(os.listdir(mixed_img_train))} 张, 验证集 {len(os.listdir(mixed_img_val))} 张！")

# 3. 自动生成本地和服务器的两个 yaml 配置文件
yaml_local = os.path.join(PROJECT_ROOT, "dji_mixed_yolo_local.yaml")
with open(yaml_local, "w", encoding="utf-8") as f:
    f.write(f"""# 本地 Windows 混合数据集配置
path: {mixed_dir.replace('\\', '/')}
train: images/train
val: images/val
names:
  0: dji_camera
""")

yaml_server = os.path.join(PROJECT_ROOT, "dji_mixed_yolo_server.yaml")
with open(yaml_server, "w", encoding="utf-8") as f:
    f.write("""# 服务器 Linux 混合数据集配置
path: /mnt/data/home/zhoujiayan/BoxDreamer/yolo_mixed_dataset
train: images/train
val: images/val
names:
  0: dji_camera
""")

print(f"--> [Step 3/3 成功] 已自动生成本地与服务器数据集配置文件：\n  - {yaml_local}\n  - {yaml_server}")

