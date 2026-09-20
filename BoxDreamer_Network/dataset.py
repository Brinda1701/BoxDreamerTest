import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def apply_comprehensive_aug(img_bgr, pts_2d, img_size=224, p_aug=0.90):
    """
    全方位工业级多模态数据增强管线：
    1. 仿射旋转与微倾斜 (同步严格更新 8 个角点坐标)
    2. 极端暗光、超低对比度与非线性 Gamma
    3. 模拟手掌/手指的不规则局部遮挡 (Random Cutout)
    4. 模拟玻璃屏幕日光灯局部反光白斑 (Specular Flare)
    5. 色温与白平衡漂移 (HSV Jitter)
    6. 高 ISO 传感器弱光噪点与动态运动拖影 (Motion Blur)
    """
    if np.random.uniform(0, 1) > p_aug:
        return img_bgr, pts_2d

    img = img_bgr.copy()
    pts = pts_2d.copy()

    # =========================================================================
    # 1. 几何增强：随机平面旋转 (±22 度) 与微仿射 (同步变换 8 个角点)
    # =========================================================================
    if np.random.uniform(0, 1) < 0.65:
        angle = np.random.uniform(-22.0, 22.0)
        scale = np.random.uniform(0.92, 1.08)
        center = (img_size / 2.0, img_size / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 变换图像
        img = cv2.warpAffine(img, M, (img_size, img_size), borderMode=cv2.BORDER_REFLECT)

        # 严格同步变换 8 个角点
        pts_homo = np.hstack([pts, np.ones((8, 1))])
        pts = (M @ pts_homo.T).T

    # =========================================================================
    # 2. 光学与环境：暗光、低对比度、Gamma 压缩
    # =========================================================================
    img_float = img.astype(np.float32)

    # 亮度波动 (0.35 ~ 1.45)
    brightness_factor = np.random.uniform(0.35, 1.45)
    img_float = img_float * brightness_factor

    # 对比度压低 (0.30 ~ 1.40)
    contrast_factor = np.random.uniform(0.30, 1.40)
    mean_val = np.mean(img_float, axis=(0, 1), keepdims=True)
    img_float = (img_float - mean_val) * contrast_factor + mean_val

    # Gamma 变换 (模拟暗区动态压缩)
    if np.random.uniform(0, 1) < 0.6:
        gamma = np.random.uniform(0.5, 2.2)
        inv_gamma = 1.0 / gamma
        img_float = np.clip(img_float, 0, 255) / 255.0
        img_float = np.power(img_float, inv_gamma) * 255.0

    img = np.clip(img_float, 0, 255).astype(np.uint8)

    # =========================================================================
    # 3. 颜色与白平衡漂移 (HSV 空间变换)
    # =========================================================================
    if np.random.uniform(0, 1) < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-15, 15)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.65, 1.35), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # =========================================================================
    # 4. 模拟手掌/手指遮挡 (Random Cutout / Erasing)
    # 随机在机身上涂抹 1~2 块随机灰度/肤色块，迫使网络学会用未遮挡角点推算全局
    # =========================================================================
    if np.random.uniform(0, 1) < 0.55:
        num_holes = np.random.randint(1, 3)
        for _ in range(num_holes):
            hole_w = np.random.randint(25, 65)
            hole_h = np.random.randint(25, 65)
            hx = np.random.randint(10, img_size - hole_w - 10)
            hy = np.random.randint(10, img_size - hole_h - 10)
            # 随机模拟手部肤色或阴影黑色
            color_choice = np.random.choice(["skin", "shadow", "noise"])
            if color_choice == "skin":
                fill_color = [np.random.randint(130, 200), np.random.randint(150, 220), np.random.randint(180, 240)]
            elif color_choice == "shadow":
                fill_color = [np.random.randint(15, 50), np.random.randint(15, 50), np.random.randint(15, 50)]
            else:
                fill_color = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
            cv2.rectangle(img, (hx, hy), (hx + hole_w, hy + hole_h), fill_color, -1)

    # =========================================================================
    # 5. 模拟日光灯局部反光白斑 (Specular Glare)
    # =========================================================================
    if np.random.uniform(0, 1) < 0.4:
        center_g = (np.random.randint(30, img_size - 30), np.random.randint(30, img_size - 30))
        axes = (np.random.randint(15, 45), np.random.randint(8, 25))
        angle_g = np.random.randint(0, 180)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.ellipse(mask, center_g, axes, angle_g, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        alpha = np.random.uniform(0.35, 0.75)
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) + mask * alpha, 0, 255).astype(np.uint8)

    # =========================================================================
    # 6. 高 ISO 传感器弱光噪点与动态模糊
    # =========================================================================
    if np.random.uniform(0, 1) < 0.35:
        noise = np.random.normal(0, np.random.uniform(3.0, 10.0), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if np.random.uniform(0, 1) < 0.3:
        ksize = np.random.choice([3, 5])
        kernel = np.zeros((ksize, ksize))
        kernel[int((ksize - 1) / 2), :] = np.ones(ksize) / ksize
        img = cv2.filter2D(img, -1, kernel)

    return img, pts


class DJIActionPoseDataset(Dataset):
    def __init__(
        self,
        bop_scene_dir: str,
        corners_3d_path: str,
        img_size: int = 224,
        sigma: float = 3.0,
        is_train: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.sigma = sigma
        self.is_train = is_train

        self.corners_3d = np.load(corners_3d_path) * 1000.0

        gt_path = os.path.join(bop_scene_dir, "scene_gt.json")
        cam_path = os.path.join(bop_scene_dir, "scene_camera.json")
        with open(gt_path, "r") as f:
            self.scene_gt = json.load(f)
        with open(cam_path, "r") as f:
            self.scene_cam = json.load(f)

        self.frame_ids = sorted(list(self.scene_gt.keys()), key=lambda x: int(x))
        self.rgb_dir = os.path.join(bop_scene_dir, "rgb")

    def __len__(self):
        return len(self.frame_ids)

    def _project_3d_to_2d(self, R, t, K):
        pts_cam = R @ self.corners_3d.T + t
        pts_homo = K @ pts_cam
        u = pts_homo[0] / pts_homo[2]
        v = pts_homo[1] / pts_homo[2]
        pts_2d = np.stack([u, v], axis=1)
        return pts_2d

    def _crop_and_resize(self, img, pts_2d_orig, padding_ratio=1.35, is_train=True):
        h, w = img.shape[:2]
        min_xy = np.min(pts_2d_orig, axis=0)
        max_xy = np.max(pts_2d_orig, axis=0)

        cx = (min_xy[0] + max_xy[0]) / 2.0
        cy = (min_xy[1] + max_xy[1]) / 2.0
        box_w = max_xy[0] - min_xy[0]
        box_h = max_xy[1] - min_xy[1]

        crop_size = int(max(box_w, box_h) * padding_ratio)
        crop_size = max(crop_size, 32)

        if is_train:
            scale_jitter = np.random.uniform(0.9, 1.15)
            crop_size = int(crop_size * scale_jitter)
            shift_max = 0.10 * crop_size
            cx += np.random.uniform(-shift_max, shift_max)
            cy += np.random.uniform(-shift_max, shift_max)

        half = crop_size // 2
        x1, y1 = int(cx - half), int(cy - half)
        x2, y2 = x1 + crop_size, y1 + crop_size

        pad_l = max(0, -x1)
        pad_t = max(0, -y1)
        pad_r = max(0, x2 - w)
        pad_b = max(0, y2 - h)

        if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
            padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT)
            crop_img = padded[y1 + pad_t: y2 + pad_t, x1 + pad_l: x2 + pad_l]
        else:
            crop_img = img[y1: y2, x1: x2]

        pts_2d_crop = (pts_2d_orig - np.array([x1, y1])) * (self.img_size / float(crop_size))
        crop_resized = cv2.resize(crop_img, (self.img_size, self.img_size))
        return crop_resized, pts_2d_crop

    def _generate_gaussian_heatmap(self, pts_2d):
        heatmaps = np.zeros((8, self.img_size, self.img_size), dtype=np.float32)
        grid_x, grid_y = np.meshgrid(np.arange(self.img_size), np.arange(self.img_size))
        for i in range(8):
            u, v = pts_2d[i]
            dist_sq = (grid_x - u) ** 2 + (grid_y - v) ** 2
            heatmaps[i] = np.exp(-dist_sq / (2 * self.sigma ** 2))
        return heatmaps

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        img_name = f"{int(frame_id):06d}.jpg"
        img_path = os.path.join(self.rgb_dir, img_name)
        if not os.path.exists(img_path):
            img_name = f"{int(frame_id):06d}.png"
            img_path = os.path.join(self.rgb_dir, img_name)

        img_bgr = cv2.imread(img_path)

        gt_info = self.scene_gt[frame_id][0]
        cam_info = self.scene_cam[frame_id]

        R = np.array(gt_info["cam_R_m2c"]).reshape(3, 3)
        t = np.array(gt_info["cam_t_m2c"]).reshape(3, 1)
        K = np.array(cam_info["cam_K"]).reshape(3, 3)

        pts_2d_orig = self._project_3d_to_2d(R, t, K)
        crop_bgr, pts_2d_crop = self._crop_and_resize(img_bgr, pts_2d_orig, padding_ratio=1.35, is_train=self.is_train)

        # 训练期应用全方位多模态增强（旋转、遮挡、反光、暗光、白平衡）
        if self.is_train:
            crop_bgr, pts_2d_crop = apply_comprehensive_aug(crop_bgr, pts_2d_crop, self.img_size, p_aug=0.90)

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0

        heatmaps = self._generate_gaussian_heatmap(pts_2d_crop)
        heatmaps_tensor = torch.from_numpy(heatmaps).float()

        return {
            "image": img_tensor,
            "heatmap": heatmaps_tensor,
            "pts_2d": torch.from_numpy(pts_2d_crop).float(),
            "frame_id": frame_id
        }


class RealDJIDataset(Dataset):
    def __init__(self, dataset_dir: str, img_size: int = 224, sigma: float = 3.0, is_train: bool = True):
        super().__init__()
        labels_path = os.path.join(dataset_dir, "labels.json")
        self.img_dir = os.path.join(dataset_dir, "images")
        self.img_size = img_size
        self.sigma = sigma
        self.is_train = is_train

        with open(labels_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        self.grid_x, self.grid_y = np.meshgrid(np.arange(img_size), np.arange(img_size))

    def __len__(self):
        return len(self.samples)

    def _generate_gaussian_heatmap(self, pts_2d):
        heatmaps = np.zeros((8, self.img_size, self.img_size), dtype=np.float32)
        for i in range(8):
            u, v = pts_2d[i]
            dist_sq = (self.grid_x - u) ** 2 + (self.grid_y - v) ** 2
            heatmaps[i] = np.exp(-dist_sq / (2 * self.sigma ** 2))
        return heatmaps

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_file = item.get("image_file") or item.get("img_name")
        img_path = os.path.join(self.img_dir, img_file)
        bgr = cv2.imread(img_path)

        if bgr.shape[0] != self.img_size or bgr.shape[1] != self.img_size:
            bgr = cv2.resize(bgr, (self.img_size, self.img_size))

        pts_2d = np.array(item.get("corners_224") or item.get("corners_2d"), dtype=np.float32)

        # 真实数据在训练时同样应用全套增强（同步旋转与角点映射）
        if self.is_train:
            bgr, pts_2d = apply_comprehensive_aug(bgr, pts_2d, self.img_size, p_aug=0.90)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        heatmaps = self._generate_gaussian_heatmap(pts_2d)

        return {
            "image": img_tensor,
            "heatmap": torch.from_numpy(heatmaps).float(),
            "pts_2d": torch.from_numpy(pts_2d).float(),
            "frame_id": str(item.get("frame_idx", idx))
        }
