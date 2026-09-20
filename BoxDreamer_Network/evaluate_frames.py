import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
for _p in [str(_SCRIPT_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .models import BoxDreamerModel
    from .dataset import DJIActionPoseDataset
except (ImportError, ValueError):
    from models import BoxDreamerModel
    from dataset import DJIActionPoseDataset


EDGES_12 = [
    (0, 1), (2, 3), (4, 5), (6, 7), # 沿 Z 轴 4 条棱
    (0, 2), (1, 3), (4, 6), (5, 7), # 沿 Y 轴 4 条棱
    (0, 4), (1, 5), (2, 6), (3, 7)  # 沿 X 轴 4 条棱
]


def draw_box(img, pts_2d, color, radius=4, thickness=2):
    for pt in pts_2d:
        p = (int(round(pt[0])), int(round(pt[1])))
        cv2.circle(img, p, radius=radius, color=color, thickness=-1)

    for i1, i2 in EDGES_12:
        p1 = (int(round(pts_2d[i1][0])), int(round(pts_2d[i1][1])))
        p2 = (int(round(pts_2d[i2][0])), int(round(pts_2d[i2][1])))
        cv2.line(img, p1, p2, color=color, thickness=thickness)


def run_batch_evaluation(
    weights_path: str = None,
    scene_dir: str = None,
    corners_path: str = None,
    output_dir: str = None,
    num_samples_visualize: int = 6,
    batch_size: int = 16
):
    if weights_path is None:
        weights_path = str(_SCRIPT_DIR / 'best_boxdreamer.pth')
    if scene_dir is None:
        scene_dir = str(_PROJECT_ROOT / 'bop_datasets' / 'dji' / 'train_pbr' / '000000')
    if corners_path is None:
        corners_path = str(_PROJECT_ROOT / 'dji_bbox_corners.npy')
    if output_dir is None:
        output_dir = str(_SCRIPT_DIR / 'eval_results')

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--> [Evaluation] Using device: {device}')
    print(f'--> [Evaluation] Loading weights from: {weights_path}')

    # 1. 载入数据集并划分出严格未参与训练的验证集 (20% 验证集)
    full_dataset = DJIActionPoseDataset(scene_dir, corners_path, img_size=224, sigma=3.0)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f'--> [Evaluation] Total validation samples: {len(val_dataset)}')

    # 2. 载入训练好的模型
    model = BoxDreamerModel(img_size=224, patch_size=14, d_model=384, device=device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 统计全体验证集的量化指标
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    all_corner_errs = [[] for _ in range(8)]
    all_mean_errs = []

    scale_x = 640.0 / 224.0
    scale_y = 480.0 / 224.0

    print('--> [Evaluation] Running full quantitative evaluation on validation set...')
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['image'].to(device)
            gt_pts = batch['pts_2d'].to(device)

            _, pred_coords = model(imgs, return_coords=True)

            pred_rescaled = pred_coords.clone()
            pred_rescaled[:, :, 0] *= scale_x
            pred_rescaled[:, :, 1] *= scale_y

            gt_rescaled = gt_pts.clone()
            gt_rescaled[:, :, 0] *= scale_x
            gt_rescaled[:, :, 1] *= scale_y

            dists = torch.norm(pred_rescaled - gt_rescaled, dim=-1).cpu().numpy()
            for b in range(dists.shape[0]):
                for c in range(8):
                    all_corner_errs[c].append(dists[b, c])
                all_mean_errs.append(np.mean(dists[b]))

    all_mean_errs = np.array(all_mean_errs)
    overall_mean = np.mean(all_mean_errs)
    overall_median = np.median(all_mean_errs)

    pct_under_10px = np.mean(all_mean_errs < 10.0) * 100
    pct_under_20px = np.mean(all_mean_errs < 20.0) * 100
    pct_under_30px = np.mean(all_mean_errs < 30.0) * 100

    print('\n' + '=' * 65)
    print('           BoxDreamer 验证集量化指标统计报告')
    print('=' * 65)
    print(f'验证集总样本数:         {len(val_dataset)} 帧')
    print(f'全图 8 角点平均误差:    {overall_mean:.2f} 像素 (在 640x480 分辨率下)')
    print(f'全图中位数误差:         {overall_median:.2f} 像素')
    print(f'误差 < 10 像素占比:     {pct_under_10px:.1f}%')
    print(f'误差 < 20 像素占比:     {pct_under_20px:.1f}%')
    print(f'误差 < 30 像素占比:     {pct_under_30px:.1f}%')
    print('-' * 65)
    print('各角点细分平均物理误差:')
    for c in range(8):
        c_mean = np.mean(all_corner_errs[c])
        print(f'  角点 {c} 平均误差: {c_mean:5.2f} 像素')
    print('=' * 65 + '\n')

    # 4. 可视化精选典型视角样本并生成多视角拼接总览大图
    print(f'--> [Evaluation] Rendering {num_samples_visualize} diverse visual comparison samples...')
    vis_images = []
    vis_indices = np.linspace(0, len(val_dataset) - 1, num_samples_visualize, dtype=int)

    for idx, v_i in enumerate(vis_indices):
        sample = val_dataset[v_i]
        frame_id = sample['frame_id']

        img_name = f'{int(frame_id):06d}.jpg'
        raw_img_path = Path(scene_dir) / 'rgb' / img_name
        if not raw_img_path.exists():
            raw_img_path = Path(scene_dir) / 'rgb' / f'{int(frame_id):06d}.png'

        bgr = cv2.imread(str(raw_img_path))
        orig_h, orig_w = bgr.shape[:2]

        img_tensor = sample['image'].unsqueeze(0).to(device)
        with torch.no_grad():
            _, pred_c = model(img_tensor, return_coords=True)

        p_coords = pred_c[0].cpu().numpy()
        p_coords[:, 0] *= (orig_w / 224.0)
        p_coords[:, 1] *= (orig_h / 224.0)

        gt_c = sample['pts_2d'].numpy()
        gt_c[:, 0] *= (orig_w / 224.0)
        gt_c[:, 1] *= (orig_h / 224.0)

        frame_err = np.mean(np.linalg.norm(p_coords - gt_c, axis=-1))

        canvas = bgr.copy()
        draw_box(canvas, gt_c, color=(255, 120, 0), radius=3, thickness=2)
        draw_box(canvas, p_coords, color=(0, 255, 0), radius=4, thickness=2)

        tag = f'Frame {int(frame_id):04d} | Err: {frame_err:.1f}px'
        cv2.putText(canvas, tag, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        single_path = Path(output_dir) / f'eval_frame_{int(frame_id):06d}.jpg'
        cv2.imwrite(str(single_path), canvas)
        vis_images.append(canvas)

    if len(vis_images) >= 6:
        row1 = np.hstack(vis_images[:3])
        row2 = np.hstack(vis_images[3:6])
        montage = np.vstack([row1, row2])
        montage_path = Path(output_dir) / 'montage_multi_angles.jpg'
        cv2.imwrite(str(montage_path), montage)
        print(f'--> [Evaluation] Multi-angle montage comparison saved to:\n    {montage_path}')

    print(f'--> [Evaluation] All evaluation samples successfully saved to:\n    {output_dir}')


if __name__ == '__main__':
    run_batch_evaluation()
