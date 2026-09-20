import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
for _p in [_SCRIPT_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from .models import BoxDreamerModel
except (ImportError, ValueError):
    from models import BoxDreamerModel


# 3D 立方体的 12 条棱连接关系（根据 dji_bbox_corners.npy 顶点的对应坐标推导）
# 8 个顶点中：
# 0~3 为物体左侧 (x_min)，4~7 为物体右侧 (x_max)
# 0,1,4,5 为下表面 (y_min)，2,3,6,7 为上表面 (y_max)
# 0,2,4,6 为后表面 (z_min)，1,3,5,7 为前表面 (z_max)
EDGES_12 = [
    # 沿 Z 轴的 4 条棱
    (0, 1), (2, 3), (4, 5), (6, 7),
    # 沿 Y 轴的 4 条棱
    (0, 2), (1, 3), (4, 6), (5, 7),
    # 沿 X 轴的 4 条棱
    (0, 4), (1, 5), (2, 6), (3, 7)
]


def infer_and_draw_3d_box(
    image_path: str,
    weights_path: str,
    output_path: str = "output_3d_box.jpg",
    img_size: int = 224
):
    """
    纯单目 2D 推理并连线绘制 3D 边界框
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device}")

    # ==========================================================================
    # 1. 读取原图并进行网络前处理
    # ==========================================================================
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到测试图片: {image_path}")

    bgr_orig = cv2.imread(image_path)
    orig_h, orig_w = bgr_orig.shape[:2]

    # --------------------------------------------------------------------------
    # 图像预处理
    # --------------------------------------------------------------------------
    # 提示步骤：
    # 1. 将 BGR 转为 RGB 颜色空间；
    # 2. 用 cv2.resize 将图像缩放到网络要求的尺寸 (img_size, img_size)；
    # 3. 转换为 PyTorch 张量：维度调整为 (1, 3, img_size, img_size)，像素值归一化到 [0, 1]；
    # 4. 迁移到对应 device。
    # 
    rgb = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2RGB)
    rgb_resized = cv2.resize(rgb, (img_size, img_size))
    input_tensor = torch.from_numpy(rgb_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    # ==========================================================================
    # 2. 载入模型与权重
    # ==========================================================================
    print("--> Loading model...")
    model = BoxDreamerModel(img_size=img_size, patch_size=14, d_model=384, device=device)

    if os.path.exists(weights_path):
        print(f"--> Loading trained weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"--> [Warning] 权重文件不存在: {weights_path}，将使用未训练随机权重进行流程演示。")

    model.to(device)

    # --------------------------------------------------------------------------
    # 执行无内参前向推理
    # --------------------------------------------------------------------------
    # 提示：
    # 1. 切换模型至评估模式 model.eval()；
    # 2. 在 torch.no_grad() 上下文中，将 input_tensor 输入模型，得到 pred_heatmaps (1, 8, 224, 224)。
    # 
    model.eval()
    with torch.no_grad():
        pred_heatmaps, pred_coords = model(input_tensor, return_coords=True)

    # ==========================================================================
    # 3. 提取 8 个角点的 2D 像素坐标并还原回原图分辨率
    # ==========================================================================
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size

    # 使用端到端由 3D 刚体先验监督的亚像素可微坐标
    coords_np = pred_coords[0].cpu().numpy() # (8, 2)
    corners_2d = []
    for i in range(8):
        px = int(round(coords_np[i, 0] * scale_x))
        py = int(round(coords_np[i, 1] * scale_y))
        corners_2d.append([px, py])

    print(f"--> Extracted 8 corners (on original image {orig_w}x{orig_h}):")
    for i, pt in enumerate(corners_2d):
        print(f"    Corner {i}: ({pt[0]}, {pt[1]})")

    # ==========================================================================
    # 4. 尝试读取 Ground Truth (真值) 进行量化比对
    # ==========================================================================
    gt_corners_2d = None
    img_p = Path(image_path).resolve()
    scene_dir = img_p.parent.parent
    gt_file = scene_dir / "scene_gt.json"
    cam_file = scene_dir / "scene_camera.json"
    corners_npy = PROJECT_ROOT / "dji_bbox_corners.npy"

    frame_id = img_p.stem.lstrip("0")
    if frame_id == "":
        frame_id = "0"

    if gt_file.exists() and cam_file.exists() and Path(corners_npy).exists():
        try:
            import json
            with open(gt_file, "r") as f:
                scene_gt = json.load(f)
            with open(cam_file, "r") as f:
                scene_cam = json.load(f)

            if frame_id in scene_gt and frame_id in scene_cam:
                corners_3d = np.load(corners_npy) * 1000.0
                gt_info = scene_gt[frame_id][0]
                cam_info = scene_cam[frame_id]
                R = np.array(gt_info["cam_R_m2c"]).reshape(3, 3)
                t = np.array(gt_info["cam_t_m2c"]).reshape(3, 1)
                K = np.array(cam_info["cam_K"]).reshape(3, 3)

                pts_cam = R @ corners_3d.T + t
                pts_homo = K @ pts_cam
                u = pts_homo[0] / pts_homo[2]
                v = pts_homo[1] / pts_homo[2]
                gt_corners_2d = np.stack([u, v], axis=1)

                print("\n--> [Ground Truth Comparison & Pixel Errors]:")
                errs = []
                for i in range(8):
                    dist = np.linalg.norm(np.array(corners_2d[i]) - gt_corners_2d[i])
                    errs.append(dist)
                    print(f"    Corner {i} | Pred: ({corners_2d[i][0]:3d}, {corners_2d[i][1]:3d}) | GT: ({gt_corners_2d[i][0]:5.1f}, {gt_corners_2d[i][1]:5.1f}) | Err: {dist:5.1f}px")
                print(f"--> [Average 2D Pixel Error]: {np.mean(errs):.2f} px\n")
        except Exception as e:
            print(f"--> [Warning] 尝试加载 GT 失败: {e}")

    # ==========================================================================
    # 5. 在原图上绘制 3D 边界框 (蓝虚线: GT, 绿实线: 预测)
    # ==========================================================================
    vis_img = bgr_orig.copy()

    # 绘制 GT 真实框 (蓝色)
    if gt_corners_2d is not None:
        for pt in gt_corners_2d:
            cv2.circle(vis_img, (int(round(pt[0])), int(round(pt[1]))), radius=3, color=(255, 120, 0), thickness=-1)
        for idx1, idx2 in EDGES_12:
            pt1 = (int(round(gt_corners_2d[idx1][0])), int(round(gt_corners_2d[idx1][1])))
            pt2 = (int(round(gt_corners_2d[idx2][0])), int(round(gt_corners_2d[idx2][1])))
            cv2.line(vis_img, pt1, pt2, color=(255, 120, 0), thickness=2)

    # 绘制网络预测框 (实心红点 + 翠绿实线)
    for pt in corners_2d:
        cv2.circle(vis_img, tuple(pt), radius=4, color=(0, 0, 255), thickness=-1)

    for idx1, idx2 in EDGES_12:
        pt1 = tuple(corners_2d[idx1])
        pt2 = tuple(corners_2d[idx2])
        cv2.line(vis_img, pt1, pt2, color=(0, 255, 0), thickness=2)

    # 保存并展示结果
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis_img)
    print(f"--> 3D wireframe box visualization successfully saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    parser = argparse.ArgumentParser(description="BoxDreamer 3D 边界框推理与连线可视化")
    parser.add_argument(
        "--image",
        type=str,
        default=str(PROJECT_ROOT / "bop_datasets" / "dji" / "train_pbr" / "000000" / "rgb" / "000000.jpg"),
        help="待测试 RGB 图像路径"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(SCRIPT_DIR / "best_boxdreamer.pth"),
        help="已训练的模型权重文件 (.pth)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(SCRIPT_DIR / "output_3d_box.jpg"),
        help="绘制 3D 框后的结果图片输出路径"
    )
    args = parser.parse_args()

    infer_and_draw_3d_box(
        image_path=args.image,
        weights_path=args.weights,
        output_path=args.output
    )