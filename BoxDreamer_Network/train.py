import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

# 将当前目录添加进 sys.path，支持独立脚本执行与跨级调用
_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

try:
    from .dataset import DJIActionPoseDataset, RealDJIDataset
    from .models import BoxDreamerModel
    from .loss import HeatmapLoss
except (ImportError, ValueError):
    from dataset import DJIActionPoseDataset, RealDJIDataset
    from models import BoxDreamerModel
    from loss import HeatmapLoss


def parse_args():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    parser = argparse.ArgumentParser(description="BoxDreamer 3D 边界框热力图预测网络 - 训练脚本")

    parser.add_argument(
        "--scene_dir",
        type=str,
        default=str(PROJECT_ROOT / "bop_datasets" / "dji" / "train_pbr" / "000000"),
        help="BOP 格式渲染数据集场景目录"
    )
    parser.add_argument(
        "--corners_path",
        type=str,
        default=str(PROJECT_ROOT / "dji_bbox_corners.npy"),
        help="DJI Action 相机自身坐标系下的 8 个 3D 边界框角点坐标"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(SCRIPT_DIR / "best_boxdreamer.pth"),
        help="权重保存路径"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="已有模型权重路径 (用于微调)"
    )

    # 训练超参数
    parser.add_argument("--img_size", type=int, default=224, help="网络输入图像分辨率")
    parser.add_argument("--batch_size", type=int, default=16, help="单批次大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="AdamW 优化器初始学习率")
    parser.add_argument("--num_epochs", type=int, default=80, help="训练轮数")
    parser.add_argument("--sigma", type=float, default=3.0, help="GT 高斯热力图的方差半径")
    parser.add_argument("--fg_weight", type=float, default=2.0, help="关键点前景加权系数")
    parser.add_argument("--loss_type", type=str, default="smooth_l1", choices=["focal", "smooth_l1"])
    parser.add_argument("--coord_weight", type=float, default=3.0, help="亚像素坐标损失权重 (大幅强化坐标惩罚)")
    parser.add_argument("--geom_weight", type=float, default=1.5, help="3D 刚体几何拓扑损失权重")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="Transformer 梯度裁剪阈值")

    # 数据集模式
    parser.add_argument("--real_dataset", type=str, default=None, help="真实数据集路径")
    parser.add_argument("--mix_real", action="store_true", help="是否混合训练")
    parser.add_argument("--unfreeze_dino", action="store_true", help="是否解冻 DINOv2 后半段深入微调 (冲刺 < 5px)")

    default_workers = 4 if (os.name != "nt" and torch.cuda.is_available()) else 0
    parser.add_argument("--num_workers", type=int, default=default_workers)
    parser.add_argument("--max_batches", type=int, default=None)

    return parser.parse_args()


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Using device: {device}")
    if device == "cuda":
        print(f"--> GPU Device: {torch.cuda.get_device_name(0)}")

    print(f"--> Save Path:    {args.save_path}")
    print(f"--> Loss Config:  Loss={args.loss_type} | Coord_W={args.coord_weight} | Geom_W={args.geom_weight}")
    print(f"--> Unfreeze DINO: {args.unfreeze_dino}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)

    # 1. 准备数据
    print("--> Loading dataset...")
    if args.real_dataset is not None:
        real_train = RealDJIDataset(args.real_dataset, img_size=args.img_size, sigma=args.sigma, is_train=True)
        real_val = RealDJIDataset(args.real_dataset, img_size=args.img_size, sigma=args.sigma, is_train=False)
        print(f"--> [Real Dataset] 成功载入全视频真实切片: {len(real_train)} 张")

        n_total = len(real_train)
        indices = list(range(n_total))
        np.random.seed(42)
        np.random.shuffle(indices)
        split_idx = int(0.85 * n_total)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_real_subset = Subset(real_train, train_indices)
        val_dataset = Subset(real_val, val_indices)

        if args.mix_real:
            bop_data = DJIActionPoseDataset(
                bop_scene_dir=args.scene_dir,
                corners_3d_path=args.corners_path,
                img_size=args.img_size,
                sigma=args.sigma,
                is_train=True
            )
            # 训练集：BOP 渲染 + 3x 真实全量视频切片
            train_dataset = ConcatDataset([bop_data, train_real_subset, train_real_subset, train_real_subset])
            print(f"--> [Mixed Train] BOP({len(bop_data)}) + 真实切片x3({len(train_real_subset)*3}) = 共 {len(train_dataset)} 张")
        else:
            train_dataset = train_real_subset

        print(f"--> Split: Train = {len(train_dataset)}, Val (Clean) = {len(val_dataset)}")
    else:
        full_dataset = DJIActionPoseDataset(
            bop_scene_dir=args.scene_dir,
            corners_3d_path=args.corners_path,
            img_size=args.img_size,
            sigma=args.sigma,
            is_train=True
        )
        train_size = int(0.85 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"--> Split: Train = {train_size}, Val = {val_size}")

    pin_mem = (device == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    # 2. 初始化模型
    print("--> Initializing model...")
    model = BoxDreamerModel(
        img_size=args.img_size,
        patch_size=14,
        d_model=384,
        unfreeze_dino=args.unfreeze_dino,
        device=device
    )

    if args.weights is not None and os.path.exists(args.weights):
        print(f"--> [Fine-tuning] 载入已有权重: {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

    model.to(device)

    criterion = HeatmapLoss(
        loss_type=args.loss_type,
        fg_weight=args.fg_weight,
        coord_weight=args.coord_weight,
        geom_weight=args.geom_weight,
        img_size=args.img_size
    )

    # 差异化学习率配置：解冻的 DINOv2 用更小的 lr 保护底层预训练特征，Decoder 用主力 lr
    dino_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "encoder" in name:
            dino_params.append(param)
        else:
            decoder_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decoder_params, "lr": args.lr},
    ]
    if len(dino_params) > 0:
        optimizer_grouped_parameters.append({"params": dino_params, "lr": args.lr * 0.2})
        print(f"--> [Optimizer] DINOv2 参数启用低学习率: {args.lr * 0.2:.2e}")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> Trainable parameters: {trainable_count / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    best_val_px_err = float("inf")

    print("\n--> 开始冲刺 < 5px 极致精度训练大循环...")
    for epoch in range(args.num_epochs):
        model.train()
        running_train_loss = 0.0
        running_train_px_err = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=pin_mem)
            targets = batch["heatmap"].to(device, non_blocking=pin_mem)
            target_coords = batch["pts_2d"].to(device, non_blocking=pin_mem)

            outputs, pred_coords = model(images, return_coords=True)
            loss = criterion(outputs, targets, pred_coords=pred_coords, target_coords=target_coords)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=args.grad_clip)

            optimizer.step()

            running_train_loss += loss.item()
            running_train_px_err += criterion.last_metrics.get("pixel_error", 0.0)

        scheduler.step()

        n_train = len(train_loader)
        avg_train_loss = running_train_loss / max(1, n_train)
        avg_train_px_err = running_train_px_err / max(1, n_train)

        # 验证
        model.eval()
        running_val_loss = 0.0
        running_val_px_err = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=pin_max) if 'pin_max' in locals() else batch["image"].to(device, non_blocking=pin_mem)
                targets = batch["heatmap"].to(device, non_blocking=pin_mem)
                target_coords = batch["pts_2d"].to(device, non_blocking=pin_mem)

                outputs, pred_coords = model(images, return_coords=True)
                val_loss = criterion(outputs, targets, pred_coords=pred_coords, target_coords=target_coords)

                running_val_loss += val_loss.item()
                running_val_px_err += criterion.last_metrics.get("pixel_error", 0.0)

        n_val = len(val_loader)
        avg_val_loss = running_val_loss / max(1, n_val)
        avg_val_px_err = running_val_px_err / max(1, n_val)

        curr_lr = scheduler.get_last_lr()[0]
        improved = avg_val_px_err < best_val_px_err

        if improved:
            best_val_px_err = avg_val_px_err
            torch.save(model.state_dict(), args.save_path)
            flag = " [NEW BEST - MODEL SAVED]"
        else:
            flag = ""

        print(
            f"Epoch [{epoch+1:02d}/{args.num_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (Err: {avg_train_px_err:5.2f}px) | "
            f"Val Loss: {avg_val_loss:.4f} (Err: {avg_val_px_err:5.2f}px) | "
            f"LR: {curr_lr:.2e} {flag}"
        )

    print(f"\n--> Training Complete! Best model saved at: {args.save_path} (Best Val Err: {best_val_px_err:.2f}px)")


if __name__ == "__main__":

    
    train(parse_args())
