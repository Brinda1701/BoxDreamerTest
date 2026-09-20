import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

try:
    from .encoder.dinov2 import DinoV2Wrapper
    from .backbone.betr import BETR
except (ImportError, ValueError):
    from encoder.dinov2 import DinoV2Wrapper
    from backbone.betr import BETR


class SpatialSoftArgmax2d(nn.Module):
    """
    二维空间可微积分 (Spatial Soft-Argmax) 亚像素坐标提取模块
    增加自适应高锐度温度，压制多峰扩散，直冲 5px 以内精度
    """
    def __init__(self, temperature: float = 60.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmaps: torch.Tensor) -> torch.Tensor:
        B, C, H, W = heatmaps.shape
        device = heatmaps.device
        dtype = heatmaps.dtype

        # 提高温度系数，使概率分布呈极尖锐的狄拉克脉冲状态，消除副峰拖拽
        flat_hm = heatmaps.view(B, C, -1) * self.temperature
        probs = F.softmax(flat_hm, dim=-1).view(B, C, H, W)

        pos_y, pos_x = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )

        expected_x = torch.sum(probs * pos_x, dim=(-2, -1))
        expected_y = torch.sum(probs * pos_y, dim=(-2, -1))

        coords = torch.stack([expected_x, expected_y], dim=-1)
        return coords


class BoxDreamerModel(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        d_model: int = 384,
        num_decoder_layers: int = 6,
        unfreeze_dino: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.d_model = d_model
        self.device_type = device

        # 允许选择性解冻 DINOv2 以突破 5px 瓶颈
        dino_cfg = {
            'model_type': 'dinov2_vits14_reg',
            'freeze': not unfreeze_dino,
            'device': device
        }
        print(f"Initializing DINOv2 Encoder (Freeze={not unfreeze_dino})...")
        self.encoder = DinoV2Wrapper(ckpt_path=None, cfg=dino_cfg)
        self.encoder.to_device(device)

        if unfreeze_dino:
            # 解冻 DINOv2 后半部分层 (最后 4 个 blocks)，让网络直接学习相机的底层物理特征
            for name, param in self.encoder.model.named_parameters():
                if "blocks.8" in name or "blocks.9" in name or "blocks.10" in name or "blocks.11" in name or "norm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("--> [DINOv2] 成功解冻最后 4 个 Transformer Blocks 进行端到端高精微调！")

        betr_cfg = {
            "decoder_only": True,
            "patch_size": self.patch_size,
            "img_size": self.img_size,
            "nvs_supervision": False,
            "ray_supervision": True,
            "pose_representation": "bb8",
            "bbox_representation": "heatmap",
            "diff_emb": False,
            "use_pretrained": True
        }

        self.decoder = BETR(d_model=self.d_model, nhead=6, num_decoder_layers=num_decoder_layers, **betr_cfg)
        # 锐化温度设为 60.0，锁定精准亚像素且保证平滑充沛梯度
        self.soft_argmax = SpatialSoftArgmax2d(temperature=60.0)

    def extract_subpixel_coords(self, heatmaps: torch.Tensor) -> torch.Tensor:
        return self.soft_argmax(heatmaps)

    def forward(self, x: torch.Tensor, return_coords: bool = False):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        if hasattr(self.encoder, 'model') and any(p.requires_grad for p in self.encoder.model.parameters()):
            # 若解冻 DINOv2，保留梯度传播
            x_norm = self.encoder._resnet_normalize_image(x)
            patch_tokens = self.encoder.model.forward_features(x_norm)['x_norm_patchtokens']
        else:
            patch_tokens = self.encoder.predict(x)

        # BETR 解码器接口规范：
        # pose_feat: (B, 1, 8, H, W)
        # rgbs: (B, 1, 3, H, W)
        # masks: (B, 1) 布尔型，True 代表需要查询预测的视角
        # pretrain_rgb_feat: (B, 1, P, d_model)
        rgbs = x.unsqueeze(1)
        masks = torch.ones((B, 1), dtype=torch.bool, device=x.device)
        dummy_pose_feat = torch.zeros((B, 1, 8, H, W), dtype=x.dtype, device=x.device)
        patch_tokens = patch_tokens.unsqueeze(1)

        raw_heatmaps = self.decoder(
            pose_feat=dummy_pose_feat,
            rgbs=rgbs,
            masks=masks,
            pretrain_rgb_feat=patch_tokens
        )
        # BETR 内部输出为 2 * sigmoid - 1 (取值 [-1, 1])，将其映射回真实热力图概率区间 [0, 1]
        heatmaps = torch.clamp((raw_heatmaps + 1.0) / 2.0, 0.0, 1.0)

        if return_coords:
            coords = self.extract_subpixel_coords(heatmaps)
            return heatmaps, coords
        return heatmaps
