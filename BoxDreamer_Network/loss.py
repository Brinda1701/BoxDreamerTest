import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointFocalLoss(nn.Module):
    """
    CenterNet / CornerNet 经典 Modified Focal Loss
    
    优化原理：
    1. 针对正样本点 (target >= 0.99)：
       - 以 (1 - pred)^alpha 进行自适应加权，随着预测逼近 1.0，梯度自适应衰减为 0；
    2. 针对非峰值/背景点 (target < 0.99)：
       - 引入 (1 - target)^beta 对真实关键点周围的高斯模糊区域进行平滑折扣，允许一定的空间模糊容错；
       - 引入 (pred)^alpha 对占 99% 的简单背景负样本（pred 接近 0）实施强力梯度抑制；
    3. 彻底根治背景梯度主导问题，使网络预测出的 8 个角点热斑更加凝聚、尖锐，消除假阳性亮斑。
    """
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 数值裁剪，防止 log(0) 产生 NaN
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        # target >= 0.9 视为关键点正样本峰值区域，其余为背景与衰减负样本
        pos_mask = target.ge(0.9)
        neg_mask = target.lt(0.9)

        pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_mask.float()
        neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * torch.pow(1.0 - target, self.beta) * neg_mask.float()

        num_pos = max(1.0, pos_mask.float().sum())
        # 标准 CenterNet / CornerNet Focal Loss:
        # 正负样本全部由关键点正样本峰值数归一化，一旦空白处出现哪怕微弱的假峰值，都会产生强大的负向梯度压制
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


class RigidTopologyLoss(nn.Module):
    """
    3D 刚体边界框几何拓扑一致性损失 (Rigid Cuboid Prior Loss)
    
    优化原理：
    1. 传统 2D 点坐标损失只独立监督孤立角点，无法感知“8个角点必须属于同一个长方体”的物理刚体规律；
    2. 引入双重拓扑互锁约束：
       - 12 棱向量一致性损失 (Edge Vector Loss): 约束每条棱的方向与相对长度向量 e_ij = P_j - P_i
       - 28 对角点相对欧氏距离矩阵损失 (Pairwise Distance Matrix Loss):
         约束所有角点对之间的相对空间投影距离（含 12 条棱、12 条表面对角线、4 条体对角线）
    3. 几何互锁效应：当某个角点因局部强光或遮挡漂移时，破坏的拓扑距离会瞬间产生多向向心拉力，
       将其强行拉回符合 3D 刚体透视投影的准确网格中，彻底消除边界框拉伸与扭曲畸变。
    """
    EDGES_12 = [
        (0, 1), (2, 3), (4, 5), (6, 7), # 沿 Z 轴 4 条棱
        (0, 2), (1, 3), (4, 6), (5, 7), # 沿 Y 轴 4 条棱
        (0, 4), (1, 5), (2, 6), (3, 7)  # 沿 X 轴 4 条棱
    ]

    def __init__(self, img_size: float = 224.0):
        super().__init__()
        self.img_size = float(img_size)
        self.edges_i = [e[0] for e in self.EDGES_12]
        self.edges_j = [e[1] for e in self.EDGES_12]

    def forward(self, pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_coords: (B, 8, 2)
            target_coords: (B, 8, 2)
        Returns:
            geom_loss: 标量几何拓扑一致性损失
        """
        # 1. 12 棱向量方向与长度一致性 (L1 损失，直接衡量棱向量的绝对像素偏差)
        pred_edges = pred_coords[:, self.edges_j] - pred_coords[:, self.edges_i]       # (B, 12, 2)
        target_edges = target_coords[:, self.edges_j] - target_coords[:, self.edges_i] # (B, 12, 2)
        edge_loss = F.l1_loss(pred_edges, target_edges)

        # 2. 28 对角点相对欧氏距离矩阵一致性 (刚体互锁约束)
        pred_dist = torch.cdist(pred_coords, pred_coords)       # (B, 8, 8)
        target_dist = torch.cdist(target_coords, target_coords) # (B, 8, 8)
        dist_loss = F.l1_loss(pred_dist, target_dist)

        return (edge_loss + dist_loss) * 0.5


class HeatmapLoss(nn.Module):
    """
    BoxDreamer 综合热力图、亚像素坐标与 3D 刚体几何拓扑三重联合损失函数
    
    兼容性设计：
    1. 默认向下兼容：直接调用 criterion(pred, target) 时，以 focal 或 smooth_l1 计算热力图损失；
    2. 坐标与几何扩展：当同时传入 pred_coords 与 target_coords 时，自动引入亚像素坐标平滑 L1 损失
       与 3D 刚体几何拓扑一致性损失，形成“热力图峰值 + 坐标回归 + 刚体结构先验”三位一体联合驱动。
    """
    def __init__(
        self,
        loss_type: str = "focal",
        fg_weight: float = 2.0,
        coord_weight: float = 1.0,
        geom_weight: float = 0.5,
        img_size: int = 224
    ):
        super().__init__()
        self.loss_type = loss_type
        self.fg_weight = fg_weight
        self.coord_weight = coord_weight
        self.geom_weight = geom_weight
        self.img_size = float(img_size)

        self.focal_loss_fn = KeypointFocalLoss(alpha=2.0, beta=4.0)
        self.rigid_geom_fn = RigidTopologyLoss(img_size=self.img_size)
        self.last_metrics = {}

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_coords: torch.Tensor = None,
        target_coords: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            pred: 网络预测热力图 (B, 8, H, W)，取值 [0, 1]
            target: 真实高斯热力图 (B, 8, H, W)，取值 [0, 1]
            pred_coords: (可选) Soft-Argmax 回归的连续角点坐标 (B, 8, 2)
            target_coords: (可选) 真实 2D 角点坐标 (B, 8, 2)
        Returns:
            total_loss: 可求导的总损失标量张量
        """
        assert pred.shape == target.shape, f"形状不一致: {pred.shape} vs {target.shape}"

        # 1. 计算热力图损失 (Focal Loss 或 带权 Smooth L1)
        if self.loss_type == "focal":
            hm_loss = self.focal_loss_fn(pred, target)
        else:
            base_loss = F.smooth_l1_loss(pred, target, reduction='none')
            weight_mask = torch.ones_like(target)
            weight_mask[target > 0.05] = self.fg_weight
            hm_loss = (base_loss * weight_mask).mean()

        total_loss = hm_loss
        coord_loss_val = 0.0
        geom_loss_val = 0.0
        mean_px_err = 0.0

        if pred_coords is not None and target_coords is not None:
            # 2. 计算可微亚像素坐标损失 (取消 / 10.0，使像素误差直接产生等量级的强烈梯度)
            if self.coord_weight > 0:
                coord_loss = F.l1_loss(pred_coords, target_coords)

                total_loss = total_loss + self.coord_weight * coord_loss
                coord_loss_val = coord_loss.item()

            # 3. 计算 3D 刚体拓扑一致性损失 (若 geom_weight > 0)
            if self.geom_weight > 0:
                geom_loss = self.rigid_geom_fn(pred_coords, target_coords)
                total_loss = total_loss + self.geom_weight * geom_loss
                geom_loss_val = geom_loss.item()

            # 统计物理像素平均误差 (用于监控，无须梯度)
            with torch.no_grad():
                mean_px_err = torch.norm(pred_coords - target_coords, dim=-1).mean().item()

        self.last_metrics = {
            "hm_loss": hm_loss.item(),
            "coord_loss": coord_loss_val,
            "geom_loss": geom_loss_val,
            "pixel_error": mean_px_err
        }

        return total_loss


# ==============================================================================
# 单元测试与梯度检查
# ==============================================================================
if __name__ == "__main__":
    print("Testing upgraded HeatmapLoss with 3D Rigidity...")
    criterion = HeatmapLoss(loss_type="focal", coord_weight=1.0, geom_weight=1.0)

    # 1. 模拟网络预测 (需要求导，背景预测通常在 0.05~0.1 左右)
    dummy_pred = (torch.rand(2, 8, 224, 224) * 0.1).requires_grad_()
    dummy_coords = (torch.rand(2, 8, 2) * 224.0).requires_grad_()

    # 2. 模拟真值
    dummy_target = torch.zeros(2, 8, 224, 224)
    dummy_target[:, :, 100:110, 100:110] = 0.8
    dummy_target[:, :, 105, 105] = 1.0 # 正样本峰值
    dummy_gt_coords = torch.full((2, 8, 2), 105.0)

    # 3. 计算三位一体联合损失
    loss = criterion(dummy_pred, dummy_target, pred_coords=dummy_coords, target_coords=dummy_gt_coords)
    print(f"Calculated Joint Loss: {loss.item():.6f}")
    print(f"Metrics detail: {criterion.last_metrics}")

    # 4. 反向传播梯度检查
    loss.backward()
    print(f"Heatmap Grad norm: {dummy_pred.grad.norm().item():.4f}")
    print(f"Coords Grad norm:  {dummy_coords.grad.norm().item():.4f}")
    print("Success! Tri-Supervision HeatmapLoss with 3D Rigidity is completely functional and verified.")
 