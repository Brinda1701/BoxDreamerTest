# BoxDreamer 3D 目标边界框训练与视频推理系统解析

## 1. 项目背景与技术路线约束

本项目的核心目标是对视频序列中的目标（DJI Action 4 相机）进行高精度的 **3D 边界框检测与追踪（3D Bounding Box Tracking）**。

在传统 3D 视觉管线中，通常依赖相机标定内参矩阵 $K$ 以及 PnP（Perspective-n-Point）刚体位姿求解器。但根据导师明确要求：
* **核心约束**：**严禁使用任何相机内参参数矩阵 $K$**，只允许基于纯 RGB 图像进行 2D 关键角点回归并直接连接成长方体线框。
* **技术演进**：前端 YOLO 检测 2D 目标视口，后端 DINOv2 驱动的 BoxDreamer 热图网络回归 8 个 3D 角点投影坐标。
* **工程挑战**：不使用 3D 刚体模型与相机几何的前提下，神经网络独立回归 8 个角点极易产生**视口跳动、角点空间错乱（蝴蝶结领结形打叉）、以及遮挡/反光漂移**等问题。

为彻底解决上述缺陷并重构训练管线，本文档系统总结当前落地的训练与推理方案。

---

## 2. 训练优化与推理方案总览图谱

### 训练阶段改进 (Training Pipeline)
```
[BOP 渲染/真实数据]
     │
     ▼
[3D 角点投影至 2D 原图]
     │
     ▼
[核心改进 1: 随机平移与缩放抖动 (Jitter Crop)]
(中心随机偏移 ±10% crop_size, 尺度随机缩放 0.9~1.15，同步严格更新角点坐标)
     │
     ▼
[生成 224x224 局部高斯热力图]
     │
     ▼
[前向传播: DINOv2 + BETR 解码 + Spatial Soft-Argmax]
     │
     ▼
[核心改进 2: 无缩水真实尺度损失计算]
Loss = FocalLoss(Heatmaps) + 1.0 * L1(pred_coords, gt_coords) + 0.5 * RigidTopologyLoss
(废除 / 10.0 的缩小操作，使关键点亚像素定位获得充沛反向梯度)
```

### 视频推理阶段管线 (Inference Pipeline)
```
[原始视频帧]
     │
     ▼
[YOLOv8 2D 检测] + 几何长宽比约束 (去除腕带误检，纯逐帧处理)
     │
     ▼
[截取正方形 RoI + 边界镜像填充]
     │
     ▼
[DINOv2 + BETR 解码器] -> 预测 8 通道热图与亚像素坐标
     │
     ▼
[224x224 局部角点映射回全图原始分辨率像素]
     │
     ▼
[优化一（方案 B）] 基于热图置信度门控与三棱对称几何闭环互补 (解决反光与遮挡漂移)
     │
     ▼
[优化二（策略二）] 纯 2D 拓扑几何解缠绕 (向量叉乘防“X”打叉)
     │
     ▼
[最终高质量稳定 3D 线框渲染]
```

---

## 3. 详细方案拆解（含逐行代码注解）

---

### 一、训练阶段核心优化：随机视口抖动增强与真实量纲 Loss 回归

#### 1. 解决的痛点
* **训练与推理域不一致（Domain Gap）**：此前训练集中相机永远在 224x224 正中间完美居中，而推理阶段 YOLO 的检测框中心会随视角、反光产生 5~15 像素随机漂移，导致模型对偏移极度敏感。
* **关键点梯度被严重压制**：此前 `coord_loss` 人为除以了 `10.0`，导致其数值只有 0.5~1.0，而 Focal Loss 高达 4.0~6.0。模型主要依靠热图范围拿分，缺乏将角点收敛到 1~2 像素以内的驱动力。

#### 2. 代码实现（已落地于 `dataset.py` 与 `loss.py`）
```python
# -------------------------------------------------------------
# 1. dataset.py: 训练期随机平移与尺度扰动 (Jitter Data Augmentation)
# -------------------------------------------------------------
# 尺度微小扰动: 0.9 ~ 1.15
if is_train:
    scale_jitter = np.random.uniform(0.9, 1.15)
    crop_size = int(crop_size * scale_jitter)
    # 中心点微小平移扰动: 模拟 ±10% crop_size 的视口偏移 (约 10~20 像素)
    shift_max = 0.10 * crop_size
    cx += np.random.uniform(-shift_max, shift_max)
    cy += np.random.uniform(-shift_max, shift_max)

half = crop_size // 2
x1, y1 = int(cx - half), int(cy - half)
x2, y2 = x1 + crop_size, y1 + crop_size

# 关键：严格将 8 个角点同步映射到抖动后的新视口中
pts_2d_crop = (pts_2d_orig - np.array([x1, y1])) * (self.img_size / float(crop_size))


# -------------------------------------------------------------
# 2. loss.py: 取消人为除以 10.0，使坐标误差直接释放真实量纲梯度
# -------------------------------------------------------------
# 直接采用完整 L1 损失，误差每偏差 1 像素即产生 1.0 的强力惩罚
coord_loss = F.l1_loss(pred_coords, target_coords)
total_loss = hm_loss + self.coord_weight * coord_loss + self.geom_weight * geom_loss
```

---

### 二、推理阶段优化方案：方案 B（热图置信度门控对称补全）

#### 1. 痛点问题
在视频旋转时，背光面、底面或手部遮挡角点在热图上的激活峰值较低，直接回归容易向外漂移变形。

#### 2. 数学原理与代码实现
利用长方体在弱透视下的平行四边形面向量外推：
$$\vec{p}_{\text{target}} \approx \vec{p}_{\text{base}} + (\vec{p}_{\text{ref\_end}} - \vec{p}_{\text{ref\_start}})$$

```python
def complete_occluded_corners_by_symmetry(pts, confs, conf_thresh=0.25):
    """
    长方体 3 组平行棱几何闭环修复：
    当某些角点因反光、视角遮挡导致热图置信度极低（conf < conf_thresh）时，
    放弃网络预测的假象杂波，利用长方体平行四边形面向量关系：
        p_dst = p_src + (p_opp2 - p_opp1)
    通过高置信度角点推导补全被遮挡的角点位置。
    """
    fixed = pts.copy()
    
    relations = [
        # X 轴向补全: (0-1), (2-3), (4-5), (6-7)
        (1, 0, 2, 3), (0, 1, 3, 2), (3, 2, 0, 1), (2, 3, 1, 0),
        (5, 4, 6, 7), (4, 5, 7, 6), (7, 6, 4, 5), (6, 7, 5, 4),
        # Y 轴向补全: (0-2), (1-3), (4-6), (5-7)
        (2, 0, 1, 3), (0, 2, 3, 1), (3, 1, 0, 2), (1, 3, 2, 0),
        (6, 4, 5, 7), (4, 6, 7, 5), (7, 5, 4, 6), (5, 7, 6, 4),
        # Z 轴向补全: (0-4), (1-5), (2-6), (3-7)
        (4, 0, 1, 5), (0, 4, 5, 1), (5, 1, 0, 4), (1, 5, 4, 0),
        (6, 2, 3, 7), (2, 6, 7, 3), (7, 3, 2, 6), (3, 7, 6, 2),
    ]

    for target, base, r_start, r_end in relations:
        if confs[target] < conf_thresh and confs[base] >= conf_thresh and confs[r_start] >= conf_thresh and confs[r_end] >= conf_thresh:
            fixed[target] = fixed[base] + (fixed[r_end] - fixed[r_start])
            confs[target] = 0.5  # 恢复为中等可信度

    return fixed
```

---

### 三、推理阶段优化方案：策略二（纯 2D 拓扑几何解缠绕）

#### 1. 痛点问题
网络在旋转大角度时可能将同一面内的相邻编号混淆，导致连线在 2D 平面出现“X”领结形交叉。

#### 2. 数学原理与代码实现
长方体任意矩形面在 2D 透视投影下均为简单四边形，同面内相对的两条边在 2D 平面上绝不可能相交。若检测到相交，对调错序顶点即可消除打叉。

```python
def segments_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def untangle_cuboid_2d(pts):
    p = pts.copy()
    face_checks = [
        (0, 1, 2, 3, (1, 3)), (0, 2, 1, 3, (2, 3)), # 左侧面
        (4, 5, 6, 7, (5, 7)), (4, 6, 5, 7, (6, 7)), # 右侧面
        (0, 1, 4, 5, (1, 5)), (0, 4, 1, 5, (4, 5)), # 底面
        (2, 3, 6, 7, (3, 7)), (2, 6, 3, 7, (6, 7)), # 顶面
        (0, 2, 4, 6, (2, 6)), (0, 4, 2, 6, (4, 6)), # 后面
        (1, 3, 5, 7, (3, 7)), (1, 5, 3, 7, (5, 7)), # 前面
    ]
    for a1, a2, b1, b2, (s1, s2) in face_checks:
        if segments_intersect(p[a1], p[a2], p[b1], p[b2]):
            p[[s1, s2]] = p[[s2, s1]]
    return p
```

---

## 4. 重新训练启动指令

当前代码已准备就绪，包含**抗抖数据增强**与**高权重亚像素坐标监督**：

```powershell
# 启动重新训练（推荐使用较优超参）
python BoxDreamer_Network/train.py --num_epochs 20 --batch_size 8 --lr 5e-5 --coord_weight 1.0 --geom_weight 0.5
```

训练完成后，使用最新推理脚本进行视频检测：
```powershell
python BoxDreamer_Network/infer_video.py --video test_video/head_left_rgb_raw.mp4 --output test_video/final_tracked_v3.mp4 --max_frames 120
```
