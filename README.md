# BoxDreamer —— 单目无内参 3D 边界框追踪系统

基于 DINOv2 + BETR 架构，实现对手持 DJI Action 4 相机的单目 3D 边界框回归与视频追踪。**推理阶段无需任何相机内参矩阵**，直接从单张 2D 图像回归 8 个角点的绝对像素坐标。

## 项目背景

本项目的目标场景是：以第三视角对两台手持 DJI Action 4 相机进行三维边界框追踪（8 个角点构成的长方体线框），用于叠衣服等手工操作任务的动作捕捉。

主要挑战包括：
- 弱光 / 低对比度环境（暗光桌面）
- 手部大面积遮挡
- 日光灯局部强反光白斑
- 动态运动模糊

**严格约束**：推理期间绝对不使用相机内参 $K$ 或 PnP 算法，纯粹依靠单目 2D 关键点回归直接输出三维长方体。

## 系统架构

```
输入视频帧
   ↓
YOLO 目标检测（2D 边界框定位两台相机）
   ↓
自适应视口裁切（与训练期完全一致的 YOLO 框驱动裁切）
   ↓
DINOv2（ViT-S/14-reg，可选解冻后 4 个 Block 微调）
   ↓
BETR 解码器（Transformer，8 通道热力图回归）
   ↓
SpatialSoftArgmax2d（亚像素级坐标提取，温度 60.0）
   ↓
时序平滑 + 3D 刚体解缠（untangle_cuboid_2d）
   ↓
输出：8 个 2D 角点 + 12 条棱线的三维透视线框
```

## 项目结构

```
Test/
├── BoxDreamer_Network/                  # 核心神经网络（3D 角点预测）
│   ├── backbone/                        # BETR Transformer 解码器
│   ├── encoder/                         # DINOv2 封装
│   ├── models.py                        # BoxDreamerModel 主模型
│   ├── dataset.py                       # 训练数据集与数据增强
│   ├── train.py                         # 训练脚本
│   ├── loss.py                          # Focal Loss + 刚体几何拓扑损失
│   ├── infer_video.py                   # 完整视频推理与 3D 线框渲染
│   ├── extract_full_video_dataset.py    # 全视频 AprilTag 真值挖掘脚本
│   ├── evaluate_frames.py               # 单帧评估工具
│   ├── demo_infer.py                    # 单图推理演示
│   └── README.md                        # 网络详细文档
├── bop_datasets/            # [不追踪] BOP 格式合成渲染数据集
├── real_dataset/            # [不追踪] 原始真实手持样本（147 张）
├── real_dataset_full/       # [不追踪] 全视频 YOLO 对齐样本（307 张）
├── runs/                    # [不追踪] YOLO 训练输出权重
├── test_video/              # [不追踪] 输入/输出视频
├── dji_bbox_corners.npy     # DJI Action 4 相机坐标系下 8 个 3D 角点（毫米）
├── dji_camera.yaml          # YOLO 训练配置（本地）
├── dji_camera_server.yaml   # YOLO 训练配置（服务器）
├── requirements.txt         # Python 依赖列表
├── s2_p1_gen_pbr_data.py    # BOP 合成数据生成脚本
├── export_yolo_labels.py    # YOLO 标签导出工具
├── BoxDreamer.pdf           # 参考论文
└── BoxDreamerModel.py       # 原版多视图 BoxDreamer（参考实现）
```

## 快速开始

### 依赖安装

```bash
pip install -r requirements.txt
```

### 1. 对视频进行 3D 追踪推理

```bash
cd BoxDreamer_Network
python infer_video.py --video ../test_video/head_left_rgb_raw.mp4 --output output.mp4
```

### 2. 从头训练

```bash
cd BoxDreamer_Network
python train.py \
    --real_dataset ../real_dataset_full \
    --mix_real \
    --num_epochs 60 \
    --lr 5e-5 \
    --save_path best_boxdreamer.pth
```

### 3. 原视频专属微调（Video-Specific Adaptation）

```bash
# 第一步：从视频中的 AprilTag 自动提取毫米级精准 3D 真值
python extract_full_video_dataset.py

# 第二步：在视频专属数据集上极速微调
python train.py \
    --real_dataset ../real_dataset_full \
    --mix_real \
    --unfreeze_dino \
    --weights best_boxdreamer.pth \
    --loss_type smooth_l1 \
    --num_epochs 30 \
    --lr 2e-5 \
    --coord_weight 3.0 \
    --save_path best_boxdreamer_adapted.pth
```

## 训练精度演进

| 训练阶段 | 验证误差（224px 切片空间） | 对应原视频实际误差 |
|---|---|---|
| 初始 BOP 合成渲染 | 17.28 px | ~9.2 px |
| + 真实手持样本（147 张） | 11.56 px | ~6.1 px |
| + DINOv2 解冻 + 全套数据增强 | 9.64 px | ~5.1 px |
| + 全视频专属微调（307 张） | **5.48 px** | **~2.9 px** |

## 数据增强策略

训练期对每帧图像随机施加以下 6 种工业级增强（概率 90%）：

1. **$\pm 22°$ 仿射旋转**：同步变换 8 个 3D 角点坐标
2. **手部遮挡打洞**：模拟肤色 / 阴影 / 噪声矩形遮挡块
3. **日光灯局部反光白斑**：随机椭圆高斯高光
4. **暗光亮度压低**：亮度系数随机 0.3 ~ 1.5 倍
5. **高 ISO 传感器噪点**：随机高斯噪声 + 运动模糊
6. **HSV 白平衡抖动**：色调 / 饱和度 / 明度随机扰动

## 服务器运行说明

> GPU 1 硬件故障，**必须**在服务器端执行前设置：
> ```bash
> export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES=0
> ```

模型权重（`.pth`）、视频文件（`.mp4`）及数据集目录均已在 `.gitignore` 中排除，请通过服务器 `scp` 或单独渠道分发。
