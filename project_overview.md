# 项目整体概览与设计文档

> 本文档面向 **BoxDreamer** 项目（位于 `d:/zjy/Deep_Learning/Test`），系统化梳理 **项目结构、关键模块、数据流、训练/推理流程、以及实现的优化方案**，帮助快速上手并在实验室服务器上完成可视化。

---

## 1️⃣ 项目目录结构（关键文件）
```
Test/
├─ BoxDreamer_Network/                # 核心模型、数据集、loss、训练脚本、推理脚本
│   ├─ models.py                     # BoxDreamerModel（ViT‑style backbone + 预测头）
│   ├─ dataset.py                    # DJIActionPoseDataset / RealDJIDataset
│   ├─ loss.py                       # HeatmapLoss、KeypointFocalLoss、RigidTopologyLoss
│   ├─ train.py                      # 训练入口，解析参数、加载数据、循环训练、保存最佳模型
│   ├─ infer_video.py                # 视频/图片推理，完成对称补全、纠缠消除、绘制3D盒子
│   └─ best_boxdreamer.pth           # 训练后保存的最佳权重（默认路径）
│
├─ bop_datasets/                     # BOP 格式的合成渲染数据（默认使用）
│   └─ dji/train_pbr/000000/        # 每帧的 rgb、depth、scene_gt.json 等
│
├─ dji_bbox_corners.npy              # 8 个 3D 顶点的固定坐标（相机坐标系）
├─ export_yolo_labels.py             # 将 YOLO 标注导出为 BOP 所需的 JSON 格式（可选）
├─ OPTIMIZATION_SUMMARY.md          # 项目优化方案、实验记录（已更新）
└─ work/                             # 额外素材（如 .mtl、.obj）
```

> **核心入口**：`BoxDreamer_Network/train.py`（训练）和 `BoxDreamer_Network/infer_video.py`（推理）。其余脚本为辅助工具。

---

## 2️⃣ 关键模块功能说明
| 模块 | 功能 | 主要实现点 |
|------|------|------------|
| **dataset.py** | 数据读取 & 增强 | - `DJIActionPoseDataset` 支持 **随机 Scale/Shift**（训练时 jitter）<br>- `is_train` 标记控制是否使用 jitter<br>- 自动加载 BOP 合成数据或真实切片数据 |
| **loss.py** | 多任务损失 | - `HeatmapLoss` 包含 **热力图 loss**、**亚像素坐标 loss (`coord_weight`)**、**3D 几何拓扑 loss (`geom_weight`)**<br>- 移除了之前对 loss 的 `/10.0` 缩放，保证关键点 loss 权重真实体现 |
| **train.py** | 训练循环 | - 读取命令行参数（支持 `--save_path`、`--weights`、`--mix_real` 等）<br>- 划分 **80% 训练 / 20% 验证**<br>- 使用 `AdamW` + **余弦学习率衰减**<br>- **仅在验证像素误差下降时保存模型** (`best_boxdreamer.pth`) |
| **infer_video.py** | 推理 Pipeline | - 读取 YOLO 检测框 → Crop → Resize → BoxDreamer 前向 → 软 Argmax 获得亚像素坐标<br>- **对称补全** `complete_occluded_corners_by_symmetry`
- **纠缠消除** `untangle_cuboid_2d`
- **绘制 3D 立方体** `draw_3d_box`
- **不再使用** `SimpleBoxTracker` 与 `regularize_cuboid_affine_2d`（已删除） |

---

## 3️⃣ 训练流程（在实验室服务器上）
### 3.1 环境准备（一次性）
```bash
# 1. 进入项目根目录（服务器上）
cd /mnt/data/home/zhoujiayan/BoxDreamer/BoxDreamer_Network

# 2. 安装依赖（推荐使用 conda）
conda create -n boxdreamer python=3.10 -y
conda activate boxdreamer
pip install torch torchvision opencv-python tqdm  # 以及你项目中可能需要的其他库，例如 'ultralytics'
```
> **注意**：如果已有 conda 环境，可直接 `conda activate <env>`。

### 3.2 启动训练（示例）
```bash
# 推荐使用 nohup + & 让任务在后台运行
nohup python train.py \
    --num_epochs 40 \
    --batch_size 16 \
    --lr 5e-5 \
    --coord_weight 1.0 \
    --geom_weight 0.5 \
    --num_workers 4 \
    --save_path best_boxdreamer_v2.pth \
    > train_boxdreamer.log 2>&1 &

# 查看实时日志
tail -f train_boxdreamer.log
```
- **日志关键字段**：`Epoch`, `Train Loss`, `Val Loss`, `Err`（像素误差），以及 `Saved Best Model` 标记。
- 训练结束后或中途 **Ctrl+C**（若想手动停止）后，最佳模型会保存在 `best_boxdreamer_v2.pth`。

---

## 4️⃣ 可视化推理（在实验室服务器上）
### 4.1 单帧图片推理
```bash
python infer_video.py \
    --model_path best_boxdreamer_v2.pth \
    --input_image /path/to/test_image.jpg \
    --output_dir ./visualization
```
- 结果图片会保存在 `./visualization`，文件名默认 `frame_00000.png`（若只推一帧则为 `frame_00000.png`），其中 **3D 立方体** 已经绘制在原图上。

### 4.2 视频推理（逐帧）
```bash
python infer_video.py \
    --model_path best_boxdreamer_v2.pth \
    --input_video /path/to/video.mp4 \
    --output_dir ./video_out
```
- 程序会遍历每一帧，生成 `frame_00000.png、frame_00001.png …`，保存在 `video_out` 目录。

### 4.3 合成可播放的视频（ffmpeg）
```bash
# 进入输出目录
cd video_out
# 将帧序列合成为 mp4（帧率 15 fps，可自行调节）
ffmpeg -y -framerate 15 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p result.mp4
```
- `result.mp4` 即为 **可视化后的视频**，可以直接在服务器上用 `mpv`、`ffplay` 播放，或 `scp` 下载到本地观看。

### 4.4 常用调参参数（可选）
| 参数 | 含义 | 推荐值 |
|------|------|--------|
| `--model_path` | 待推理的权重文件 | 训练产生的最佳模型路径 |
| `--input_image` / `--input_video` | 输入路径 | 任何 JPEG/PNG 或 MP4 文件 |
| `--output_dir` | 可视化结果保存目录 | 新建一个专用文件夹即可 |
| `--batch_size`（推理时） | 同时处理多少帧（默认 1） | 小显存机器保持 1，GPU 充足可调到 4 |

---

## 5️⃣ 项目设计思路与优化方案回顾
| 章节 | 内容概述 |
|------|----------|
| **1. 数据增强** | 在 `DJIActionPoseDataset` 中加入 **随机 Scale (0.9‑1.15) & Shift (±10% crop)**，缓解训练‑推理域差距。
| **2. 损失权重** | 将 **关键点坐标 loss** (`coord_weight`) 设为 `1.0`，**几何拓扑 loss** (`geom_weight`) 设为 `0.5`，并去掉不合理的 `/10.0` 缩放，提升关键点精度。
| **3. 去除不必要的后处理** | 删除 **角点局部果冻抖**（方案 1）和 **仿射正则化**（方案 4），简化推理流水线，减少误差累积。
| **4. 学习率调度** | 采用 **CosineAnnealingLR**（从 `lr` 逐渐衰减至 `1e-6`），防止后期震荡并自然实现 **学习率 warm‑up** 效果。
| **5. 模型保存策略** | 只保存 **验证像素误差最小** 的 checkpoint，避免保存大量无用模型，节约磁盘空间。
| **6. 可视化实现** | `infer_video.py` 完整实现 **对称补全 → 纠缠消除 → 3D 绘制**，直接输出带立方体的图像/视频。

---

## 6️⃣ 常见问题与排查
| 场景 | 可能原因 | 解决方案 |
|------|----------|----------|
| **推理报 `FileNotFoundError`** | 输入路径写错或文件未上传至服务器 | 确认路径使用绝对路径，使用 `ls` 检查文件是否存在。
| **模型权重加载失败** (`RuntimeError: Unexpected key ...`) | 权重文件和当前模型结构不匹配（比如改动了 `BoxDreamerModel`） | 确保使用与训练相同的模型代码；若改动后想继续训练，使用 `strict=False` 加载。
| **可视化图像中没有立方体** | 检查 `infer_video.py` 中是否成功检测到 YOLO 框（`clean_boxes` 为空） | 先在同目录下运行 `python infer_video.py --input_image …` 并观察日志 `Detected N boxes`，若为 0，检查 YOLO 权重或图片质量。
| **ffmpeg 合成视频报错** | `ffmpeg` 未安装或帧文件命名不匹配 | 在服务器上 `apt-get install ffmpeg`（Linux）或 `conda install -c conda-forge ffmpeg`；确保帧文件名为 `frame_00000.png` 连续递增。

---

## 7️⃣ 学习资源 & 推荐阅读
- **ViT 与 Transformer**：`Attention is All You Need`、`An Image is Worth 16x16 Words`（Vision Transformer）
- **BOP 数据集**：<https://bop.felk.cvut.cz/>（官方文档、评测指标）
- **BoxDreamer 论文/技术报告**（如果有内部文档，请查阅）
- **Git Bash**：前面提供的 `git_bash_tutorial.md`（快速入门）
- **ffmpeg 官方手册**：<https://ffmpeg.org/documentation.html>

---

## 8️⃣ 快速上手 Checklist（服务器）
1. **环境**：`conda activate boxdreamer`、`pip install -r requirements.txt`（若有）
2. **数据**：确保 `/mnt/data/home/zhoujiayan/BoxDreamer/bop_datasets/dji/train_pbr/000000/` 完整。
3. **训练**：执行 `nohup python train.py … > train.log 2>&1 &`，实时 `tail -f train.log`。
4. **模型**：训练结束后检查 `best_boxdreamer*.pth` 是否生成。
5. **推理**：`python infer_video.py --model_path best_boxdreamer.pth --input_video xxx.mp4 --output_dir ./out`
6. **可视化**：`ffmpeg -framerate 15 -i out/frame_%05d.png -c:v libx264 result.mp4`。
7. **下载结果**（如需要），使用 `scp` 把 `result.mp4` 拉回本地。

---

**祝你在实验室服务器上顺利完成训练、推理与可视化** 🎉 如还有其他细节想要深入了解，随时告诉我！
