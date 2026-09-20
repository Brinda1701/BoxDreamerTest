# BoxDreamer - Monocular 3D Bounding Box Tracking System

An uncalibrated monocular 3D bounding box regression and tracking system for DJI Action 4 cameras, based on DINOv2 + BETR architecture with zero camera intrinsics at inference time.

## Project Overview

This project achieves accurate 3D corner prediction (8 keypoints forming a cuboid) for handheld DJI Action 4 cameras in challenging conditions:
- Dark lighting / low contrast
- Hand occlusion
- Specular glare
- Dynamic motion blur

**Architecture**: YOLO (2D detection) → DINOv2 (feature extraction) → BETR (8-corner heatmap regression) → SpatialSoftArgmax2d (subpixel localization)

**Key constraint**: No camera intrinsics K or PnP at inference time — pure monocular 2D keypoint regression.

## Project Structure

```
Test/
├── BoxDreamer_Network/       # Core neural network for 3D corner prediction
│   ├── backbone/             # BETR transformer decoder
│   ├── encoder/              # DINOv2 wrapper
│   ├── models.py             # BoxDreamerModel (main model class)
│   ├── dataset.py            # Training dataset loaders with augmentations
│   ├── train.py              # Training script
│   ├── loss.py               # Focal loss + Rigid topology loss
│   ├── infer_video.py        # Full video inference & 3D wireframe rendering
│   ├── extract_full_video_dataset.py  # AprilTag-based GT extraction
│   └── README.md             # Detailed network documentation
├── bop_datasets/             # [LARGE, not tracked] BOP synthetic render datasets
├── real_dataset/             # [not tracked] Original real handheld samples (147 imgs)
├── real_dataset_full/        # [not tracked] Full-video YOLO-aligned samples (307 imgs)
├── runs/                     # [not tracked] YOLO training weights
├── test_video/               # [not tracked] Input/output videos
├── yolo_full_scene_dataset/  # [not tracked] YOLO training dataset
├── dji_bbox_corners.npy      # 3D corner coordinates in camera frame (mm)
├── dji_camera.yaml           # YOLO training config (local)
├── dji_camera_server.yaml    # YOLO training config (server)
├── requirements.txt          # Python dependencies
├── s2_p1_gen_pbr_data.py     # BOP synthetic data generation script
├── export_yolo_labels.py     # YOLO label exporter
├── BoxDreamer.pdf            # Reference paper
└── BoxDreamerModel.py        # Original multi-view BoxDreamer (reference)
```

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
```

### 1. Run 3D tracking on a video
```bash
cd BoxDreamer_Network
python infer_video.py --video ../test_video/head_left_rgb_raw.mp4 --output output.mp4
```

### 2. Train from scratch
```bash
cd BoxDreamer_Network
python train.py \
    --real_dataset ../real_dataset_full \
    --mix_real \
    --num_epochs 60 \
    --lr 5e-5 \
    --save_path best_boxdreamer.pth
```

### 3. Fine-tune on a specific video (Video-Specific Adaptation)
```bash
# Step 1: Extract YOLO-aligned ground truth from video AprilTags
python extract_full_video_dataset.py

# Step 2: Fine-tune with video-specific data
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

## Training Progress History

| Stage | Val Error (224px crop) | Approx. on Full Frame |
|---|---|---|
| Initial BOP-only | 17.28 px | ~9.2 px |
| + Real data (147 samples) | 11.56 px | ~6.1 px |
| + DINOv2 unfreeze + augmentation | 9.64 px | ~5.1 px |
| + Video-specific adaptation (307 samples) | 5.48 px | ~2.9 px |

## Notes

- GPU: Server requires `export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES=0`
- Model weights (.pth), videos (.mp4), and datasets are excluded from git (see `.gitignore`)

