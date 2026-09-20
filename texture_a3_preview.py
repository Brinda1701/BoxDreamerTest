import blenderproc as bproc

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# 固定预览脚本: 只用于检查烘焙后的顶点颜色方向是否正确。
# 运行时需在 BlenderProc 环境: blenderproc run texture_a3_preview.py

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:\zjy\Deep_Learning\Test")
PLY_PATH = ROOT / "work" / "obj_000001_colored.ply"
OUT_DIR = ROOT / "work"

parser = argparse.ArgumentParser()
parser.add_argument("--ply", type=str, default=str(PLY_PATH))
parser.add_argument("--out_prefix", type=str, default="preview")
args = parser.parse_args()

bproc.init()

# 使用与训练一致的内参。
cam = json.load(open(ROOT / "bop_datasets" / "dji" / "camera.json", encoding="utf-8"))
K = np.array(
    [[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]]
)
bproc.camera.set_intrinsics_from_K_matrix(K, cam["width"], cam["height"])

# 加载带顶点颜色的模型。
objs = bproc.loader.load_obj(args.ply)
if len(objs) != 1:
    raise RuntimeError("预览要求模型是单一网格。")
obj = objs[0]
obj.set_scale([0.001, 0.001, 0.001])
obj.set_location([0.0, 0.0, 0.0])
obj.set_rotation_euler([0.0, 0.0, 0.0])

# 简单灯光 + 环境背景, 让黑色机身可辨认。
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0.5, -0.6, 1.2])
light.set_energy(300)
try:
    bproc.world.set_world_background_hdr_img(
        str(ROOT / "resources" / "hdris" / "small_empty_room_3_1k.hdr"),
        strength=1.2,
    )
except Exception as exc:
    print("HDR 设置失败, 仅使用点光源:", exc)

views = {
    0: np.array([0.0, -0.7, 0.05]),  # 从 -Y 看镜头面
    1: np.array([0.0, 0.7, 0.05]),   # 从 +Y 看主屏面
    2: np.array([0.45, -0.35, 0.22]),  # 前右+顶部过渡(侧视1)
    3: np.array([-0.45, -0.3, -0.18]), # 前左+底部过渡(侧视2)
}
for frame_id, location in views.items():
    poi = np.array([0.0, 0.0, 0.0])
    rotation = bproc.camera.rotation_from_forward_vec(
        poi - location, inplane_rot=0.0
    )
    cam2world = bproc.math.build_transformation_mat(location.tolist(), rotation)
    bproc.camera.add_camera_pose(cam2world, frame=frame_id)

bproc.renderer.set_max_amount_of_samples(20)
data = bproc.renderer.render()

OUT_DIR.mkdir(parents=True, exist_ok=True)
names = [
    "%s_lens_face.png" % args.out_prefix,
    "%s_screen_face.png" % args.out_prefix,
    "%s_side1_face.png" % args.out_prefix,
    "%s_side2_face.png" % args.out_prefix,
]
for idx, name in enumerate(names):
    rgb = data["colors"][idx]
    bgr = rgb[:, :, ::-1]
    out_path = OUT_DIR / name
    cv2.imwrite(str(out_path), bgr)
    print("已保存: %s" % out_path)
