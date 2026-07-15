#!/usr/bin/env python3
"""V2 calibration: require ALL parts Z>0.05 AND project into image."""
import math, numpy as np, yaml, sys

LAYOUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "src/challenge_cup_simulator/test/collect_scene2_dataset/generated_layouts/scene2_seed_3390_parts.yaml"

with open(LAYOUT_PATH) as f:
    raw = yaml.safe_load(f)
parts = {}
for name, data in raw["parts"].items():
    parts[name] = {"world_xyz": [float(v) for v in data["world_xyz"]]}

fx = fy = 392.87106038; cx = cy = 640.0; W = 1280; H = 720

best_score = -1
best_params = None

for cam_x in np.linspace(0.02, 0.40, 30):
    for cam_z in np.linspace(1.05, 1.80, 30):
        for look_x in np.linspace(-0.40, 0.00, 20):
            for look_z in np.linspace(0.75, 0.88, 10):
                pos = np.array([cam_x, 0.0, cam_z])
                target = np.array([look_x, 0.0, look_z])
                forward = target - pos
                f_norm = np.linalg.norm(forward)
                if f_norm < 1e-6 or forward[2] >= 0:
                    continue
                forward = forward / f_norm
                world_up = np.array([0, 0, 1])
                right = np.cross(forward, world_up)
                if np.linalg.norm(right) < 1e-6:
                    continue
                right = right / np.linalg.norm(right)
                up = np.cross(right, forward)
                R = np.array([right, up, forward])
                
                all_z_ok = True
                in_frame = 0
                total_z = 0
                for name, data in parts.items():
                    pw = np.array(data["world_xyz"])
                    pc = R @ pw - R @ pos
                    z = pc[2]
                    if z <= 0.05:
                        all_z_ok = False
                        break
                    u = fx * pc[0] / z + cx
                    v = fy * pc[1] / z + cy
                    if 20 <= u < W-20 and 20 <= v < H-20:
                        in_frame += 1
                    total_z += z
                if all_z_ok and in_frame >= 3:
                    score = in_frame * 100 + total_z
                    if score > best_score:
                        best_score = score
                        best_params = (pos, target, R, in_frame)

if best_params:
    p, t, R, n = best_params
    print(f"Best: {n}/6 parts in frame (score={best_score:.0f})")
    print(f"  camera position: [{p[0]:.4f}, 0.0, {p[2]:.4f}]")
    print(f"  camera look-at:  [{t[0]:.4f}, 0.0, {t[2]:.4f}]")

    trace = R[0,0]+R[1,1]+R[2,2]
    if trace > 0:
        s = math.sqrt(trace+1)*2; w=s/4; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=s/4; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=s/4; z=(R[1,2]+R[2,1])/s
    else:
        s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=s/4

    print(f"\n# For head_camera_extrinsics.yaml:")
    print(f"world_pos: [{p[0]:.4f}, 0.0, {p[2]:.4f}]")
    print(f"world_quat_xyzw: [{x:.6f}, {y:.6f}, {z:.6f}, {w:.6f}]")
else:
    print("No solution found with all parts Z>0.05 and >=3 in frame")
