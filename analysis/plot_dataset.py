"""수집한 LeRobot 데이터셋 요약 그림 생성

  1) <name>_spawn.png      : 에피소드별 캐비넷 위치 분포 (도달 가능 영역 위에 표시)
  2) <name>_trajectory.png : 한 에피소드의 state / action 시계열
  3) <name>_frames.png     : 한 에피소드의 손목 카메라 프레임 스트립
  + 콘솔에 에피소드 수, 길이 통계 출력

실행: cd analysis && python plot_dataset.py ../dataset/cabinet_opening_random_spawn [episode_index]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = os.path.abspath(sys.argv[1])
ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
name = os.path.basename(root.rstrip("/"))
OUT_DIR = os.path.abspath("../docs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

ds = LeRobotDataset("OMY", root=root)
hf = ds.hf_dataset.with_format(None)
ep_idx = np.array(hf["episode_index"])
n_eps = ds.meta.total_episodes
lengths = np.bincount(ep_idx, minlength=n_eps)
print(f"[{name}] episodes={n_eps} frames={len(ds)} fps={ds.fps} | "
      f"episode length: mean {lengths.mean()/ds.fps:.1f}s, min {lengths.min()/ds.fps:.1f}s, max {lengths.max()/ds.fps:.1f}s")

# ---- 1) spawn distribution --------------------------------------------------
if "cabinet_pos" in ds.features:
    cab = np.array(hf["cabinet_pos"])
    first = np.array([np.argmax(ep_idx == e) for e in range(n_eps)])
    pos = cab[first]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    csv = os.path.join(OUT_DIR, "workspace_reachability.csv")
    if os.path.exists(csv):
        w = np.genfromtxt(csv, delimiter=",", names=True)
        ok = w["open_ok"] > 0.5
        ax.scatter(w["cab_x"][ok], w["cab_y"][ok], s=150, marker="s", color="#c5e0b4", label="reachable (grasp+open)")
        ax.scatter(w["cab_x"][~ok], w["cab_y"][~ok], s=150, marker="s", color="#e7e6e6", label="not reachable")
    uniq, counts = np.unique(np.round(pos[:, :2], 3), axis=0, return_counts=True)
    ax.scatter(pos[:, 0], pos[:, 1], s=40, color="#1f4e79", zorder=3, label=f"episodes (n={n_eps})")
    for (x, y), c in zip(uniq, counts):
        if c > 1:
            ax.annotate(f"x{c}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlim(-0.2, 0.2); ax.set_ylim(-0.2, 0.1); ax.set_aspect("equal")
    ax.set_xlabel("cabinet x offset [m]"); ax.set_ylabel("cabinet y offset [m]")
    ax.set_title(f"Cabinet positions per episode: {name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    fig.savefig(os.path.join(OUT_DIR, f"{name}_spawn.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---- 2) state / action trajectory -------------------------------------------
sel = np.where(ep_idx == ep)[0]
state = np.array(hf["observation.state"])[sel]
action = np.array(hf["action"])[sel]
t = np.arange(len(sel)) / ds.fps
dim = state.shape[1]
if dim == 10:
    groups = [("EE position [m]", slice(0, 3), ["x", "y", "z"]),
              ("EE rotation (6D)", slice(3, 9), [f"r{i}" for i in range(6)]),
              ("gripper", slice(9, 10), ["grip"])]
else:
    groups = [("joint angle [rad]", slice(0, dim - 1), [f"q{i+1}" for i in range(dim - 1)]),
              ("gripper", slice(dim - 1, dim), ["grip"])]
fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2.6 * len(groups)), sharex=True)
for ax, (title, s, labels) in zip(axes, groups):
    colors = plt.cm.tab10(np.arange(s.stop - s.start))
    for k, lab in enumerate(labels):
        ax.plot(t, state[:, s][:, k], color=colors[k], lw=1.6, label=f"state {lab}")
        ax.plot(t, action[:, s][:, k], color=colors[k], lw=1.0, ls="--")
    ax.set_ylabel(title); ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
axes[-1].set_xlabel("time [s]  (solid: state, dashed: action)")
fig.suptitle(f"{name} - episode {ep}")
fig.savefig(os.path.join(OUT_DIR, f"{name}_trajectory.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- 3) wrist camera frames -------------------------------------------------
idx = np.linspace(sel[0], sel[-1], 6).astype(int)
fig, axes = plt.subplots(1, len(idx), figsize=(3.2 * len(idx), 2.8))
for ax, i in zip(axes, idx):
    img = ds[int(i)]["observation.wrist_image"].permute(1, 2, 0).numpy()
    ax.imshow(img); ax.set_axis_off()
    ax.set_title(f"t = {(i - sel[0]) / ds.fps:.1f}s", fontsize=10)
fig.suptitle(f"{name} - episode {ep} wrist camera")
fig.savefig(os.path.join(OUT_DIR, f"{name}_frames.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print("saved figures to", OUT_DIR)
