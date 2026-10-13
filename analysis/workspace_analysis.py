"""캐비넷 위치별 도달 가능성(Reachability) 분석

캐비넷 루트(cabinet_root)를 (x, y) 격자로 옮겨가며, 로봇이
  1) 손잡이를 잡는 자세(grasp)와
  2) 서랍을 15cm 당긴 자세(open)
에 IK로 도달할 수 있는지 계산한다. 결과는 히트맵(png)과 csv로 저장한다.

- 제어점은 Pinocchio의 tcp_link(= link6, 손목 플랜지)이며, 손가락 끝(TCP)은 그보다 0.22m 앞에 있다.
- 그리퍼는 수평(roll = -0.4727 rad)으로 맞춘 자세를 기준으로 한다.

실행: cd analysis && python workspace_analysis.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pinocchio as pin

sys.path.insert(0, os.path.abspath("../src"))
from ik import solve_ik_safe
from cabinet_spawn import GRID_POSITIONS

OUT_DIR = os.path.abspath("../docs/figures")
os.makedirs(OUT_DIR, exist_ok=True)

xml_path = os.path.abspath("../asset/scene.xml")
urdf_path = os.path.abspath("../asset/urdf_omy/omy.urdf")
mesh_path = os.path.abspath("../asset/urdf_omy")

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
cabinet_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cabinet_root")
handle_gid = mj_model.geom(name="handle").id

pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_path])
pin_data = pin_model.createData()
frame_id = pin_model.getFrameId("tcp_link")

q0 = np.array([1.57079629, 0.325819591, -1.73266281, 0.930127424, -1.57079296, -1.57080310, 0, 0, 0, 0])
ee_rotvec_0 = np.array([0.37008376, -1.53892958, -0.37008376])
LEVEL_ROLL = -0.4727          # 그리퍼를 수평으로 만드는 roll (U 키)
FINGER_REACH = 0.215         # link6 -> 손가락 사이 파지점 거리 (+y 방향 오프셋)
PULL = 0.15                  # 서랍 최대 이동량
POS_TOL, ROT_TOL = 5e-3, 5e-2

R_level = pin.rpy.rpyToMatrix(np.array([LEVEL_ROLL, 0, 0])) @ pin.exp3(ee_rotvec_0)
lower, upper = pin_model.lowerPositionLimit, pin_model.upperPositionLimit


def handle_pos(x, y):
    mj_model.body_pos[cabinet_bid] = [x, y, 0.0]
    mj_data.qpos[:] = 0
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data.geom_xpos[handle_gid].copy()


def reach(target_pos, q_init):
    """IK로 target에 도달 가능한지: (성공 여부, 해, 위치오차, 자세오차)."""
    T = pin.SE3(R_level, target_pos)
    _, q, _ = solve_ik_safe(pin_model, pin_data, frame_id, q_init, T, max_iters=300)
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    cur = pin_data.oMf[frame_id]
    pos_err = np.linalg.norm(cur.translation - target_pos)
    rot_err = np.linalg.norm(pin.log3(cur.rotation.T @ R_level))
    in_limits = np.all(q[:6] >= lower[:6] - 1e-6) and np.all(q[:6] <= upper[:6] + 1e-6)
    return (pos_err < POS_TOL and rot_err < ROT_TOL and in_limits), q, pos_err, rot_err


xs = np.round(np.arange(-0.25, 0.2501, 0.025), 3)
ys = np.round(np.arange(-0.30, 0.2001, 0.025), 3)
grid = np.zeros((len(ys), len(xs)), dtype=int)   # 0: 불가, 1: 잡기만 가능, 2: 잡기+열기 가능
rows = []
for iy, y in enumerate(ys):
    for ix, x in enumerate(xs):
        h = handle_pos(x, y)
        ok_g, q_g, eg, _ = reach(np.array([h[0], h[1] + FINGER_REACH, h[2]]), q0)
        ok_o, _, eo, _ = reach(np.array([h[0], h[1] + FINGER_REACH + PULL, h[2]]), q_g) if ok_g else (False, None, np.nan, np.nan)
        grid[iy, ix] = 2 if (ok_g and ok_o) else (1 if ok_g else 0)
        rows.append((x, y, h[0], h[1], int(ok_g), int(ok_o), eg, eo))

np.savetxt(os.path.join(OUT_DIR, "workspace_reachability.csv"), np.array(rows), delimiter=",", fmt="%.4f",
           header="cab_x,cab_y,handle_x,handle_y,grasp_ok,open_ok,grasp_pos_err,open_pos_err", comments="")

fig, ax = plt.subplots(figsize=(10.5, 7))
cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#f4b183", "#70ad47"])
dx, dy = xs[1] - xs[0], ys[1] - ys[0]
ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=2, aspect="equal",
          extent=[xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2])
ref = plt.Rectangle((-0.1, -0.125), 0.2, 0.125, fill=False, ec="#1f4e79", lw=2.5, ls="--")
ax.add_patch(ref)
ax.plot(0, 0, marker="*", ms=14, color="#1f4e79")
gx, gy = zip(*[(p[0], p[1]) for p in GRID_POSITIONS])
ax.scatter(gx, gy, s=60, facecolors="white", edgecolors="black", zorder=3)
ax.set_xlabel("cabinet_root x offset [m]")
ax.set_ylabel("cabinet_root y offset [m]  (+y: closer to robot)")
ax.set_title("Reachability of drawer handle vs. cabinet position")
handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in
           zip(["#d9d9d9", "#f4b183", "#70ad47"], ["unreachable", "grasp only", "grasp + open (15cm)"])]
handles += [matplotlib.lines.Line2D([], [], color="#1f4e79", ls="--", lw=2.5, label="reference random range"),
            matplotlib.lines.Line2D([], [], color="#1f4e79", marker="*", ls="", ms=12, label="default position"),
            matplotlib.lines.Line2D([], [], color="black", marker="o", mfc="white", ls="", ms=8, label="grid spawn positions")]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))
fig.savefig(os.path.join(OUT_DIR, "workspace_reachability.png"), dpi=150, bbox_inches="tight")

full = grid == 2
print(f"grid {len(xs)}x{len(ys)} | grasp+open feasible cells: {full.sum()} / {full.size}")
for y in (-0.2, -0.125, 0.0, 0.1):
    iy = int(np.argmin(np.abs(ys - y)))
    ok_x = xs[full[iy]]
    print(f"  y={y:+.3f}: feasible x range = {ok_x.min() if len(ok_x) else None} .. {ok_x.max() if len(ok_x) else None}")
ref_mask = (xs[None, :] >= -0.1) & (xs[None, :] <= 0.1) & (ys[:, None] >= -0.125) & (ys[:, None] <= 0.0)
print(f"reference range cells feasible: {full[ref_mask].sum()} / {ref_mask.sum()}")
