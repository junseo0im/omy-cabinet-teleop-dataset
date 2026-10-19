"""캐비넷 배치(spawn) 유틸리티

collect_data.py / teleoperation.py 에서 에피소드마다 캐비넷(cabinet_root) 위치를 바꿔 주기 위해 사용한다.

mode
  - 'three'  : 기본/왼쪽 뒤/오른쪽 뒤 위치에서 3/3/3회 수집
  - 'fixed'  : 항상 기본 위치 (0, 0, 0)
  - 'grid'   : GRID_POSITIONS 를 에피소드마다 순서대로 순환 (위치별 데이터 수를 균등하게)
  - 'random' : X_RANGE x Y_RANGE 에서 균등 분포로 샘플링

모든 후보 위치는 IK 로 '손잡이 잡기 + 서랍 15cm 당기기' 가 가능한지 검사하고,
불가능한 위치는 버린다(random 은 다시 샘플링, grid 는 시작 시 제외).
도달 가능 영역은 analysis/workspace_analysis.py 로 계산한 결과를 근거로 한다.
"""
import mujoco
import numpy as np
import pinocchio as pin

from ik import solve_ik_safe

# 참고 코드의 랜덤 범위
X_RANGE = (-0.1, 0.1)
Y_RANGE = (-0.125, 0.0)

# 3 x 3 격자 (x: 좌/우, y: 앞/뒤)
GRID_POSITIONS = [(x, y, 0.0) for y in (-0.1, -0.05, 0.0) for x in (-0.1, 0.0, 0.1)]

# cabinet_root 이동량(m)과 위치별 수집 개수
THREE_POSITION_PLAN = (
    ((0.0, 0.0, 0.0), 3),
    ((-0.06, -0.05, 0.0), 3),
    ((0.06, -0.05, 0.0), 3),
)
THREE_POSITION_EPISODES = sum(count for _, count in THREE_POSITION_PLAN)


Q0 = np.array([1.57079629, 0.325819591, -1.73266281, 0.930127424, -1.57079296, -1.57080310, 0, 0, 0, 0])
EE_ROTVEC_0 = np.array([0.37008376, -1.53892958, -0.37008376])
LEVEL_ROLL = -0.4727     # 그리퍼를 수평으로 만드는 roll
FINGER_REACH = 0.215     # link6(제어점) -> 손가락 사이 파지점 거리
PULL = 0.15              # 서랍 최대 이동량 (drawer_slide range)


class CabinetSpawner:
    def __init__(self, mj_model, pin_model, frame_id, mode="random",
                 x_range=X_RANGE, y_range=Y_RANGE, grid=GRID_POSITIONS, check_reach=True, start_episode=0):
        assert mode in ("fixed", "grid", "random", "three"), mode
        self.mj_model, self.mode = mj_model, mode
        self.x_range, self.y_range = x_range, y_range
        self.check_reach = check_reach
        self.cabinet_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "cabinet_root")

        # 손잡이 위치 = 캐비넷 위치 + 고정 오프셋 (cabinet_root 는 회전 없이 평행이동만 하므로)
        tmp = mujoco.MjData(mj_model)
        mj_model.body_pos[self.cabinet_bid] = [0.0, 0.0, 0.0]
        mujoco.mj_forward(mj_model, tmp)
        self.handle_offset = tmp.geom_xpos[mj_model.geom(name="handle").id].copy()

        # IK 검사는 수집 루프와 상태를 공유하지 않도록 별도 data 사용
        self.pin_model, self.pin_data, self.frame_id = pin_model, pin_model.createData(), frame_id
        self.R_level = pin.rpy.rpyToMatrix(np.array([LEVEL_ROLL, 0, 0])) @ pin.exp3(EE_ROTVEC_0)

        self.grid = [np.array(p, dtype=float) for p in grid if self.is_reachable(np.array(p, dtype=float))]
        dropped = len(grid) - len(self.grid)
        if mode == "grid" and dropped:
            print(f"[CabinetSpawner] {dropped} grid position(s) are unreachable and were removed.")
        self.plan = []
        if mode == "three":
            for position, count in THREE_POSITION_PLAN:
                pos = np.array(position, dtype=float)
                if not self.is_reachable(pos):
                    raise ValueError(f"Planned cabinet position is unreachable: {position}")
                self.plan.extend([pos.copy() for _ in range(count)])
            if not 0 <= start_episode <= len(self.plan):
                raise ValueError("Saved episode count exceeds the three-position plan")
        self.count = start_episode

    def handle_pos(self, cabinet_pos):
        return np.asarray(cabinet_pos) + self.handle_offset

    def _reach(self, target, q_init):
        T = pin.SE3(self.R_level, target)
        _, q, _ = solve_ik_safe(self.pin_model, self.pin_data, self.frame_id, q_init, T, max_iters=300)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        cur = self.pin_data.oMf[self.frame_id]
        pos_ok = np.linalg.norm(cur.translation - target) < 5e-3
        rot_ok = np.linalg.norm(pin.log3(cur.rotation.T @ self.R_level)) < 5e-2
        lim_ok = np.all(q[:6] >= self.pin_model.lowerPositionLimit[:6] - 1e-6) and \
                 np.all(q[:6] <= self.pin_model.upperPositionLimit[:6] + 1e-6)
        return pos_ok and rot_ok and lim_ok, q

    def is_reachable(self, cabinet_pos):
        """손잡이 파지 자세와, 서랍을 끝까지 당긴 자세 모두 IK 해가 있는지."""
        if not self.check_reach:
            return True
        h = self.handle_pos(cabinet_pos)
        ok_grasp, q_grasp = self._reach(np.array([h[0], h[1] + FINGER_REACH, h[2]]), Q0)
        if not ok_grasp:
            return False
        ok_open, _ = self._reach(np.array([h[0], h[1] + FINGER_REACH + PULL, h[2]]), q_grasp)
        return ok_open

    def next_position(self):
        if self.mode == "three":
            if self.count >= len(self.plan):
                raise StopIteration(f"All {len(self.plan)} planned episodes have been collected")
            pos = self.plan[self.count].copy()
        elif self.mode == "fixed":
            pos = np.zeros(3)
        elif self.mode == "grid":
            pos = self.grid[self.count % len(self.grid)].copy()
        else:
            for _ in range(100):
                pos = np.array([np.random.uniform(*self.x_range), np.random.uniform(*self.y_range), 0.0])
                if self.is_reachable(pos):
                    break
            else:
                raise RuntimeError("No reachable cabinet position found in the given range.")
        self.count += 1
        return pos

    def apply(self, mj_data, cabinet_pos):
        self.mj_model.body_pos[self.cabinet_bid] = cabinet_pos
        mujoco.mj_forward(self.mj_model, mj_data)
