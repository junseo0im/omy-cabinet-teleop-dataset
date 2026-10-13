import os
import time
import cv2
import glfw
import mujoco
import mujoco.viewer as mj_view
import numpy as np
import pinocchio as pin
import shutil

from ik import solve_ik_safe
from cabinet_spawn import CabinetSpawner

from lerobot.datasets.lerobot_dataset import LeRobotDataset

def key_callback(keycode):
    global pre_keycode, key_state, gripper_state

    if keycode == glfw.KEY_SPACE:
        gripper_state = not gripper_state

    if keycode == glfw.KEY_LEFT_SHIFT:
        print("Reset")

    if keycode == pre_keycode:
        key_state[keycode] = True
    else:
        key_state = {}
        key_state[keycode] = True

    pre_keycode = keycode

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out
    
def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

# 캐비넷 배치 방식 (cabinet_spawn.py 참고)
#   'fixed' : 기본 위치 고정
#   'grid'  : 3x3 격자 위치를 에피소드마다 순환
#   'random': 범위 내 무작위 (도달 불가능한 위치는 IK 검사로 제외)
#   실행 시 환경변수로 변경 가능: SPAWN_MODE=grid python collect_data.py
SPAWN_MODE = os.environ.get('SPAWN_MODE', 'random')

NUM_DEMO  = int(os.environ.get('NUM_DEMO', 50))
ROOT = {
    'fixed':  '../dataset/cabinet_opening',
    'grid':   '../dataset/cabinet_opening_grid_spawn',
    'random': '../dataset/cabinet_opening_random_spawn',
}[SPAWN_MODE]

REPO_NAME = 'OMY'
TASK_NAME = 'Opening the Cabinet'

np.random.seed(20)

xml_path = os.path.abspath("../asset/scene.xml")
    
urdf_path = os.path.abspath("../asset/urdf_omy/omy.urdf")
mesh_path = os.path.abspath("../asset/urdf_omy")

while True:
    anwser = input(f"Select one of the following options to proceed:\n"
                f"1. Create a new dataset at {ROOT} (this will remove pre dataset).\n"
                f"2. Append to the existing dataset at {ROOT}.\n"
                f"Enter 1 or 2: ")

    if anwser == '1':
        if os.path.exists(ROOT):
            shutil.rmtree(ROOT)
            
        # LeRobotDataset.create 함수를 사용하여 새로운 데이터셋 생성
        # state  = 현재 EE 위치(3) + 6D 회전(6) + 그리퍼(1)
        # action = 목표 EE 위치(3) + 6D 회전(6) + 목표 그리퍼(1)
        dataset = LeRobotDataset.create(
            repo_id=REPO_NAME,
            root = ROOT,
            robot_type="omy",
            fps=50,
            features={
                "observation.wrist_image": {
                    "dtype": "image",
                    "shape": (480, 640, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float64",
                    "shape": (10,),
                    "names": ["state"],
                },
                "action": {
                    "dtype": "float64",
                    "shape": (10,),
                    "names": ["action"],
                },
                # 에피소드의 캐비넷 위치 기록용 (분포 확인/위치별 분석).
                # 'observation.' 로 시작하지 않으므로 ACT 정책 입력으로는 사용되지 않음.
                "cabinet_pos": {
                    "dtype": "float32",
                    "shape": (3,),
                    "names": ["x", "y", "z"],
                },
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )

        print("Creating a new dataset...")
            
        break
    
    elif anwser == '2':
        if os.path.exists(ROOT):
            dataset = LeRobotDataset(
                repo_id=REPO_NAME,
                root = ROOT,
            )
            print("Appending to the existing dataset...")
            
        else:
            print(f"The folder {ROOT} does not exist.")
        
        break
    
    else:
        print("Invalid input. Please enter 1 or 2.")
        continue
        

mj_model = mujoco.MjModel.from_xml_path(xml_path)
mj_data = mujoco.MjData(mj_model)
drawer_slide_idx = mj_model.joint(name="drawer_slide").jntid

pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, package_dirs=[mesh_path])
pin_data = pin_model.createData()
frame_id = pin_model.getFrameId("tcp_link")

spawner = CabinetSpawner(mj_model, pin_model, frame_id, mode=SPAWN_MODE)
cabinet_pos = spawner.next_position()
spawner.apply(mj_data, cabinet_pos)
print(f"Cabinet position ({SPAWN_MODE}): {np.round(cabinet_pos, 3)}")

action = np.zeros(7)
episode_id = 0
record_flag = False

q0 = np.array([1.57079629, 0.325819591, -1.73266281, 0.930127424, -1.57079296, -1.57080310,0,0,0,0])
mj_data.qpos[:pin_model.nq] = q0
ee_position_0 = np.array([0.113, -0.1895, 0.535])
ee_rotvec_0 = np.array([0.37008376, -1.53892958, -0.37008376])

delta = 0.002
max_grip = 0.9
min_grip = 0.0
key_state = {}
pre_keycode = None
gripper_state = False

dt = 0.02 # 50Hz

t0 = time.time()
t = 0.0
try:
    with mj_view.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        renderer = mujoco.Renderer(mj_model, 480, 640)
        
        viewer.cam.azimuth = 200
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.2, -0.3, 0.5]

        pin.forwardKinematics(pin_model, pin_data, q0)
        pin.updateFramePlacements(pin_model, pin_data)
        current_SE3 = pin_data.oMf[frame_id]
        ee_position = ee_position_0.copy()

        eulers = np.zeros(3)

        while viewer.is_running() and episode_id < NUM_DEMO:
            # End Effector의 z 좌표를 위아래로 움직이기 위한 코드 (J: 아래, K: 위)
            if key_state.get(glfw.KEY_J):
                ee_position[2] -= delta
            elif key_state.get(glfw.KEY_K):
                ee_position[2] += delta
            elif key_state.get(glfw.KEY_LEFT):
                ee_position[1] -= delta
            elif key_state.get(glfw.KEY_RIGHT):
                ee_position[1] += delta
            elif key_state.get(glfw.KEY_UP):
                ee_position[0] -= delta
            elif key_state.get(glfw.KEY_DOWN):
                ee_position[0] += delta
            elif key_state.get(glfw.KEY_Y):
                eulers[0] += delta 
            elif key_state.get(glfw.KEY_U):
                eulers[0] -= delta 
            elif key_state.get(glfw.KEY_LEFT_SHIFT):
                if dataset.episode_buffer["size"] == 0:
                    print("Please start recording first by pressing Right Shift + any movement key")
                    key_state = {}
                    continue

                print(f"Resetting the environment for the next episode: {episode_id}/{NUM_DEMO}")
                episode_id += 1
                record_flag = False
                
                dataset.save_episode()

                # End Effector의 위치와 자세, 로봇의 관절각도, Cabinet을 초기 상태로 리셋
                ee_position = ee_position_0.copy()
                eulers = np.zeros(3)
                gripper_state = False
                key_state = {}

                mj_data.qpos[:pin_model.nq] = q0
                mj_data.qpos[mj_model.jnt_qposadr[drawer_slide_idx]] = 0.0
                mj_data.qvel[:] = 0.0
                mj_data.qacc[:] = 0.0
                mj_data.ctrl[:pin_model.nq] = q0

                # 다음 에피소드의 캐비넷 위치 재배치 (spawner.apply 내부에서 mj_forward 수행)
                cabinet_pos = spawner.next_position()
                spawner.apply(mj_data, cabinet_pos)
                print(f"Cabinet position ({SPAWN_MODE}): {np.round(cabinet_pos, 3)}")

            elif key_state.get(glfw.KEY_RIGHT_SHIFT):
                if not record_flag and sum(action) != 0:
                    record_flag = True
                    print("Start recording")
            
            R_0 = pin.exp3(ee_rotvec_0)
            R_target = pin.rpy.rpyToMatrix(eulers) @ R_0
            T_target = pin.SE3(R_target, ee_position)

            ee_rotvec_target = pin.log3(R_target)
            # print(f"ee position: {ee_position}, ee rotvec: {ee_rotvec_target}, joint pos: {mj_data.qpos[:pin_model.nq]}")
            q = np.copy(mj_data.qpos[:pin_model.nq])
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)
            T_current = pin_data.oMf[frame_id]
            current_ee_pose = np.concatenate([T_current.translation,
                                      mat_to_rot6d(T_current.rotation)], axis=0)

            success, q_sol, _ = solve_ik_safe(pin_model, pin_data, frame_id, q, T_target)

            target_ee_pose = np.concatenate([T_target.translation,
                                          mat_to_rot6d(T_target.rotation)], axis=0)

            if gripper_state:
                q_sol[-2] = min(q_sol[-2] + delta * 20, max_grip)
            else:
                q_sol[-2] = max(q_sol[-2] - delta * 20, min_grip)
            q_sol[-4] = q_sol[-2]

            q_sol[-1] = 0.0
            q_sol[-3] = 0.0

            mj_data.ctrl[:pin_model.nq] = q_sol

            renderer.update_scene(mj_data, camera="egocentric")
            rgb_overlay = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("Wrist Camera", rgb_overlay)

            wrist_image = cv2.cvtColor(rgb_overlay, cv2.COLOR_BGR2RGB)
            # wrist_image = cv2.resize(wrist_image, (256, 256), interpolation=cv2.INTER_LINEAR)

            state = np.concatenate([current_ee_pose, [q[-2]]], axis=0)
            action = np.concatenate([target_ee_pose, [q_sol[-2]]], axis=0)
                
            cv2.waitKey(1)
            if record_flag:
                # 데이터셋에 프레임 추가: 손목 카메라 이미지 + 현재 상태(state) + 목표 행동(action)
                dataset.add_frame( {
                        "observation.wrist_image": wrist_image,
                        "observation.state": state,
                        "action": action,
                        "cabinet_pos": cabinet_pos.astype(np.float32),
                    }, task = TASK_NAME
                )

            for _ in range(int(dt / 0.002)):
                # mujoco.mj_forward(mj_model, mj_data)
                mujoco.mj_step(mj_model, mj_data)

            t += dt
            t_sleep = t - (time.time() - t0)

            viewer.sync()       
            if t_sleep > 0:
                time.sleep(t_sleep)

except KeyboardInterrupt:
    print("\nSimulation interrupted. Closing viewer...")

renderer.close()
viewer.close()
cv2.destroyAllWindows()

shutil.rmtree(dataset.root / 'images')