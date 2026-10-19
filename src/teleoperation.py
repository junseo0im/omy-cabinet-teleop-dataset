import os
import time
import cv2
import glfw
import mujoco
import mujoco.viewer as mj_view
import numpy as np
import pinocchio as pin

from ik import solve_ik_safe
from cabinet_spawn import CabinetSpawner
from teleop_options import control_options, configure_rendering

def key_callback(keycode):
    global pre_keycode, key_state, gripper_state

    if keycode == glfw.KEY_SPACE:
        gripper_state = not gripper_state

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

# 캐비넷 배치 방식: 'fixed' | 'grid' | 'random' | 'three' (cabinet_spawn.py 참고)
SPAWN_MODE = os.environ.get('SPAWN_MODE', 'random')

REPO_NAME = 'OMY'
TASK_NAME = 'Opening the Cabinet'

np.random.seed(325)

xml_path = os.path.abspath("../asset/scene.xml")
    
urdf_path = os.path.abspath("../asset/urdf_omy/omy.urdf")
mesh_path = os.path.abspath("../asset/urdf_omy")
        

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
move_delta, rotation_delta, fast_render = control_options(dt)

t0 = time.time()
t = 0.0
try:
    with mj_view.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        renderer = mujoco.Renderer(mj_model, 480, 640)
        configure_rendering(renderer.scene, fast_render)
        configure_rendering(viewer.user_scn, fast_render)
        
        viewer.cam.azimuth = 200
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = [0.2, -0.3, 0.5]

        pin.forwardKinematics(pin_model, pin_data, q0)
        pin.updateFramePlacements(pin_model, pin_data)
        current_SE3 = pin_data.oMf[frame_id]
        ee_position = ee_position_0.copy()

        eulers = np.zeros(3)

        while viewer.is_running():
            # 키 입력에 따른 목표 위치와 자세 변경
            if key_state.get(glfw.KEY_J):
                ee_position[2] -= move_delta
            elif key_state.get(glfw.KEY_K):
                ee_position[2] += move_delta
            elif key_state.get(glfw.KEY_LEFT):
                ee_position[1] -= move_delta
            elif key_state.get(glfw.KEY_RIGHT):
                ee_position[1] += move_delta
            elif key_state.get(glfw.KEY_UP):
                ee_position[0] -= move_delta
            elif key_state.get(glfw.KEY_DOWN):
                ee_position[0] += move_delta
            elif key_state.get(glfw.KEY_Y):
                eulers[0] += rotation_delta
            elif key_state.get(glfw.KEY_U):
                eulers[0] -= rotation_delta
            elif key_state.get(glfw.KEY_LEFT_SHIFT):
                print(f"Resetting the environment.")

                # 로봇과 서랍을 초기 상태로 되돌린다.
                ee_position = ee_position_0.copy()
                eulers = np.zeros(3)
                gripper_state = False
                key_state = {}

                mj_data.qpos[:pin_model.nq] = q0
                mj_data.qpos[mj_model.jnt_qposadr[drawer_slide_idx]] = 0.0
                mj_data.qvel[:] = 0.0
                mj_data.qacc[:] = 0.0
                mj_data.ctrl[:pin_model.nq] = q0

                # 캐비넷 위치 재배치 (spawner.apply 내부에서 mj_forward 수행)
                if SPAWN_MODE == 'three' and spawner.count >= len(spawner.plan):
                    spawner.count = 0
                cabinet_pos = spawner.next_position()
                spawner.apply(mj_data, cabinet_pos)
                print(f"Cabinet position ({SPAWN_MODE}): {np.round(cabinet_pos, 3)}")
            
            R_0 = pin.exp3(ee_rotvec_0)
            R_target = pin.rpy.rpyToMatrix(eulers) @ R_0
            T_target = pin.SE3(R_target, ee_position)

            ee_rotvec_target = pin.log3(R_target)
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
                
            cv2.waitKey(1)

            for _ in range(int(dt / 0.002)):
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
