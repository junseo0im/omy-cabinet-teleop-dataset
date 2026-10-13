import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin

def solve_ik_safe(model, data, ee_frame_id, q_init, target_SE3,
                  max_iters=1000,
                  eps=1e-4,
                  dt=0.1,
                  damp=1e-4,
                  max_step_size=0.1,
                  verbose=False):

    q = q_init.copy()
    success = False

    for i in range(max_iters):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)

        current_SE3 = data.oMf[ee_frame_id]

        err_SE3 = current_SE3.actInv(target_SE3)
        err = pin.log(err_SE3).vector  # 6D (3 for rotation, 3 for translation)

        if norm(err) < eps:
            success = True
            break

        J = pin.computeFrameJacobian(model, data, q, ee_frame_id, pin.ReferenceFrame.LOCAL)
        J_log_inv = pin.Jlog6(err_SE3.inverse())  # mapping for log(SE3)
        J = -J_log_inv @ J

        # damped least squares
        JJt = J @ J.T + damp * np.eye(6)
        v = -J.T @ solve(JJt, err)

        # max speed
        if norm(v) > max_step_size:
            v = v / norm(v) * max_step_size

        # update q
        q = pin.integrate(model, q, v * dt)

        if verbose and i % 10 == 0:
            print(f"[{i:03d}] err norm: {norm(err):.6f}")

    return success, q, err