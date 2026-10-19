"""키보드 제어 속도와 렌더링 설정."""
import math
import os
from pathlib import Path

import mujoco


def configure_qt_fonts():
    # cv2를 불러온 뒤, 없는 Qt 폰트 경로를 시스템 폰트 경로로 바꾼다.
    if os.name != 'posix':
        return
    current = os.environ.get('QT_QPA_FONTDIR')
    if current and Path(current).is_dir():
        return
    for directory in ['/usr/share/fonts/truetype/dejavu', '/usr/share/fonts']:
        if Path(directory).is_dir():
            os.environ['QT_QPA_FONTDIR'] = directory
            return


def control_options(dt):
    configure_qt_fonts()
    def positive_speed(name, default):
        value = float(os.environ.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number')
        return value

    move_speed = positive_speed('MOVE_SPEED', 0.1)  # m/s
    rotation_speed = positive_speed('ROT_SPEED', 0.1)  # rad/s
    fast_render = os.environ.get('FAST_RENDER', '0') == '1'
    print(f'Controls: MOVE_SPEED={move_speed:g} m/s, '
          f'ROT_SPEED={rotation_speed:g} rad/s, FAST_RENDER={int(fast_render)}')
    return move_speed * dt, rotation_speed * dt, fast_render


def configure_rendering(scene, fast_render):
    if scene is not None and fast_render:
        # 그림자와 반사를 끈다.
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
