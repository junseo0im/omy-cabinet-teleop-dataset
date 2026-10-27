"""동시에 기록한 관절값으로 동일 시연의 관절공간 데이터셋을 만든다."""
import argparse
from pathlib import Path
import shutil
import tempfile

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def convert(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Destination already exists: {destination}; use a new --output')
    dataset = LeRobotDataset('OMY', root=source)
    required = {'joint_state', 'joint_action', 'cabinet_pos'}
    if not required <= set(dataset.features):
        raise ValueError('Source has no paired joint values. Collect with JOINT_SPACE=0 first.')
    for key in ['joint_state', 'joint_action']:
        if tuple(dataset.features[key]['shape']) != (7,):
            raise ValueError(f'{key} must have seven components')
    table = dataset.hf_dataset.with_format(None)
    episodes = np.asarray(table['episode_index'])
    states = np.asarray(table['joint_state'])
    actions = np.asarray(table['joint_action'])
    positions = np.asarray(table['cabinet_pos'], dtype=np.float32)
    tasks = table['task_index']
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('Source contains nonfinite joint values')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.paired_joint_', dir=destination.parent) as temporary:
        root = Path(temporary) / 'dataset'
        joint = LeRobotDataset.create(
            repo_id='OMY', root=root, robot_type=dataset.meta.robot_type, fps=dataset.fps,
            features={
                'observation.wrist_image': dataset.features['observation.wrist_image'],
                'observation.state': {'dtype': 'float64', 'shape': (7,), 'names': ['state']},
                'action': {'dtype': 'float64', 'shape': (7,), 'names': ['action']},
                'cabinet_pos': dataset.features['cabinet_pos'],
            }, image_writer_threads=0, image_writer_processes=0)
        for episode in range(dataset.meta.total_episodes):
            for index in np.flatnonzero(episodes == episode):
                image = dataset[int(index)]['observation.wrist_image']
                image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                joint.add_frame({
                    'observation.wrist_image': image,
                    'observation.state': states[index],
                    'action': actions[index],
                    'cabinet_pos': positions[index],
                }, task=dataset.meta.tasks[tasks[index]])
            joint.save_episode()
            print(f'Converted {episode + 1}/{dataset.meta.total_episodes}', flush=True)
        shutil.rmtree(root / 'images', ignore_errors=True)
        root.rename(destination)
    print(f'Saved {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    destination = args.output or args.source.resolve().with_name(args.source.resolve().name + '_joint')
    convert(args.source, destination)


if __name__ == '__main__':
    main()
