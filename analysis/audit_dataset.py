"""LeRobot 데이터셋의 상태·행동·영상과 에피소드별 위치를 검사한다."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def audit(root):
    dataset = LeRobotDataset('OMY', root=root.resolve())
    table = dataset.hf_dataset.with_format(None)
    state = np.asarray(table['observation.state'])
    action = np.asarray(table['action'])
    positions = np.asarray(table['cabinet_pos'])
    episodes = np.asarray(table['episode_index'])
    frames = np.asarray(table['frame_index'])
    timestamps = np.asarray(table['timestamp'])
    assert state.shape == action.shape == (len(dataset), 10)
    assert positions.shape == (len(dataset), 3)
    assert all(np.isfinite(values).all() for values in (state, action, positions, timestamps))
    assert np.array_equal(np.unique(episodes), np.arange(dataset.meta.total_episodes))

    records = []
    counts = Counter()
    for episode in range(dataset.meta.total_episodes):
        indices = np.flatnonzero(episodes == episode)
        assert len(indices) > 0
        assert np.allclose(positions[indices], positions[indices[0]])
        assert np.array_equal(frames[indices], np.arange(len(indices)))
        assert np.allclose(timestamps[indices], np.arange(len(indices)) / dataset.fps, atol=1e-5)
        position = tuple(np.round(positions[indices[0]], 4).tolist())
        counts[position] += 1
        records.append({
            'episode_index': episode,
            'frames': len(indices),
            'dataset_duration_seconds': len(indices) / dataset.fps,
            'cabinet_root_offset_m': list(position),
        })

    for index in range(len(dataset)):
        image = dataset[index]['observation.wrist_image']
        assert tuple(image.shape) == (3, 480, 640)
        if index % 500 == 0:
            print(f'Decoded {index}/{len(dataset)} frames', flush=True)

    return {
        'episodes': dataset.meta.total_episodes,
        'frames': len(dataset),
        'fps': dataset.fps,
        'dataset_duration_seconds': len(dataset) / dataset.fps,
        'position_counts': [{'cabinet_root_offset_m': list(pos), 'episodes': count}
                            for pos, count in counts.items()],
        'checks_passed': ['all camera frames decoded', 'finite 10D state/action',
                          'constant cabinet position within each episode',
                          'continuous frame indices', 'timestamps match dataset fps'],
        'episode_details': records,
        'limitations': 'Drawer displacement and task-success labels were not recorded. '
                       'These checks validate data structure, not task success.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('../docs/dataset_summary.json'))
    args = parser.parse_args()
    summary = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Validated {summary["episodes"]} episodes, {summary["frames"]} frames')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
