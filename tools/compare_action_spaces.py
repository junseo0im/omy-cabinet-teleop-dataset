"""동일 시연의 작업공간·관절공간 ACT를 같은 하이퍼파라미터로 학습한다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, default=PROJECT / 'docs/act_selection.json')
    parser.add_argument('--dataset-root', type=Path, default=PROJECT / 'dataset')
    parser.add_argument('--out-root', type=Path, default=PROJECT / 'train/space_comparison')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'xpu', 'auto'], default='auto')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-steps-per-epoch', type=int)
    parser.add_argument('--val-frames-per-episode', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    selected = json.loads(args.selection.read_text())['best']
    chunk = int(selected['chunk_size'])
    lr, kl = float(selected['learning_rate']), float(selected['kl_weight'])
    assert chunk >= 25 and np.isfinite([lr, kl]).all() and lr > 0 and kl >= 0
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    frame_count = None
    split_counts = None
    for name, dataset_name, width in [
        ('workspace', 'cabinet_opening_three_positions_paired', 10),
        ('joint', 'cabinet_opening_three_positions_paired_joint', 7),
    ]:
        dataset = args.dataset_root / dataset_name
        info = json.loads((dataset / 'meta/info.json').read_text())
        assert info['total_episodes'] == 9
        if frame_count is None:
            frame_count = info['total_frames']
        else:
            assert info['total_frames'] == frame_count, 'The two datasets must contain the same frames'
        assert info['features']['action']['shape'] == [width]
        command = [sys.executable, str(PROJECT / 'src/train.py'), '--dataset', str(dataset.resolve()),
            '--out-root', str(args.out_root.resolve()), '--run-name', name,
            '--epochs', str(args.epochs), '--batch-size', str(args.batch_size),
            '--chunk-size', str(chunk), '--lr', str(lr), '--kl-weight', str(kl),
            '--val-episodes', '2,5,8', '--eval-horizon', '25', '--seed', '0',
            '--device', args.device, '--num-workers', str(args.num_workers)]
        for option in ['max_steps_per_epoch', 'val_frames_per_episode']:
            if getattr(args, option) is not None:
                command += ['--' + option.replace('_', '-'), str(getattr(args, option))]
        print('RUN ' + ' '.join(command), flush=True)
        if args.dry_run:
            continue
        result_path = args.out_root / name / 'result.json'
        if not result_path.exists():
            log_path = args.out_root / f'{name}.log'
            with log_path.open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8', errors='replace',
                    env={**os.environ, 'MPLBACKEND': 'Agg', 'PYTHONUNBUFFERED': '1'})
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if 'epoch' in line or 'device=' in line or 'Train Done' in line or 'Error' in line:
                        print(line.rstrip(), flush=True)
                if process.wait():
                    print(log_path.read_text(encoding='utf-8')[-16000:], flush=True)
                    raise RuntimeError(f'{name} failed; log: {log_path}')
        result = json.loads(result_path.read_text())
        for key, expected in {'epochs': args.epochs, 'batch_size': args.batch_size, 'chunk_size': chunk,
            'lr': lr, 'kl_weight': kl, 'seed': 0, 'eval_horizon': 25,
            'max_steps_per_epoch': args.max_steps_per_epoch,
            'val_frames_per_episode': args.val_frames_per_episode}.items():
            assert result[key] == expected, f'Existing {name} result has different {key}'
        assert result['train_episodes'] == [0, 1, 3, 4, 6, 7]
        assert result['val_episodes'] == [2, 5, 8] and not result['no_augment']
        counts = (result['train_frames'], result['val_frames'])
        if split_counts is None:
            split_counts = counts
        else:
            assert counts == split_counts, 'The two runs must use the same train/validation split'
        if args.val_frames_per_episode is None:
            assert sum(counts) == frame_count
        rows.append({'space': name, 'dimension': width,
            'epochs': result['epochs'], 'optimizer_updates': result['optimizer_updates'],
            'best_val_l1': result['best_validation']['val_l1_loss'],
            'best_epoch': result['best_validation']['epoch'],
            'final_train_l1': result['final_losses']['l1_loss'],
            'final_total_loss': result['final_losses']['total'], 'seconds': result['elapsed_seconds']})
    if not rows:
        return
    summary = {'shared_hyperparameters': {'chunk_size': chunk, 'learning_rate': lr, 'kl_weight': kl,
                'batch_size': args.batch_size, 'epochs': args.epochs, 'seed': 0, 'eval_horizon': 25},
               'data': {'episodes': 9, 'frames': frame_count,
                        'train_frames': split_counts[0], 'val_frames': split_counts[1]},
               'runs': rows,
               'interpretation': 'Normalized L1 has different action coordinates and dimensions (10 vs 7). '
                   'Compare early convergence and representation-dependent errors; do not infer task-success superiority.'}
    (args.out_root / 'space_comparison.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for row in rows:
        with np.load(args.out_root / row['space'] / 'epoch_latest/loss_log.npz') as log:
            for axis, key in zip(axes, ['l1_loss', 'val_l1_loss', 'kld_loss']):
                axis.plot(log[key][:, 0], log[key][:, 1], marker='o', label=row['space'])
    for axis, title in zip(axes, ['Training L1', 'Validation L1 (first 25 steps)', 'KL divergence']):
        axis.set(title=title, xlabel='Epoch', ylabel='Loss')
        axis.grid(alpha=.3)
        axis.legend()
    fig.suptitle('Same demonstrations and settings; different state/action representations')
    fig.tight_layout()
    fig.savefig(args.out_root / 'space_comparison.png', dpi=160)
    plt.close(fig)
    print('SPACE_COMPARISON_COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
