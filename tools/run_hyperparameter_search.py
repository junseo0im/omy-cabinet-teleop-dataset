"""동일한 데이터 분할과 학습 예산으로 네 ACT 설정을 비교한다."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
CASES = [('A_chunk100_lr1e-4', 100, 1e-4, 10),
         ('B_chunk50_lr1e-4', 50, 1e-4, 10),
         ('C_chunk50_lr1e-5', 50, 1e-5, 10),
         ('D_chunk50_lr1e-4_kl100', 50, 1e-4, 100)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT / 'dataset/cabinet_opening_three_positions')
    parser.add_argument('--out-root', type=Path, default=PROJECT / 'train/hyperparameter_search')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps-per-epoch', type=int, default=0, help='0: use every training batch')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--val-frames-per-episode', type=int, default=0, help='0: use every validation frame')
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'xpu'], default='auto')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, chunk, lr, kl in CASES:
        command = [sys.executable, str(PROJECT / 'src/train.py'), '--dataset', str(args.dataset.resolve()),
                   '--out-root', str(args.out_root.resolve()), '--run-name', name,
                   '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--chunk-size', str(chunk), '--lr', str(lr),
                   '--kl-weight', str(kl), '--val-episodes', '2,5,8', '--eval-horizon', '25',
                   '--val-every', '1', '--num-workers', str(args.num_workers),
                   '--torch-threads', str(args.torch_threads), '--device', args.device, '--seed', '0']
        if args.max_steps_per_epoch:
            command += ['--max-steps-per-epoch', str(args.max_steps_per_epoch)]
        if args.val_frames_per_episode:
            command += ['--val-frames-per-episode', str(args.val_frames_per_episode)]
        if args.no_augment:
            command.append('--no-augment')
        print('RUN ' + ' '.join(command), flush=True)
        if args.dry_run:
            continue
        result_path = args.out_root / name / 'result.json'
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            for key in ['epochs', 'max_steps_per_epoch', 'batch_size', 'val_frames_per_episode']:
                expected = getattr(args, key) or None if key in ['max_steps_per_epoch', 'val_frames_per_episode'] else getattr(args, key)
                assert previous[key] == expected, f'Existing result has different {key}'
            assert previous['no_augment'] == args.no_augment
            for key, expected in {'chunk_size': chunk, 'lr': lr, 'kl_weight': kl,
                                  'eval_horizon': 25, 'seed': 0, 'val_episodes': [2, 5, 8]}.items():
                assert previous[key] == expected, f'Existing result has different {key}'
        else:
            with (args.out_root / f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if 'epoch' in line or 'device=' in line or 'Train Done' in line or 'Error' in line:
                        print(line.rstrip(), flush=True)
                if process.wait():
                    raise RuntimeError(f'{name} failed; inspect its log')
        result = json.loads(result_path.read_text())
        rows.append({'run': name, 'chunk_size': chunk, 'learning_rate': lr,
                     'kl_weight': kl, 'batch_size': args.batch_size,
                     'optimizer_updates': result['optimizer_updates'],
                     'best_epoch': result['best_validation']['epoch'],
                     'best_val_l1': result['best_validation']['val_l1_loss'],
                     'final_train_l1': result['final_losses']['l1_loss'],
                     'final_total_loss': result['final_losses']['total'],
                     'seconds': result['elapsed_seconds']})
    if not rows:
        return
    with (args.out_root / 'comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = min(rows, key=lambda row: row['best_val_l1'])
    (args.out_root / 'comparison.json').write_text(json.dumps({'runs': rows, 'best': best}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for row in rows:
        with np.load(args.out_root / row['run'] / 'epoch_latest/loss_log.npz') as log:
            values = log['val_l1_loss']
            axes[0].plot(values[:, 0], values[:, 1], marker='o', label=row['run'])
    epoch_label = f'Epoch (up to {args.max_steps_per_epoch} updates)' if args.max_steps_per_epoch else 'Epoch'
    axes[0].set(xlabel=epoch_label, ylabel='Validation L1 (normalized)',
                title='Same validation frames, first 25 action steps')
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=.3)
    axes[1].bar([row['run'][0] for row in rows], [row['best_val_l1'] for row in rows], color='#285878')
    axes[1].set(xlabel='Configuration', ylabel='Minimum validation L1', title='Best checkpoint per configuration')
    for index, row in enumerate(rows):
        axes[1].annotate(f'{row["best_val_l1"]:.4f}', (index, row['best_val_l1']),
                         xytext=(0, 4), textcoords='offset points', ha='center')
    fig.tight_layout()
    fig.savefig(args.out_root / 'comparison.png', dpi=160)
    plt.close(fig)
    with np.load(args.out_root / best['run'] / 'epoch_latest/loss_log.npz') as log:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        for axis, keys, title in zip(axes, [('total',), ('l1_loss', 'val_l1_loss'), ('kld_loss',)],
                                    ['Total loss', 'L1 losses', 'KL divergence']):
            for key in keys:
                axis.plot(log[key][:, 0], log[key][:, 1], marker='o', label=key)
            axis.set(title=title, xlabel='Epoch', ylabel='Loss')
            axis.grid(alpha=.3)
            axis.legend(fontsize=8)
        fig.suptitle(best['run'])
        fig.tight_layout()
        fig.savefig(args.out_root / 'best_loss_components.png', dpi=160)
        plt.close(fig)
    print('COMPARISON_COMPLETE ' + json.dumps({'best': best, 'runs': rows}), flush=True)


if __name__ == '__main__':
    main()
