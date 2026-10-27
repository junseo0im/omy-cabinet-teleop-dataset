"""서랍 열기 시연으로 ACT 학습 및 loss 기록."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

# Colab 커널의 inline 백엔드가 학습 가상환경에 전달되지 않도록 한다.
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2
from tqdm import tqdm
import yaml

from lerobot.configs.types import FeatureType
from lerobot.constants import ACTION, OBS_IMAGES
from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy

PROJECT = Path(__file__).resolve().parents[1]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=PROJECT / 'dataset/cabinet_opening_three_positions')
    parser.add_argument('--out-root', type=Path, default=PROJECT / 'train')
    parser.add_argument('--run-name', default='workspace_act')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--checkpoint-epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-backbone', type=float, default=None)
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--n-action-steps', type=int, default=None)
    parser.add_argument('--kl-weight', type=float, default=10.)
    parser.add_argument('--dim-model', type=int, default=512)
    parser.add_argument('--dim-feedforward', type=int, default=3200)
    parser.add_argument('--val-ratio', type=float, default=.1)
    parser.add_argument('--val-episodes', default=None, help='comma-separated held-out episode indices')
    parser.add_argument('--val-every', type=int, default=1)
    parser.add_argument('--eval-horizon', type=int, default=None)
    parser.add_argument('--val-frames-per-episode', type=int, default=None)
    parser.add_argument('--max-steps-per-epoch', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--torch-threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'xpu'], default='auto')
    parser.add_argument('--no-augment', action='store_true')
    args = parser.parse_args()
    for key in ['epochs', 'checkpoint_epochs', 'batch_size', 'chunk_size', 'val_every', 'torch_threads']:
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    for key in ['eval_horizon', 'val_frames_per_episode', 'max_steps_per_epoch']:
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if not 0 < args.val_ratio < 1 or args.lr <= 0 or args.num_workers < 0:
        parser.error('Check val-ratio, lr and num-workers')
    if args.eval_horizon is not None and args.eval_horizon > args.chunk_size:
        parser.error('eval-horizon must not exceed chunk-size')
    return args


def get_device(requested):
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu')
        return torch.device('cpu')
    return torch.device(requested)


def save_loss_log_to_npz(loss_log, save_dir):
    np.savez(save_dir, **{key: np.asarray(values) for key, values in loss_log.items()})


def visualize_loss(loss_log, save_dir):
    plt.figure(figsize=(10, 6))
    for k, values in loss_log.items():
        if not values:
            continue
        steps, losses = zip(*values)
        plt.plot(steps, losses, label=k)
    plt.title('Loss over Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate(policy, loader, device, horizon):
    """공통 예측 구간에서 패딩을 제외한 정규화 L1을 계산한다."""
    policy.eval()
    absolute_error = 0.
    valid_elements = 0
    for batch in loader:
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        normalized = dict(policy.normalize_inputs(batch))
        normalized[OBS_IMAGES] = [normalized[key] for key in policy.config.image_features]
        normalized = policy.normalize_targets(normalized)
        predicted, _ = policy.model(normalized)
        target = normalized[ACTION][:, :horizon]
        mask = (~normalized['action_is_pad'][:, :horizon]).unsqueeze(-1).expand_as(target)
        error = F.l1_loss(predicted[:, :horizon], target, reduction='none')
        absolute_error += error.masked_select(mask).sum().item()
        valid_elements += mask.sum().item()
    policy.train()
    if valid_elements == 0:
        raise ValueError('No valid validation actions')
    return absolute_error / valid_elements


def main():
    args = get_args()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = get_device(args.device)
    output = args.out_root / args.run_name
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new --run-name; output already exists: {output}')
    meta = LeRobotDatasetMetadata('OMY', root=args.dataset.resolve())
    episode_ids = list(range(meta.total_episodes))
    if len(episode_ids) < 2:
        raise ValueError('At least two episodes are needed for train/validation split')
    if args.val_episodes:
        val_episodes = sorted({int(value) for value in args.val_episodes.split(',')})
    else:
        n_val = max(1, min(len(episode_ids) - 1, round(len(episode_ids) * args.val_ratio)))
        val_episodes = sorted(np.random.RandomState(args.seed).permutation(episode_ids)[:n_val].tolist())
    if not val_episodes or not set(val_episodes) < set(episode_ids):
        raise ValueError('Validation episodes must be a nonempty proper subset of episode indices')
    train_episodes = sorted(set(episode_ids) - set(val_episodes))
    # 검증 에피소드의 통계는 학습 시 정규화에 사용하지 않는다.
    train_stats = aggregate_stats([meta.episodes_stats[episode] for episode in train_episodes])
    features = dataset_to_policy_features(meta.features)
    cfg = ACTConfig(
        input_features={key: value for key, value in features.items() if value.type != FeatureType.ACTION},
        output_features={key: value for key, value in features.items() if value.type == FeatureType.ACTION},
        chunk_size=args.chunk_size, n_action_steps=args.n_action_steps or max(1, args.chunk_size // 2),
        kl_weight=args.kl_weight, dim_model=args.dim_model, dim_feedforward=args.dim_feedforward,
        optimizer_lr=args.lr, optimizer_lr_backbone=args.lr_backbone or args.lr,
        device=str(device), push_to_hub=False)
    policy = ACTPolicy(cfg, dataset_stats=train_stats).to(device).train()
    deltas = resolve_delta_timestamps(cfg, meta)
    augmentation = None if args.no_augment else v2.Compose([
        v2.RandomAffine(degrees=10, translate=(.05, .05)),
        v2.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.1),
        v2.GaussianBlur(kernel_size=3, sigma=(.1, 2.)),
    ])
    full_train = LeRobotDataset('OMY', root=args.dataset.resolve(), delta_timestamps=deltas,
                                image_transforms=augmentation)
    full_val = LeRobotDataset('OMY', root=args.dataset.resolve(), delta_timestamps=deltas)
    frame_episodes = np.asarray(full_train.hf_dataset['episode_index'])
    train_indices = np.flatnonzero(np.isin(frame_episodes, train_episodes)).tolist()
    val_indices = []
    for episode in val_episodes:
        indices = np.flatnonzero(frame_episodes == episode)
        if args.val_frames_per_episode and len(indices) > args.val_frames_per_episode:
            indices = indices[np.linspace(0, len(indices) - 1, args.val_frames_per_episode).round().astype(int)]
        val_indices.extend(indices.tolist())
    common = dict(num_workers=args.num_workers, pin_memory=device.type == 'cuda',
                  persistent_workers=args.num_workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_train, train_indices),
        batch_size=args.batch_size, shuffle=True, drop_last=False, generator=generator, **common)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(full_val, val_indices),
        batch_size=args.batch_size, shuffle=False, **common)
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=args.lr)
    horizon = args.eval_horizon or args.chunk_size
    output.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(device=str(device), train_episodes=train_episodes, val_episodes=val_episodes,
        train_frames=len(train_indices), val_frames=len(val_indices), eval_horizon=horizon,
        normalization='training episodes only',
        partial_epochs=args.max_steps_per_epoch is not None and args.max_steps_per_epoch < len(train_loader))
    (output / 'train_args.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    loss_log = defaultdict(list)
    best = {'val_l1_loss': float('inf'), 'epoch': None}
    updates = 0

    def checkpoint(name):
        directory = output / name
        policy.save_pretrained(directory)
        config = json.loads((directory / 'config.json').read_text())
        (directory / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        save_loss_log_to_npz(loss_log, directory / 'loss_log.npz')
        visualize_loss(loss_log, directory / 'loss_log.png')
        (directory / 'train_args.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
        return directory

    print(f'device={device}; train={train_episodes}; validation={val_episodes}; '
          f'train_frames={len(train_indices)}; validation_frames={len(val_indices)}', flush=True)
    start = time.perf_counter()
    for epoch in tqdm(range(1, args.epochs + 1), mininterval=10):
        epoch_losses = defaultdict(float)
        samples = 0
        for step, batch in enumerate(train_loader):
            if args.max_steps_per_epoch is not None and step >= args.max_steps_per_epoch:
                break
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, details = policy.forward(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            optimizer.step()
            count = len(batch[ACTION])
            samples += count
            epoch_losses['total'] += loss.item() * count
            for key, value in details.items():
                epoch_losses[key] += float(value) * count
            updates += 1
        for key, value in epoch_losses.items():
            loss_log[key].append((epoch, value / samples))
        if epoch % args.val_every == 0 or epoch == args.epochs:
            value = evaluate(policy, val_loader, device, horizon)
            loss_log['val_l1_loss'].append((epoch, value))
            if value < best['val_l1_loss']:
                best = {'val_l1_loss': value, 'epoch': epoch}
                checkpoint('epoch_best')
        if epoch % args.checkpoint_epochs == 0:
            checkpoint(f'epoch_{epoch}')
        print(json.dumps({'epoch': epoch, 'optimizer_updates': updates,
                          **{key: values[-1][1] for key, values in loss_log.items()}}), flush=True)
    final = checkpoint('epoch_latest')
    result = {**settings, 'optimizer_updates': updates, 'elapsed_seconds': time.perf_counter() - start,
              'best_validation': best, 'final_losses': {key: values[-1][1] for key, values in loss_log.items()}}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'Train Done! Saved {final}', flush=True)


if __name__ == '__main__':
    main()
