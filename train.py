"""
ClearNet Training Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent OpenMP conflicts on some systems
import json
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from configs.config import *
from models.image_model import Restormer, build_gt_pyramid
from data.datasets import MultiTaskLoader, create_train_eval_loader
from data.transforms import PairedTransform
from utils.helpers import seed_everything, count_trainable_parameters, create_directories
from utils.metrics import evaluate_model_train
from utils.plotting import save_training_plots


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ClearNet Training')

# Device
parser.add_argument('--gpu', type=str, default="0",
                    help='GPU id(s) to use')
parser.add_argument('--device', type=str, default=DEVICE,
                    help='Device (cuda or cpu)')

# Training hyperparameters
parser.add_argument('--num_iters', type=int, default=NUM_ITERS,
                    help='Total number of training iterations')
parser.add_argument('--warmup_iters', type=int, default=WARMUP_ITERS,
                    help='Number of warmup iterations')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                    help='Batch size per GPU')
parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                    help='Learning rate')
parser.add_argument('--patch_size', type=int, default=IMAGE_SIZE[0],
                    help='Training patch size')
parser.add_argument('--fourier_weight', type=float, default=FOURIER_LOSS_WEIGHT,
                    help='Weight for Fourier loss')
parser.add_argument('--seed', type=int, default=SEED,
                    help='Random seed')

# Logging
parser.add_argument('--iter_jump', type=int, default=ITER_JUMP,
                    help='Logging frequency (iterations)')
parser.add_argument('--checkpoint_iter', type=int, default=CHECKPOINT_ITER,
                    help='Checkpoint saving frequency (iterations)')

# Data paths
parser.add_argument('--base_dir', type=str, default=BASE_DIR,
                    help='Root directory containing training datasets')
parser.add_argument('--data_file_dir', type=str, default='data_dir/',
                    help='Directory containing .txt file lists for datasets')

# Task configuration
parser.add_argument('--de_type', nargs='+', default=ALL_TASKS,
                    help='Degradation types to train on')

# Output paths
parser.add_argument('--save_dir', type=str, default=CHECKPOINT_DIR,
                    help='Directory to save checkpoints and logs')

# Resume
parser.add_argument('--pretrained', type=str, default='',
                    help='Path to pretrained checkpoint for resuming training')

# Model
parser.add_argument('--width', type=int, default=WIDTH,
                    help='Model base channel width')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def pyramid_l1_loss(outputs, gt_list, criterion):
    total = sum(criterion(pred, gt) for pred, gt in zip(outputs, gt_list))
    return total / len(outputs)


def fourier_loss(pred, target, criterion):
    pred_fft   = torch.fft.fft2(pred,   dim=(-2, -1))
    target_fft = torch.fft.fft2(target, dim=(-2, -1))
    pred_fft   = torch.stack((pred_fft.real,   pred_fft.imag),   -1)
    target_fft = torch.stack((target_fft.real, target_fft.imag), -1)
    return criterion(pred_fft, target_fft)



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_joint(image_model, image_loader, train_eval_loaders, device,
                checkpoint_dir):

    print("\n" + "=" * 80)
    print("ClearNet Training")
    print("=" * 80 + "\n")

    criterion_image = nn.L1Loss()

    optimizer = torch.optim.AdamW(
        [{'params': image_model.parameters()}],
        lr=args.lr
    )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_iters
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters - args.warmup_iters, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_iters]
    )

    training_logs = {
        'iteration':    [],
        'loss_pyramid': [],
        'loss_fourier': [],
        'loss_total':   [],
        'learning_rate': []
    }

    start_iter = 1

    # ----------------------------------------------------------------
    # Resume from checkpoint
    # ----------------------------------------------------------------
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"Resuming from checkpoint: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)

        image_model.load_state_dict(checkpoint['image_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint['iteration'] + 1

        if 'training_logs' in checkpoint:
            training_logs = checkpoint['training_logs']

        print(f"Resumed from iteration {checkpoint['iteration']}")
        print(f"Continuing from iteration {start_iter}\n")

    image_model.train()

    for itern in range(start_iter, args.num_iters + 1):

        # ----------------------------------------------------------------
        # Data + task setup
        # ----------------------------------------------------------------
        task     = random.choice(args.de_type)
        idx_task = args.de_type.index(task)

        inputs, targets = image_loader.get_batch(task)
        inputs  = inputs.to(device)
        targets = targets.to(device)

        logits = F.one_hot(torch.tensor(idx_task),
                           num_classes=len(args.de_type)).float().to(device)
        logits = logits.unsqueeze(0).expand(inputs.size(0), -1).contiguous()

        # Forward — returns list of outputs (aux + final)
        outputs = image_model(inputs, logits)

        # Build matching GT tensors at each pyramid level
        gt_list = build_gt_pyramid(targets, outputs)

        # Losses
        loss_pyramid = pyramid_l1_loss(outputs, gt_list, criterion_image)
        loss_fft     = fourier_loss(outputs[-1], targets, criterion_image)
        loss_total   = loss_pyramid + args.fourier_weight * loss_fft

        # Backward
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        # ----------------------------------------------------------------
        # Logging
        # ----------------------------------------------------------------
        current_lr = scheduler.get_last_lr()[0]
        training_logs['iteration'].append(itern)
        training_logs['loss_pyramid'].append(loss_pyramid.item())
        training_logs['loss_fourier'].append(loss_fft.item())
        training_logs['loss_total'].append(loss_total.item())
        training_logs['learning_rate'].append(current_lr)

        # ----------------------------------------------------------------
        # Periodic print + log save
        # ----------------------------------------------------------------
        if itern % args.iter_jump == 0:
            print(f"\nIteration {itern}:")
            print(f"  Pyramid L1 Loss : {loss_pyramid.item():.4f}  "
                  f"(avg over {len(outputs)} levels)")
            print(f"  Fourier Loss    : {loss_fft.item():.4f}  "
                  f"(final output only)")
            print(f"  Total Loss      : {loss_total.item():.4f}")
            print(f"  Learning Rate   : {current_lr:.2e}")

            logs_path = os.path.join(checkpoint_dir, "training_logs.json")
            with open(logs_path, 'w') as f:
                json.dump({
                    'training_logs': training_logs,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)

        # ----------------------------------------------------------------
        # Checkpoint + evaluation
        # ----------------------------------------------------------------
        if itern % args.checkpoint_iter == 0 or itern == args.num_iters:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"joint_iter_{itern}.pth"
            )
            torch.save({
                'image_model':   image_model.state_dict(),
                'optimizer':     optimizer.state_dict(),
                'scheduler':     scheduler.state_dict(),
                'iteration':     itern,
                'training_logs': training_logs,
            }, checkpoint_path)
            print(f"Saved checkpoint at iteration {itern}")

            print("*" * 100)
            print(f"Iteration {itern}")
            print("*" * 100)

            eval_model = _FinalOutputWrapper(image_model)

            print("Train eval")
            train_psnr, train_ssim = evaluate_model_train(
                eval_model, train_eval_loaders, device, args.de_type
            )



            eval_results = {
                'iteration': itern,
                'timestamp': datetime.now().isoformat(),
                'num_pyramid_levels': len(outputs),
                'train_eval': {
                    'psnr': {t: float(p) for t, p in train_psnr.items()},
                    'ssim': {t: float(s) for t, s in train_ssim.items()},
                    'avg_psnr': float(sum(train_psnr.values()) / len(train_psnr)),
                    'avg_ssim': float(sum(train_ssim.values()) / len(train_ssim))
                }
            }

            eval_path = os.path.join(checkpoint_dir, f"eval_iter_{itern}.json")
            with open(eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            print(f"Saved evaluation results to {eval_path}")

            image_model.train()

    # ----------------------------------------------------------------
    # End of training — final plots
    # ----------------------------------------------------------------
    print("\nTraining Complete! Saving final plots...")

    final_logs_path = os.path.join(checkpoint_dir, "final_training_logs.json")
    with open(final_logs_path, 'w') as f:
        json.dump({
            'training_logs':   training_logs,
            'final_timestamp': datetime.now().isoformat(),
            'total_iterations': args.num_iters
        }, f, indent=2)

    save_training_plots(training_logs, checkpoint_dir)

    return training_logs


class _FinalOutputWrapper(nn.Module):
    """Wraps Restormer so model(inp, logits) returns outputs[-1] only."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp, logits):
        return self.model(inp, logits)[-1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    seed_everything(args.seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    create_directories(args.save_dir)

    print("Initializing data loaders...")
    transform    = PairedTransform(size=(args.patch_size, args.patch_size))
    image_loader = MultiTaskLoader(
        batch_size=args.batch_size,
        transform=transform,
        seed=args.seed,
        base_dir=args.base_dir,
        data_file_dir=args.data_file_dir,
    )

    print("Creating train eval loaders...")
    train_eval_loaders = create_train_eval_loader(
        base_dir=args.base_dir,
        batch_size=1,
        max_images_per_task=100
    )

    image_model = Restormer(
        inp_channels=IMG_CHANNEL,
        out_channels=IMG_CHANNEL,
        dim=args.width,
        num_blocks=NUM_BLKS,
        num_refinement_blocks=REFINE_NUM_BLK,
        heads=HEADS,
        num_tasks=len(args.de_type)
    ).to(device)

    print(f"Image model trainable parameters: "
          f"{count_trainable_parameters(image_model):,}")
    print("Model initialized!\n")

    training_logs = train_joint(
        image_model,
        image_loader,
        train_eval_loaders,
        device,
        args.save_dir,
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()