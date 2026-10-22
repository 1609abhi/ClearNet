"""
ClearNet Test / Evaluation Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent OpenMP conflicts on some systems
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
import pandas as pd

# Local imports
from configs.config import *
from models.image_model import Restormer
from utils.helpers import count_trainable_parameters
from utils.metrics import AverageMeter, compute_psnr_ssim
from data.datasets import EvalDataset


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ClearNet Testing')

# Device
parser.add_argument('--gpu', type=str, default="0",
                    help='GPU id(s) to use')
parser.add_argument('--device', type=str, default=DEVICE,
                    help='Device (cuda or cpu)')

# Model
parser.add_argument('--pretrained', type=str, required=True,
                    help='Path to pretrained checkpoint')
parser.add_argument('--width', type=int, default=WIDTH,
                    help='Model base channel width')

# Task configuration
parser.add_argument('--de_type', nargs='+', default=ALL_TASKS,
                    help='Degradation types to test on')

# Output
parser.add_argument('--output_path', type=str, default='evals/',
                    help='Output directory for evaluation results')

args = parser.parse_args()


# ---------------------------------------------------------------------------
# Model evaluator
# ---------------------------------------------------------------------------
class ModelEvaluator:
    def __init__(self, checkpoint_path, device):
        print("Initializing model...")

        self.device = device
        self.image_model = Restormer(
            inp_channels=IMG_CHANNEL,
            out_channels=IMG_CHANNEL,
            dim=args.width,
            num_blocks=NUM_BLKS,
            num_refinement_blocks=REFINE_NUM_BLK,
            heads=HEADS,
            num_tasks=len(args.de_type)
        ).to(device)

        print(f"Trainable parameters: {count_trainable_parameters(self.image_model):,}")

        state_dict = torch.load(checkpoint_path, map_location=device)
        self.image_model.load_state_dict(state_dict, strict=True)
        self.image_model.eval()
        print("Model initialized!\n")

    def evaluate_dataset(self, input_dir, target_dir, task_name):
        dataset = EvalDataset(input_dir, target_dir)
        loader = DataLoader(dataset, batch_size=1, pin_memory=True,
                            shuffle=False, num_workers=0)

        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        with torch.no_grad():
            for inputs, targets, filenames in tqdm(loader, desc=f"{task_name}", unit="img"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                idx_task = args.de_type.index(task_name)
                logits = F.one_hot(torch.tensor(idx_task),
                                   num_classes=len(args.de_type)).float().to(self.device)
                logits = logits.unsqueeze(0).expand(inputs.size(0), -1).contiguous()

                outputs = self.image_model(inputs, logits)[-1]

                temp_psnr, temp_ssim, N = compute_psnr_ssim(outputs, targets)
                psnr_meter.update(temp_psnr, N)
                ssim_meter.update(temp_ssim, N)

        return psnr_meter.avg, ssim_meter.avg


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_report(checkpoint_path, dataset_paths, device, output_file="evaluation_report.csv"):
    all_results = []

    evaluator = ModelEvaluator(checkpoint_path, device)

    for task_name, datasets in dataset_paths.items():
        if task_name not in args.de_type:
            continue

        for dataset in datasets:
            target_dir, input_dir, *extra = dataset
            noise_level = extra[0] if extra else ""

            print(f"Evaluating {task_name} - {noise_level}")
            psnr, ssim = evaluator.evaluate_dataset(input_dir, target_dir, task_name)

            all_results.append({
                'Task': task_name,
                'Dataset': os.path.basename(input_dir),
                'Noise Level': noise_level,
                'PSNR': psnr,
                'SSIM': ssim
            })

    df = pd.DataFrame(all_results)
    print(tabulate(df, headers='keys', tablefmt='grid'))

    final_df = df.groupby('Task', as_index=False)[['PSNR', 'SSIM']].mean()
    final_df.to_csv(output_file, index=False)

    print("\n===== Final Evaluation Summary =====\n")
    print(tabulate(final_df, headers='keys', tablefmt='grid'))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    os.makedirs(args.output_path, exist_ok=True)

    output_file = os.path.join(args.output_path, "evaluation_report.csv")

    generate_report(
        args.pretrained,
        dataset_paths=DATASET_PATHS,
        device=device,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
