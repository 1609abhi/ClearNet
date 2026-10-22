"""
Metrics and evaluation utilities for ClearNet.
Canonical home for AverageMeter, compute_psnr_ssim, and evaluate_model_train.
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tabulate import tabulate

from configs.config import ALL_TASKS


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_psnr_ssim(recovered, clean):
    """Compute PSNR and SSIM between recovered and clean tensors."""
    assert recovered.shape == clean.shape
    recovered = np.clip(recovered.detach().cpu().numpy(), 0, 1)
    clean     = np.clip(clean.detach().cpu().numpy(),     0, 1)
    recovered = recovered.transpose(0, 2, 3, 1)
    clean     = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0
    for i in range(recovered.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recovered[i], data_range=1)
        ssim += structural_similarity(clean[i],   recovered[i],
                                      data_range=1, channel_axis=2)
    return psnr / recovered.shape[0], ssim / recovered.shape[0], recovered.shape[0]


def evaluate_model_train(image_model, train_eval_loaders, device, de_type):
    """
    Evaluate the model on train-eval loaders.
    Preprocessing identical to standalone eval script (crop_img base=16).
    """
    image_model.eval()
    table_data   = []
    psnr_results = []
    ssim_results = []
    task_names   = []

    with torch.no_grad():
        for task_name, loader in train_eval_loaders.items():
            if task_name not in de_type:
                continue

            psnr_meter = AverageMeter()
            ssim_meter = AverageMeter()

            idx_task     = de_type.index(task_name)
            logits       = F.one_hot(torch.tensor(idx_task),
                                     num_classes=len(de_type)).float().to(device)

            for inputs, targets in loader:
                inputs       = inputs.to(device)
                targets      = targets.to(device)
                logits_batch = logits.unsqueeze(0).expand(inputs.size(0), -1).contiguous()
                outputs      = image_model(inputs, logits_batch)

                b_psnr, b_ssim, N = compute_psnr_ssim(outputs, targets)
                psnr_meter.update(b_psnr, N)
                ssim_meter.update(b_ssim, N)

            task_names.append(task_name)
            table_data.append([f"{task_name} (train)",
                                f"{psnr_meter.avg:.2f} dB",
                                f"{ssim_meter.avg:.4f}"])
            psnr_results.append(psnr_meter.avg)
            ssim_results.append(ssim_meter.avg)

    print("\n" + tabulate(table_data, headers=["Task", "PSNR", "SSIM"], tablefmt="grid"))
    return (
        {t: psnr for t, psnr in zip(task_names, psnr_results)},
        {t: ssim for t, ssim in zip(task_names, ssim_results)}
    )
