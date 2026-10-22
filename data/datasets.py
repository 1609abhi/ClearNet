"""
Dataset classes for ClearNet training.
Uses .txt file lists from data_dir/ to index training images,
following the DFPIR convention.
"""

import os
import random
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision import transforms

from configs.config import TASK_FOLDERS, ALL_TASKS
from utils.helpers import crop_img, seed_worker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SIGMAS = [15, 25, 50]
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}


def _add_gaussian_noise(clean_np, sigma):
    """Add gaussian noise on uint8 numpy array [0,255], return noisy uint8."""
    noise = np.random.randn(*clean_np.shape)
    noisy = np.clip(clean_np + noise * sigma, 0, 255).astype(np.uint8)
    return noisy


def _read_file_list(txt_path):
    """Read a .txt file and return list of stripped, non-empty, non-comment lines."""
    entries = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                entries.append(line)
    return entries


# ---------------------------------------------------------------------------
# Training datasets  (txt-file-list driven)
# ---------------------------------------------------------------------------
class TaskDataset(Dataset):
    """
    Standard inputs/targets dataset for derain, deblur, lol.
    File list comes from a .txt file in data_dir/.
    """
    def __init__(self, input_dir, target_dir, file_list_path=None, transform=None):
        self.input_dir  = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.transform  = transform

        if file_list_path and os.path.exists(file_list_path):
            # Load filenames from .txt file list
            self.filenames = _read_file_list(file_list_path)
            print(f"  Loaded {len(self.filenames)} filenames from {file_list_path}")
        else:
            # Fallback: scan directory
            self.filenames = sorted([
                f.name for f in self.input_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            print(f"  Scanned {len(self.filenames)} filenames from {self.input_dir}")

        if len(self.filenames) == 0:
            raise RuntimeError(f"No images found for {self.input_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename   = self.filenames[idx]
        input_img  = Image.open(self.input_dir  / filename).convert('RGB')
        target_img = Image.open(self.target_dir / filename).convert('RGB')

        if self.transform:
            return self.transform(input_img, target_img)
        return TF.to_tensor(np.array(input_img)), TF.to_tensor(np.array(target_img))


class TaskDatasetNoise(Dataset):
    """
    Denoising dataset — only clean images in targets/.
    Noisy input generated on-the-fly at a randomly chosen sigma (15, 25, 50).
    File list comes from a .txt file in data_dir/.
    """
    def __init__(self, target_dir, file_list_path=None, transform=None):
        self.target_dir = Path(target_dir)
        self.transform  = transform

        if file_list_path and os.path.exists(file_list_path):
            self.filenames = _read_file_list(file_list_path)
            print(f"  Loaded {len(self.filenames)} filenames from {file_list_path}")
        else:
            self.filenames = sorted([
                f.name for f in self.target_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS
            ])
            print(f"  Scanned {len(self.filenames)} filenames from {self.target_dir}")

        if len(self.filenames) == 0:
            raise RuntimeError(f"No clean images found in {self.target_dir}")

        print(f"  Denoising dataset: {len(self.filenames)} clean images, "
              f"noise added on-the-fly at sigma = {SIGMAS}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename  = self.filenames[idx]
        clean_np  = np.array(Image.open(self.target_dir / filename).convert('RGB'))

        sigma     = random.choice(SIGMAS)
        noisy_np  = _add_gaussian_noise(clean_np, sigma)

        noisy_pil = Image.fromarray(noisy_np)
        clean_pil = Image.fromarray(clean_np)

        if self.transform:
            return self.transform(noisy_pil, clean_pil)
        return TF.to_tensor(noisy_np), TF.to_tensor(clean_np)


# ---------------------------------------------------------------------------
# MultiTaskLoader  (used during training)
# ---------------------------------------------------------------------------

# Map task names to data_dir .txt file paths (relative to data_file_dir)
TASK_FILE_LISTS = {
    'noise': 'noisy/denoise.txt',
    'rain':  'rainy/rainTrain.txt',
    'haze':  'hazy/hazy_outside.txt',
    'blur':  'gopro/train_gopro.txt',
    'lol':   'lol/train_lol.txt',
}


class MultiTaskLoader:
    """
    Provides infinite batches for each task via get_batch(task_name).

    Expected base_dir structure:
        denoising_train/targets/
        dehazing_train/inputs/ & targets/
        deraining_train/inputs/ & targets/
        lol_train/inputs/ & targets/
        deblurring_train/inputs/ & targets/
    """
    def __init__(self, base_dir, batch_size=16, transform=None, seed=42,
                 data_file_dir='data_dir/'):
        base_path    = Path(base_dir)
        task_folders = TASK_FOLDERS

        self.tasks = {}

        for task_name, folder_name in task_folders.items():
            task_path = base_path / folder_name

            if not task_path.exists():
                print(f"Warning: {task_name} not found at {task_path}, skipping...")
                continue

            # Resolve .txt file list path
            txt_rel = TASK_FILE_LISTS.get(task_name, '')
            txt_path = os.path.join(data_file_dir, txt_rel) if txt_rel else None

            try:
                if task_name == 'noise':
                    dataset = TaskDatasetNoise(
                        task_path / 'targets',
                        file_list_path=txt_path,
                        transform=transform
                    )
                else:
                    dataset = TaskDataset(
                        task_path / 'inputs',
                        task_path / 'targets',
                        file_list_path=txt_path,
                        transform=transform
                    )

                self.tasks[task_name] = dataset
                print(f"Loaded {task_name}: {len(dataset)} samples")

            except Exception as e:
                print(f"Error loading {task_name}: {e}")
                continue

        if len(self.tasks) == 0:
            raise RuntimeError(f"No valid datasets found in {base_path}")

        self.loaders = {
            task_name: DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
                generator=torch.Generator().manual_seed(seed + i),
                worker_init_fn=seed_worker
            )
            for i, (task_name, dataset) in enumerate(self.tasks.items())
        }

        self.iterators = {
            task: iter(loader) for task, loader in self.loaders.items()
        }

        print(f"\nMultiTaskLoader initialized with {len(self.tasks)} tasks: "
              f"{list(self.tasks.keys())}")

    def get_batch(self, key):
        if key not in self.iterators:
            raise KeyError(f"Task '{key}' not found. "
                           f"Available: {list(self.iterators.keys())}")
        try:
            return next(self.iterators[key])
        except StopIteration:
            self.iterators[key] = iter(self.loaders[key])
            return next(self.iterators[key])

    def get_available_tasks(self):
        return list(self.tasks.keys())

    def __getitem__(self, key):
        return self.loaders[key]


# ---------------------------------------------------------------------------
# Train-eval datasets  (crop_img base=16, no augmentation, 100 images)
# ---------------------------------------------------------------------------
class TrainEvalDataset(Dataset):
    """
    Mirrors eval script preprocessing exactly:
      PIL load -> np.array -> crop_img(base=16) -> ToTensor
    """
    def __init__(self, input_dir, target_dir, max_images=100):
        self.input_dir  = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.toTensor   = transforms.ToTensor()

        all_files = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        rng = random.Random(42)
        self.filenames = sorted(rng.sample(all_files, min(max_images, len(all_files))))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename   = self.filenames[idx]
        input_img  = np.array(Image.open(self.input_dir  / filename).convert('RGB'))
        target_img = np.array(Image.open(self.target_dir / filename).convert('RGB'))
        input_img  = crop_img(input_img,  base=16)
        target_img = crop_img(target_img, base=16)
        return self.toTensor(input_img), self.toTensor(target_img)


class TrainEvalDatasetNoise(Dataset):
    """
    Train-eval for denoising — sigma_25 only, crop_img(base=16).
    Noise added on-the-fly on uint8 BEFORE ToTensor.
    """
    def __init__(self, target_dir, sigma=25, max_images=100):
        self.target_dir = Path(target_dir)
        self.sigma      = sigma
        self.toTensor   = transforms.ToTensor()

        all_files = sorted([
            f.name for f in self.target_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        rng = random.Random(42)
        self.filenames = sorted(rng.sample(all_files, min(max_images, len(all_files))))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename  = self.filenames[idx]
        clean_np  = np.array(Image.open(self.target_dir / filename).convert('RGB'))
        clean_np  = crop_img(clean_np, base=16)
        noisy_np  = _add_gaussian_noise(clean_np, self.sigma)
        return self.toTensor(noisy_np), self.toTensor(clean_np)


# ---------------------------------------------------------------------------
# Train-eval loader factory
# ---------------------------------------------------------------------------
def create_train_eval_loader(base_dir, batch_size=1, max_images_per_task=100):
    """
    Returns flat dict: task_name -> DataLoader
    Noise uses sigma_25 only.
    All datasets use crop_img(base=16), no augmentation.
    """
    base_path = Path(base_dir)
    loaders   = {}

    for task_name, folder_name in TASK_FOLDERS.items():
        task_path = base_path / folder_name

        if not task_path.exists():
            print(f"Warning: {task_name} not found at {task_path}, skipping...")
            continue

        try:
            if task_name == 'noise':
                dataset = TrainEvalDatasetNoise(
                    task_path / 'targets',
                    sigma=25,
                    max_images=max_images_per_task
                )
                print(f"Train eval - noise/sigma_25: {len(dataset)} images")

            else:
                dataset = TrainEvalDataset(
                    task_path / 'inputs',
                    task_path / 'targets',
                    max_images=max_images_per_task
                )
                print(f"Train eval - {task_name}: {len(dataset)} images")

            loaders[task_name] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

        except Exception as e:
            print(f"Error creating train eval loader for {task_name}: {e}")
            continue

    return loaders


# ---------------------------------------------------------------------------
# Test / Evaluation dataset
# ---------------------------------------------------------------------------
class EvalDataset(Dataset):
    """
    Evaluation dataset: loads paired input/target images,
    applies crop_img(base=16), converts to tensors.
    """
    def __init__(self, input_dir, target_dir):
        self.input_dir  = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.filenames = sorted([
            f.name for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        self.toTensor = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename    = self.filenames[idx]
        input_img   = np.array(Image.open(self.input_dir  / filename).convert('RGB'))
        target_img  = np.array(Image.open(self.target_dir / filename).convert('RGB'))
        input_img   = crop_img(input_img,  base=16)
        target_img  = crop_img(target_img, base=16)
        return self.toTensor(input_img), self.toTensor(target_img), filename
