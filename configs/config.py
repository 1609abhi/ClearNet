"""
Default configuration values for ClearNet.
These are used as argparse defaults in train.py and test.py.
Override any value via command-line arguments.
"""

# Random seed
SEED = 42

# Device
DEVICE = "cuda"

# Training parameters
NUM_ITERS    = 500_000
WARMUP_ITERS = 50_000
BATCH_SIZE = 2
ITER_JUMP = 2_000       # Logging frequency
CHECKPOINT_ITER = 10_000  # Saving frequency

# Model parameters
IMG_CHANNEL = 3
WIDTH = 48
NUM_BLKS = [4, 6, 6, 8]
REFINE_NUM_BLK = 4
HEADS = [1, 2, 4, 8]

# Data paths
BASE_DIR = 'datasets'

# Training paths
CHECKPOINT_DIR = "saved_logs/d5"

# Image transform parameters
IMAGE_SIZE = (256, 256)

# Optimizer parameters
LEARNING_RATE = 2e-4

# Fourier loss weight
FOURIER_LOSS_WEIGHT = 0.1

# Task configuration
ALL_TASKS = ['blur', 'haze', 'lol', 'noise', 'rain']

TASK_FOLDERS = {
    'blur':  'deblurring_train',
    'haze':  'dehazing_train',
    'lol':   'lol_train',
    'noise': 'denoising_train',
    'rain':  'deraining_train',
}

# Test dataset paths: task -> list of [target_dir, input_dir, label]
DATASET_PATHS = {
    "blur": [
        ["test_sandbox/datasets/test-data/GoPro/target", "test_sandbox/datasets/test-data/GoPro/input", "gopro"]
    ],
    "haze": [
        ["test_sandbox/datasets/test-data/SOTS/GT", "test_sandbox/datasets/test-data/SOTS/IN", "sots"]
    ],
    "lol": [
        ["test_sandbox/datasets/test-data/LOL/high", "test_sandbox/datasets/test-data/LOL/low", "lol"]
    ],
    "noise": [
        ["test_sandbox/datasets/test-data/denoising_testsets/CBSD68", "test_sandbox/datasets/test-data/denoising_testsets/CBSD68_25", "25"]
    ],
    "rain": [
        ["test_sandbox/datasets/test-data/Rain100L/target", "test_sandbox/datasets/test-data/Rain100L/input", "rain100l"]
    ],
}

# Resume configuration
RESUME_TRAINING = False
RESUME_CHECKPOINT = ""
