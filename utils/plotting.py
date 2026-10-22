"""
Training visualization utilities for ClearNet.
"""

import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def save_training_plots(training_logs, checkpoint_dir):
    """Save training loss and learning rate summary plots."""
    plots_dir = os.path.join(checkpoint_dir, "training_plots")
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(training_logs['iteration'], training_logs['loss_total'],
                 'r-', linewidth=2, label='Total')
    axes[0].plot(training_logs['iteration'], training_logs['loss_pyramid'],
                 'b--', linewidth=1, label='Pyramid', alpha=0.7)
    axes[0].set_title('Total vs Pyramid Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(training_logs['iteration'], training_logs['loss_fourier'],
                 'b-', linewidth=2)
    axes[1].set_title('Fourier Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(training_logs['iteration'], training_logs['learning_rate'],
                 color='brown', linewidth=2)
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Iteration')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plots saved to {plots_dir}")
