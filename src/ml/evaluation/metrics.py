"""
Metrics - Track và visualize training metrics
"""
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


class MetricsVisualizer:
    """
    Visualize training metrics
    """
    
    @staticmethod
    def plot_training_history(history_path: str, save_path: str = None):
        """
        Plot training history
        
        Args:
            history_path: Path to training_history.pkl
            save_path: Where to save plot (optional)
        """
        # Load history
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        train_losses = history['train_losses']
        val_losses = history['val_losses']
        epochs = list(range(1, len(train_losses) + 1))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = val_losses.index(min(val_losses)) + 1
        best_loss = min(val_losses)
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
        ax.text(best_epoch, best_loss, f'{best_loss:.3f}', ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"💾 Plot saved to {save_path}")
        else:
            plt.show()
    
    @staticmethod
    def print_summary(history_path: str):
        """Print training summary"""
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        print("="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Total epochs:    {len(history['train_losses'])}")
        print(f"Best val loss:   {history.get('best_val_loss', 'N/A'):.4f}")
        print(f"Final train loss: {history['train_losses'][-1]:.4f}")
        print(f"Final val loss:   {history['val_losses'][-1]:.4f}")
        print("="*60)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--history', type=str, required=True, help='Path to training_history.pkl')
    parser.add_argument('--output', type=str, default=None, help='Output plot path')
    
    args = parser.parse_args()
    
    MetricsVisualizer.print_summary(args.history)
    MetricsVisualizer.plot_training_history(args.history, args.output)