"""
Training Callbacks - Monitor và log training
"""
import time
from typing import List, Dict, Optional
from pathlib import Path
import json


class TrainingCallback:
    """Base callback class"""
    
    def on_epoch_begin(self, epoch: int):
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        pass
    
    def on_train_begin(self):
        pass
    
    def on_train_end(self):
        pass


class ProgressLogger(TrainingCallback):
    """Log training progress"""
    
    def __init__(self, log_every: int = 1):
        self.log_every = log_every
        self.start_time = None
    
    def on_train_begin(self):
        self.start_time = time.time()
        print("🚀 Training started")
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        if (epoch + 1) % self.log_every == 0:
            elapsed = time.time() - self.start_time
            
            train_loss = logs.get('train_loss', 0)
            val_loss = logs.get('val_loss', 0)
            lr = logs.get('lr', 0)
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {lr:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    def on_train_end(self):
        total_time = time.time() - self.start_time
        print(f"✅ Training completed in {total_time:.1f}s")


class MetricTracker(TrainingCallback):
    """Track metrics over time"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch': [],
        }
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(logs.get('train_loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        self.history['lr'].append(logs.get('lr', 0))
    
    def get_history(self) -> Dict:
        return self.history
    
    def save_history(self, filepath: str):
        """Save history to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"💾 History saved to {filepath}")


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on validation loss"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        val_loss = logs.get('val_loss', float('inf'))
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
    
    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False


class ModelCheckpoint(TrainingCallback):
    """Save model checkpoints"""
    
    def __init__(
        self,
        save_dir: str,
        save_best_only: bool = True,
        save_every: int = 10
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.save_every = save_every
        
        self.best_loss = float('inf')
    
    def on_epoch_end(self, epoch: int, logs: Dict):
        val_loss = logs.get('val_loss', float('inf'))
        
        # Save best
        if self.save_best_only and val_loss < self.best_loss:
            self.best_loss = val_loss
            filepath = self.save_dir / 'best_model.pth'
            # Note: actual save handled by trainer
            logs['save_best'] = True
        
        # Periodic save
        if (epoch + 1) % self.save_every == 0:
            filepath = self.save_dir / f'checkpoint_epoch{epoch+1}.pth'
            logs['save_checkpoint'] = str(filepath)


# ==================== TESTS ====================

def test_callbacks():
    """Test callbacks"""
    print("Testing Callbacks...")
    
    # Test ProgressLogger
    logger = ProgressLogger(log_every=1)
    logger.on_train_begin()
    
    logs = {'train_loss': 1.5, 'val_loss': 2.0, 'lr': 0.001}
    logger.on_epoch_end(0, logs)
    
    logger.on_train_end()
    print("  ✅ ProgressLogger OK")
    
    # Test MetricTracker
    tracker = MetricTracker()
    
    for epoch in range(5):
        logs = {
            'train_loss': 10 - epoch,
            'val_loss': 12 - epoch,
            'lr': 0.001 * (1 - epoch * 0.1)
        }
        tracker.on_epoch_end(epoch, logs)
    
    history = tracker.get_history()
    assert len(history['train_loss']) == 5
    assert history['train_loss'][0] == 10
    assert history['train_loss'][4] == 6
    
    print("  ✅ MetricTracker OK")
    
    # Test EarlyStopping
    early_stop = EarlyStoppingCallback(patience=3)
    
    for epoch in range(10):
        logs = {'val_loss': 5.0 if epoch < 3 else 5.1}  # No improvement after epoch 3
        early_stop.on_epoch_end(epoch, logs)
        
        if early_stop.should_stop:
            print(f"     Stopped at epoch {epoch+1}")
            break
    
    assert early_stop.should_stop
    assert epoch >= 3  # Should stop after patience epochs
    
    print("  ✅ EarlyStopping OK")
    
    print("✅ Callbacks tests passed!")


if __name__ == "__main__":
    test_callbacks()