"""
Trainer V3 - Production-grade training pipeline
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../networks'))

from ml.core import StateEncoderV2, ActionDecoderV2, RewardCalculator
from ml.networks import EnsembleNetwork


class TrainerV3:
    """
    Advanced trainer with:
    - Multi-objective loss (Q-value + validity + bonus)
    - Early stopping with patience
    - Learning rate scheduling
    - Gradient clipping
    - Checkpoint management
    """
    
    def __init__(
        self,
        dataset_path: str,
        network_type: str = 'ensemble',
        experiment_name: Optional[str] = None,
        device: str = 'auto'
    ):
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Device: {self.device}")
        
        # Experiment name
        if experiment_name is None:
            experiment_name = f"maubinh_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.save_dir = Path(f"data/models/{experiment_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print(f"📂 Loading dataset: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.full_dataset = pickle.load(f)
        
        print(f"   Total samples: {len(self.full_dataset)}")
        
        # Train/val split (90/10)
        split_idx = int(len(self.full_dataset) * 0.9)
        random.shuffle(self.full_dataset)
        
        self.train_data = self.full_dataset[:split_idx]
        self.val_data = self.full_dataset[split_idx:]
        
        print(f"   Train: {len(self.train_data)}")
        print(f"   Val:   {len(self.val_data)}")
        
        # Network
        if network_type == 'ensemble':
            from ml.networks import EnsembleNetwork
            self.network = EnsembleNetwork().to(self.device)
        elif network_type == 'dqn':
            from ml.networks import DQNNetwork
            self.network = DQNNetwork().to(self.device)
        elif network_type == 'transformer':
            from ml.networks import TransformerNetwork
            self.network = TransformerNetwork().to(self.device)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"🧠 Network: {network_type}")
        print(f"   Parameters: {total_params:,}")
        
        # Components
        self.encoder = StateEncoderV2()
        self.decoder = ActionDecoderV2()
        self.reward_calc = RewardCalculator()
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epochs_trained = 0
    
    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 15,
        warmup_epochs: int = 5,
        grad_clip: float = 1.0,
        save_every: int = 10
    ):
        """
        Main training loop
        """
        print("\n" + "="*60)
        print(f"🚀 TRAINING: {self.experiment_name}")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Patience: {patience}")
        print()
        
        # Optimizer
        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training loop
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(batch_size, optimizer, grad_clip)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate_epoch(batch_size)
            self.val_losses.append(val_loss)
            
            # LR step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s", end="")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save best
                self._save_checkpoint('best_model.pth')
                print(" 💾 BEST!", end="")
            else:
                patience_counter += 1
            
            print()
            
            # Periodic save
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'checkpoint_epoch{epoch+1}.pth')
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
            
            self.epochs_trained = epoch + 1
        
        # Final save
        self._save_checkpoint('final_model.pth')
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epochs_trained,
        }
        
        with open(self.save_dir / 'training_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Epochs trained: {self.epochs_trained}")
        print(f"Models saved to: {self.save_dir}")
        print("="*60)
    
    def _train_epoch(self, batch_size: int, optimizer, grad_clip: float) -> float:
        """Train one epoch"""
        self.network.train()
        
        # Shuffle
        random.shuffle(self.train_data)
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(self.train_data), batch_size):
            batch = self.train_data[i:i+batch_size]
            
            if len(batch) < 4:  # Min batch size
                continue
            
            # Prepare batch
            states = torch.FloatTensor(
                np.array([s['state'] for s in batch])
            ).to(self.device)
            
            rewards = torch.FloatTensor(
                np.array([s['reward'] for s in batch])
            ).to(self.device)
            
            # Forward
            q_values = self.network(states)
            
            # Loss: MSE between max Q and reward
            predicted_values = q_values.max(dim=1)[0]
            
            # Main loss
            mse_loss = nn.functional.mse_loss(predicted_values, rewards)
            
            # Regularization: encourage valid arrangements (bonus for high Q on high reward)
            # Ranking loss
            ranking_loss = self._ranking_loss(q_values, rewards)
            
            # Total loss
            total_loss_batch = mse_loss + 0.1 * ranking_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
            
            optimizer.step()
            
            total_loss += mse_loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, batch_size: int) -> float:
        """Validate one epoch"""
        self.network.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(self.val_data), batch_size):
                batch = self.val_data[i:i+batch_size]
                
                if len(batch) < 4:
                    continue
                
                states = torch.FloatTensor(
                    np.array([s['state'] for s in batch])
                ).to(self.device)
                
                rewards = torch.FloatTensor(
                    np.array([s['reward'] for s in batch])
                ).to(self.device)
                
                q_values = self.network(states)
                predicted_values = q_values.max(dim=1)[0]
                
                loss = nn.functional.mse_loss(predicted_values, rewards)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _ranking_loss(self, q_values, rewards):
        """Ranking loss: samples with higher rewards should have higher Q-values"""
        batch_size = rewards.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0).to(self.device)
        
        # Max Q values
        max_q = q_values.max(dim=1)[0]
        
        # Sample pairs
        num_pairs = min(batch_size // 2, 16)
        indices = torch.randperm(batch_size)[:num_pairs * 2]
        
        idx1 = indices[:num_pairs]
        idx2 = indices[num_pairs:num_pairs * 2]
        
        # Compute differences
        reward_diff = rewards[idx1] - rewards[idx2]
        q_diff = max_q[idx1] - max_q[idx2]
        
        # Hinge loss
        loss = torch.mean(torch.relu(-reward_diff * q_diff + 0.1))
        
        return loss
    
    def _save_checkpoint(self, filename: str):
        """Save checkpoint"""
        filepath = self.save_dir / filename
        
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epochs_trained,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        torch.save(checkpoint, filepath)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Mau Binh ML Agent')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset')
    parser.add_argument('--network', type=str, default='ensemble', choices=['dqn', 'transformer', 'ensemble'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--name', type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = TrainerV3(
        dataset_path=args.data,
        network_type=args.network,
        experiment_name=args.name
    )
    
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )