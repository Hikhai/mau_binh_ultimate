"""
Expert Training V2
- Constraint-aware loss function
- Better reward shaping
- Curriculum learning
- Comprehensive validation
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))
sys.path.insert(0, os.path.dirname(__file__))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints
from network_v2 import ConstraintAwareDQN


class ExpertTrainerV2:
    """
    Advanced trainer with constraint-aware learning
    """
    
    def __init__(
        self,
        dataset_path: str,
        experiment_name: str = None,
        save_dir: str = "../../data/models",
        log_dir: str = "../../data/logs"
    ):
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Device: {self.device}")
        
        # Experiment
        if experiment_name is None:
            experiment_name = f"expert_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Network
        self.network = ConstraintAwareDQN(state_size=52, action_size=1000).to(self.device)
        
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"🧠 Network parameters: {total_params:,}")
        
        # Load dataset
        print(f"📂 Loading dataset: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            self.full_dataset = pickle.load(f)
        
        print(f"   Total samples: {len(self.full_dataset)}")
        
        # Split train/val (90/10)
        split_idx = int(len(self.full_dataset) * 0.9)
        random.shuffle(self.full_dataset)
        self.train_data = self.full_dataset[:split_idx]
        self.val_data = self.full_dataset[split_idx:]
        
        print(f"   Train: {len(self.train_data)}")
        print(f"   Val:   {len(self.val_data)}")
        
        # Normalize rewards
        all_rewards = [d['reward'] for d in self.full_dataset]
        self.reward_mean = np.mean(all_rewards)
        self.reward_std = max(np.std(all_rewards), 0.01)
        
        print(f"   Reward mean: {self.reward_mean:.2f}")
        print(f"   Reward std:  {self.reward_std:.2f}")
        
        # Stats
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def normalize_reward(self, reward):
        """Normalize reward to ~0 mean, ~1 std"""
        return (reward - self.reward_mean) / self.reward_std
    
    def train(
        self,
        num_epochs: int = 200,
        batch_size: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 20,
        warmup_epochs: int = 5
    ):
        """
        Train with advanced features
        """
        print("\n" + "="*60)
        print(f"🚀 EXPERT TRAINING V2: {self.experiment_name}")
        print("="*60)
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Patience: {patience}")
        print(f"Warmup epochs: {warmup_epochs}")
        print()
        
        # Optimizer with weight decay (regularization)
        optimizer = optim.AdamW(
            self.network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (warmup + cosine)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(batch_size, optimizer)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self._validate()
            self.val_losses.append(val_loss)
            
            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}", end="")
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                self._save_model("best_model.pth")
                print(" 💾 BEST!", end="")
            else:
                patience_counter += 1
            
            print()
            
            # Checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_model(f"checkpoint_epoch{epoch+1}.pth")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
        
        # Save final
        self._save_model("final_model.pth")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }
        
        with open(self.save_dir / "training_history.pkl", 'wb') as f:
            pickle.dump(history, f)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Models saved to: {self.save_dir}")
    
    def _train_epoch(self, batch_size, optimizer):
        """Train one epoch"""
        self.network.train()
        
        # Shuffle
        random.shuffle(self.train_data)
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(self.train_data), batch_size):
            batch = self.train_data[i:i+batch_size]
            
            if len(batch) < 4:  # Min batch for BatchNorm
                continue
            
            # Prepare data
            states = torch.FloatTensor(
                np.array([item['state'] for item in batch])
            ).to(self.device)
            
            # Normalize rewards
            rewards = torch.FloatTensor(
                np.array([self.normalize_reward(item['reward']) for item in batch])
            ).to(self.device)
            
            # Forward
            q_values = self.network(states)
            
            # Get Q value for "best action" (action 0 as proxy)
            predicted = q_values[:, 0]
            
            # Main loss: MSE between predicted Q and normalized reward
            mse_loss = nn.functional.mse_loss(predicted, rewards)
            
            # Ranking loss: ensure higher reward = higher Q value
            ranking_loss = self._ranking_loss(q_values, rewards)
            
            # Total loss
            total_loss_batch = mse_loss + 0.1 * ranking_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
            
            optimizer.step()
            
            total_loss += mse_loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _ranking_loss(self, q_values, rewards):
        """
        Ranking loss: Ensure that samples with higher rewards 
        get higher Q-values
        """
        batch_size = rewards.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0).to(self.device)
        
        # Get Q values for action 0
        q = q_values[:, 0]
        
        # Random pairs
        num_pairs = min(batch_size // 2, 32)
        indices = torch.randperm(batch_size)[:num_pairs * 2]
        
        idx1 = indices[:num_pairs]
        idx2 = indices[num_pairs:num_pairs * 2]
        
        # For each pair, higher reward should have higher Q
        reward_diff = rewards[idx1] - rewards[idx2]
        q_diff = q[idx1] - q[idx2]
        
        # Hinge loss: penalize when Q ordering disagrees with reward ordering
        loss = torch.mean(torch.relu(-reward_diff * q_diff + 0.1))
        
        return loss
    
    def _validate(self):
        """Validate on held-out data"""
        self.network.eval()
        
        with torch.no_grad():
            states = torch.FloatTensor(
                np.array([item['state'] for item in self.val_data])
            ).to(self.device)
            
            rewards = torch.FloatTensor(
                np.array([self.normalize_reward(item['reward']) for item in self.val_data])
            ).to(self.device)
            
            # Handle large validation set (process in batches)
            batch_size = 256
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(self.val_data), batch_size):
                batch_states = states[i:i+batch_size]
                batch_rewards = rewards[i:i+batch_size]
                
                if len(batch_states) < 4:
                    continue
                
                q_values = self.network(batch_states)
                predicted = q_values[:, 0]
                
                loss = nn.functional.mse_loss(predicted, batch_rewards)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _save_model(self, filename):
        """Save model checkpoint"""
        filepath = self.save_dir / filename
        
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        torch.save(checkpoint, filepath)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Expert Training V2')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--name', type=str, default=None)
    
    args = parser.parse_args()
    
    trainer = ExpertTrainerV2(
        dataset_path=args.dataset,
        experiment_name=args.name
    )
    
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience
    )