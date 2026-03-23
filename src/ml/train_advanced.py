"""
Advanced Training Pipeline with:
- Learning rate scheduling
- Early stopping
- Checkpointing
- TensorBoard logging
- Curriculum learning
"""
import sys
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dqn_agent import DQNAgent


class AdvancedTrainer:
    """Professional training pipeline"""
    
    def __init__(
        self,
        agent: DQNAgent,
        dataset_path: str,
        experiment_name: str = None,
        save_dir: str = "../../data/models",
        log_dir: str = "../../data/logs"
    ):
        self.agent = agent
        self.dataset_path = dataset_path
        
        # Experiment name
        if experiment_name is None:
            experiment_name = f"dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Directories
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Load dataset (SIMPLE FORMAT - list of dicts)
        print(f"📂 Loading dataset from {dataset_path}...")
        with open(dataset_path, 'rb') as f:
            self.dataset = pickle.load(f)
        
        print(f"   Loaded {len(self.dataset)} samples")
        
        # Verify format
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            if isinstance(sample, dict):
                print(f"   ✅ Dataset format: dictionary")
                print(f"   Keys: {list(sample.keys())}")
            else:
                print(f"   ⚠️  Unknown format: {type(sample)}")
        
        # Training stats
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        lr_schedule: str = "cosine",
        early_stopping_patience: int = 10,
        save_every: int = 5,
        eval_every: int = 1
    ):
        """
        Train with advanced features
        """
        print("="*60)
        print(f"🚀 ADVANCED TRAINING: {self.experiment_name}")
        print("="*60)
        print(f"Dataset size: {len(self.dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"LR schedule: {lr_schedule}")
        print(f"Early stopping: {early_stopping_patience}")
        print()
        
        # Setup optimizer
        optimizer = optim.Adam(self.agent.policy_net.parameters(), lr=learning_rate)
        
        # Setup LR scheduler
        if lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif lr_schedule == "step":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        elif lr_schedule == "exponential":
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_loss = self._train_epoch(batch_size, optimizer)
            
            # Log
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # LR scheduling
            if scheduler:
                scheduler.step()
            
            # Evaluation
            if (epoch + 1) % eval_every == 0:
                eval_loss = self._evaluate()
                self.writer.add_scalar('Loss/eval', eval_loss, epoch)
                print(f"   Eval Loss: {eval_loss:.4f}")
                
                # Early stopping check
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self.patience_counter = 0
                    
                    # Save best model
                    best_path = self.save_dir / "best_model.pth"
                    self.agent.save(str(best_path))
                    print(f"   💾 New best model saved!")
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= early_stopping_patience:
                        print(f"\n⏹️  Early stopping triggered (patience={early_stopping_patience})")
                        break
            
            # Checkpointing
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.save_dir / f"checkpoint_epoch{epoch+1}.pth"
                self.agent.save(str(checkpoint_path))
        
        # Final save
        final_path = self.save_dir / "final_model.pth"
        self.agent.save(str(final_path))
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Models saved to: {self.save_dir}")
        print(f"Logs saved to: {self.log_dir}")
        
        self.writer.close()
    
    def _train_epoch(self, batch_size, optimizer):
        """Train one epoch with constraint penalty"""
        self.agent.policy_net.train()
        
        np.random.shuffle(self.dataset)
        
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(self.dataset), batch_size):
            batch = self.dataset[i:i+batch_size]
            
            if len(batch) < 2:
                continue
            
            states = torch.FloatTensor(np.array([item['state'] for item in batch]))
            rewards = torch.FloatTensor(np.array([item['reward'] for item in batch]))
            
            # Add validity flag
            valid_flags = torch.FloatTensor(np.array([
                1.0 if item.get('metadata', {}).get('valid', False) else 0.0 
                for item in batch
            ]))
            
            states = states.to(self.agent.device)
            rewards = rewards.to(self.agent.device)
            valid_flags = valid_flags.to(self.agent.device)
            
            actions = torch.zeros(len(batch), dtype=torch.long).to(self.agent.device)
            
            q_values = self.agent.policy_net(states)
            predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            # Loss with validity penalty
            loss = torch.nn.functional.mse_loss(predicted_q, rewards)
            
            # Add penalty for invalid arrangements
            validity_loss = torch.mean((1 - valid_flags) * predicted_q**2)
            total_loss_batch = loss + 0.5 * validity_loss  # Weight validity
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.policy_net.parameters(), 10.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _evaluate(self):
        """Evaluate on validation set"""
        self.agent.policy_net.eval()
        
        # Use last 10% as validation
        val_size = len(self.dataset) // 10
        val_data = self.dataset[-val_size:]
        
        total_loss = 0
        
        with torch.no_grad():
            states = torch.FloatTensor(np.array([item['state'] for item in val_data]))
            rewards = torch.FloatTensor(np.array([item['reward'] for item in val_data]))
            
            states = states.to(self.agent.device)
            rewards = rewards.to(self.agent.device)
            
            actions = torch.zeros(len(val_data), dtype=torch.long).to(self.agent.device)
            
            q_values = self.agent.policy_net(states)
            predicted_q = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            loss = torch.nn.functional.mse_loss(predicted_q, rewards)
            total_loss = loss.item()
        
        return total_loss


# ==================== CLI ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced DQN Training')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset pickle file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential', 'none'])
    parser.add_argument('--early-stop', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Create agent
    agent = DQNAgent(
        state_size=52,
        action_size=1000,
        use_dueling=True,
        learning_rate=args.lr,
        buffer_size=100000,
        batch_size=args.batch_size
    )
    
    # Create trainer
    trainer = AdvancedTrainer(
        agent=agent,
        dataset_path=args.dataset,
        experiment_name=args.name
    )
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_schedule=args.lr_schedule if args.lr_schedule != 'none' else None,
        early_stopping_patience=args.early_stop
    )


if __name__ == "__main__":
    main()