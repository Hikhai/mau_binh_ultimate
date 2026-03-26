"""
Quick ML Training - Mau Binh V3
Tự generate training data + Train model
KHÔNG phụ thuộc vào ultimate_solver (bypass bug None)
"""
import sys
import os
import random
import time
from pathlib import Path
from collections import Counter
from itertools import combinations
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "core"))
sys.path.insert(0, str(project_root / "src" / "ml" / "core"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from card import Card, Rank, Suit, Deck
from evaluator import HandEvaluator
from hand_types import HandType
from state_encoder import StateEncoderV3
from reward_calculator import RewardCalculatorV2
from action_decoder import ActionDecoderV3
from arrangement_validator import ArrangementValidatorV2


# ============================================================
# SMART ARRANGER - Tự xếp bài KHÔNG cần solver
# ============================================================

class SmartArranger:
    """
    Xếp 13 lá bài thành 3 chi (front/middle/back) tối ưu
    
    Strategies:
    1. Greedy by rank
    2. Pair-aware: Giữ pairs/trips trong đúng chi
    3. Flush-aware: Detect và xếp flush
    4. Straight-aware: Detect và xếp straight
    5. Random search: Random shuffle tìm best
    
    Đảm bảo: back >= middle >= front (không LỦNG)
    """
    
    def __init__(self):
        self.validator = ArrangementValidatorV2()
        self.reward_calc = RewardCalculatorV2()
    
    def arrange(self, cards, num_attempts=50):
        """
        Xếp 13 cards thành arrangement tối ưu
        
        Returns:
            (back, middle, front, reward) hoặc None nếu thất bại
        """
        if len(cards) != 13:
            return None
        
        best_arr = None
        best_reward = -1000.0
        
        # Strategy 1: Greedy by rank
        arr = self._arrange_greedy(cards)
        if arr:
            reward = self._evaluate(arr)
            if reward > best_reward:
                best_reward = reward
                best_arr = arr
        
        # Strategy 2: Pair-aware
        arr = self._arrange_pairs(cards)
        if arr:
            reward = self._evaluate(arr)
            if reward > best_reward:
                best_reward = reward
                best_arr = arr
        
        # Strategy 3: Flush-aware
        arr = self._arrange_flush(cards)
        if arr:
            reward = self._evaluate(arr)
            if reward > best_reward:
                best_reward = reward
                best_arr = arr
        
        # Strategy 4: Straight-aware
        arr = self._arrange_straight(cards)
        if arr:
            reward = self._evaluate(arr)
            if reward > best_reward:
                best_reward = reward
                best_arr = arr
        
        # Strategy 5: Random search
        for _ in range(num_attempts):
            arr = self._arrange_random(cards)
            if arr:
                reward = self._evaluate(arr)
                if reward > best_reward:
                    best_reward = reward
                    best_arr = arr
        
        # Strategy 6: Smart combinations
        for _ in range(min(num_attempts, 20)):
            arr = self._arrange_smart_combo(cards)
            if arr:
                reward = self._evaluate(arr)
                if reward > best_reward:
                    best_reward = reward
                    best_arr = arr
        
        if best_arr:
            return (*best_arr, best_reward)
        
        return None
    
    def _evaluate(self, arr):
        """Evaluate arrangement, return reward"""
        back, middle, front = arr
        
        is_valid, _, _ = self.validator.is_valid_detailed(back, middle, front, track_stats=False)
        
        if not is_valid:
            return -1000.0
        
        result = self.reward_calc.calculate_reward(back, middle, front)
        return result['total_reward']
    
    def _arrange_greedy(self, cards):
        """Sort by rank, split 5-5-3"""
        sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        
        back = sorted_cards[:5]
        middle = sorted_cards[5:10]
        front = sorted_cards[10:13]
        
        return self._validate_and_return(back, middle, front)
    
    def _arrange_pairs(self, cards):
        """Put pairs/trips strategically"""
        rank_groups = {}
        for card in cards:
            if card.rank not in rank_groups:
                rank_groups[card.rank] = []
            rank_groups[card.rank].append(card)
        
        # Sort groups by count then rank (quads > trips > pairs > singles)
        sorted_groups = sorted(
            rank_groups.items(),
            key=lambda x: (len(x[1]), x[0].value),
            reverse=True
        )
        
        # Distribute
        back_cards = []
        middle_cards = []
        front_cards = []
        
        for rank, group in sorted_groups:
            count = len(group)
            
            if count == 4:
                # Quad → back (bonus!)
                if len(back_cards) + 4 <= 5:
                    back_cards.extend(group)
                else:
                    # Split quad
                    remaining = 5 - len(back_cards)
                    back_cards.extend(group[:remaining])
                    middle_cards.extend(group[remaining:])
                    
            elif count == 3:
                # Triple
                if len(front_cards) == 0 and len(back_cards) >= 5 and len(middle_cards) >= 5:
                    # Front triple (bonus +6!)
                    front_cards.extend(group)
                elif len(back_cards) + 3 <= 5:
                    back_cards.extend(group)
                elif len(middle_cards) + 3 <= 5:
                    middle_cards.extend(group)
                else:
                    # Split
                    for card in group:
                        if len(back_cards) < 5:
                            back_cards.append(card)
                        elif len(middle_cards) < 5:
                            middle_cards.append(card)
                        else:
                            front_cards.append(card)
                    
            elif count == 2:
                # Pair
                if len(back_cards) + 2 <= 5:
                    back_cards.extend(group)
                elif len(middle_cards) + 2 <= 5:
                    middle_cards.extend(group)
                elif len(front_cards) + 2 <= 3:
                    front_cards.extend(group)
                else:
                    for card in group:
                        if len(back_cards) < 5:
                            back_cards.append(card)
                        elif len(middle_cards) < 5:
                            middle_cards.append(card)
                        else:
                            front_cards.append(card)
            else:
                # Single
                if len(back_cards) < 5:
                    back_cards.extend(group)
                elif len(middle_cards) < 5:
                    middle_cards.extend(group)
                elif len(front_cards) < 3:
                    front_cards.extend(group)
        
        # Verify counts
        if len(back_cards) != 5 or len(middle_cards) != 5 or len(front_cards) != 3:
            return None
        
        return self._validate_and_return(back_cards, middle_cards, front_cards)
    
    def _arrange_flush(self, cards):
        """Try to make flush in back or middle"""
        suit_groups = {}
        for card in cards:
            if card.suit not in suit_groups:
                suit_groups[card.suit] = []
            suit_groups[card.suit].append(card)
        
        # Find suit with 5+ cards
        for suit, group in sorted(suit_groups.items(), key=lambda x: len(x[1]), reverse=True):
            if len(group) >= 5:
                # Use top 5 for back (flush)
                flush_cards = sorted(group, key=lambda c: c.rank.value, reverse=True)[:5]
                remaining = [c for c in cards if c not in flush_cards]
                
                # Sort remaining
                remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
                
                middle = remaining_sorted[:5]
                front = remaining_sorted[5:8]
                
                arr = self._validate_and_return(flush_cards, middle, front)
                if arr:
                    return arr
        
        return None
    
    def _arrange_straight(self, cards):
        """Try to make straight in back"""
        sorted_cards = sorted(cards, key=lambda c: c.rank.value)
        ranks = [c.rank.value for c in sorted_cards]
        
        # Find 5 consecutive ranks
        for start_idx in range(len(sorted_cards) - 4):
            window = sorted_cards[start_idx:start_idx + 5]
            window_ranks = [c.rank.value for c in window]
            
            # Check consecutive
            if window_ranks[-1] - window_ranks[0] == 4 and len(set(window_ranks)) == 5:
                remaining = [c for c in cards if c not in window]
                remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
                
                middle = remaining_sorted[:5]
                front = remaining_sorted[5:8]
                
                arr = self._validate_and_return(window, middle, front)
                if arr:
                    return arr
        
        # Check wheel (A-2-3-4-5)
        rank_set = set(c.rank.value for c in cards)
        if {14, 2, 3, 4, 5}.issubset(rank_set):
            wheel = []
            for r in [14, 2, 3, 4, 5]:
                for c in cards:
                    if c.rank.value == r and c not in wheel:
                        wheel.append(c)
                        break
            
            if len(wheel) == 5:
                remaining = [c for c in cards if c not in wheel]
                remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
                
                middle = remaining_sorted[:5]
                front = remaining_sorted[5:8]
                
                arr = self._validate_and_return(wheel, middle, front)
                if arr:
                    return arr
        
        return None
    
    def _arrange_random(self, cards):
        """Random shuffle and split"""
        shuffled = cards.copy()
        random.shuffle(shuffled)
        
        back = sorted(shuffled[:5], key=lambda c: c.rank.value, reverse=True)
        middle = sorted(shuffled[5:10], key=lambda c: c.rank.value, reverse=True)
        front = sorted(shuffled[10:13], key=lambda c: c.rank.value, reverse=True)
        
        return self._validate_and_return(back, middle, front)
    
    def _arrange_smart_combo(self, cards):
        """Try random front combination, greedy for rest"""
        # Random front (3 cards)
        indices = random.sample(range(13), 3)
        front = [cards[i] for i in indices]
        remaining = [cards[i] for i in range(13) if i not in indices]
        
        # Sort remaining by rank
        remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
        back = remaining_sorted[:5]
        middle = remaining_sorted[5:10]
        
        return self._validate_and_return(back, middle, front)
    
    def _validate_and_return(self, back, middle, front):
        """Validate and return if valid"""
        try:
            is_valid, _, _ = self.validator.is_valid_detailed(
                back, middle, front, track_stats=False
            )
            if is_valid:
                return (back, middle, front)
        except:
            pass
        return None


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class MauBinhNet(nn.Module):
    """
    Neural network cho Mau Binh
    Input: 130-dim state (V3 features)
    Output: Front action logits (286 dims)
    """
    
    def __init__(self, state_size=130, action_size=286, hidden_size=512):
        super().__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, action_size),
        )
    
    def forward(self, state):
        return self.network(state)


# ============================================================
# DATASET
# ============================================================

class MauBinhDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            'state': torch.FloatTensor(ex['state']),
            'front_action': torch.LongTensor([ex['front_action']]),
            'reward': torch.FloatTensor([ex['reward']]),
        }


# ============================================================
# DATA GENERATION
# ============================================================

def generate_training_data(num_samples=5000, num_attempts=50):
    """Generate training data bằng SmartArranger"""
    
    print(f"\n{'='*60}")
    print(f"🎲 GENERATING TRAINING DATA")
    print(f"{'='*60}")
    print(f"Target samples: {num_samples:,}")
    print(f"{'='*60}\n")
    
    encoder = StateEncoderV3()
    decoder = ActionDecoderV3()
    arranger = SmartArranger()
    
    def random_13_cards():
        """Random 13 cards từ 52 lá"""
        full = Deck.full_deck()
        return random.sample(full, 13)
    
    examples = []
    failed = 0
    
    start_time = time.time()
    pbar = tqdm(total=num_samples, desc="Generating")
    
    while len(examples) < num_samples:
        cards = random_13_cards()
        
        result = arranger.arrange(cards, num_attempts=num_attempts)
        
        if result is None:
            failed += 1
            continue
        
        back, middle, front, reward = result
        
        state = encoder.encode(cards)
        arrangement = (back, middle, front)
        front_action, back_action = decoder.encode_arrangement(arrangement, cards)
        
        examples.append({
            'state': state,
            'front_action': front_action,
            'back_action': back_action,
            'reward': reward,
        })
        
        pbar.update(1)
    
    pbar.close()
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Generated {len(examples):,} in {elapsed:.1f}s")
    print(f"   Failed: {failed:,}")
    print(f"   Rate: {len(examples)/(len(examples)+failed):.1%}")
    
    rewards = [ex['reward'] for ex in examples]
    print(f"\n📊 Rewards: min={min(rewards):.1f} max={max(rewards):.1f} mean={np.mean(rewards):.1f}")
    
    return examples

# ============================================================
# TRAINING
# ============================================================

def train_model(
    num_samples=5000,
    epochs=50,
    batch_size=64,
    learning_rate=1e-3,
    device='auto',
    save_dir='models/quick_ml',
):
    """Main training function"""
    
    print("\n" + "="*60)
    print("🚀 MAU BINH ML TRAINING - V3")
    print("="*60)
    
    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Device: {device}")
    
    # Generate data
    examples = generate_training_data(num_samples)
    
    if len(examples) < 100:
        print("❌ Not enough data!")
        return
    
    # Split train/val
    random.shuffle(examples)
    split_idx = int(len(examples) * 0.9)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    print(f"\n📦 Dataset:")
    print(f"   Train: {len(train_examples):,}")
    print(f"   Val:   {len(val_examples):,}")
    
    # DataLoaders
    train_dataset = MauBinhDataset(train_examples)
    val_dataset = MauBinhDataset(val_examples)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0
    )
    
    # Model
    model = MauBinhNet(state_size=130, action_size=286).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {total_params:,} parameters")
    
    # Optimizer + Scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6
    )
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print(f"\n{'='*60}")
    print(f"🏋️ TRAINING")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    best_accuracy = 0.0
    no_improve_count = 0
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    training_log = []
    
    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            state = batch['state'].to(device)
            action = batch['front_action'].squeeze().to(device)
            
            # Forward
            logits = model(state)
            loss = criterion(logits, action)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            train_correct += (pred == action).sum().item()
            train_total += action.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # ===== VALIDATE =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                action = batch['front_action'].squeeze().to(device)
                
                logits = model(state)
                loss = criterion(logits, action)
                
                val_loss += loss.item()
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == action).sum().item()
                val_total += action.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # LR Schedule
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'lr': current_lr,
        }
        training_log.append(log_entry)
        
        # Print
        print(
            f"Epoch {epoch+1:3d}/{epochs} │ "
            f"Train: loss={train_loss:.4f} acc={train_acc:.2%} │ "
            f"Val: loss={val_loss:.4f} acc={val_acc:.2%} │ "
            f"LR={current_lr:.1e}",
            end=""
        )
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracy = val_acc
            no_improve_count = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'state_size': 130,
                'action_size': 286,
                'training_log': training_log,
            }, save_path / 'best_model.pt')
            
            print(f" ⭐ BEST", end="")
        else:
            no_improve_count += 1
        
        print()
        
        # Early stopping
        if no_improve_count >= 15:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc,
        'state_size': 130,
        'action_size': 286,
        'training_log': training_log,
    }, save_path / 'final_model.pt')
    
    # Save training log
    import json
    with open(save_path / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Best val loss:     {best_val_loss:.4f}")
    print(f"Best val accuracy: {best_accuracy:.2%}")
    print(f"Models saved to:   {save_path}")
    print(f"{'='*60}\n")
    
    return model, training_log


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Mau Binh ML Model V3")
    parser.add_argument('--samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='models/quick_ml', help='Save directory')
    
    args = parser.parse_args()
    
    print("\n" + "🃏" * 30)
    print("MAU BINH ML TRAINING PIPELINE V3")
    print("🃏" * 30)
    
    model, log = train_model(
        num_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )