"""
Expert Data Generator V3 - IMPROVED Production Grade
Generates 100% valid expert-level data with HIGH QUALITY
"""
import sys
import os
import numpy as np
import pickle
import random
from pathlib import Path
from typing import List, Tuple
from itertools import combinations
import multiprocessing as mp
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

from card import Card, Deck
from evaluator import HandEvaluator
from ml.core import RewardCalculator, StateEncoderV2


class ExpertDataGeneratorV3:
    """
    Generate expert-quality training data - IMPROVED
    
    Improvements over V2:
    1. More diverse arrangement strategies
    2. Better reward filtering (only keep good arrangements)
    3. Bonus-focused generation
    4. 100% valid, high-quality data
    """
    
    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = StateEncoderV2()
        self.reward_calc = RewardCalculator()
    
    def generate_valid_arrangements(
        self,
        cards: List[Card],
        max_arrangements: int = 200,
        timeout: float = 2.0
    ) -> List[Tuple[List[Card], List[Card], List[Card]]]:
        """
        Generate MANY valid arrangements using DIVERSE strategies
        """
        import time
        start_time = time.time()
        
        valid_arrs = []
        
        # Analyze hand
        rank_counts = Counter(c.rank for c in cards)
        suit_counts = Counter(c.suit for c in cards)
        
        pairs = sorted([r for r, c in rank_counts.items() if c == 2],
                       key=lambda r: r.value, reverse=True)
        trips = [r for r, c in rank_counts.items() if c == 3]
        quads = [r for r, c in rank_counts.items() if c == 4]
        
        # ===== STRATEGY 1: TRIPS IN FRONT (chi cuối xám → +6 bonus!) =====
        for trip_rank in trips:
            if time.time() - start_time > timeout:
                break
            
            trip_cards = [c for c in cards if c.rank == trip_rank][:3]
            remaining = [c for c in cards if c not in trip_cards]
            
            # Try multiple back/middle splits
            for _ in range(10):
                random.shuffle(remaining)
                back = remaining[:5]
                middle = remaining[5:10]
                
                if self._is_valid(back, middle, trip_cards):
                    valid_arrs.append((back, middle, trip_cards))
            
            # Also try sorted remaining
            sorted_rem = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
            back = sorted_rem[:5]
            middle = sorted_rem[5:10]
            
            if self._is_valid(back, middle, trip_cards):
                valid_arrs.append((back, middle, trip_cards))
        
        # ===== STRATEGY 2: HIGH PAIRS IN FRONT =====
        for pair_rank in pairs[:3]:
            if time.time() - start_time > timeout:
                break
            
            pair_cards = [c for c in cards if c.rank == pair_rank][:2]
            other_cards = [c for c in cards if c not in pair_cards]
            
            # Try different kickers
            for kicker in other_cards[:5]:
                front = pair_cards + [kicker]
                remaining = [c for c in other_cards if c != kicker]
                
                # Try sorted
                sorted_rem = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
                back = sorted_rem[:5]
                middle = sorted_rem[5:10]
                
                if self._is_valid(back, middle, front):
                    valid_arrs.append((back, middle, front))
                
                # Try random
                for _ in range(3):
                    random.shuffle(remaining)
                    back = remaining[:5]
                    middle = remaining[5:10]
                    
                    if self._is_valid(back, middle, front):
                        valid_arrs.append((back, middle, front))
        
        # ===== STRATEGY 3: FIND STRAIGHTS IN BACK/MIDDLE =====
        sorted_cards = sorted(cards, key=lambda c: c.rank.value)
        
        for start_val in range(2, 11):  # 2-10
            if time.time() - start_time > timeout:
                break
            
            straight = []
            for target_val in range(start_val, start_val + 5):
                for card in cards:
                    if card.rank.value == target_val and card not in straight:
                        straight.append(card)
                        break
            
            if len(straight) == 5:
                other_cards = [c for c in cards if c not in straight]
                
                # Straight in back
                for _ in range(5):
                    random.shuffle(other_cards)
                    middle = other_cards[:5]
                    front = other_cards[5:8]
                    
                    if self._is_valid(straight, middle, front):
                        valid_arrs.append((straight, middle, front))
                        break
                
                # Straight in middle
                for _ in range(5):
                    random.shuffle(other_cards)
                    back = other_cards[:5]
                    front = other_cards[5:8]
                    
                    if self._is_valid(back, straight, front):
                        valid_arrs.append((back, straight, front))
                        break
        
        # ===== STRATEGY 4: FIND FLUSHES IN BACK/MIDDLE =====
        for suit, count in suit_counts.items():
            if count >= 5:
                if time.time() - start_time > timeout:
                    break
                
                suit_cards = [c for c in cards if c.suit == suit]
                flush = suit_cards[:5]
                other_cards = [c for c in cards if c not in flush]
                
                # Flush in back
                for _ in range(5):
                    random.shuffle(other_cards)
                    middle = other_cards[:5]
                    front = other_cards[5:8]
                    
                    if self._is_valid(flush, middle, front):
                        valid_arrs.append((flush, middle, front))
                        break
        
        # ===== STRATEGY 5: TWO PAIRS IN BACK/MIDDLE =====
        if len(pairs) >= 2:
            for i in range(min(2, len(pairs))):
                for j in range(i+1, min(4, len(pairs))):
                    if time.time() - start_time > timeout:
                        break
                    
                    pair1_cards = [c for c in cards if c.rank == pairs[i]][:2]
                    pair2_cards = [c for c in cards if c.rank == pairs[j]][:2]
                    
                    two_pair = pair1_cards + pair2_cards
                    other_cards = [c for c in cards if c not in two_pair]
                    
                    # Two pair in back
                    for kicker in other_cards[:3]:
                        back = two_pair + [kicker]
                        remaining = [c for c in other_cards if c != kicker]
                        
                        for _ in range(3):
                            random.shuffle(remaining)
                            middle = remaining[:5]
                            front = remaining[5:8]
                            
                            if self._is_valid(back, middle, front):
                                valid_arrs.append((back, middle, front))
                                break
        
        # ===== STRATEGY 6: RANDOM VALID (fill up to max) =====
        attempts = 0
        max_attempts = max_arrangements * 3
        
        while len(valid_arrs) < max_arrangements and attempts < max_attempts:
            if time.time() - start_time > timeout:
                break
            
            attempts += 1
            
            shuffled = cards.copy()
            random.shuffle(shuffled)
            
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            if self._is_valid(back, middle, front):
                valid_arrs.append((back, middle, front))
        
        # Remove duplicates
        unique = []
        seen = set()
        
        for arr in valid_arrs:
            key = self._arrangement_key(arr)
            if key not in seen:
                seen.add(key)
                unique.append(arr)
        
        return unique
    
    def _arrangement_key(self, arr):
        """Create unique key for arrangement"""
        back, middle, front = arr
        return (
            tuple(sorted([c.to_index() for c in back])),
            tuple(sorted([c.to_index() for c in middle])),
            tuple(sorted([c.to_index() for c in front]))
        )
    
    def _is_valid(self, back, middle, front):
        """Check if arrangement is valid"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        try:
            is_valid, _ = HandEvaluator.is_valid_arrangement(back, middle, front)
            return is_valid
        except Exception:
            return False
    
    def select_best_arrangement(
        self,
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]]
    ) -> Tuple[Tuple[List[Card], List[Card], List[Card]], float]:
        """
        Select BEST arrangement - IMPROVED filtering
        """
        if not arrangements:
            return None, -100.0
        
        # Calculate rewards for all arrangements
        scored_arrs = []
        for arr in arrangements:
            reward = self.reward_calc.calculate_reward(*arr)
            if reward > 0:  # Only keep valid ones with positive reward
                scored_arrs.append((arr, reward))
        
        # If no good arrangements found, take best from all
        if not scored_arrs:
            best_arr = None
            best_reward = -float('inf')
            
            for arr in arrangements:
                reward = self.reward_calc.calculate_reward(*arr)
                if reward > best_reward:
                    best_reward = reward
                    best_arr = arr
            
            return best_arr, best_reward
        
        # Sort by reward descending
        scored_arrs.sort(key=lambda x: x[1], reverse=True)
        
        # Return best
        return scored_arrs[0]
    
    def generate_single_sample(self, seed: int = None) -> dict:
        """Generate one training sample"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Random hand
        deck = Deck.full_deck()
        cards = random.sample(deck, 13)
        
        # Generate valid arrangements (MORE = BETTER)
        valid_arrs = self.generate_valid_arrangements(
            cards,
            max_arrangements=200,
            timeout=2.0
        )
        
        if not valid_arrs:
            return None
        
        # Select best
        best_arr, reward = self.select_best_arrangement(valid_arrs)
        
        if best_arr is None or reward < 0:
            return None
        
        # Encode state
        state = self.encoder.encode(cards)
        
        # Create sample
        sample = {
            'state': state,
            'arrangement': best_arr,
            'reward': float(reward),
            'num_alternatives': len(valid_arrs),
        }
        
        return sample
    
    def generate_batch(self, num_samples: int, worker_id: int = 0) -> List[dict]:
        """Generate batch of samples (for multiprocessing)"""
        samples = []
        
        for i in range(num_samples):
            seed = worker_id * 100000 + i
            sample = self.generate_single_sample(seed)
            
            if sample is not None:
                samples.append(sample)
            
            # Progress report
            if (i + 1) % 1000 == 0:
                avg_reward = np.mean([s['reward'] for s in samples[-100:]]) if samples else 0
                print(f"   Worker {worker_id}: {i+1}/{num_samples} "
                      f"(samples: {len(samples)}, avg_reward: {avg_reward:.2f})")
        
        return samples
    
    def generate_dataset(
        self,
        num_samples: int = 100000,
        num_workers: int = 4,
        output_name: str = None
    ) -> str:
        """
        Generate full dataset with multiprocessing
        
        Returns:
            Path to saved dataset
        """
        print(f"🎓 Generating {num_samples} expert samples (IMPROVED)")
        print(f"   Workers: {num_workers}")
        print(f"   Strategies: trips, pairs, straights, flushes, two-pairs, random")
        
        samples_per_worker = num_samples // num_workers
        
        # Multiprocessing
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                self.generate_batch,
                [(samples_per_worker, i) for i in range(num_workers)]
            )
        
        # Combine
        all_samples = []
        for batch in results:
            all_samples.extend(batch)
        
        print(f"✅ Generated {len(all_samples)} samples")
        
        # Statistics
        rewards = [s['reward'] for s in all_samples]
        valid_count = sum(1 for r in rewards if r > 0)
        bonus_count = sum(1 for r in rewards if r > 50)
        high_quality = sum(1 for r in rewards if r > 10)
        
        print(f"\n📊 Dataset statistics:")
        print(f"   Valid: {valid_count}/{len(all_samples)} ({valid_count/len(all_samples)*100:.1f}%)")
        print(f"   High quality (reward>10): {high_quality}/{len(all_samples)} ({high_quality/len(all_samples)*100:.1f}%)")
        print(f"   With bonus (reward>50): {bonus_count}/{len(all_samples)} ({bonus_count/len(all_samples)*100:.1f}%)")
        print(f"   Reward range: {min(rewards):.2f} - {max(rewards):.2f}")
        print(f"   Reward mean: {np.mean(rewards):.2f}")
        print(f"   Reward median: {np.median(rewards):.2f}")
        
        # Save
        if output_name is None:
            output_name = f"expert_v3_{len(all_samples)}.pkl"
        
        output_path = self.output_dir / output_name
        
        with open(output_path, 'wb') as f:
            pickle.dump(all_samples, f)
        
        print(f"💾 Saved to {output_path}")
        
        return str(output_path)


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate expert training data')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    parser.add_argument('--output-dir', type=str, default='data/training', help='Output directory')
    
    args = parser.parse_args()
    
    generator = ExpertDataGeneratorV3(output_dir=args.output_dir)
    dataset_path = generator.generate_dataset(
        num_samples=args.samples,
        num_workers=args.workers,
        output_name=args.output
    )
    
    print(f"\n✅ Dataset ready at: {dataset_path}")