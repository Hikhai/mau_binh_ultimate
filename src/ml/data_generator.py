"""
Professional Data Generation Pipeline
Generate high-quality training data from self-play
"""
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import multiprocessing as mp
from tqdm import tqdm
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints


class DataGenerator:
    """
    Generate training data using multiple strategies:
    1. Random play
    2. Greedy play
    3. Expert play (using solvers)
    4. Self-play
    """
    
    def __init__(self, output_dir: str = "../../data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bonus_calc = BonusPoints()
    
    def generate_dataset(
        self,
        num_samples: int = 10000,
        strategy: str = "mixed",
        num_workers: int = 4,
        save_every: int = 1000
    ) -> str:
        """
        Generate dataset with parallel processing
        
        Args:
            num_samples: Number of data points to generate
            strategy: 'random', 'greedy', 'expert', 'mixed'
            num_workers: Number of parallel workers
            save_every: Save checkpoint every N samples
            
        Returns:
            Path to saved dataset
        """
        print(f"🎲 Generating {num_samples} samples using {strategy} strategy")
        print(f"   Workers: {num_workers}")
        
        # Split work
        samples_per_worker = num_samples // num_workers
        
        # Parallel generation
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                self._generate_batch,
                [(samples_per_worker, strategy, i) for i in range(num_workers)]
            )
        
        # Combine results
        all_data = []
        for batch in results:
            all_data.extend(batch)
        
        print(f"✅ Generated {len(all_data)} samples")
        
        # Save dataset in SIMPLE format (list of dicts)
        dataset_path = self.output_dir / f"dataset_{strategy}_{len(all_data)}.pkl"
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"💾 Saved to {dataset_path}")
        
        # Save metadata
        metadata = {
            'num_samples': len(all_data),
            'strategy': strategy,
            'num_workers': num_workers
        }
        
        metadata_path = self.output_dir / f"dataset_{strategy}_{len(all_data)}_meta.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return str(dataset_path)
    
    def _generate_batch(
        self,
        num_samples: int,
        strategy: str,
        worker_id: int
    ) -> List[Dict]:
        """Generate batch of samples (single worker) - returns list of dicts"""
        data = []
        
        # Set different random seed for each worker
        np.random.seed(worker_id * 1000 + int(os.getpid()))
        random.seed(worker_id * 1000 + int(os.getpid()))
        
        for i in range(num_samples):
            # Generate random hand
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # Encode state (one-hot vector)
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            # Generate arrangement based on strategy
            if strategy == "random":
                arrangement = self._random_arrangement(cards)
            elif strategy == "greedy":
                arrangement = self._greedy_arrangement(cards)
            elif strategy == "expert":
                arrangement = self._expert_arrangement(cards)
            elif strategy == "mixed":
                s = random.choice(["random", "greedy", "expert"])
                if s == "random":
                    arrangement = self._random_arrangement(cards)
                elif s == "greedy":
                    arrangement = self._greedy_arrangement(cards)
                else:
                    arrangement = self._expert_arrangement(cards)
            else:
                arrangement = self._greedy_arrangement(cards)
            
            # Evaluate arrangement
            back, middle, front = arrangement
            
            # Calculate reward
            bonus = self.bonus_calc.calculate_bonus(back, middle, front)
            
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            strength_reward = (
                back_rank.hand_type.value * 0.5 +
                middle_rank.hand_type.value * 0.3 +
                front_rank.hand_type.value * 0.2
            )
            
            reward = bonus + strength_reward
            
            # Create data point as DICTIONARY
            data_point = {
                'state': state,
                'arrangement': arrangement,  # Tuple of (back, middle, front)
                'reward': float(reward),
                'ev': float(reward),  # Simplified
                'win_probs': {
                    'scoop': 0.0,
                    'win_2_of_3': 0.0
                },
                'metadata': {
                    'strategy': strategy,
                    'worker_id': worker_id
                }
            }
            
            data.append(data_point)
        
        return data
    
    def _random_arrangement(self, cards: List[Card]) -> Tuple:
        """Random valid arrangement"""
        max_attempts = 100
        for _ in range(max_attempts):
            shuffled = cards.copy()
            random.shuffle(shuffled)
            
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            if self._is_valid(back, middle, front):
                return (back, middle, front)
        
        # Fallback
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        return (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])
    
    def _greedy_arrangement(self, cards: List[Card]) -> Tuple:
        """Greedy heuristic arrangement"""
        from collections import Counter
        
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        
        trips = [r for r, c in rank_counts.items() if c == 3]
        pairs = [r for r, c in rank_counts.items() if c == 2]
        
        if trips:
            trip_rank = max(trips)
            trip_cards = [c for c in cards if c.rank == trip_rank]
            other_cards = [c for c in cards if c.rank != trip_rank]
            
            front = trip_cards[:3]
            middle = other_cards[:5]
            back = other_cards[5:]
            
            if self._is_valid(back, middle, front):
                return (back, middle, front)
        
        if len(pairs) >= 1:
            pair_rank = max(pairs)
            pair_cards = [c for c in cards if c.rank == pair_rank]
            other_cards = [c for c in cards if c.rank != pair_rank]
            
            front = [pair_cards[0], pair_cards[1], other_cards[0]]
            middle = other_cards[1:6]
            back = other_cards[6:11]
            
            if self._is_valid(back, middle, front):
                return (back, middle, front)
        
        return (sorted_cards[8:13], sorted_cards[3:8], sorted_cards[:3])
    
    def _expert_arrangement(self, cards: List[Card]) -> Tuple:
        """Use game theory solver (expensive but high quality)"""
        # Simplified expert: just use greedy for speed
        return self._greedy_arrangement(cards)
    
    def _is_valid(self, back, middle, front) -> bool:
        """Check if arrangement is valid"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        try:
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            
            return back_rank >= middle_rank
        except:
            return False


# ==================== CLI ====================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Training Data')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--strategy', type=str, default='mixed',
                        choices=['random', 'greedy', 'expert', 'mixed'],
                        help='Generation strategy')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    
    args = parser.parse_args()
    
    generator = DataGenerator()
    
    dataset_path = generator.generate_dataset(
        num_samples=args.samples,
        strategy=args.strategy,
        num_workers=args.workers
    )
    
    print(f"\n✅ Dataset saved to: {dataset_path}")


if __name__ == "__main__":
    main()