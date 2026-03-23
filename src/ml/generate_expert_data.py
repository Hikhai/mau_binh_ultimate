"""
Generate HIGH-QUALITY expert data with valid arrangements only
"""
import sys
import os
import numpy as np
import pickle
from pathlib import Path
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints
from itertools import combinations


class ExpertDataGenerator:
    """Generate expert-quality training data"""
    
    def __init__(self, output_dir: str = "../../data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bonus_calc = BonusPoints()
    
    def generate_valid_arrangements(self, cards):
        """Generate ALL valid arrangements (brute force)"""
        valid = []
        
        # Generate some valid arrangements (sample for speed)
        for _ in range(50):  # Try 50 random arrangements
            shuffled = cards.copy()
            random.shuffle(shuffled)
            
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            try:
                back_rank = HandEvaluator.evaluate(back)
                middle_rank = HandEvaluator.evaluate(middle)
                
                if back_rank >= middle_rank:
                    # Calculate reward
                    bonus = self.bonus_calc.calculate_bonus(back, middle, front)
                    
                    back_strength = back_rank.hand_type.value
                    middle_strength = middle_rank.hand_type.value
                    front_rank = HandEvaluator.evaluate(front)
                    front_strength = front_rank.hand_type.value
                    
                    reward = (
                        bonus * 2.0 +  # Weight bonus higher!
                        back_strength * 0.5 +
                        middle_strength * 0.3 +
                        front_strength * 0.2
                    )
                    
                    valid.append((back, middle, front, reward))
            except:
                pass
        
        return valid
    
    def generate_dataset(self, num_samples: int = 10000):
        """Generate expert dataset"""
        print(f"🎓 Generating {num_samples} EXPERT samples...")
        print(f"   Strategy: Valid arrangements only + Bonus optimization")
        
        data = []
        
        for i in range(num_samples):
            if (i + 1) % 1000 == 0:
                print(f"   Progress: {i+1}/{num_samples}")
            
            # Generate random hand
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # Generate valid arrangements
            valid_arrs = self.generate_valid_arrangements(cards)
            
            if not valid_arrs:
                continue
            
            # Pick BEST arrangement (highest reward)
            best_arr = max(valid_arrs, key=lambda x: x[3])
            back, middle, front, reward = best_arr
            
            # Encode state
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            # Create data point
            data_point = {
                'state': state,
                'arrangement': (back, middle, front),
                'reward': float(reward),
                'ev': float(reward),
                'win_probs': {'scoop': 0.0, 'win_2_of_3': 0.0},
                'metadata': {'strategy': 'expert', 'valid': True}
            }
            
            data.append(data_point)
        
        # Save
        dataset_path = self.output_dir / f"dataset_expert_valid_{len(data)}.pkl"
        
        with open(dataset_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Generated {len(data)} valid samples")
        print(f"💾 Saved to {dataset_path}")
        
        # Stats
        bonuses = [d['reward'] for d in data if d['reward'] > 5]
        print(f"\n📊 Dataset statistics:")
        print(f"   Samples with bonus: {len(bonuses)}/{len(data)} ({len(bonuses)/len(data)*100:.1f}%)")
        print(f"   Average reward: {np.mean([d['reward'] for d in data]):.2f}")
        
        return str(dataset_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    
    gen = ExpertDataGenerator()
    dataset_path = gen.generate_dataset(args.samples)
    
    print(f"\n✅ Expert dataset ready!")
    print(f"   Path: {dataset_path}")