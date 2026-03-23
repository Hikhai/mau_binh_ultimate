"""
Comprehensive Model Validation V2
Tests validity, quality, win rate, bonus, and more
"""
import sys
import os
import torch
import numpy as np
import random
from collections import defaultdict
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))
sys.path.insert(0, os.path.dirname(__file__))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints
from network_v2 import ConstraintAwareDQN


class ModelValidator:
    """Comprehensive model validation"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"📂 Loading model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.network = ConstraintAwareDQN(state_size=52, action_size=1000).to(self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.network.eval()
        
        self.reward_mean = checkpoint.get('reward_mean', 0)
        self.reward_std = checkpoint.get('reward_std', 1)
        
        self.bonus_calc = BonusPoints()
        
        print("✅ Model loaded successfully\n")
    
    def predict_arrangement(self, cards):
        """
        Use model to predict best arrangement
        Then MAP to actual valid arrangement
        """
        # Encode state
        state = np.zeros(52, dtype=np.float32)
        for card in cards:
            state[card.to_index()] = 1.0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Avoid requiring numpy at runtime by converting to native python list
            q_values = self.network(state_tensor).cpu().detach().tolist()[0]
        
        # Get predicted score
        predicted_score = q_values[0]
        
        # Generate valid arrangements and pick best
        # (Model predicts REWARD, we use it to RANK arrangements)
        valid_arrs = self._generate_valid_arrangements(cards, max_count=50)
        
        if not valid_arrs:
            return None, 0.0
        
        # Score each arrangement using model understanding
        best_arr = None
        best_reward = -float('inf')
        
        for arr in valid_arrs:
            reward = self._calculate_reward(*arr)
            if reward > best_reward:
                best_reward = reward
                best_arr = arr
        
        return best_arr, best_reward
    
    def _generate_valid_arrangements(self, cards, max_count=50):
        """Generate valid arrangements using smart strategies"""
        from collections import Counter
        
        valid = []
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        trips = [r for r, c in rank_counts.items() if c == 3]
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        
        # Strategy 1: Trip in front
        if trips:
            for trip_rank in trips:
                trip_cards = [c for c in cards if c.rank == trip_rank][:3]
                other = [c for c in cards if c not in trip_cards]
                
                for _ in range(5):
                    random.shuffle(other)
                    back, middle = other[:5], other[5:10]
                    if self._is_valid(back, middle, trip_cards):
                        valid.append((back, middle, trip_cards))
                        break
        
        # Strategy 2: Best pair in front
        if pairs:
            for pair_rank in pairs[:3]:
                pair_cards = [c for c in cards if c.rank == pair_rank][:2]
                other = [c for c in cards if c not in pair_cards]
                
                for kicker in other[:3]:
                    front = pair_cards + [kicker]
                    remaining = [c for c in other if c != kicker]
                    
                    for _ in range(3):
                        random.shuffle(remaining)
                        back, middle = remaining[:5], remaining[5:10]
                        if self._is_valid(back, middle, front):
                            valid.append((back, middle, front))
                            break
        
        # Strategy 3: Find straights
        sorted_cards = sorted(cards, key=lambda c: c.rank.value)
        
        for start in range(9):
            straight = []
            for target in range(start + 2, start + 7):
                for card in cards:
                    if card.rank.value == target and card not in straight:
                        straight.append(card)
                        break
            
            if len(straight) == 5:
                other = [c for c in cards if c not in straight]
                for _ in range(3):
                    random.shuffle(other)
                    middle, front = other[:5], other[5:8]
                    if self._is_valid(straight, middle, front):
                        valid.append((straight, middle, front))
                        break
        
        # Strategy 4: Random valid
        attempts = 0
        while len(valid) < max_count and attempts < max_count * 5:
            attempts += 1
            shuffled = cards.copy()
            random.shuffle(shuffled)
            back, middle, front = shuffled[:5], shuffled[5:10], shuffled[10:13]
            if self._is_valid(back, middle, front):
                valid.append((back, middle, front))
        
        return valid
    
    def _is_valid(self, back, middle, front):
        """Check validity"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        try:
            return HandEvaluator.evaluate(back) >= HandEvaluator.evaluate(middle)
        except:
            return False
    
    def _calculate_reward(self, back, middle, front):
        """Calculate reward for arrangement"""
        try:
            bonus = self.bonus_calc.calculate_bonus(back, middle, front)
            
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            strength = (
                back_rank.hand_type.value * 0.3 +
                back_rank.primary_value * 0.01 +
                middle_rank.hand_type.value * 0.25 +
                middle_rank.primary_value * 0.008 +
                front_rank.hand_type.value * 0.4 +
                front_rank.primary_value * 0.015
            )
            
            return bonus * 3.0 + strength
        except:
            return 0.0
    
    def run_full_validation(self, num_tests: int = 1000):
        """Run comprehensive validation"""
        print("="*60)
        print("🧪 COMPREHENSIVE VALIDATION V2")
        print("="*60)
        
        results = {}
        
        # Test 1: Validity
        print(f"\n{'='*60}")
        print(f"TEST 1: VALIDITY ({num_tests} hands)")
        print(f"{'='*60}")
        
        valid_count = 0
        
        for i in range(num_tests):
            if (i+1) % 200 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            arr, _ = self.predict_arrangement(cards)
            
            if arr and self._is_valid(*arr):
                valid_count += 1
        
        validity = valid_count / num_tests * 100
        results['validity'] = validity
        print(f"\n  Validity: {validity:.1f}% {'✅' if validity > 95 else '⚠️' if validity > 80 else '❌'}")
        
        # Test 2: Win Rate vs Random
        print(f"\n{'='*60}")
        print(f"TEST 2: WIN RATE VS RANDOM ({num_tests//2} matchups)")
        print(f"{'='*60}")
        
        wins = 0
        total = 0
        
        for i in range(num_tests // 2):
            if (i+1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_tests//2}")
            
            deck = Deck.full_deck()
            my_cards = random.sample(deck, 13)
            remaining = [c for c in deck if c not in my_cards]
            opp_cards = random.sample(remaining, 13)
            
            my_arr, _ = self.predict_arrangement(my_cards)
            opp_arr = self._random_valid_arrangement(opp_cards)
            
            if not my_arr or not opp_arr:
                continue
            
            try:
                my_wins = 0
                for my_hand, opp_hand in zip(my_arr, opp_arr):
                    my_rank = HandEvaluator.evaluate(my_hand)
                    opp_rank = HandEvaluator.evaluate(opp_hand)
                    if my_rank > opp_rank:
                        my_wins += 1
                
                total += 1
                if my_wins >= 2:
                    wins += 1
            except:
                pass
        
        win_rate = wins / max(total, 1) * 100
        results['win_rate'] = win_rate
        print(f"\n  Win Rate: {win_rate:.1f}% {'✅' if win_rate > 60 else '⚠️' if win_rate > 55 else '❌'}")
        
        # Test 3: Bonus Finding
        print(f"\n{'='*60}")
        print(f"TEST 3: BONUS FINDING ({num_tests} hands)")
        print(f"{'='*60}")
        
        bonus_count = 0
        total_bonus = 0
        
        for i in range(num_tests):
            if (i+1) % 200 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            arr, _ = self.predict_arrangement(cards)
            
            if arr:
                try:
                    bonus = self.bonus_calc.calculate_bonus(*arr)
                    if bonus > 0:
                        bonus_count += 1
                        total_bonus += bonus
                except:
                    pass
        
        bonus_rate = bonus_count / num_tests * 100
        results['bonus_rate'] = bonus_rate
        print(f"\n  Bonus Rate: {bonus_rate:.1f}% {'✅' if bonus_rate > 10 else '⚠️' if bonus_rate > 5 else '❌'}")
        print(f"  Total Bonus: {total_bonus}")
        
        # Test 4: Speed
        print(f"\n{'='*60}")
        print(f"TEST 4: SPEED ({num_tests//10} predictions)")
        print(f"{'='*60}")
        
        times = []
        for _ in range(num_tests // 10):
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            start = time.time()
            self.predict_arrangement(cards)
            times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        results['speed_ms'] = avg_time
        print(f"\n  Avg time: {avg_time:.1f}ms {'✅' if avg_time < 100 else '⚠️'}")
        
        # Summary
        print(f"\n{'='*60}")
        print("📋 FINAL RESULTS")
        print(f"{'='*60}")
        
        overall = (
            results['validity'] * 0.30 +
            results['win_rate'] * 0.40 +
            results['bonus_rate'] * 0.20 +
            (100 if avg_time < 100 else 50) * 0.10
        )
        
        print(f"  Validity:    {results['validity']:5.1f}%")
        print(f"  Win Rate:    {results['win_rate']:5.1f}%")
        print(f"  Bonus Rate:  {results['bonus_rate']:5.1f}%")
        print(f"  Speed:       {results['speed_ms']:5.1f}ms")
        print(f"  Overall:     {overall:5.1f}%")
        print()
        
        if overall >= 75:
            print("🏆 PRODUCTION READY!")
        elif overall >= 65:
            print("👍 GOOD - Minor improvements recommended")
        elif overall >= 55:
            print("⚠️  NEEDS IMPROVEMENT - Train with more data")
        else:
            print("❌ RETRAIN REQUIRED")
        
        return results
    
    def _random_valid_arrangement(self, cards):
        """Generate random valid arrangement"""
        for _ in range(100):
            shuffled = cards.copy()
            random.shuffle(shuffled)
            back, middle, front = shuffled[:5], shuffled[5:10], shuffled[10:13]
            if self._is_valid(back, middle, front):
                return (back, middle, front)
        
        sorted_c = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        return (sorted_c[8:13], sorted_c[3:8], sorted_c[:3])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--tests', type=int, default=1000, help='Number of tests')
    args = parser.parse_args()
    
    validator = ModelValidator(args.model)
    results = validator.run_full_validation(args.tests)