"""
Comprehensive Model Testing Suite
Test validity, win rate, bonus optimization, and more
"""
import sys
import os
import numpy as np
import random
from collections import defaultdict
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))
sys.path.insert(0, os.path.dirname(__file__))

from card import Card, Deck
from dqn_agent import DQNAgent
from game_theory import GameTheoryEngine, BonusPoints
from evaluator import HandEvaluator
from hand_types import HandType


class ModelTester:
    """Comprehensive testing for trained model"""
    
    def __init__(self, model_path: str):
        print("="*60)
        print("🔬 INITIALIZING MODEL TESTER")
        print("="*60)
        
        self.agent = DQNAgent(state_size=52, action_size=1000, use_dueling=True)
        self.agent.load(model_path)
        
        self.bonus_calc = BonusPoints()
        
        print(f"✅ Model loaded successfully\n")
    
    def test_validity(self, num_tests: int = 1000):
        """
        Test 1: Validity Rate
        Model phải tạo ra arrangements hợp lệ
        """
        print("="*60)
        print(f"TEST 1: VALIDITY RATE ({num_tests} hands)")
        print("="*60)
        
        valid = 0
        invalid = 0
        error_types = defaultdict(int)
        
        for i in range(num_tests):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            # Generate random hand
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # Get ML prediction
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            action = self.agent.select_action(state, epsilon=0.0)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            
            back, middle, front = arrangement
            
            # Validate
            try:
                # Check card counts
                if len(back) != 5:
                    error_types['back_wrong_count'] += 1
                    invalid += 1
                    continue
                
                if len(middle) != 5:
                    error_types['middle_wrong_count'] += 1
                    invalid += 1
                    continue
                
                if len(front) != 3:
                    error_types['front_wrong_count'] += 1
                    invalid += 1
                    continue
                
                # Check all cards used
                all_cards = set(back + middle + front)
                if len(all_cards) != 13:
                    error_types['duplicate_cards'] += 1
                    invalid += 1
                    continue
                
                # Check back >= middle
                back_rank = HandEvaluator.evaluate(back)
                middle_rank = HandEvaluator.evaluate(middle)
                
                if back_rank < middle_rank:
                    error_types['back_weaker_than_middle'] += 1
                    invalid += 1
                    continue
                
                # All checks passed
                valid += 1
                
            except Exception as e:
                error_types[f'exception_{type(e).__name__}'] += 1
                invalid += 1
        
        validity_rate = valid / num_tests * 100
        
        print(f"\n📊 Results:")
        print(f"  ✅ Valid:   {valid:4d}/{num_tests} ({validity_rate:5.1f}%)")
        print(f"  ❌ Invalid: {invalid:4d}/{num_tests} ({100-validity_rate:5.1f}%)")
        
        if error_types:
            print(f"\n  Error breakdown:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {error}: {count}")
        
        # Grade
        if validity_rate >= 95:
            grade = "A+ 🌟"
        elif validity_rate >= 90:
            grade = "A  ⭐"
        elif validity_rate >= 80:
            grade = "B  👍"
        elif validity_rate >= 70:
            grade = "C  ⚠️"
        else:
            grade = "D  ❌"
        
        print(f"\n  Grade: {grade}\n")
        
        return validity_rate
    
    def test_win_rate(self, num_tests: int = 500):
        """
        Test 2: Win Rate vs Random Baseline
        Model phải thắng > 50% vs random player
        """
        print("="*60)
        print(f"TEST 2: WIN RATE VS RANDOM ({num_tests} matchups)")
        print("="*60)
        
        wins = 0
        losses = 0
        ties = 0
        
        chi_wins = defaultdict(int)
        chi_losses = defaultdict(int)
        
        for i in range(num_tests):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            # Generate matchup
            deck = Deck.full_deck()
            my_cards = random.sample(deck, 13)
            remaining = [c for c in deck if c not in my_cards]
            opp_cards = random.sample(remaining, 13)
            
            # My arrangement (ML)
            state = np.zeros(52, dtype=np.float32)
            for card in my_cards:
                state[card.to_index()] = 1.0
            
            action = self.agent.select_action(state, epsilon=0.0)
            my_arr = self.agent.action_encoder.decode_action(action, my_cards)
            
            # Opponent arrangement (random valid)
            opp_arr = self._random_valid_arrangement(opp_cards)
            
            if not opp_arr:
                continue
            
            # Compare
            try:
                my_back_rank = HandEvaluator.evaluate(my_arr[0])
                my_mid_rank = HandEvaluator.evaluate(my_arr[1])
                my_front_rank = HandEvaluator.evaluate(my_arr[2])
                
                opp_back_rank = HandEvaluator.evaluate(opp_arr[0])
                opp_mid_rank = HandEvaluator.evaluate(opp_arr[1])
                opp_front_rank = HandEvaluator.evaluate(opp_arr[2])
                
                # Count chi wins
                my_chi_wins = 0
                
                if my_back_rank > opp_back_rank:
                    my_chi_wins += 1
                    chi_wins['back'] += 1
                elif my_back_rank < opp_back_rank:
                    chi_losses['back'] += 1
                
                if my_mid_rank > opp_mid_rank:
                    my_chi_wins += 1
                    chi_wins['middle'] += 1
                elif my_mid_rank < opp_mid_rank:
                    chi_losses['middle'] += 1
                
                if my_front_rank > opp_front_rank:
                    my_chi_wins += 1
                    chi_wins['front'] += 1
                elif my_front_rank < opp_front_rank:
                    chi_losses['front'] += 1
                
                # Overall result
                if my_chi_wins >= 2:
                    wins += 1
                elif my_chi_wins <= 1:
                    losses += 1
                else:
                    ties += 1
                    
            except Exception as e:
                pass
        
        total_games = wins + losses + ties
        win_rate = wins / total_games * 100 if total_games > 0 else 0
        
        print(f"\n📊 Results:")
        print(f"  🏆 Wins:   {wins:3d}/{total_games} ({win_rate:5.1f}%)")
        print(f"  💀 Losses: {losses:3d}/{total_games} ({losses/total_games*100:5.1f}%)")
        print(f"  🤝 Ties:   {ties:3d}/{total_games}")
        
        print(f"\n  Win rate by chi:")
        for chi in ['back', 'middle', 'front']:
            total_chi = chi_wins[chi] + chi_losses[chi]
            if total_chi > 0:
                chi_wr = chi_wins[chi] / total_chi * 100
                print(f"    {chi:8s}: {chi_wins[chi]:3d}/{total_chi:3d} ({chi_wr:5.1f}%)")
        
        # Grade
        if win_rate >= 70:
            grade = "A+ 🌟"
        elif win_rate >= 60:
            grade = "A  ⭐"
        elif win_rate >= 55:
            grade = "B  👍"
        elif win_rate >= 50:
            grade = "C  ⚠️"
        else:
            grade = "D  ❌"
        
        print(f"\n  Grade: {grade}\n")
        
        return win_rate
    
    def test_bonus_optimization(self, num_tests: int = 1000):
        """
        Test 3: Bonus Optimization
        Model phải tìm được bonus opportunities
        """
        print("="*60)
        print(f"TEST 3: BONUS OPTIMIZATION ({num_tests} hands)")
        print("="*60)
        
        hands_with_bonus = 0
        total_bonus_points = 0
        bonus_breakdown = defaultdict(int)
        
        for i in range(num_tests):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # ML arrangement
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            action = self.agent.select_action(state, epsilon=0.0)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            
            try:
                bonus = self.bonus_calc.calculate_bonus(*arrangement)
                
                if bonus > 0:
                    hands_with_bonus += 1
                    total_bonus_points += bonus
                    bonus_breakdown[bonus] += 1
                    
            except:
                pass
        
        bonus_rate = hands_with_bonus / num_tests * 100
        avg_bonus = total_bonus_points / num_tests
        
        print(f"\n📊 Results:")
        print(f"  Hands with bonus: {hands_with_bonus}/{num_tests} ({bonus_rate:5.1f}%)")
        print(f"  Total bonus points: {total_bonus_points}")
        print(f"  Average bonus per hand: {avg_bonus:.2f}")
        
        if bonus_breakdown:
            print(f"\n  Bonus point breakdown:")
            for points, count in sorted(bonus_breakdown.items(), reverse=True):
                print(f"    {points:2d} points: {count:3d} hands")
        
        # Grade
        if bonus_rate >= 15:
            grade = "A+ 🌟"
        elif bonus_rate >= 10:
            grade = "A  ⭐"
        elif bonus_rate >= 7:
            grade = "B  👍"
        elif bonus_rate >= 5:
            grade = "C  ⚠️"
        else:
            grade = "D  ❌"
        
        print(f"\n  Grade: {grade}\n")
        
        return bonus_rate
    
    def test_hand_strength_distribution(self, num_tests: int = 1000):
        """
        Test 4: Hand Strength Distribution
        Kiểm tra model có ưu tiên đúng không
        """
        print("="*60)
        print(f"TEST 4: HAND STRENGTH DISTRIBUTION ({num_tests} hands)")
        print("="*60)
        
        front_types = defaultdict(int)
        middle_types = defaultdict(int)
        back_types = defaultdict(int)
        
        for i in range(num_tests):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{num_tests}")
            
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            action = self.agent.select_action(state, epsilon=0.0)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            
            try:
                back_rank = HandEvaluator.evaluate(arrangement[0])
                middle_rank = HandEvaluator.evaluate(arrangement[1])
                front_rank = HandEvaluator.evaluate(arrangement[2])
                
                back_types[str(back_rank.hand_type)] += 1
                middle_types[str(middle_rank.hand_type)] += 1
                front_types[str(front_rank.hand_type)] += 1
                
            except:
                pass
        
        print(f"\n📊 Front (3 cards) distribution:")
        for hand_type, count in sorted(front_types.items()):
            pct = count / num_tests * 100
            print(f"  {hand_type:15s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n📊 Middle (5 cards) distribution:")
        for hand_type, count in sorted(middle_types.items()):
            pct = count / num_tests * 100
            print(f"  {hand_type:15s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n📊 Back (5 cards) distribution:")
        for hand_type, count in sorted(back_types.items()):
            pct = count / num_tests * 100
            print(f"  {hand_type:15s}: {count:4d} ({pct:5.1f}%)")
        
        print()
        
        return True
    
    def test_inference_speed(self, num_tests: int = 1000):
        """
        Test 5: Inference Speed
        Model phải đủ nhanh cho real-time use
        """
        print("="*60)
        print(f"TEST 5: INFERENCE SPEED ({num_tests} predictions)")
        print("="*60)
        
        times = []
        
        for _ in range(num_tests):
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            state = np.zeros(52, dtype=np.float32)
            for card in cards:
                state[card.to_index()] = 1.0
            
            start = time.time()
            action = self.agent.select_action(state, epsilon=0.0)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            elapsed = time.time() - start
            
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        predictions_per_sec = 1 / avg_time
        
        print(f"\n📊 Results:")
        print(f"  Average time:   {avg_time*1000:6.2f} ms")
        print(f"  Std deviation:  {std_time*1000:6.2f} ms")
        print(f"  Min time:       {min_time*1000:6.2f} ms")
        print(f"  Max time:       {max_time*1000:6.2f} ms")
        print(f"  Throughput:     {predictions_per_sec:6.0f} predictions/sec")
        
        # Grade
        if avg_time < 0.05:  # < 50ms
            grade = "A+ 🌟"
        elif avg_time < 0.1:  # < 100ms
            grade = "A  ⭐"
        elif avg_time < 0.2:  # < 200ms
            grade = "B  👍"
        elif avg_time < 0.5:  # < 500ms
            grade = "C  ⚠️"
        else:
            grade = "D  ❌"
        
        print(f"\n  Grade: {grade}\n")
        
        return avg_time
    
    def _random_valid_arrangement(self, cards: List[Card]) -> Tuple:
        """Generate random valid arrangement"""
        max_attempts = 100
        
        for _ in range(max_attempts):
            shuffled = cards.copy()
            random.shuffle(shuffled)
            
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            try:
                back_rank = HandEvaluator.evaluate(back)
                middle_rank = HandEvaluator.evaluate(middle)
                
                if back_rank >= middle_rank:
                    return (back, middle, front)
            except:
                pass
        
        # Fallback
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        return (sorted_cards[8:13], sorted_cards[3:8], sorted_cards[:3])
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE MODEL TEST SUITE")
        print("="*60)
        print()
        
        results = {}
        
        # Test 1: Validity
        results['validity'] = self.test_validity(1000)
        
        # Test 2: Win rate
        results['win_rate'] = self.test_win_rate(500)
        
        # Test 3: Bonus optimization
        results['bonus_rate'] = self.test_bonus_optimization(1000)
        
        # Test 4: Distribution
        self.test_hand_strength_distribution(1000)
        
        # Test 5: Speed
        results['avg_speed'] = self.test_inference_speed(1000)
        
        # Overall summary
        print("="*60)
        print("📋 FINAL SUMMARY")
        print("="*60)
        print()
        
        print("Metric                    Score        Grade")
        print("-"*60)
        
        # Validity
        v = results['validity']
        if v >= 95:
            v_grade = "A+"
        elif v >= 90:
            v_grade = "A "
        elif v >= 80:
            v_grade = "B "
        else:
            v_grade = "C "
        print(f"Validity Rate             {v:5.1f}%      {v_grade}  {'✅' if v >= 90 else '⚠️'}")
        
        # Win rate
        w = results['win_rate']
        if w >= 70:
            w_grade = "A+"
        elif w >= 60:
            w_grade = "A "
        elif w >= 55:
            w_grade = "B "
        else:
            w_grade = "C "
        print(f"Win Rate vs Random        {w:5.1f}%      {w_grade}  {'✅' if w >= 55 else '⚠️'}")
        
        # Bonus
        b = results['bonus_rate']
        if b >= 15:
            b_grade = "A+"
        elif b >= 10:
            b_grade = "A "
        elif b >= 7:
            b_grade = "B "
        else:
            b_grade = "C "
        print(f"Bonus Finding Rate        {b:5.1f}%      {b_grade}  {'✅' if b >= 7 else '⚠️'}")
        
        # Speed
        s = results['avg_speed'] * 1000
        if s < 50:
            s_grade = "A+"
        elif s < 100:
            s_grade = "A "
        elif s < 200:
            s_grade = "B "
        else:
            s_grade = "C "
        print(f"Inference Speed           {s:5.1f} ms    {s_grade}  {'✅' if s < 200 else '⚠️'}")
        
        print()
        
        # Overall score
        overall = (
            results['validity'] * 0.35 +  # 35% weight
            results['win_rate'] * 0.35 +   # 35% weight
            results['bonus_rate'] * 0.20 + # 20% weight
            (100 if s < 100 else 50) * 0.10  # 10% weight
        )
        
        print(f"Overall Score:            {overall:5.1f}%")
        print()
        
        if overall >= 80:
            print("🏆 Grade: A - PRODUCTION READY! 🚀")
            print("   ✅ Model is ready for deployment")
        elif overall >= 70:
            print("👍 Grade: B - GOOD, minor improvements recommended")
            print("   💡 Consider training with more data for better results")
        elif overall >= 60:
            print("⚠️  Grade: C - NEEDS IMPROVEMENT")
            print("   🔧 Recommend retraining with better hyperparameters")
        else:
            print("❌ Grade: D - RETRAIN REQUIRED")
            print("   🔄 Model needs significant improvement before deployment")
        
        print()
        print("="*60)
        
        return results


if __name__ == "__main__":
    model_path = "../../data/models/pro_training_v1/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train a model first!")
        exit(1)
    
    tester = ModelTester(model_path)
    results = tester.run_all_tests()