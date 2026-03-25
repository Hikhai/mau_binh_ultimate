"""
Model Validator - Validate trained models
"""
import sys
import os
import random
import numpy as np
from typing import List, Dict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../agent'))

from card import Deck
from ml.agent import MauBinhAgent
from ml.core import RewardCalculator


class ModelValidator:
    """
    Validate trained model với nhiều test cases
    """
    
    def __init__(self, model_path: str):
        self.agent = MauBinhAgent(model_path=model_path)
        self.reward_calc = RewardCalculator()
    
    def validate(self, num_tests: int = 1000) -> Dict:
        """
        Run comprehensive validation
        
        Returns:
            {
                'valid_rate': float,
                'avg_reward': float,
                'bonus_rate': float,
                'perfect_rate': float,
                'stats': dict
            }
        """
        print(f"🧪 Validating model with {num_tests} tests...")
        
        valid_count = 0
        bonus_count = 0
        perfect_count = 0  # reward > 100
        total_reward = 0
        
        rewards = []
        
        for i in range(num_tests):
            # Random hand
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # Solve
            arrangement = self.agent.solve(cards, mode='best')
            
            # Evaluate
            reward = self.reward_calc.calculate_reward(*arrangement)
            
            if reward > -50:
                valid_count += 1
                total_reward += reward
                rewards.append(reward)
                
                if reward > 50:
                    bonus_count += 1
                
                if reward > 100:
                    perfect_count += 1
            
            if (i + 1) % 100 == 0:
                print(f"   Progress: {i+1}/{num_tests}")
        
        # Statistics
        valid_rate = valid_count / num_tests
        avg_reward = total_reward / max(valid_count, 1)
        bonus_rate = bonus_count / max(valid_count, 1)
        perfect_rate = perfect_count / max(valid_count, 1)
        
        stats = {
            'total_tests': num_tests,
            'valid_count': valid_count,
            'bonus_count': bonus_count,
            'perfect_count': perfect_count,
            'min_reward': min(rewards) if rewards else 0,
            'max_reward': max(rewards) if rewards else 0,
            'median_reward': np.median(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
        }
        
        result = {
            'valid_rate': valid_rate,
            'avg_reward': avg_reward,
            'bonus_rate': bonus_rate,
            'perfect_rate': perfect_rate,
            'stats': stats
        }
        
        self._print_results(result)
        
        return result
    
    def _print_results(self, result: Dict):
        """Print validation results"""
        print("\n" + "="*60)
        print("📊 VALIDATION RESULTS")
        print("="*60)
        print(f"Valid Rate:      {result['valid_rate']*100:6.2f}%")
        print(f"Avg Reward:      {result['avg_reward']:8.2f}")
        print(f"Bonus Rate:      {result['bonus_rate']*100:6.2f}%")
        print(f"Perfect Rate:    {result['perfect_rate']*100:6.2f}%")
        print()
        print("Statistics:")
        stats = result['stats']
        print(f"  Min reward:    {stats['min_reward']:8.2f}")
        print(f"  Max reward:    {stats['max_reward']:8.2f}")
        print(f"  Median reward: {stats['median_reward']:8.2f}")
        print(f"  Std reward:    {stats['std_reward']:8.2f}")
        print("="*60)
        
        # Assessment
        if result['valid_rate'] >= 0.99:
            print("✅ EXCELLENT - Model produces valid arrangements")
        elif result['valid_rate'] >= 0.95:
            print("⚠️  GOOD - Some invalid arrangements")
        else:
            print("❌ POOR - Too many invalid arrangements")
        
        if result['avg_reward'] >= 15:
            print("✅ EXCELLENT - High average reward")
        elif result['avg_reward'] >= 10:
            print("✅ GOOD - Decent average reward")
        else:
            print("⚠️  FAIR - Low average reward")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate trained model')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--tests', type=int, default=1000, help='Number of tests')
    
    args = parser.parse_args()
    
    validator = ModelValidator(args.model)
    validator.validate(num_tests=args.tests)