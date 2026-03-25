"""
Benchmark - So sánh với baselines
"""
import sys
import os
import time
import random
from typing import List, Dict
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../agent'))

from card import Deck
from ml.agent import MauBinhAgent
from ml.core import RewardCalculator


class Benchmark:
    """
    Benchmark ML agent vs baselines
    """
    
    def __init__(self, model_path: str = None):
        self.ml_agent = MauBinhAgent(model_path=model_path) if model_path else None
        self.reward_calc = RewardCalculator()
    
    def random_baseline(self, cards) -> tuple:
        """Random arrangement (very weak)"""
        shuffled = cards.copy()
        random.shuffle(shuffled)
        return (shuffled[:5], shuffled[5:10], shuffled[10:13])
    
    def greedy_baseline(self, cards) -> tuple:
        """Greedy by rank (baseline)"""
        sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        return (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])
    
    def run_benchmark(self, num_hands: int = 100) -> Dict:
        """
        Run benchmark comparison
        
        Returns:
            {
                'random': {...},
                'greedy': {...},
                'ml': {...}
            }
        """
        print(f"🏁 Running benchmark with {num_hands} hands...")
        
        results = defaultdict(lambda: {
            'total_reward': 0,
            'valid_count': 0,
            'bonus_count': 0,
            'avg_time': 0,
        })
        
        for i in range(num_hands):
            # Random hand
            deck = Deck.full_deck()
            cards = random.sample(deck, 13)
            
            # Test random
            start = time.time()
            arr_random = self.random_baseline(cards)
            time_random = time.time() - start
            
            reward_random = self.reward_calc.calculate_reward(*arr_random)
            if reward_random > -50:
                results['random']['valid_count'] += 1
                results['random']['total_reward'] += reward_random
                if reward_random > 50:
                    results['random']['bonus_count'] += 1
            results['random']['avg_time'] += time_random
            
            # Test greedy
            start = time.time()
            arr_greedy = self.greedy_baseline(cards)
            time_greedy = time.time() - start
            
            reward_greedy = self.reward_calc.calculate_reward(*arr_greedy)
            if reward_greedy > -50:
                results['greedy']['valid_count'] += 1
                results['greedy']['total_reward'] += reward_greedy
                if reward_greedy > 50:
                    results['greedy']['bonus_count'] += 1
            results['greedy']['avg_time'] += time_greedy
            
            # Test ML
            if self.ml_agent and self.ml_agent.network:
                start = time.time()
                arr_ml = self.ml_agent.solve(cards, mode='best')
                time_ml = time.time() - start
                
                reward_ml = self.reward_calc.calculate_reward(*arr_ml)
                if reward_ml > -50:
                    results['ml']['valid_count'] += 1
                    results['ml']['total_reward'] += reward_ml
                    if reward_ml > 50:
                        results['ml']['bonus_count'] += 1
                results['ml']['avg_time'] += time_ml
            
            if (i + 1) % 20 == 0:
                print(f"   Progress: {i+1}/{num_hands}")
        
        # Compute averages
        for method in results:
            valid = max(results[method]['valid_count'], 1)
            results[method]['avg_reward'] = results[method]['total_reward'] / valid
            results[method]['valid_rate'] = results[method]['valid_count'] / num_hands
            results[method]['bonus_rate'] = results[method]['bonus_count'] / valid
            results[method]['avg_time'] = results[method]['avg_time'] / num_hands * 1000  # ms
        
        self._print_comparison(results)
        
        return dict(results)
    
    def _print_comparison(self, results: Dict):
        """Print benchmark comparison"""
        print("\n" + "="*70)
        print("🏆 BENCHMARK COMPARISON")
        print("="*70)
        print(f"{'Method':<15} {'Valid%':>8} {'Avg Reward':>12} {'Bonus%':>8} {'Time(ms)':>10}")
        print("-"*70)
        
        for method in ['random', 'greedy', 'ml']:
            if method not in results or results[method]['valid_count'] == 0:
                continue
            
            r = results[method]
            print(f"{method.capitalize():<15} "
                  f"{r['valid_rate']*100:>7.1f}% "
                  f"{r['avg_reward']:>12.2f} "
                  f"{r['bonus_rate']*100:>7.1f}% "
                  f"{r['avg_time']:>10.2f}")
        
        print("="*70)
        
        # Winner
        if 'ml' in results and results['ml']['valid_count'] > 0:
            if results['ml']['avg_reward'] > results['greedy']['avg_reward']:
                improvement = (results['ml']['avg_reward'] - results['greedy']['avg_reward']) / results['greedy']['avg_reward'] * 100
                print(f"✅ ML Agent wins! +{improvement:.1f}% better than greedy")
            else:
                print("⚠️  ML Agent needs more training")
        else:
            print("ℹ️  ML Agent not tested (no model loaded)")


# ==================== CLI ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark ML agent')
    parser.add_argument('--model', type=str, default=None, help='Path to model (optional)')
    parser.add_argument('--hands', type=int, default=100, help='Number of hands')
    
    args = parser.parse_args()
    
    benchmark = Benchmark(model_path=args.model)
    benchmark.run_benchmark(num_hands=args.hands)