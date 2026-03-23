"""
Benchmark ML model vs traditional solvers
"""
import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))
sys.path.insert(0, os.path.dirname(__file__))

from card import Deck
from dqn_agent import DQNAgent
from game_theory import GameTheoryEngine
import numpy as np


def benchmark():
    print("="*60)
    print("ML MODEL vs TRADITIONAL SOLVER BENCHMARK")
    print("="*60)
    
    # Load ML model
    agent = DQNAgent(state_size=52, action_size=1000, use_dueling=True)
    agent.load("../../data/models/pro_training_v1/best_model.pth")
    
    num_hands = 100
    
    print(f"\nTesting {num_hands} random hands...\n")
    
    # ML timings
    ml_times = []
    
    # Traditional timings
    trad_times = []
    
    for i in range(num_hands):
        # Generate random hand
        deck = Deck.full_deck()
        cards = random.sample(deck, 13)
        
        # Test ML
        state = np.zeros(52, dtype=np.float32)
        for card in cards:
            state[card.to_index()] = 1.0
        
        start = time.time()
        action = agent.select_action(state, epsilon=0.0)
        arrangement = agent.action_encoder.decode_action(action, cards)
        ml_time = time.time() - start
        ml_times.append(ml_time)
        
        # Test Traditional (fast mode)
        start = time.time()
        gt_engine = GameTheoryEngine(cards, verbose=False)
        # Use simple greedy for fair comparison
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        trad_arrangement = (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])
        trad_time = time.time() - start
        trad_times.append(trad_time)
    
    # Results
    print(f"{'Method':<15} {'Avg Time':<12} {'Speed':<15}")
    print("-"*60)
    print(f"{'ML Model':<15} {np.mean(ml_times)*1000:>8.2f} ms   {1/np.mean(ml_times):>10.0f} hands/sec")
    print(f"{'Traditional':<15} {np.mean(trad_times)*1000:>8.2f} ms   {1/np.mean(trad_times):>10.0f} hands/sec")
    print()
    print(f"Speedup: {np.mean(trad_times)/np.mean(ml_times):.2f}x faster")


if __name__ == "__main__":
    benchmark()
