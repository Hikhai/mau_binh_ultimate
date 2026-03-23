"""
Test ML integration
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engines'))

from card import Deck
from ultimate_solver import UltimateSolver, SolverMode

def test_ml_integration():
    print("="*60)
    print("🧪 TESTING ML INTEGRATION")
    print("="*60)
    
    # Test hands
    test_cases = [
        "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S",
        "7S 7H 7D 7C AS KH QD JC 10S 9H 8D 6C 5S",
        "AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S"
    ]
    
    for i, hand_str in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {hand_str}")
        print('='*60)
        
        cards = Deck.parse_hand(hand_str)
        
        # Test each mode
        for mode in [SolverMode.FAST, SolverMode.ML_ONLY, SolverMode.ULTIMATE]:
            print(f"\n--- {mode.value.upper()} MODE ---")
            
            solver = UltimateSolver(cards, mode=mode, verbose=True)
            result = solver.solve()
            
            print(f"Back:   {Deck.cards_to_string(result.back)}")
            print(f"Middle: {Deck.cards_to_string(result.middle)}")
            print(f"Front:  {Deck.cards_to_string(result.front)}")
            print(f"EV: {result.ev:+.2f}, Bonus: +{result.bonus}, Time: {result.computation_time:.2f}s")
    
    print("\n" + "="*60)
    print("✅ ML INTEGRATION TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_ml_integration()