"""
Compare different solver modes
"""
import sys
import os
import time
from tabulate import tabulate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
from card import Deck
from ultimate_solver import UltimateSolver, SolverMode

def compare_modes(hand_str):
    """Compare all modes on one hand"""
    
    print(f"\n{'='*70}")
    print(f"📊 COMPARING MODES")
    print(f"{'='*70}")
    print(f"Hand: {hand_str}\n")
    
    cards = Deck.parse_hand(hand_str)
    
    results = []
    modes = [
        SolverMode.FAST,
        SolverMode.BALANCED,
        SolverMode.ML_ONLY,
        SolverMode.ACCURATE,
        SolverMode.ULTIMATE
    ]
    
    for mode in modes:
        print(f"Testing {mode.value}...", end=" ")
        
        start = time.time()
        solver = UltimateSolver(cards, mode=mode, verbose=False)
        result = solver.solve()
        elapsed = time.time() - start
        
        results.append({
            'Mode': mode.value.upper(),
            'EV': f"{result.ev:+.2f}",
            'Bonus': f"+{result.bonus}",
            'Scoop %': f"{result.p_scoop*100:.1f}%",
            'Time (s)': f"{elapsed:.2f}",
            'Front': Deck.cards_to_string(result.front)[:15] + "..."
        })
        
        print("✓")
    
    # Display table
    print("\n")
    print(tabulate(results, headers='keys', tablefmt='grid'))
    print()

if __name__ == "__main__":
    # Test with premium hand
    hand = "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S"
    compare_modes(hand)
    
    # Install tabulate if needed: pip install tabulate