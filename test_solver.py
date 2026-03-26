"""
Test Script - Test Ultimate Solver V3
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'src/core'))

from card import Deck
from ultimate_solver import UltimateSolver, SolverMode, get_available_modes


def test_normal_hand():
    """Test bài thường"""
    print("="*60)
    print("🧪 TEST NORMAL HAND")
    print("="*60)
    
    # Bài test
    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    cards = Deck.parse_hand(hand_str)
    
    print(f"\n📋 Input: {hand_str}\n")
    
    # Test balanced mode
    print("🔹 Mode: BALANCED")
    solver = UltimateSolver(cards, mode=SolverMode.BALANCED, verbose=False)
    result = solver.solve()
    print(result)
    
    # Test hybrid mode (nếu có)
    available_modes = get_available_modes()
    if 'ml_hybrid' in available_modes:
        print("\n🔹 Mode: ML_HYBRID")
        solver = UltimateSolver(cards, mode=SolverMode.ML_HYBRID, verbose=True)
        result = solver.solve()
        print(result)


def test_special_hand():
    """Test binh đặc biệt"""
    print("\n" + "="*60)
    print("🧪 TEST SPECIAL HAND")
    print("="*60)
    
    # Sảnh rồng
    hand_str = "2♠ 3♥ 4♦ 5♣ 6♠ 7♥ 8♦ 9♣ 10♠ J♥ Q♦ K♣ A♠"
    cards = Deck.parse_hand(hand_str)
    
    print(f"\n📋 Input: {hand_str}\n")
    
    solver = UltimateSolver(cards, mode=SolverMode.FAST, verbose=False)
    result = solver.solve()
    print(result)


def test_all_modes():
    """Test tất cả modes"""
    print("\n" + "="*60)
    print("🧪 TEST ALL MODES")
    print("="*60)
    
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    cards = Deck.parse_hand(hand_str)
    
    print(f"\n📋 Input: {hand_str}\n")
    
    available = get_available_modes()
    print(f"Available modes: {available}\n")
    
    for mode_str in ['fast', 'balanced', 'accurate', 'ultimate']:
        mode = SolverMode(mode_str)
        solver = UltimateSolver(cards, mode=mode, verbose=False)
        result = solver.solve()
        
        if result.back:
            print(f"✅ {mode_str.upper():12s} | Score: {result.total_score:6.2f} | "
                  f"Bonus: +{result.bonus:2d} | Time: {result.computation_time:.3f}s")
        else:
            print(f"🎉 {mode_str.upper():12s} | SPECIAL HAND")


if __name__ == "__main__":
    test_normal_hand()
    test_special_hand()
    test_all_modes()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)