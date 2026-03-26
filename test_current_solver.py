"""
Test current solver performance - Xem nó "ngu" ở đâu
"""
import sys
sys.path.insert(0, 'src')

from core.card import Deck
from ultimate_solver import UltimateSolver, SolverMode
from ml.core.reward_calculator import RewardCalculatorV2
from ml.core.arrangement_validator import ArrangementValidatorV2

def test_current_performance():
    """Test với 10 hands random"""
    
    print("\n" + "="*60)
    print("🧪 TESTING CURRENT SOLVER PERFORMANCE")
    print("="*60)
    
    validator = ArrangementValidatorV2()
    reward_calc = RewardCalculatorV2()
    
    modes = ['fast', 'balanced', 'accurate', 'ultimate']
    
    # Test cases
    test_hands = [
        # Case 1: Hand có potential special (6 đôi)
        "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ J♣ 10♠ 10♥ 9♦ 9♣ 8♠",
        
        # Case 2: Hand có thùng phá sảnh potential
        "A♠ K♠ Q♠ J♠ 10♠ 9♠ 8♠ 2♥ 3♦ 4♣ 5♥ 6♦ 7♣",
        
        # Case 3: Hand balanced
        "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠",
        
        # Case 4: Hand yếu
        "9♠ 8♥ 7♦ 6♣ 5♠ 4♥ 3♦ 2♣ K♠ Q♥ J♦ 10♣ 9♥",
        
        # Case 5: Hand có tứ quý
        "A♠ A♥ A♦ A♣ K♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠",
    ]
    
    results = {}
    
    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"Testing mode: {mode.upper()}")
        print(f"{'─'*60}")
        
        mode_results = []
        
        for i, hand_str in enumerate(test_hands, 1):
            cards = Deck.parse_hand(hand_str)
            
            try:
                solver = UltimateSolver(cards, mode=SolverMode(mode))
                result = solver.solve()
                
                # Validate
                is_valid, msg, metadata = validator.is_valid_detailed(
                    result.back, result.middle, result.front
                )
                
                # Calculate reward với V3
                reward_result = reward_calc.calculate_reward(
                    result.back, result.middle, result.front
                )
                
                print(f"\nHand {i}: {hand_str[:40]}...")
                print(f"  Valid:  {is_valid}")
                print(f"  Bonus:  {reward_result['bonus']:.0f}")
                print(f"  Reward: {reward_result['total_reward']:.2f}")
                print(f"  Time:   {result.computation_time:.3f}s")
                
                if not is_valid:
                    print(f"  ❌ LỦNG: {metadata.get('issue', 'unknown')}")
                
                mode_results.append({
                    'valid': is_valid,
                    'bonus': reward_result['bonus'],
                    'reward': reward_result['total_reward'],
                    'time': result.computation_time,
                })
                
            except Exception as e:
                print(f"\n❌ Hand {i} ERROR: {e}")
                mode_results.append({
                    'valid': False,
                    'bonus': 0,
                    'reward': -200,
                    'time': 0,
                })
        
        results[mode] = mode_results
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY - Avg metrics per mode")
    print("="*60)
    
    for mode, mode_results in results.items():
        valid_rate = sum(r['valid'] for r in mode_results) / len(mode_results)
        avg_bonus = sum(r['bonus'] for r in mode_results) / len(mode_results)
        avg_reward = sum(r['reward'] for r in mode_results) / len(mode_results)
        avg_time = sum(r['time'] for r in mode_results) / len(mode_results)
        
        print(f"\n{mode.upper():12s}:")
        print(f"  Valid rate:  {valid_rate:.1%}")
        print(f"  Avg bonus:   {avg_bonus:.1f}")
        print(f"  Avg reward:  {avg_reward:.1f}")
        print(f"  Avg time:    {avg_time:.3f}s")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    test_current_performance()