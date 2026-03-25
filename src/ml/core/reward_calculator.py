"""
Reward Calculator - ĐÚNG 100% LUẬT MẬU BINH
Version: 2.0 - Production Ready
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../engines'))

from typing import List
from card import Card
from evaluator import HandEvaluator
from hand_types import HandType


class RewardCalculator:
    """
    Tính reward CHÍNH XÁC theo luật Mậu Binh
    
    Luật chi tiết:
    - Chi cuối: Xám → +6 chi/người
    - Chi 2: Cù lũ → +4, Tứ quý → +16, Thùng phá sảnh → +20
    - Chi đầu: Tứ quý → +8, Thùng phá sảnh → +10
    
    Reward = Bonus × 10 + Hand Strength × 1
    """
    
    # Constants
    INVALID_PENALTY = -100.0
    BONUS_WEIGHT = 10.0
    STRENGTH_WEIGHT = 1.0
    
    # Bonus points (theo luật)
    BONUS_FRONT_TRIP = 6
    BONUS_MIDDLE_FULL_HOUSE = 4
    BONUS_MIDDLE_FOUR_KIND = 16
    BONUS_MIDDLE_STRAIGHT_FLUSH = 20
    BONUS_BACK_FOUR_KIND = 8
    BONUS_BACK_STRAIGHT_FLUSH = 10
    
    @staticmethod
    def calculate_reward(
        back: List[Card],
        middle: List[Card],
        front: List[Card],
        num_opponents: int = 3
    ) -> float:
        """
        Tính reward tổng hợp
        
        Args:
            back: 5 lá chi đầu
            middle: 5 lá chi 2
            front: 3 lá chi cuối
            num_opponents: số đối thủ (để scale bonus)
            
        Returns:
            reward: float (cao = tốt, -100 = invalid)
        """
        # STEP 1: VALIDATE
        is_valid, _ = HandEvaluator.is_valid_arrangement(back, middle, front)
        
        if not is_valid:
            return RewardCalculator.INVALID_PENALTY
        
        # STEP 2: BONUS
        bonus = RewardCalculator._calculate_bonus(back, middle, front)
        
        # STEP 3: STRENGTH
        strength = RewardCalculator._calculate_strength(back, middle, front)
        
        # STEP 4: COMBINE
        # Bonus × 10 (vì bonus = tiền thật!)
        # Strength × 1 (chỉ để rank các arrangement cùng bonus)
        reward = bonus * RewardCalculator.BONUS_WEIGHT + strength * RewardCalculator.STRENGTH_WEIGHT
        
        return reward
    
    @staticmethod
    def _calculate_bonus(back: List[Card], middle: List[Card], front: List[Card]) -> int:
        """Tính bonus ĐÚNG luật"""
        bonus = 0
        
        back_eval = HandEvaluator.evaluate(back)
        middle_eval = HandEvaluator.evaluate(middle)
        front_eval = HandEvaluator.evaluate(front)
        
        # CHI CUỐI (front) - CHỈ có xám mới có bonus!
        if front_eval.hand_type == HandType.THREE_OF_KIND:
            bonus += RewardCalculator.BONUS_FRONT_TRIP
        
        # CHI 2 (middle)
        if middle_eval.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += RewardCalculator.BONUS_MIDDLE_STRAIGHT_FLUSH
        elif middle_eval.hand_type == HandType.FOUR_OF_KIND:
            bonus += RewardCalculator.BONUS_MIDDLE_FOUR_KIND
        elif middle_eval.hand_type == HandType.FULL_HOUSE:
            bonus += RewardCalculator.BONUS_MIDDLE_FULL_HOUSE
        
        # CHI ĐẦU (back)
        if back_eval.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += RewardCalculator.BONUS_BACK_STRAIGHT_FLUSH
        elif back_eval.hand_type == HandType.FOUR_OF_KIND:
            bonus += RewardCalculator.BONUS_BACK_FOUR_KIND
        
        return bonus
    
    @staticmethod
    def _calculate_strength(back: List[Card], middle: List[Card], front: List[Card]) -> float:
        """
        Tính hand strength ĐÚNG
        
        Chiến lược:
        - Front (chi cuối) QUAN TRỌNG NHẤT (50%) vì quyết định thắng thua
        - Back (chi đầu) thứ 2 (30%)
        - Middle (chi 2) thứ 3 (20%)
        """
        back_eval = HandEvaluator.evaluate(back)
        middle_eval = HandEvaluator.evaluate(middle)
        front_eval = HandEvaluator.evaluate(front)
        
        # FRONT: HandType chỉ có 0/1/3, primary_value 2-14
        front_type_score = front_eval.hand_type.value * 2.0  # 0/2/6
        front_rank_score = front_eval.primary_value / 14.0    # 0-1
        front_strength = front_type_score + front_rank_score
        
        # MIDDLE: HandType 0-9, primary_value 2-14
        middle_type_score = middle_eval.hand_type.value * 2.0
        middle_rank_score = middle_eval.primary_value / 14.0
        middle_strength = middle_type_score + middle_rank_score
        
        # BACK: HandType 0-9, primary_value 2-14
        back_type_score = back_eval.hand_type.value * 2.0
        back_rank_score = back_eval.primary_value / 14.0
        back_strength = back_type_score + back_rank_score
        
        # WEIGHTED COMBINATION
        total_strength = (
            front_strength * 0.5 +
            back_strength * 0.3 +
            middle_strength * 0.2
        )
        
        return total_strength


# ==================== TESTS ====================

def test_reward_calculator():
    """Test RewardCalculator"""
    print("Testing RewardCalculator...")
    from card import Deck
    
    # Test 1: Valid + Bonus
    back = Deck.parse_hand("A♠ A♥ A♦ K♣ K♠")     # Cù lũ A-K
    middle = Deck.parse_hand("Q♠ Q♥ Q♦ J♣ J♠")   # Cù lũ Q-J → +4
    front = Deck.parse_hand("10♠ 10♥ 10♦")       # Xám 10 → +6
    
    reward = RewardCalculator.calculate_reward(back, middle, front)
    expected_bonus = 4 + 6
    
    print(f"  Test 1: Bonus arrangement")
    print(f"    Bonus: {expected_bonus} → Reward: {reward:.2f}")
    assert reward > 90, f"Expected > 90, got {reward}"
    
    # Test 2: LỦNG
    back = Deck.parse_hand("K♠ K♥ 5♦ 4♣ 2♠")
    middle = Deck.parse_hand("A♠ A♥ A♦ Q♣ Q♠")
    front = Deck.parse_hand("J♠ J♥ 3♦")
    
    reward = RewardCalculator.calculate_reward(back, middle, front)
    print(f"  Test 2: Invalid (LỦNG)")
    print(f"    Reward: {reward:.2f}")
    assert reward == -100.0
    
    # Test 3: Valid no bonus
    back = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠")
    middle = Deck.parse_hand("9♠ 9♥ 8♦ 8♣ 2♠")
    front = Deck.parse_hand("7♠ 7♥ 6♦")
    
    reward = RewardCalculator.calculate_reward(back, middle, front)
    print(f"  Test 3: Valid no bonus")
    print(f"    Reward: {reward:.2f}")
    assert 0 < reward < 50
    
    print("✅ RewardCalculator tests passed!")


if __name__ == "__main__":
    test_reward_calculator()