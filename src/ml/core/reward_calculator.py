"""
Reward Calculator - FINAL FIXED VERSION
Tính reward ĐÚNG HOÀN TOÀN theo luật Mậu Binh
"""
from typing import List
import sys
import os

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
core_dir = os.path.join(project_root, 'src', 'core')

sys.path.insert(0, core_dir)

from card import Card
from evaluator import HandEvaluator
from hand_types import HandType, HandRank


class RewardCalculator:
    """
    Reward Calculator cho ML training - FINAL VERSION
    
    FIXED:
    - Import paths correct
    - Bonus calculation đúng 100%
    - Validation chặt chẽ
    - NO hardcoded threshold (-50)
    """
    
    def __init__(self):
        """Initialize reward calculator"""
        self.type_scores = {
            HandType.HIGH_CARD: 0,
            HandType.PAIR: 2,
            HandType.TWO_PAIR: 4,
            HandType.THREE_OF_KIND: 6,
            HandType.STRAIGHT: 8,
            HandType.FLUSH: 10,
            HandType.FULL_HOUSE: 12,
            HandType.FOUR_OF_KIND: 16,
            HandType.STRAIGHT_FLUSH: 20,
            HandType.ROYAL_FLUSH: 25,
        }
    
    def calculate_reward(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> float:
        """
        Tính reward tổng
        
        Args:
            back: Chi 1 (5 lá)
            middle: Chi 2 (5 lá)
            front: Chi cuối (3 lá)
        
        Returns:
            reward (float):
                -100.0 nếu invalid (lủng)
                0-200 nếu valid
        """
        # Validate arrangement
        is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
        
        if not is_valid:
            # INVALID → penalty cực lớn
            return -100.0
        
        # Valid → calculate components
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        # Components
        bonus = self._calculate_bonus(back, middle, front)
        strength = self._calculate_strength(back_rank, middle_rank, front_rank)
        balance = self._calculate_balance(back_rank, middle_rank, front_rank)
        
        # Total reward
        # Bonus có trọng số CỰC CAO vì có thể +6, +16, +20 chi!
        reward = (
            bonus * 4.0 +        # Bonus = TOP priority!
            strength * 1.0 +     # Strength
            balance * 0.5        # Balance (minor)
        )
        
        return reward
    
    def _calculate_bonus(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> float:
        """
        Tính bonus ĐÚNG THEO LUẬT Mậu Binh Miền Nam
        
        Returns:
            bonus (int): 0-42
                - Xám chi cuối: +6
                - Cù lũ chi giữa: +4
                - Tứ quý chi đầu: +8
                - Tứ quý chi giữa: +16
                - Thùng phá sảnh chi đầu: +10
                - Thùng phá sảnh chi giữa: +20
                - Max combo: +6 +20 +10 = +36 (hoặc +6 +16 +10 = +32)
        """
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        bonus = 0
        
        # === FRONT (CHI CUỐI - 3 LÁ) ===
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            bonus += 6  # Xám chi cuối
        
        # === MIDDLE (CHI GIỮA - 5 LÁ) ===
        if middle_rank.hand_type == HandType.FULL_HOUSE:
            bonus += 4  # Cù lũ chi giữa
        elif middle_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 16  # Tứ quý chi giữa
        elif middle_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 20  # Thùng phá sảnh chi giữa (bao gồm Royal Flush)
        
        # === BACK (CHI ĐẦU - 5 LÁ) ===
        if back_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 8  # Tứ quý chi đầu
        elif back_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 10  # Thùng phá sảnh chi đầu (bao gồm Royal Flush)
        
        return float(bonus)
    
    def _calculate_strength(
        self,
        back_rank: HandRank,
        middle_rank: HandRank,
        front_rank: HandRank
    ) -> float:
        """
        Tính sức mạnh tổng hợp của arrangement
        
        Returns:
            strength (float): 0-100
        """
        # === FRONT STRENGTH ===
        front_type = front_rank.hand_type
        front_primary = front_rank.primary_value
        
        if front_type == HandType.THREE_OF_KIND:
            # Xám → CỰC MẠNH!
            front_strength = 30.0 + front_primary * 1.5
        elif front_type == HandType.PAIR:
            # Đôi → mạnh trung bình
            front_strength = 15.0 + front_primary * 1.0
        else:
            # Mậu thầu → yếu
            front_strength = front_primary * 0.5
        
        # === BACK STRENGTH ===
        back_type_score = self.type_scores.get(back_rank.hand_type, 0)
        back_strength = back_type_score * 2.0 + back_rank.primary_value * 0.5
        
        # === MIDDLE STRENGTH ===
        middle_type_score = self.type_scores.get(middle_rank.hand_type, 0)
        middle_strength = middle_type_score * 1.5 + middle_rank.primary_value * 0.4
        
        # === WEIGHTED TOTAL ===
        # Front quan trọng nhất (40%)
        # Back thứ 2 (35%)
        # Middle thứ 3 (25%)
        total_strength = (
            front_strength * 0.4 +
            back_strength * 0.35 +
            middle_strength * 0.25
        )
        
        return total_strength
    
    def _calculate_balance(
        self,
        back_rank: HandRank,
        middle_rank: HandRank,
        front_rank: HandRank
    ) -> float:
        """
        Tính độ cân bằng (tránh gap quá lớn → dễ bị rớt)
        
        Returns:
            balance (float): 0-20
                Gap nhỏ → balance cao
                Gap lớn → balance thấp
        """
        back_val = back_rank.hand_type.value
        middle_val = middle_rank.hand_type.value
        front_val = front_rank.hand_type.value
        
        # Tính gap giữa các chi
        gap = abs(back_val - middle_val) + abs(middle_val - front_val)
        
        # Gap càng lớn → balance càng thấp
        # Gap = 0 (perfect balance) → balance = 20
        # Gap = 13 (max) → balance = 0.5 (gần 0)
        balance = max(0, 20.0 - gap * 1.5)
        
        return balance
    
    def get_description(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> str:
        """
        Mô tả chi tiết reward
        
        Returns:
            description (str)
        """
        is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
        
        if not is_valid:
            return f"INVALID: {msg}"
        
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        bonus = self._calculate_bonus(back, middle, front)
        strength = self._calculate_strength(back_rank, middle_rank, front_rank)
        balance = self._calculate_balance(back_rank, middle_rank, front_rank)
        total = self.calculate_reward(back, middle, front)
        
        desc = f"""
Reward Breakdown:
- Bonus:    {bonus:6.1f} (×4.0 = {bonus*4.0:6.1f})
- Strength: {strength:6.1f} (×1.0 = {strength:6.1f})
- Balance:  {balance:6.1f} (×0.5 = {balance*0.5:6.1f})
─────────────────────────────
TOTAL:      {total:6.1f}

Hands:
- Back:   {back_rank}
- Middle: {middle_rank}
- Front:  {front_rank}
"""
        return desc.strip()


# ==================== TESTS ====================

def test_reward_calculator():
    """Test RewardCalculator"""
    print("="*60)
    print("🧪 TESTING REWARD CALCULATOR")
    print("="*60)
    
    from card import Deck
    
    calc = RewardCalculator()
    
    # Test 1: Valid arrangement with bonus
    print("\n📝 Test 1: Valid arrangement (Xám front + Sảnh back)")
    back = Deck.parse_hand("J♦ 10♣ 9♠ 8♥ 7♦")     # Sảnh
    middle = Deck.parse_hand("K♦ K♣ Q♠ Q♥ 6♣")   # Thú
    front = Deck.parse_hand("5♠ 5♥ 5♦")          # Xám
    
    reward = calc.calculate_reward(back, middle, front)
    bonus = calc._calculate_bonus(back, middle, front)
    
    print(f"  Reward: {reward:.2f}")
    print(f"  Bonus: +{bonus:.0f}")
    print(f"  Expected bonus: +6 (Xám chi cuối)")
    
    assert bonus == 6, f"Expected bonus 6, got {bonus}"
    assert reward > 0, f"Expected positive reward, got {reward}"
    print("  ✅ PASS")
    
    # Test 2: Invalid arrangement (lủng)
    print("\n📝 Test 2: Invalid arrangement (lủng)")
    back = Deck.parse_hand("A♠ A♥ K♦ Q♣ J♠")     # Đôi A
    middle = Deck.parse_hand("9♠ 9♥ 9♦ 8♣ 8♠")   # Cù lũ 9-8
    front = Deck.parse_hand("7♠ 7♥ 6♦")          # Đôi 7
    
    reward = calc.calculate_reward(back, middle, front)
    
    print(f"  Reward: {reward:.2f}")
    print(f"  Expected: -100 (invalid)")
    
    assert reward == -100.0, f"Expected -100, got {reward}"
    print("  ✅ PASS")
    
    # Test 3: High bonus combo
    print("\n📝 Test 3: High bonus combo")
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")    # Royal Flush
    middle = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♥")   # Tứ quý 7
    front = Deck.parse_hand("5♠ 5♥ 5♦")          # Xám 5
    
    reward = calc.calculate_reward(back, middle, front)
    bonus = calc._calculate_bonus(back, middle, front)
    
    print(f"  Reward: {reward:.2f}")
    print(f"  Bonus: +{bonus:.0f}")
    print(f"  Expected: +6 (Xám) +16 (Tứ quý middle) +10 (Royal back) = +32")
    
    assert bonus == 32, f"Expected 32, got {bonus}"
    print("  ✅ PASS")
    
    # Test 4: Description
    print("\n📝 Test 4: Description")
    desc = calc.get_description(back, middle, front)
    print(desc)
    print("  ✅ PASS")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_reward_calculator()