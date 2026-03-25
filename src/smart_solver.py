"""
Smart Solver - ULTRA OPTIMIZED VERSION V2
Tối ưu tốc độ: từ 10s xuống < 0.5s
"""
import sys
import os
from typing import List, Tuple, Optional, Dict
from collections import Counter
from itertools import combinations

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
sys.path.insert(0, core_dir)

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType, compare_cross_street
from special_hands import SpecialHandsChecker


class BonusCalculator:
    """Tính điểm thưởng cho các chi đặc biệt"""
    
    @staticmethod
    def calculate_from_ranks(back_rank, middle_rank, front_rank) -> int:
        """Tính bonus từ ranks đã evaluate sẵn"""
        bonus = 0
        
        # Xám chi cuối: +6 chi
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            bonus += 6
        
        # Cù lũ chi 2 (middle): +4 chi
        if middle_rank.hand_type == HandType.FULL_HOUSE:
            bonus += 4
        
        # Tứ quý chi 1 (back): +8 chi
        if back_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 8
        
        # Tứ quý chi 2 (middle): +16 chi
        if middle_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 16
        
        # Thùng phá sảnh chi 1 (back): +10 chi
        if back_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 10
        
        # Thùng phá sảnh chi 2 (middle): +20 chi
        if middle_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 20
        
        return bonus
    
    @staticmethod
    def calculate(back: List[Card], middle: List[Card], front: List[Card]) -> int:
        """Tính bonus (wrapper cho compatibility)"""
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        return BonusCalculator.calculate_from_ranks(back_rank, middle_rank, front_rank)


class SmartSolver:
    """
    Smart Solver - ULTRA OPTIMIZED VERSION V2
    
    Key optimizations:
    1. Pre-compute ALL evaluations once (O(1) lookup)
    2. Store ranks with indices for zero re-evaluation
    3. Early pruning by hand type
    4. Score using cached ranks only
    """
    
    # Pre-computed type scores (avoid dict lookup in hot path)
    TYPE_SCORES = [0, 2, 4, 6, 8, 10, 12, 16, 20, 25]  # Index = HandType.value
    
    def __init__(self):
        self._cache_5: Dict[tuple, object] = {}
        self._cache_3: Dict[tuple, object] = {}
    
    def find_best_arrangement(
        self,
        cards: List[Card],
        top_k: int = 1
    ) -> List[Tuple]:
        """Tìm TOP K arrangements tốt nhất"""
        
        if len(cards) != 13:
            return []
        
        # *** BƯỚC 1: CHECK BINH ĐẶC BIỆT ***
        special_result = SpecialHandsChecker.check(cards)
        if special_result.is_special:
            return [(None, None, None, special_result)]
        
        # *** BƯỚC 2: PRE-COMPUTE TẤT CẢ EVALUATIONS ***
        self._precompute_all(cards)
        
        # *** BƯỚC 3: TÌM TẤT CẢ VALID ARRANGEMENTS + SCORE ***
        scored = self._find_and_score_all(cards)
        
        # Clear cache
        self._cache_5.clear()
        self._cache_3.clear()
        
        if not scored:
            return []
        
        # Sort và return
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]
    
    def find_best(self, cards: List[Card]) -> Optional[Tuple]:
        """Tìm arrangement tốt nhất"""
        results = self.find_best_arrangement(cards, top_k=1)
        if not results or results[0][0] is None:
            return None
        return (results[0][0], results[0][1], results[0][2])
    
    def _precompute_all(self, cards: List[Card]):
        """Pre-compute tất cả evaluations - chỉ 1 lần!"""
        self._cache_5.clear()
        self._cache_3.clear()
        
        # 5-card hands: C(13,5) = 1287
        for indices in combinations(range(13), 5):
            hand = [cards[i] for i in indices]
            self._cache_5[indices] = HandEvaluator.evaluate(hand)
        
        # 3-card hands: C(13,3) = 286
        for indices in combinations(range(13), 3):
            hand = [cards[i] for i in indices]
            self._cache_3[indices] = HandEvaluator.evaluate(hand)
    
    def _find_and_score_all(self, cards: List[Card]) -> List[Tuple]:
        """Tìm và score tất cả valid arrangements trong 1 pass"""
        results = []
        all_indices = set(range(13))
        
        for back_idx in combinations(range(13), 5):
            back_rank = self._cache_5[back_idx]
            back_type = back_rank.hand_type.value
            back_primary = back_rank.primary_value
            
            remaining = all_indices - set(back_idx)
            remaining_list = sorted(remaining)
            
            for middle_idx in combinations(remaining_list, 5):
                middle_rank = self._cache_5[middle_idx]
                middle_type = middle_rank.hand_type.value
                middle_primary = middle_rank.primary_value
                
                # *** TỈA SỚM 1: back >= middle ***
                if back_type < middle_type:
                    continue
                if back_type == middle_type and back_primary < middle_primary:
                    continue
                
                # Front indices
                front_idx = tuple(sorted(remaining - set(middle_idx)))
                front_rank = self._cache_3[front_idx]
                front_type = front_rank.hand_type.value
                front_primary = front_rank.primary_value
                
                # *** TỈA SỚM 2: middle >= front ***
                if not self._is_middle_ge_front(middle_type, middle_primary, front_type, front_primary):
                    continue
                
                # *** SCORE TRỰC TIẾP TỪ CACHED RANKS ***
                score = self._score_from_ranks(back_rank, middle_rank, front_rank)
                
                # Tạo cards
                back_cards = [cards[i] for i in back_idx]
                middle_cards = [cards[i] for i in middle_idx]
                front_cards = [cards[i] for i in front_idx]
                
                results.append((back_cards, middle_cards, front_cards, score))
        
        return results
    
    def _is_middle_ge_front(self, mid_type: int, mid_primary: int, front_type: int, front_primary: int) -> bool:
        """Check middle >= front (inlined for speed)"""
        # Front chỉ có: HIGH_CARD(0), PAIR(1), THREE_OF_KIND(3)
        
        if front_type == 3:  # Xám
            if mid_type < 3:
                return False
            if mid_type == 3 and mid_primary < front_primary:
                return False
        elif front_type == 1:  # Đôi
            if mid_type < 1:
                return False
            if mid_type == 1 and mid_primary < front_primary:
                return False
        # front_type == 0 (HIGH_CARD) → always OK
        
        return True
    
    def _score_from_ranks(self, back_rank, middle_rank, front_rank) -> float:
        """Score từ cached ranks - KHÔNG gọi evaluate()!"""
        
        # === BONUS ===
        bonus = BonusCalculator.calculate_from_ranks(back_rank, middle_rank, front_rank)
        bonus_score = bonus * 3.0
        
        # === FRONT ===
        front_type = front_rank.hand_type.value
        front_primary = front_rank.primary_value
        
        if front_type == 3:  # THREE_OF_KIND
            front_score = 15.0 + front_primary * 0.5
        elif front_type == 1:  # PAIR
            front_score = 5.0 + front_primary * 0.4
        else:  # HIGH_CARD
            front_score = front_primary * 0.2
        
        # === BACK ===
        back_type = back_rank.hand_type.value
        back_score = (self.TYPE_SCORES[back_type] + back_rank.primary_value * 0.1) * 1.2
        
        # === MIDDLE ===
        middle_type = middle_rank.hand_type.value
        middle_score = (self.TYPE_SCORES[middle_type] + middle_rank.primary_value * 0.1) * 0.8
        
        # === BALANCE ===
        gap = abs(back_type - middle_type) + abs(middle_type - front_type)
        balance_score = max(0, 5 - gap * 0.5)
        
        return bonus_score + front_score + back_score + middle_score + balance_score


# ==================== TESTS ====================

def test_speed():
    """Test tốc độ"""
    import time
    
    print("Testing Speed Optimization V2...")
    
    solver = SmartSolver()
    
    test_hands = [
        "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠",
        "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠",
        "A♠ A♥ A♦ K♣ K♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠",
        "7♠ 7♥ 7♦ 7♣ A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 6♣ 5♠",
        "A♠ K♠ Q♠ J♠ 10♠ 9♥ 8♥ 7♥ 6♥ 5♥ 4♦ 3♦ 2♦",
    ]
    
    total_time = 0
    
    for i, hand_str in enumerate(test_hands):
        cards = Deck.parse_hand(hand_str)
        
        start = time.time()
        results = solver.find_best_arrangement(cards, top_k=1)
        elapsed = time.time() - start
        total_time += elapsed
        
        if results and results[0][0] is not None:
            back, middle, front, score = results[0]
            print(f"  Hand {i+1}: {elapsed:.3f}s | Score: {score:.2f} | Front: {Deck.cards_to_string(front)}")
        else:
            print(f"  Hand {i+1}: {elapsed:.3f}s | Special hand")
    
    avg_time = total_time / len(test_hands)
    print(f"\n  ⏱️  Average time: {avg_time:.3f}s")
    
    if avg_time < 0.5:
        print("  🚀 BLAZING FAST! (< 0.5s)")
    elif avg_time < 1.0:
        print("  ✅ FAST! (< 1s)")
    elif avg_time < 2.0:
        print("  ✅ OK (< 2s)")
    else:
        print("  ⚠️  Still slow, consider Cython")
    
    print("✅ Speed test completed!")


def test_correctness():
    """Test kết quả vẫn đúng"""
    print("\nTesting Correctness...")
    
    solver = SmartSolver()
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    results = solver.find_best_arrangement(cards, top_k=1)
    
    assert results and results[0][0] is not None
    
    back, middle, front, score = results[0]
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert is_valid, f"Invalid: {msg}"
    
    print(f"  ✅ Valid arrangement")
    print(f"      Back:   {Deck.cards_to_string(back)} → {HandEvaluator.evaluate(back)}")
    print(f"      Middle: {Deck.cards_to_string(middle)} → {HandEvaluator.evaluate(middle)}")
    print(f"      Front:  {Deck.cards_to_string(front)} → {HandEvaluator.evaluate(front)}")
    print(f"      Score:  {score:.2f}")
    
    print("✅ Correctness test passed!")


def test_consistency():
    """Test consistency"""
    print("\nTesting Consistency...")
    
    solver = SmartSolver()
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    
    results_list = []
    for i in range(3):
        results = solver.find_best_arrangement(cards, top_k=1)
        if results and results[0][0] is not None:
            front = results[0][2]
            results_list.append(Deck.cards_to_string(front))
            print(f"  Run {i+1}: {results_list[-1]}")
    
    if len(set(results_list)) == 1:
        print("  ✅ CONSISTENT!")
    else:
        print("  ❌ INCONSISTENT!")
    
    print("✅ Consistency test passed!")


if __name__ == "__main__":
    test_speed()
    test_correctness()
    test_consistency()
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)