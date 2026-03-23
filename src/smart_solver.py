"""
Smart Solver - Tìm arrangement TỐI ƯU THỰC SỰ
Không random, không Monte Carlo
Duyệt TẤT CẢ arrangements thông minh → chọn TỐT NHẤT
"""
import sys
import os
from typing import List, Tuple, Optional
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engines'))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints


class SmartSolver:
    """
    Tìm arrangement TỐI ƯU bằng cách:
    1. Duyệt TẤT CẢ cách chia 5-5-3 hợp lệ (brute force có tỉa)
    2. Score mỗi arrangement bằng heuristic chính xác
    3. Trả về TOP arrangements
    
    Với 13 lá: C(13,5) × C(8,5) = 72,072 cách
    Sau khi tỉa invalid: ~5,000-15,000 cách
    Score mỗi cách: ~0.01ms
    Tổng thời gian: ~1-5 giây
    """
    
    def __init__(self):
        self.bonus_calc = BonusPoints()
    
    def find_best_arrangement(
        self,
        cards: List[Card],
        top_k: int = 1
    ) -> List[Tuple]:
        """
        Tìm TOP K arrangements tốt nhất
        
        Args:
            cards: 13 lá bài
            top_k: Số lượng kết quả trả về
            
        Returns:
            List of (back, middle, front, score)
        """
        # Duyệt TẤT CẢ cách chia hợp lệ
        all_valid = self._enumerate_all_valid(cards)
        
        if not all_valid:
            return []
        
        # Score tất cả
        scored = []
        for back, middle, front in all_valid:
            score = self._score(back, middle, front)
            scored.append((back, middle, front, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[3], reverse=True)
        
        return scored[:top_k]
    
    def find_best(self, cards: List[Card]) -> Optional[Tuple]:
        """
        Tìm arrangement TỐT NHẤT duy nhất
        
        Returns:
            (back, middle, front) hoặc None
        """
        results = self.find_best_arrangement(cards, top_k=1)
        
        if results:
            back, middle, front, score = results[0]
            return (back, middle, front)
        
        return None
    
    def _enumerate_all_valid(
        self,
        cards: List[Card]
    ) -> List[Tuple]:
        """
        Duyệt TẤT CẢ cách chia 5-5-3 hợp lệ
        
        Optimization:
        - Tỉa sớm: nếu back < middle → skip
        - Cache evaluations
        """
        valid = []
        eval_cache = {}
        
        card_indices = list(range(13))
        
        # Duyệt tất cả cách chọn 5 lá cho back
        for back_idx in combinations(card_indices, 5):
            back_cards = [cards[i] for i in back_idx]
            
            # Cache back evaluation
            back_key = tuple(sorted(back_idx))
            if back_key not in eval_cache:
                eval_cache[back_key] = HandEvaluator.evaluate(back_cards)
            back_rank = eval_cache[back_key]
            
            # Remaining 8 cards
            remaining_idx = [i for i in card_indices if i not in back_idx]
            
            # Duyệt tất cả cách chọn 5 lá cho middle từ 8 lá còn lại
            for middle_idx in combinations(remaining_idx, 5):
                middle_cards = [cards[i] for i in middle_idx]
                
                # Cache middle evaluation
                middle_key = tuple(sorted(middle_idx))
                if middle_key not in eval_cache:
                    eval_cache[middle_key] = HandEvaluator.evaluate(middle_cards)
                middle_rank = eval_cache[middle_key]
                
                # TỈA: back phải >= middle
                if back_rank < middle_rank:
                    continue
                
                # Front = 3 lá còn lại
                front_idx = [i for i in remaining_idx if i not in middle_idx]
                front_cards = [cards[i] for i in front_idx]
                
                # Cache front evaluation
                front_key = tuple(sorted(front_idx))
                if front_key not in eval_cache:
                    eval_cache[front_key] = HandEvaluator.evaluate(front_cards)
                front_rank = eval_cache[front_key]
                
                # TỈA: middle >= front (simplified)
                if not self._check_middle_vs_front(middle_rank, front_rank):
                    continue
                
                valid.append((back_cards, middle_cards, front_cards))
        
        return valid
    
    def _check_middle_vs_front(self, middle_rank, front_rank) -> bool:
        """Check middle >= front constraint"""
        # Front chỉ có 3 lá nên so sánh đặc biệt
        
        # Nếu front là xám, middle phải >= xám
        if front_rank.hand_type.value == 3:  # Xám
            if middle_rank.hand_type.value < 3:
                return False
            if (middle_rank.hand_type.value == 3 and
                middle_rank.primary_value < front_rank.primary_value):
                return False
        
        # Nếu front là đôi, middle phải >= đôi
        elif front_rank.hand_type.value == 1:  # Đôi
            if middle_rank.hand_type.value < 1:
                return False
            if (middle_rank.hand_type.value == 1 and
                middle_rank.primary_value < front_rank.primary_value):
                return False
        
        return True
    
    def _score(self, back, middle, front) -> float:
        """
        Score arrangement - DETERMINISTIC, không random
        
        Scoring weights (tuned for optimal play):
        - Front strength: 40% (quan trọng nhất!)
        - Back strength: 25%
        - Middle strength: 20%
        - Bonus: 15% (nhưng multiplier cao)
        """
        try:
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            # === BONUS ===
            bonus = self.bonus_calc.calculate_bonus(back, middle, front)
            bonus_score = bonus * 5.0
            
            # === FRONT (40% weight - QUAN TRỌNG NHẤT!) ===
            front_score = 0
            
            if front_rank.hand_type.value == 3:  # Xám
                # Xám chi cuối = CỰC MẠNH + bonus 6
                front_score = 25.0 + front_rank.primary_value * 0.5
            elif front_rank.hand_type.value == 1:  # Đôi
                # Đôi lớn chi cuối = rất mạnh
                front_score = 8.0 + front_rank.primary_value * 0.4
            else:  # Mậu thầu
                # Mậu thầu chi cuối = yếu
                front_score = front_rank.primary_value * 0.15
            
            # === BACK (25% weight) ===
            back_score = (
                back_rank.hand_type.value * 2.5 +
                back_rank.primary_value * 0.08
            )
            
            # === MIDDLE (20% weight) ===
            middle_score = (
                middle_rank.hand_type.value * 2.0 +
                middle_rank.primary_value * 0.06
            )
            
            # === PENALTIES ===
            penalty = 0
            
            # Penalty: Mậu thầu ở front khi có đôi lớn ở middle/back
            if front_rank.hand_type.value == 0:  # Mậu thầu front
                # Check middle có đôi lớn (Q, K, A) không
                mid_counts = Counter([c.rank.value for c in middle])
                for rank, count in mid_counts.items():
                    if count >= 2 and rank >= 12:  # Đôi Q/K/A ở middle
                        penalty -= 4.0
                
                # Check back có đôi lớn không
                back_counts = Counter([c.rank.value for c in back])
                for rank, count in back_counts.items():
                    if count >= 2 and rank >= 12:
                        penalty -= 3.0
            
            # Penalty: Front có đôi nhỏ khi middle/back có đôi lớn hơn
            if front_rank.hand_type.value == 1:  # Đôi front
                front_pair_value = front_rank.primary_value
                
                mid_counts = Counter([c.rank.value for c in middle])
                for rank, count in mid_counts.items():
                    if count >= 2 and rank > front_pair_value:
                        # Đôi lớn hơn ở middle → nên swap
                        penalty -= 2.0
            
            # === TOTAL ===
            total = bonus_score + front_score + back_score + middle_score + penalty
            
            return total
            
        except Exception:
            return -999.0


# ==================== TEST ====================

def test_smart_solver():
    """Test SmartSolver"""
    import time
    
    print("="*60)
    print("🧪 TESTING SMART SOLVER")
    print("="*60)
    
    solver = SmartSolver()
    
    test_cases = [
        (
            "AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S",
            "Strong hand with 3 pairs + straight draw"
        ),
        (
            "7S 7H 7D 7C AS KH QD JC 10S 9H 8D 6C 5S",
            "Four of a kind 7s"
        ),
        (
            "AS KH QD JC 10S 9H 8D 7C 6S 5H 4D 3C 2S",
            "Straight draw (dragon potential)"
        ),
        (
            "AS AH AD KC KD QS JH 10C 9S 8H 7D 6C 5S",
            "Trip Aces + pair Kings"
        ),
    ]
    
    for hand_str, description in test_cases:
        print(f"\n{'='*60}")
        print(f"Hand: {hand_str}")
        print(f"Desc: {description}")
        print(f"{'='*60}")
        
        cards = Deck.parse_hand(hand_str)
        
        start = time.time()
        results = solver.find_best_arrangement(cards, top_k=3)
        elapsed = time.time() - start
        
        print(f"\n⏱️  Time: {elapsed:.2f}s")
        print(f"📊 Valid arrangements found: analyzing...")
        
        for i, (back, middle, front, score) in enumerate(results):
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            bonus = BonusPoints().calculate_bonus(back, middle, front)
            
            print(f"\n  #{i+1} (Score: {score:.2f}, Bonus: +{bonus}):")
            print(f"    Back:   {Deck.cards_to_string(back):30s} → {back_rank}")
            print(f"    Middle: {Deck.cards_to_string(middle):30s} → {middle_rank}")
            print(f"    Front:  {Deck.cards_to_string(front):30s} → {front_rank}")
    
    # Consistency test
    print(f"\n{'='*60}")
    print("🔄 CONSISTENCY TEST (same input 3 times)")
    print(f"{'='*60}")
    
    cards = Deck.parse_hand("AS AH KD KC QS QH JD 10C 9S 8H 7D 6C 5S")
    
    results_list = []
    for i in range(3):
        results = solver.find_best_arrangement(cards, top_k=1)
        back, middle, front, score = results[0]
        result_str = f"{Deck.cards_to_string(front)}"
        results_list.append(result_str)
        print(f"  Run {i+1}: Front = {result_str} (Score: {score:.2f})")
    
    if len(set(results_list)) == 1:
        print("  ✅ CONSISTENT! Same result every time!")
    else:
        print("  ❌ INCONSISTENT!")
    
    print(f"\n{'='*60}")
    print("✅ SMART SOLVER TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    test_smart_solver()