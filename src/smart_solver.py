"""
Smart Solver - ULTRA OPTIMIZED VERSION V3 - FIXED
Tối ưu tốc độ + ĐÚNG LUẬT 100%
"""
import sys
import os
from typing import List, Tuple, Optional, Dict
from itertools import combinations

current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
sys.path.insert(0, core_dir)

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType, HandRank, compare_cross_street
from special_hands import SpecialHandsChecker


class BonusCalculator:
    """Tính điểm thưởng ĐÚNG THEO LUẬT"""
    
    @staticmethod
    def calculate_from_ranks(back_rank: HandRank, middle_rank: HandRank, front_rank: HandRank) -> int:
        """
        Tính bonus từ ranks
        
        LUẬT:
        - Xám chi cuối (front): +6 chi
        - Cù lũ chi giữa (middle): +4 chi
        - Tứ quý chi đầu (back): +8 chi
        - Tứ quý chi giữa (middle): +16 chi
        - Thùng phá sảnh chi đầu (back): +10 chi
        - Thùng phá sảnh chi giữa (middle): +20 chi
        """
        bonus = 0
        
        # Front (3 lá)
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            bonus += 6
        
        # Middle (5 lá)
        if middle_rank.hand_type == HandType.FULL_HOUSE:
            bonus += 4
        elif middle_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 16
        elif middle_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 20
        
        # Back (5 lá)
        if back_rank.hand_type == HandType.FOUR_OF_KIND:
            bonus += 8
        elif back_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonus += 10
        
        return bonus
    
    @staticmethod
    def calculate(back: List[Card], middle: List[Card], front: List[Card]) -> int:
        """Wrapper"""
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        return BonusCalculator.calculate_from_ranks(back_rank, middle_rank, front_rank)
    
    @staticmethod
    def get_bonus_description(back_rank: HandRank, middle_rank: HandRank, front_rank: HandRank) -> str:
        """Mô tả bonus"""
        bonuses = []
        
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            bonuses.append(f"Xám chi cuối (+6)")
        
        if middle_rank.hand_type == HandType.FULL_HOUSE:
            bonuses.append(f"Cù lũ chi giữa (+4)")
        elif middle_rank.hand_type == HandType.FOUR_OF_KIND:
            bonuses.append(f"Tứ quý chi giữa (+16)")
        elif middle_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonuses.append(f"Thùng phá sảnh chi giữa (+20)")
        
        if back_rank.hand_type == HandType.FOUR_OF_KIND:
            bonuses.append(f"Tứ quý chi đầu (+8)")
        elif back_rank.hand_type in [HandType.STRAIGHT_FLUSH, HandType.ROYAL_FLUSH]:
            bonuses.append(f"Thùng phá sảnh chi đầu (+10)")
        
        return ", ".join(bonuses) if bonuses else "Không có bonus"


class SmartSolver:
    """
    Smart Solver V3 - ĐÚNG LUẬT 100%
    
    Optimizations:
    - Pre-compute tất cả evaluations 1 lần
    - Cache với indices để O(1) lookup
    - Early pruning thông minh
    - Scoring ưu tiên bonus
    """
    
    TYPE_SCORES = {
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
    
    def __init__(self):
        self._cache_5: Dict[tuple, HandRank] = {}
        self._cache_3: Dict[tuple, HandRank] = {}
    
    def find_best_arrangement(
        self,
        cards: List[Card],
        top_k: int = 1
    ) -> List[Tuple[Optional[List[Card]], Optional[List[Card]], Optional[List[Card]], float]]:
        """
        Tìm TOP K arrangements tốt nhất
        
        Returns:
            List of (back, middle, front, score)
        """
        if len(cards) != 13:
            return []
        
        # Check binh đặc biệt
        special_result = SpecialHandsChecker.check(cards)
        if special_result.is_special:
            return [(None, None, None, special_result)]
        
        # Pre-compute
        self._precompute_all(cards)
        
        # Find và score
        scored = self._find_and_score_all(cards)
        
        # Clear cache
        self._cache_5.clear()
        self._cache_3.clear()
        
        if not scored:
            return []
        
        scored.sort(key=lambda x: x[3], reverse=True)
        return scored[:top_k]
    
    def find_best(self, cards: List[Card]) -> Optional[Tuple[List[Card], List[Card], List[Card]]]:
        """Tìm arrangement tốt nhất"""
        results = self.find_best_arrangement(cards, top_k=1)
        if not results or results[0][0] is None:
            return None
        return (results[0][0], results[0][1], results[0][2])
    
    def _precompute_all(self, cards: List[Card]):
        """Pre-compute tất cả evaluations"""
        self._cache_5.clear()
        self._cache_3.clear()
        
        # 5-card: C(13,5) = 1287
        for indices in combinations(range(13), 5):
            hand = [cards[i] for i in indices]
            self._cache_5[indices] = HandEvaluator.evaluate(hand)
        
        # 3-card: C(13,3) = 286
        for indices in combinations(range(13), 3):
            hand = [cards[i] for i in indices]
            self._cache_3[indices] = HandEvaluator.evaluate(hand)
    
    def _find_and_score_all(self, cards: List[Card]) -> List[Tuple]:
        """Tìm và score tất cả valid arrangements"""
        results = []
        all_indices = set(range(13))
        
        for back_idx in combinations(range(13), 5):
            back_rank = self._cache_5[back_idx]
            
            remaining = all_indices - set(back_idx)
            remaining_list = sorted(remaining)
            
            for middle_idx in combinations(remaining_list, 5):
                middle_rank = self._cache_5[middle_idx]
                
                # Early pruning: back >= middle
                if not self._is_back_ge_middle(back_rank, middle_rank):
                    continue
                
                # Front
                front_idx = tuple(sorted(remaining - set(middle_idx)))
                front_rank = self._cache_3[front_idx]
                
                # Early pruning: middle >= front
                try:
                    cmp = compare_cross_street(middle_rank, front_rank)
                    if cmp < 0:
                        continue
                except:
                    if not self._is_middle_ge_front_fallback(middle_rank, front_rank):
                        continue
                
                # Score
                score = self._score_from_ranks(back_rank, middle_rank, front_rank)
                
                back_cards = [cards[i] for i in back_idx]
                middle_cards = [cards[i] for i in middle_idx]
                front_cards = [cards[i] for i in front_idx]
                
                results.append((back_cards, middle_cards, front_cards, score))
        
        return results
    
    def _is_back_ge_middle(self, back_rank: HandRank, middle_rank: HandRank) -> bool:
        """Check back >= middle"""
        try:
            return back_rank >= middle_rank
        except:
            if back_rank.hand_type != middle_rank.hand_type:
                return back_rank.hand_type > middle_rank.hand_type
            if back_rank.primary_value != middle_rank.primary_value:
                return back_rank.primary_value > middle_rank.primary_value
            for k1, k2 in zip(back_rank.kickers, middle_rank.kickers):
                if k1 != k2:
                    return k1 > k2
            return True
    
    def _is_middle_ge_front_fallback(self, middle_rank: HandRank, front_rank: HandRank) -> bool:
        """Fallback check middle >= front"""
        mid_type = middle_rank.hand_type.value
        mid_primary = middle_rank.primary_value
        front_type = front_rank.hand_type.value
        front_primary = front_rank.primary_value
        
        if front_type == 3:
            if mid_type < 3:
                return False
            if mid_type == 3 and mid_primary < front_primary:
                return False
        elif front_type == 1:
            if mid_type < 1:
                return False
            if mid_type == 1 and mid_primary < front_primary:
                return False
        
        return True
    
    def _score_from_ranks(self, back_rank: HandRank, middle_rank: HandRank, front_rank: HandRank) -> float:
        """
        Score từ cached ranks
        
        Formula V3:
        Score = Bonus×4.0 + Front×1.8 + Back×1.5 + Middle×1.0 - Balance_Penalty
        """
        
        # Bonus (trọng số CỰC CAO!)
        bonus = BonusCalculator.calculate_from_ranks(back_rank, middle_rank, front_rank)
        bonus_score = bonus * 4.0
        
        # Front (quan trọng!)
        front_type = front_rank.hand_type
        front_primary = front_rank.primary_value
        
        if front_type == HandType.THREE_OF_KIND:
            front_score = 25.0 + front_primary * 1.0
        elif front_type == HandType.PAIR:
            front_score = 10.0 + front_primary * 0.6
        else:
            front_score = front_primary * 0.4
        
        front_score *= 1.8
        
        # Back
        back_type_score = self.TYPE_SCORES.get(back_rank.hand_type, 0)
        back_score = (back_type_score + back_rank.primary_value * 0.2) * 1.5
        
        # Middle
        middle_type_score = self.TYPE_SCORES.get(middle_rank.hand_type, 0)
        middle_score = (middle_type_score + middle_rank.primary_value * 0.15) * 1.0
        
        # Balance
        back_val = back_rank.hand_type.value
        middle_val = middle_rank.hand_type.value
        front_val = front_rank.hand_type.value
        
        gap = abs(back_val - middle_val) + abs(middle_val - front_val)
        balance_penalty = gap * 0.4
        
        return bonus_score + front_score + back_score + middle_score - balance_penalty


if __name__ == "__main__":
    print("smart_solver.py V3 - OK")