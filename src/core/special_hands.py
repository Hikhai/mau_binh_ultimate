"""
Special Hands Checker - FIXED VERSION
Bỏ "3 sảnh" (không tồn tại trong thực tế)
"""
from typing import List, Optional
from collections import Counter
from dataclasses import dataclass

from card import Card, Rank, Suit
from evaluator import HandEvaluator
from hand_types import HandType


@dataclass
class SpecialHandResult:
    """Kết quả check binh đặc biệt"""
    is_special: bool
    name: str
    points_per_person: int
    description: str = ""
    
    def __str__(self):
        if not self.is_special:
            return "Không phải binh đặc biệt"
        return f"🎉 {self.name} (+{self.points_per_person} chi/người)"


class SpecialHandsChecker:
    """
    Kiểm tra các binh đặc biệt thắng trắng
    
    FIXED: Bỏ "3 sảnh" (không thể xảy ra vì chi cuối chỉ 3 lá)
    
    Các binh đặc biệt hợp lệ:
    1. Sảnh rồng đồng hoa (2→A cùng chất) → +100 chi/người
    2. Sảnh rồng (2→A không đồng chất) → +50 chi/người
    3. Đồng hoa 13 lá → +10 chi/người
    4. 5 đôi + 1 xám → +10 chi/người
    5. 6 đôi → +8 chi/người
    6. 3 thùng → +8 chi/người
    """
    
    @staticmethod
    def check(cards: List[Card]) -> SpecialHandResult:
        """Kiểm tra 13 lá có phải binh đặc biệt không"""
        if len(cards) != 13:
            return SpecialHandResult(False, "", 0, "Không đủ 13 lá")
        
        # Check theo thứ tự ưu tiên
        result = SpecialHandsChecker._check_dragon_flush(cards)
        if result.is_special:
            return result
        
        result = SpecialHandsChecker._check_dragon(cards)
        if result.is_special:
            return result
        
        result = SpecialHandsChecker._check_all_same_suit(cards)
        if result.is_special:
            return result
        
        result = SpecialHandsChecker._check_five_pairs_one_trip(cards)
        if result.is_special:
            return result
        
        result = SpecialHandsChecker._check_six_pairs(cards)
        if result.is_special:
            return result
        
        result = SpecialHandsChecker._check_three_flushes(cards)
        if result.is_special:
            return result
        
        return SpecialHandResult(False, "", 0, "Không phải binh đặc biệt")
    
    @staticmethod
    def _check_dragon_flush(cards: List[Card]) -> SpecialHandResult:
        """Sảnh rồng đồng hoa: 13 lá từ 2→A cùng chất → +100 chi/người"""
        ranks = sorted([c.rank.value for c in cards])
        suits = [c.suit for c in cards]
        suit_counts = Counter(suits)
        
        is_dragon = (ranks == list(range(2, 15)))
        is_flush = (max(suit_counts.values()) == 13)
        
        if is_dragon and is_flush:
            suit = suits[0]
            return SpecialHandResult(
                True, "Sảnh rồng đồng hoa", 100,
                f"13 lá từ 2→A cùng chất {suit}"
            )
        
        return SpecialHandResult(False, "", 0)
    
    @staticmethod
    def _check_dragon(cards: List[Card]) -> SpecialHandResult:
        """Sảnh rồng: 13 lá từ 2→A → +50 chi/người"""
        ranks = sorted([c.rank.value for c in cards])
        
        if ranks == list(range(2, 15)):
            return SpecialHandResult(
                True, "Sảnh rồng", 50, "13 lá từ 2→A"
            )
        
        return SpecialHandResult(False, "", 0)
    
    @staticmethod
    def _check_all_same_suit(cards: List[Card]) -> SpecialHandResult:
        """Đồng hoa 13 lá: 13 lá cùng chất → +10 chi/người"""
        suits = [c.suit for c in cards]
        suit_counts = Counter(suits)
        
        if max(suit_counts.values()) == 13:
            suit = suits[0]
            return SpecialHandResult(
                True, "Đồng hoa 13 lá", 10,
                f"13 lá cùng chất {suit}"
            )
        
        return SpecialHandResult(False, "", 0)
    
    @staticmethod
    def _check_five_pairs_one_trip(cards: List[Card]) -> SpecialHandResult:
        """5 đôi + 1 xám → +10 chi/người"""
        ranks = [c.rank.value for c in cards]
        rank_counts = Counter(ranks)
        
        pairs = [r for r, count in rank_counts.items() if count == 2]
        trips = [r for r, count in rank_counts.items() if count == 3]
        
        if len(pairs) == 5 and len(trips) == 1:
            return SpecialHandResult(
                True, "Năm đôi một xám", 10,
                f"5 đôi + 1 xám ({Rank(trips[0])})"
            )
        
        return SpecialHandResult(False, "", 0)
    
    @staticmethod
    def _check_six_pairs(cards: List[Card]) -> SpecialHandResult:
        """6 đôi + 1 lá lẻ → +8 chi/người"""
        ranks = [c.rank.value for c in cards]
        rank_counts = Counter(ranks)
        
        pairs = [r for r, count in rank_counts.items() if count == 2]
        singles = [r for r, count in rank_counts.items() if count == 1]
        
        if len(pairs) == 6 and len(singles) == 1:
            return SpecialHandResult(
                True, "Sáu đôi", 8, "6 đôi + 1 lá lẻ"
            )
        
        return SpecialHandResult(False, "", 0)
    
    @staticmethod
    def _check_three_flushes(cards: List[Card]) -> SpecialHandResult:
        """
        3 thùng: Cả 3 chi đều là thùng → +8 chi/người
        
        LƯU Ý: Rất hiếm vì phải tìm arrangement hợp lệ
        """
        from itertools import combinations
        
        for back_indices in combinations(range(13), 5):
            back = [cards[i] for i in back_indices]
            
            back_suits = [c.suit for c in back]
            if len(set(back_suits)) != 1:
                continue
            
            remaining_indices = [i for i in range(13) if i not in back_indices]
            
            for middle_indices in combinations(remaining_indices, 5):
                middle = [cards[i] for i in middle_indices]
                
                middle_suits = [c.suit for c in middle]
                if len(set(middle_suits)) != 1:
                    continue
                
                front_indices = [i for i in remaining_indices if i not in middle_indices]
                front = [cards[i] for i in front_indices]
                
                front_suits = [c.suit for c in front]
                if len(set(front_suits)) != 1:
                    continue
                
                from evaluator import HandEvaluator
                is_valid, _ = HandEvaluator.is_valid_arrangement(back, middle, front)
                
                if is_valid:
                    return SpecialHandResult(
                        True, "Ba thùng", 8, f"Cả 3 chi đều thùng"
                    )
        
        return SpecialHandResult(False, "", 0)


if __name__ == "__main__":
    print("special_hands.py - OK")