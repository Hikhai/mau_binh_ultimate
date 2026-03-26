"""
Định nghĩa các loại tổ hợp bài - FIXED VERSION
"""
from enum import IntEnum
from typing import NamedTuple, List, Optional


class HandType(IntEnum):
    """Các loại tổ hợp bài - theo thứ tự mạnh yếu"""
    HIGH_CARD = 0        # Mậu thầu
    PAIR = 1             # Đôi
    TWO_PAIR = 2         # Thú (2 đôi)
    THREE_OF_KIND = 3    # Xám
    STRAIGHT = 4         # Sảnh
    FLUSH = 5            # Thùng
    FULL_HOUSE = 6       # Cù lũ
    FOUR_OF_KIND = 7     # Tứ quý
    STRAIGHT_FLUSH = 8   # Thùng phá sảnh
    ROYAL_FLUSH = 9      # Thùng phá sảnh lớn
    
    def __str__(self):
        names = {
            HandType.HIGH_CARD: "Mậu thầu",
            HandType.PAIR: "Đôi",
            HandType.TWO_PAIR: "Thú",
            HandType.THREE_OF_KIND: "Xám",
            HandType.STRAIGHT: "Sảnh",
            HandType.FLUSH: "Thùng",
            HandType.FULL_HOUSE: "Cù lũ",
            HandType.FOUR_OF_KIND: "Tứ quý",
            HandType.STRAIGHT_FLUSH: "Thùng phá sảnh",
            HandType.ROYAL_FLUSH: "Thùng phá sảnh lớn"
        }
        return names[self]


class HandRank(NamedTuple):
    """
    Đại diện cho rank của một tay bài
    Dùng để so sánh 2 tay bài
    """
    hand_type: HandType
    primary_value: int
    kickers: List[int]
    num_cards: int = 5
    
    def __str__(self):
        kickers_str = f" ({', '.join(map(str, self.kickers))})" if self.kickers else ""
        card_indicator = f"[{self.num_cards} lá]" if self.num_cards == 3 else ""
        return f"{self.hand_type} {self.primary_value}{kickers_str} {card_indicator}"
    
    def __lt__(self, other):
        if self.num_cards != other.num_cards:
            raise ValueError(
                f"Cannot compare {self.num_cards}-card hand with {other.num_cards}-card hand directly. "
                f"Use compare_cross_street() instead."
            )
        
        if self.hand_type != other.hand_type:
            return self.hand_type < other.hand_type
        
        if self.primary_value != other.primary_value:
            return self.primary_value < other.primary_value
        
        for k1, k2 in zip(self.kickers, other.kickers):
            if k1 != k2:
                return k1 < k2
        
        return False
    
    def __eq__(self, other):
        return (
            self.hand_type == other.hand_type and
            self.primary_value == other.primary_value and
            self.kickers == other.kickers and
            self.num_cards == other.num_cards
        )
    
    def __gt__(self, other):
        return not (self < other or self == other)
    
    def __le__(self, other):
        return self < other or self == other
    
    def __ge__(self, other):
        return self > other or self == other


def compare_cross_street(rank_5card: HandRank, rank_3card: HandRank) -> int:
    """
    So sánh 5 lá (middle/back) với 3 lá (front)
    
    Returns:
        1 if rank_5card > rank_3card
        0 if equal
        -1 if rank_5card < rank_3card
    """
    if rank_5card.num_cards != 5 or rank_3card.num_cards != 3:
        raise ValueError("Must compare 5-card with 3-card")
    
    front_type = rank_3card.hand_type.value
    middle_type = rank_5card.hand_type.value
    
    # Case 1: Front = THREE_OF_KIND (xám)
    if front_type == 3:
        if middle_type < 3:
            return -1
        if middle_type == 3:
            if rank_5card.primary_value > rank_3card.primary_value:
                return 1
            elif rank_5card.primary_value < rank_3card.primary_value:
                return -1
            else:
                return 0
        return 1
    
    # Case 2: Front = PAIR (đôi)
    if front_type == 1:
        if middle_type >= 2:
            return 1
        if middle_type == 1:
            if rank_5card.primary_value > rank_3card.primary_value:
                return 1
            elif rank_5card.primary_value < rank_3card.primary_value:
                return -1
            else:
                return 0
        return -1
    
    # Case 3: Front = HIGH_CARD
    return 1


class SpecialHandType(IntEnum):
    """Các loại thắng trắng"""
    NONE = 0
    THREE_FLUSHES = 8          # 3 thùng
    SIX_PAIRS = 8              # 6 đôi
    FIVE_PAIRS_ONE_TRIP = 10   # 5 đôi + 1 xám
    ALL_SAME_SUIT = 10         # Đồng hoa 13 lá
    DRAGON = 50                # Sảnh rồng
    DRAGON_FLUSH = 100         # Sảnh rồng đồng hoa


if __name__ == "__main__":
    print("hand_types.py - OK")