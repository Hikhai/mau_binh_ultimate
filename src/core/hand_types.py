"""
Định nghĩa các loại tổ hợp bài
"""
from enum import IntEnum
from typing import NamedTuple, List
from card import Card


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
    primary_value: int      # Giá trị chính (vd: rank của đôi, xám...)
    kickers: List[int]      # Các quân phụ (để phân định khi cùng loại)
    
    def __str__(self):
        kickers_str = f" ({', '.join(map(str, self.kickers))})" if self.kickers else ""
        return f"{self.hand_type} {self.primary_value}{kickers_str}"
    
    def __lt__(self, other):
        """So sánh 2 HandRank"""
        # So sánh loại bài trước
        if self.hand_type != other.hand_type:
            return self.hand_type < other.hand_type
        
        # Cùng loại, so sánh primary value
        if self.primary_value != other.primary_value:
            return self.primary_value < other.primary_value
        
        # Cùng primary value, so sánh kickers
        for k1, k2 in zip(self.kickers, other.kickers):
            if k1 != k2:
                return k1 < k2
        
        return False  # Bằng nhau
    
    def __eq__(self, other):
        return (
            self.hand_type == other.hand_type and
            self.primary_value == other.primary_value and
            self.kickers == other.kickers
        )
    
    def __gt__(self, other):
        return not (self < other or self == other)
    
    def __le__(self, other):
        return self < other or self == other
    
    def __ge__(self, other):
        return self > other or self == other


class SpecialHandType(IntEnum):
    """Các loại thắng trắng"""
    NONE = 0
    THREE_STRAIGHTS = 8        # 3 sảnh
    THREE_FLUSHES = 8          # 3 thùng
    SIX_PAIRS = 8              # 6 đôi
    FIVE_PAIRS_ONE_TRIP = 10   # 5 đôi + 1 xám
    ALL_SAME_SUIT = 10         # Đồng hoa 13 lá
    DRAGON = 50                # Sảnh rồng
    DRAGON_FLUSH = 100         # Sảnh rồng đồng hoa


# ==================== TESTS ====================

def test_hand_rank():
    """Test HandRank comparison"""
    print("Testing HandRank...")
    
    # Test comparison
    pair_aces = HandRank(HandType.PAIR, 14, [13, 12, 11])
    pair_kings = HandRank(HandType.PAIR, 13, [14, 12, 11])
    three_twos = HandRank(HandType.THREE_OF_KIND, 2, [14, 13])
    
    assert pair_aces > pair_kings  # A > K
    assert three_twos > pair_aces  # Xám > Đôi
    
    # Test equal
    pair_aces2 = HandRank(HandType.PAIR, 14, [13, 12, 11])
    assert pair_aces == pair_aces2
    
    # Test kickers
    pair_aces_better = HandRank(HandType.PAIR, 14, [14, 12, 11])
    assert pair_aces_better > pair_aces  # Cùng đôi A, kicker A > K
    
    print("✅ HandRank tests passed!")


if __name__ == "__main__":
    test_hand_rank()
    print("\n✅ All hand_types.py tests passed!")