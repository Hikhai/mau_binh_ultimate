"""
Định nghĩa các loại tổ hợp bài
"""
from enum import IntEnum
from typing import NamedTuple, List


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
    num_cards: int = 5      # *** THÊM MỚI: số lá (3 hoặc 5) ***
    
    def __str__(self):
        kickers_str = f" ({', '.join(map(str, self.kickers))})" if self.kickers else ""
        card_indicator = f"[{self.num_cards} lá]" if self.num_cards == 3 else ""
        return f"{self.hand_type} {self.primary_value}{kickers_str} {card_indicator}"
    
    def __lt__(self, other):
        """
        So sánh 2 HandRank
        
        *** QUAN TRỌNG ***:
        - Nếu cùng số lá (3-3 hoặc 5-5) → so sánh bình thường
        - Nếu khác số lá (3 vs 5) → KHÔNG SO SÁNH TRỰC TIẾP!
          → Cần dùng compare_cross_street()
        """
        # Nếu khác số lá → raise error (phải dùng hàm đặc biệt)
        if self.num_cards != other.num_cards:
            raise ValueError(
                f"Cannot compare {self.num_cards}-card hand with {other.num_cards}-card hand directly. "
                f"Use compare_cross_street() instead."
            )
        
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
    
    *** LOGIC ĐẶC BIỆT ***:
    - Front chỉ có: HIGH_CARD (0), PAIR (1), THREE_OF_KIND (3)
    - Middle có đầy đủ: 0-9
    
    Quy tắc:
    1. Front = THREE_OF_KIND (xám)
       → Middle phải >= THREE_OF_KIND
       
    2. Front = PAIR (đôi)
       → Middle >= TWO_PAIR (2) → OK
       → Middle = PAIR → so sánh rank
       → Middle = HIGH_CARD → middle YẾU HƠN!
       
    3. Front = HIGH_CARD
       → Middle luôn >= front
    
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
            return -1  # middle yếu hơn
        if middle_type == 3:
            # Cùng xám, so sánh rank
            if rank_5card.primary_value > rank_3card.primary_value:
                return 1
            elif rank_5card.primary_value < rank_3card.primary_value:
                return -1
            else:
                return 0
        # middle_type > 3 (STRAIGHT trở lên)
        return 1
    
    # Case 2: Front = PAIR (đôi)
    if front_type == 1:
        if middle_type >= 2:  # TWO_PAIR trở lên
            return 1
        if middle_type == 1:  # Cùng đôi
            if rank_5card.primary_value > rank_3card.primary_value:
                return 1
            elif rank_5card.primary_value < rank_3card.primary_value:
                return -1
            else:
                return 0
        # middle_type == 0 (HIGH_CARD)
        return -1  # middle yếu hơn
    
    # Case 3: Front = HIGH_CARD (mậu thầu)
    # Middle luôn >= front (vì middle có 5 lá)
    return 1


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
    
    # Test cùng số lá (5-5)
    pair_aces = HandRank(HandType.PAIR, 14, [13, 12, 11], num_cards=5)
    pair_kings = HandRank(HandType.PAIR, 13, [14, 12, 11], num_cards=5)
    three_twos = HandRank(HandType.THREE_OF_KIND, 2, [14, 13], num_cards=5)
    
    assert pair_aces > pair_kings  # A > K
    assert three_twos > pair_aces  # Xám > Đôi
    print("  ✅ 5-card comparison OK")
    
    # Test cùng số lá (3-3)
    pair_aces_3 = HandRank(HandType.PAIR, 14, [13], num_cards=3)
    pair_kings_3 = HandRank(HandType.PAIR, 13, [14], num_cards=3)
    trip_twos_3 = HandRank(HandType.THREE_OF_KIND, 2, [], num_cards=3)
    
    assert pair_aces_3 > pair_kings_3
    assert trip_twos_3 > pair_aces_3
    print("  ✅ 3-card comparison OK")
    
    # Test so sánh khác số lá → phải raise error
    try:
        result = pair_aces > pair_aces_3  # 5 lá vs 3 lá
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Cannot compare" in str(e)
        print("  ✅ Cross-comparison blocked correctly")
    
    print("✅ HandRank tests passed!")


def test_compare_cross_street():
    """Test compare_cross_street()"""
    print("\nTesting compare_cross_street()...")
    
    # Test case 1: Middle = TWO_PAIR, Front = PAIR
    middle = HandRank(HandType.TWO_PAIR, 10, [9, 2], num_cards=5)
    front = HandRank(HandType.PAIR, 14, [13], num_cards=3)  # Đôi A
    
    result = compare_cross_street(middle, front)
    assert result == 1, "TWO_PAIR should be > PAIR"
    print("  ✅ TWO_PAIR > PAIR")
    
    # Test case 2: Middle = PAIR K, Front = PAIR A
    middle = HandRank(HandType.PAIR, 13, [12, 11, 10], num_cards=5)
    front = HandRank(HandType.PAIR, 14, [13], num_cards=3)
    
    result = compare_cross_street(middle, front)
    assert result == -1, "PAIR K should be < PAIR A"
    print("  ✅ PAIR K < PAIR A")
    
    # Test case 3: Middle = THREE_OF_KIND 9, Front = THREE_OF_KIND 2
    middle = HandRank(HandType.THREE_OF_KIND, 9, [14, 13], num_cards=5)
    front = HandRank(HandType.THREE_OF_KIND, 2, [], num_cards=3)
    
    result = compare_cross_street(middle, front)
    assert result == 1, "THREE_OF_KIND 9 should be > THREE_OF_KIND 2"
    print("  ✅ THREE_OF_KIND 9 > THREE_OF_KIND 2")
    
    # Test case 4: Middle = HIGH_CARD, Front = PAIR
    middle = HandRank(HandType.HIGH_CARD, 14, [13, 12, 11, 10], num_cards=5)
    front = HandRank(HandType.PAIR, 2, [14], num_cards=3)
    
    result = compare_cross_street(middle, front)
    assert result == -1, "HIGH_CARD should be < PAIR"
    print("  ✅ HIGH_CARD < PAIR")
    
    print("✅ compare_cross_street() tests passed!")


if __name__ == "__main__":
    test_hand_rank()
    test_compare_cross_street()
    print("\n✅ All hand_types.py tests passed!")