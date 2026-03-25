"""
Hand Evaluator - Đánh giá tổ hợp bài
Đây là core của toàn bộ hệ thống!

*** FIXED VERSION ***
- Thêm num_cards vào HandRank
- Thêm compare_hands() cho cross-street comparison
- Thêm is_valid_arrangement() để validate
"""
from typing import List, Optional, Tuple
from collections import Counter
from card import Card, Rank
from hand_types import HandType, HandRank, compare_cross_street


class HandEvaluator:
    """
    Đánh giá tổ hợp bài
    Hỗ trợ cả 3 lá (chi cuối) và 5 lá (chi 1, 2)
    """
    
    @staticmethod
    def evaluate(cards: List[Card]) -> HandRank:
        """
        Đánh giá một tay bài
        
        Args:
            cards: List của 3 hoặc 5 Card objects
            
        Returns:
            HandRank object (với num_cards đúng!)
        """
        num_cards = len(cards)
        
        if num_cards == 3:
            return HandEvaluator._evaluate_3_cards(cards)
        elif num_cards == 5:
            return HandEvaluator._evaluate_5_cards(cards)
        else:
            raise ValueError(f"Invalid number of cards: {num_cards}. Expected 3 or 5.")
    
    @staticmethod
    def _evaluate_3_cards(cards: List[Card]) -> HandRank:
        """
        Đánh giá chi cuối (3 lá)
        
        Chi cuối CHỈ CÓ 3 loại:
        - HIGH_CARD (mậu thầu)
        - PAIR (đôi)
        - THREE_OF_KIND (xám)
        """
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        # Xám (Three of a kind)
        if 3 in rank_counts.values():
            trip_rank = max(r for r, count in rank_counts.items() if count == 3)
            return HandRank(
                hand_type=HandType.THREE_OF_KIND,
                primary_value=int(trip_rank),
                kickers=[],
                num_cards=3  # *** QUAN TRỌNG ***
            )
        
        # Đôi (Pair)
        if 2 in rank_counts.values():
            pair_rank = max(r for r, count in rank_counts.items() if count == 2)
            kicker = max(r for r in ranks if r != pair_rank)
            return HandRank(
                hand_type=HandType.PAIR,
                primary_value=int(pair_rank),
                kickers=[int(kicker)],
                num_cards=3  # *** QUAN TRỌNG ***
            )
        
        # Mậu thầu (High card)
        sorted_ranks = sorted(ranks, reverse=True)
        return HandRank(
            hand_type=HandType.HIGH_CARD,
            primary_value=int(sorted_ranks[0]),
            kickers=[int(r) for r in sorted_ranks[1:]],
            num_cards=3  # *** QUAN TRỌNG ***
        )
    
    @staticmethod
    def _evaluate_5_cards(cards: List[Card]) -> HandRank:
        """Đánh giá chi 1 hoặc chi 2 (5 lá)"""
        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Check flush và straight
        is_flush = max(suit_counts.values()) == 5
        is_straight, straight_high = HandEvaluator._check_straight(ranks)
        
        # Royal Flush: A-K-Q-J-10 cùng chất
        if is_flush and is_straight and straight_high == Rank.ACE:
            return HandRank(HandType.ROYAL_FLUSH, int(straight_high), [], num_cards=5)
        
        # Straight Flush: Sảnh cùng chất
        if is_flush and is_straight:
            return HandRank(HandType.STRAIGHT_FLUSH, int(straight_high), [], num_cards=5)
        
        # Four of a Kind: Tứ quý
        if 4 in rank_counts.values():
            quad_rank = max(r for r, count in rank_counts.items() if count == 4)
            kicker = max(r for r, count in rank_counts.items() if count == 1)
            return HandRank(HandType.FOUR_OF_KIND, int(quad_rank), [int(kicker)], num_cards=5)
        
        # Full House: Cù lũ
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = max(r for r, count in rank_counts.items() if count == 3)
            pair_rank = max(r for r, count in rank_counts.items() if count == 2)
            return HandRank(HandType.FULL_HOUSE, int(trip_rank), [int(pair_rank)], num_cards=5)
        
        # Flush: Thùng
        if is_flush:
            sorted_ranks = sorted(ranks, reverse=True)
            return HandRank(
                HandType.FLUSH,
                int(sorted_ranks[0]),
                [int(r) for r in sorted_ranks[1:]],
                num_cards=5
            )
        
        # Straight: Sảnh
        if is_straight:
            return HandRank(HandType.STRAIGHT, int(straight_high), [], num_cards=5)
        
        # Three of a Kind: Xám
        if 3 in rank_counts.values():
            trip_rank = max(r for r, count in rank_counts.items() if count == 3)
            kickers = sorted(
                [r for r in ranks if r != trip_rank],
                reverse=True
            )
            return HandRank(
                HandType.THREE_OF_KIND,
                int(trip_rank),
                [int(k) for k in kickers],
                num_cards=5
            )
        
        # Two Pair: Thú
        pairs = [r for r, count in rank_counts.items() if count == 2]
        if len(pairs) == 2:
            pairs_sorted = sorted(pairs, reverse=True)
            kicker = max(r for r, count in rank_counts.items() if count == 1)
            return HandRank(
                HandType.TWO_PAIR,
                int(pairs_sorted[0]),
                [int(pairs_sorted[1]), int(kicker)],
                num_cards=5
            )
        
        # Pair: Đôi
        if 2 in rank_counts.values():
            pair_rank = max(r for r, count in rank_counts.items() if count == 2)
            kickers = sorted(
                [r for r in ranks if r != pair_rank],
                reverse=True
            )
            return HandRank(
                HandType.PAIR,
                int(pair_rank),
                [int(k) for k in kickers],
                num_cards=5
            )
        
        # High Card: Mậu thầu
        sorted_ranks = sorted(ranks, reverse=True)
        return HandRank(
            HandType.HIGH_CARD,
            int(sorted_ranks[0]),
            [int(r) for r in sorted_ranks[1:]],
            num_cards=5
        )
    
    @staticmethod
    def _check_straight(ranks: List[Rank]) -> Tuple[bool, Optional[Rank]]:
        """
        Kiểm tra xem có phải sảnh không
        
        Returns:
            (is_straight, high_rank)
        """
        unique_ranks = sorted(set(ranks))
        
        # Cần đúng 5 lá khác nhau
        if len(unique_ranks) != 5:
            return False, None
        
        # Check sảnh thông thường (5 lá liên tiếp)
        if unique_ranks[-1].value - unique_ranks[0].value == 4:
            return True, unique_ranks[-1]
        
        # Check sảnh A-2-3-4-5 (wheel straight)
        if unique_ranks == [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.ACE]:
            # Trong trường hợp này, 5 là quân cao nhất
            return True, Rank.FIVE
        
        return False, None
    
    @staticmethod
    def compare(hand1: List[Card], hand2: List[Card]) -> int:
        """
        So sánh 2 tay bài CÙNG SỐ LÁ
        
        Returns:
            1 nếu hand1 > hand2
            -1 nếu hand1 < hand2
            0 nếu bằng nhau
        """
        rank1 = HandEvaluator.evaluate(hand1)
        rank2 = HandEvaluator.evaluate(hand2)
        
        # Phải cùng số lá
        if len(hand1) != len(hand2):
            raise ValueError(
                f"Cannot compare {len(hand1)}-card hand with {len(hand2)}-card hand. "
                f"Use compare_cross_street() for different card counts."
            )
        
        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        else:
            return 0
    
    @staticmethod
    def compare_hands(hand_5: List[Card], hand_3: List[Card]) -> int:
        """
        *** HÀM MỚI ***
        So sánh 5 lá (middle/back) với 3 lá (front)
        
        Returns:
            1 nếu hand_5 > hand_3
            0 nếu bằng nhau
            -1 nếu hand_5 < hand_3
        """
        if len(hand_5) != 5 or len(hand_3) != 3:
            raise ValueError("hand_5 must have 5 cards, hand_3 must have 3 cards")
        
        rank_5 = HandEvaluator.evaluate(hand_5)
        rank_3 = HandEvaluator.evaluate(hand_3)
        
        return compare_cross_street(rank_5, rank_3)
    
    @staticmethod
    def is_valid_arrangement(
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> Tuple[bool, str]:
        """
        *** HÀM MỚI ***
        Validate một cách xếp bài theo luật Mậu Binh
        
        Luật:
        1. back = 5 lá, middle = 5 lá, front = 3 lá
        2. back >= middle (5 lá so với 5 lá)
        3. middle >= front (5 lá so với 3 lá - dùng compare_cross_street)
        4. Tổng = 13 lá, không trùng
        
        Returns:
            (is_valid, error_message)
        """
        # Check số lượng lá
        if len(back) != 5:
            return False, f"Back phải có 5 lá, hiện có {len(back)}"
        if len(middle) != 5:
            return False, f"Middle phải có 5 lá, hiện có {len(middle)}"
        if len(front) != 3:
            return False, f"Front phải có 3 lá, hiện có {len(front)}"
        
        # Check không trùng lá
        all_cards = back + middle + front
        card_indices = [c.to_index() for c in all_cards]
        if len(set(card_indices)) != 13:
            return False, "Có lá bài bị trùng"
        
        # Evaluate
        try:
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
        except Exception as e:
            return False, f"Lỗi đánh giá bài: {e}"
        
        # Check back >= middle (cùng 5 lá)
        if back_rank < middle_rank:
            return False, f"LỦNG: Back ({back_rank}) < Middle ({middle_rank})"
        
        # Check middle >= front (5 lá vs 3 lá)
        cross_compare = compare_cross_street(middle_rank, front_rank)
        if cross_compare < 0:
            return False, f"LỦNG: Middle ({middle_rank}) < Front ({front_rank})"
        
        return True, "OK"
    
    @staticmethod
    def get_hand_description(cards: List[Card]) -> str:
        """Mô tả tay bài bằng tiếng Việt"""
        rank = HandEvaluator.evaluate(cards)
        return str(rank)


# ==================== TESTS ====================

def test_evaluator_3_cards():
    """Test evaluator với 3 lá"""
    print("Testing 3-card evaluation...")
    from card import Deck
    
    # Test xám
    hand = Deck.parse_hand("A♠ A♥ A♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.THREE_OF_KIND
    assert rank.primary_value == 14
    assert rank.num_cards == 3  # *** KIỂM TRA MỚI ***
    print(f"  Xám A: {rank}")
    
    # Test đôi
    hand = Deck.parse_hand("K♠ K♥ 2♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.PAIR
    assert rank.primary_value == 13
    assert rank.kickers == [2]
    assert rank.num_cards == 3  # *** KIỂM TRA MỚI ***
    print(f"  Đôi K: {rank}")
    
    # Test mậu thầu
    hand = Deck.parse_hand("A♠ K♥ Q♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.HIGH_CARD
    assert rank.primary_value == 14
    assert rank.num_cards == 3  # *** KIỂM TRA MỚI ***
    print(f"  Mậu thầu A: {rank}")
    
    print("✅ 3-card evaluation tests passed!")


def test_evaluator_5_cards():
    """Test evaluator với 5 lá"""
    print("\nTesting 5-card evaluation...")
    from card import Deck
    
    # Test Royal Flush
    hand = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.ROYAL_FLUSH
    assert rank.num_cards == 5
    print(f"  Royal Flush: {rank}")
    
    # Test Straight Flush
    hand = Deck.parse_hand("9♥ 8♥ 7♥ 6♥ 5♥")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT_FLUSH
    assert rank.primary_value == 9
    assert rank.num_cards == 5
    print(f"  Straight Flush: {rank}")
    
    # Test Four of a Kind
    hand = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FOUR_OF_KIND
    assert rank.primary_value == 7
    assert rank.num_cards == 5
    print(f"  Tứ quý 7: {rank}")
    
    # Test Full House
    hand = Deck.parse_hand("K♠ K♥ K♦ 5♣ 5♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FULL_HOUSE
    assert rank.primary_value == 13
    assert rank.num_cards == 5
    print(f"  Cù lũ K-5: {rank}")
    
    # Test Flush
    hand = Deck.parse_hand("A♦ J♦ 9♦ 5♦ 3♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FLUSH
    assert rank.num_cards == 5
    print(f"  Thùng A: {rank}")
    
    # Test Straight
    hand = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT
    assert rank.primary_value == 14
    assert rank.num_cards == 5
    print(f"  Sảnh A: {rank}")
    
    # Test Wheel Straight (A-2-3-4-5)
    hand = Deck.parse_hand("A♠ 2♥ 3♦ 4♣ 5♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT
    assert rank.primary_value == 5  # 5 is high in wheel
    assert rank.num_cards == 5
    print(f"  Sảnh thấp (wheel): {rank}")
    
    # Test Three of a Kind
    hand = Deck.parse_hand("9♠ 9♥ 9♦ A♣ K♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.THREE_OF_KIND
    assert rank.primary_value == 9
    assert rank.num_cards == 5
    print(f"  Xám 9: {rank}")
    
    # Test Two Pair
    hand = Deck.parse_hand("A♠ A♥ K♦ K♣ 2♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.TWO_PAIR
    assert rank.primary_value == 14
    assert rank.kickers == [13, 2]
    assert rank.num_cards == 5
    print(f"  Thú A-K: {rank}")
    
    # Test Pair
    hand = Deck.parse_hand("J♠ J♥ A♦ K♣ Q♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.PAIR
    assert rank.primary_value == 11
    assert rank.num_cards == 5
    print(f"  Đôi J: {rank}")
    
    # Test High Card
    hand = Deck.parse_hand("A♠ K♥ Q♦ J♣ 9♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.HIGH_CARD
    assert rank.primary_value == 14
    assert rank.num_cards == 5
    print(f"  Mậu thầu A: {rank}")
    
    print("✅ 5-card evaluation tests passed!")


def test_compare():
    """Test comparison"""
    print("\nTesting hand comparison...")
    from card import Deck
    
    # Test cùng loại, khác rank (3 lá)
    hand1 = Deck.parse_hand("A♠ A♥ K♦")
    hand2 = Deck.parse_hand("K♠ K♥ A♦")
    assert HandEvaluator.compare(hand1, hand2) == 1  # Đôi A > Đôi K
    print("  ✅ Đôi A (3 lá) > Đôi K (3 lá)")
    
    # Test khác loại (3 lá)
    hand1 = Deck.parse_hand("5♠ 5♥ 5♦")  # Xám 5
    hand2 = Deck.parse_hand("A♠ A♥ K♦")  # Đôi A
    assert HandEvaluator.compare(hand1, hand2) == 1  # Xám > Đôi
    print("  ✅ Xám 5 (3 lá) > Đôi A (3 lá)")
    
    # Test bằng nhau
    hand1 = Deck.parse_hand("A♠ K♥ Q♦")
    hand2 = Deck.parse_hand("A♥ K♠ Q♣")
    assert HandEvaluator.compare(hand1, hand2) == 0
    print("  ✅ Mậu thầu AKQ = Mậu thầu AKQ")
    
    # Test 5 lá
    hand1 = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")  # Royal Flush
    hand2 = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♠")  # Tứ quý
    assert HandEvaluator.compare(hand1, hand2) == 1
    print("  ✅ Royal Flush > Tứ quý")
    
    # Test so sánh khác số lá → phải raise error
    hand1 = Deck.parse_hand("A♠ A♥ K♦")  # 3 lá
    hand2 = Deck.parse_hand("A♠ A♥ K♦ Q♣ J♠")  # 5 lá
    try:
        HandEvaluator.compare(hand1, hand2)
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "Cannot compare" in str(e)
        print("  ✅ Cross-comparison blocked correctly")
    
    print("✅ Comparison tests passed!")


def test_compare_hands():
    """Test compare_hands() cho 5 lá vs 3 lá"""
    print("\nTesting compare_hands() (5 lá vs 3 lá)...")
    from card import Deck
    
    # Case 1: Middle = TWO_PAIR, Front = PAIR A → Middle wins
    middle = Deck.parse_hand("10♠ 10♥ 9♦ 9♣ 2♠")  # Thú 10-9
    front = Deck.parse_hand("A♠ A♥ K♦")  # Đôi A
    
    result = HandEvaluator.compare_hands(middle, front)
    assert result == 1, "TWO_PAIR > PAIR"
    print("  ✅ Thú 10-9 > Đôi A")
    
    # Case 2: Middle = PAIR K, Front = PAIR A → Front wins
    middle = Deck.parse_hand("K♠ K♥ 5♦ 4♣ 2♠")  # Đôi K
    front = Deck.parse_hand("A♠ A♥ 3♦")  # Đôi A
    
    result = HandEvaluator.compare_hands(middle, front)
    assert result == -1, "PAIR K < PAIR A"
    print("  ✅ Đôi K < Đôi A")
    
    # Case 3: Middle = HIGH_CARD, Front = PAIR → Front wins
    middle = Deck.parse_hand("A♠ K♥ Q♦ J♣ 9♠")  # Mậu thầu A
    front = Deck.parse_hand("2♠ 2♥ 3♦")  # Đôi 2
    
    result = HandEvaluator.compare_hands(middle, front)
    assert result == -1, "HIGH_CARD < PAIR"
    print("  ✅ Mậu thầu A < Đôi 2")
    
    # Case 4: Middle = THREE_OF_KIND 9, Front = THREE_OF_KIND 5 → Middle wins
    middle = Deck.parse_hand("9♠ 9♥ 9♦ A♣ K♠")  # Xám 9
    front = Deck.parse_hand("5♠ 5♥ 5♦")  # Xám 5
    
    result = HandEvaluator.compare_hands(middle, front)
    assert result == 1, "THREE_OF_KIND 9 > THREE_OF_KIND 5"
    print("  ✅ Xám 9 > Xám 5")
    
    print("✅ compare_hands() tests passed!")


def test_is_valid_arrangement():
    """Test is_valid_arrangement()"""
    print("\nTesting is_valid_arrangement()...")
    from card import Deck
    
    # Case 1: Valid arrangement
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")    # Royal Flush
    middle = Deck.parse_hand("9♥ 9♦ 8♣ 8♠ 2♥")   # Thú 9-8
    front = Deck.parse_hand("7♠ 7♥ 6♦")          # Đôi 7
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert is_valid, f"Should be valid: {msg}"
    print(f"  ✅ Valid: Royal Flush > Thú 9-8 > Đôi 7")
    
    # Case 2: LỦNG - Back < Middle
    back = Deck.parse_hand("A♠ A♥ K♦ Q♣ J♠")     # Đôi A
    middle = Deck.parse_hand("9♠ 9♥ 9♦ 8♣ 8♠")   # Cù lũ 9-8
    front = Deck.parse_hand("7♠ 7♥ 6♦")          # Đôi 7
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert not is_valid, "Should be LỦNG: Back < Middle"
    assert "LỦNG" in msg
    print(f"  ✅ Detected LỦNG: {msg}")
    
    # Case 3: LỦNG - Middle < Front
    back = Deck.parse_hand("A♠ A♥ A♦ K♣ K♠")     # Cù lũ A-K
    middle = Deck.parse_hand("9♠ 9♥ 5♦ 4♣ 2♠")   # Đôi 9
    front = Deck.parse_hand("Q♠ Q♥ Q♦")          # Xám Q
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert not is_valid, "Should be LỦNG: Middle (Đôi 9) < Front (Xám Q)"
    assert "LỦNG" in msg
    print(f"  ✅ Detected LỦNG: {msg}")
    
    # Case 4: LỦNG - Middle (Đôi K) < Front (Đôi A)
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")    # Royal Flush
    middle = Deck.parse_hand("K♥ K♦ 5♣ 4♠ 2♥")   # Đôi K
    front = Deck.parse_hand("A♥ A♦ 3♣")          # Đôi A
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert not is_valid, "Should be LỦNG: Middle (Đôi K) < Front (Đôi A)"
    assert "LỦNG" in msg
    print(f"  ✅ Detected LỦNG: {msg}")
    
    # Case 5: Wrong number of cards
    back = Deck.parse_hand("A♠ K♠ Q♠ J♠")        # 4 lá (sai!)
    middle = Deck.parse_hand("9♥ 9♦ 8♣ 8♠ 2♥")
    front = Deck.parse_hand("7♠ 7♥ 6♦ 5♣")       # 4 lá (sai!)
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(back, middle, front)
    assert not is_valid, "Should fail: wrong card count"
    print(f"  ✅ Detected wrong card count: {msg}")
    
    print("✅ is_valid_arrangement() tests passed!")


if __name__ == "__main__":
    test_evaluator_3_cards()
    print()
    test_evaluator_5_cards()
    print()
    test_compare()
    print()
    test_compare_hands()
    print()
    test_is_valid_arrangement()
    print("\n" + "="*60)
    print("✅ All evaluator.py tests passed!")
    print("="*60)