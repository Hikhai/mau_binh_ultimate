"""
Hand Evaluator - Đánh giá tổ hợp bài
Đây là core của toàn bộ hệ thống!
"""
from typing import List, Optional
from collections import Counter
from card import Card, Rank
from hand_types import HandType, HandRank


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
            HandRank object
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
        """Đánh giá chi cuối (3 lá)"""
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        # Xám (Three of a kind)
        if 3 in rank_counts.values():
            trip_rank = max(r for r, count in rank_counts.items() if count == 3)
            return HandRank(HandType.THREE_OF_KIND, int(trip_rank), [])
        
        # Đôi (Pair)
        if 2 in rank_counts.values():
            pair_rank = max(r for r, count in rank_counts.items() if count == 2)
            kicker = max(r for r in ranks if r != pair_rank)
            return HandRank(HandType.PAIR, int(pair_rank), [int(kicker)])
        
        # Mậu thầu (High card)
        sorted_ranks = sorted(ranks, reverse=True)
        return HandRank(
            HandType.HIGH_CARD,
            int(sorted_ranks[0]),
            [int(r) for r in sorted_ranks[1:]]
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
            return HandRank(HandType.ROYAL_FLUSH, int(straight_high), [])
        
        # Straight Flush: Sảnh cùng chất
        if is_flush and is_straight:
            return HandRank(HandType.STRAIGHT_FLUSH, int(straight_high), [])
        
        # Four of a Kind: Tứ quý
        if 4 in rank_counts.values():
            quad_rank = max(r for r, count in rank_counts.items() if count == 4)
            kicker = max(r for r, count in rank_counts.items() if count == 1)
            return HandRank(HandType.FOUR_OF_KIND, int(quad_rank), [int(kicker)])
        
        # Full House: Cù lũ
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trip_rank = max(r for r, count in rank_counts.items() if count == 3)
            pair_rank = max(r for r, count in rank_counts.items() if count == 2)
            return HandRank(HandType.FULL_HOUSE, int(trip_rank), [int(pair_rank)])
        
        # Flush: Thùng
        if is_flush:
            sorted_ranks = sorted(ranks, reverse=True)
            return HandRank(
                HandType.FLUSH,
                int(sorted_ranks[0]),
                [int(r) for r in sorted_ranks[1:]]
            )
        
        # Straight: Sảnh
        if is_straight:
            return HandRank(HandType.STRAIGHT, int(straight_high), [])
        
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
                [int(k) for k in kickers]
            )
        
        # Two Pair: Thú
        pairs = [r for r, count in rank_counts.items() if count == 2]
        if len(pairs) == 2:
            pairs_sorted = sorted(pairs, reverse=True)
            kicker = max(r for r, count in rank_counts.items() if count == 1)
            return HandRank(
                HandType.TWO_PAIR,
                int(pairs_sorted[0]),
                [int(pairs_sorted[1]), int(kicker)]
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
                [int(k) for k in kickers]
            )
        
        # High Card: Mậu thầu
        sorted_ranks = sorted(ranks, reverse=True)
        return HandRank(
            HandType.HIGH_CARD,
            int(sorted_ranks[0]),
            [int(r) for r in sorted_ranks[1:]]
        )
    
    @staticmethod
    def _check_straight(ranks: List[Rank]) -> tuple[bool, Optional[Rank]]:
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
        So sánh 2 tay bài
        
        Returns:
            1 nếu hand1 > hand2
            -1 nếu hand1 < hand2
            0 nếu bằng nhau
        """
        rank1 = HandEvaluator.evaluate(hand1)
        rank2 = HandEvaluator.evaluate(hand2)
        
        if rank1 > rank2:
            return 1
        elif rank1 < rank2:
            return -1
        else:
            return 0
    
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
    print(f"  Xám A: {rank}")
    
    # Test đôi
    hand = Deck.parse_hand("K♠ K♥ 2♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.PAIR
    assert rank.primary_value == 13
    assert rank.kickers == [2]
    print(f"  Đôi K: {rank}")
    
    # Test mậu thầu
    hand = Deck.parse_hand("A♠ K♥ Q♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.HIGH_CARD
    assert rank.primary_value == 14
    print(f"  Mậu thầu A: {rank}")
    
    print("✅ 3-card evaluation tests passed!")


def test_evaluator_5_cards():
    """Test evaluator với 5 lá"""
    print("Testing 5-card evaluation...")
    from card import Deck
    
    # Test Royal Flush
    hand = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.ROYAL_FLUSH
    print(f"  Royal Flush: {rank}")
    
    # Test Straight Flush
    hand = Deck.parse_hand("9♥ 8♥ 7♥ 6♥ 5♥")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT_FLUSH
    assert rank.primary_value == 9
    print(f"  Straight Flush: {rank}")
    
    # Test Four of a Kind
    hand = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FOUR_OF_KIND
    assert rank.primary_value == 7
    print(f"  Tứ quý 7: {rank}")
    
    # Test Full House
    hand = Deck.parse_hand("K♠ K♥ K♦ 5♣ 5♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FULL_HOUSE
    assert rank.primary_value == 13
    print(f"  Cù lũ K-5: {rank}")
    
    # Test Flush
    hand = Deck.parse_hand("A♦ J♦ 9♦ 5♦ 3♦")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.FLUSH
    print(f"  Thùng A: {rank}")
    
    # Test Straight
    hand = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT
    assert rank.primary_value == 14
    print(f"  Sảnh A: {rank}")
    
    # Test Wheel Straight (A-2-3-4-5)
    hand = Deck.parse_hand("A♠ 2♥ 3♦ 4♣ 5♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.STRAIGHT
    assert rank.primary_value == 5  # 5 is high in wheel
    print(f"  Sảnh thấp (wheel): {rank}")
    
    # Test Three of a Kind
    hand = Deck.parse_hand("9♠ 9♥ 9♦ A♣ K♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.THREE_OF_KIND
    assert rank.primary_value == 9
    print(f"  Xám 9: {rank}")
    
    # Test Two Pair
    hand = Deck.parse_hand("A♠ A♥ K♦ K♣ 2♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.TWO_PAIR
    assert rank.primary_value == 14
    assert rank.kickers == [13, 2]
    print(f"  Thú A-K: {rank}")
    
    # Test Pair
    hand = Deck.parse_hand("J♠ J♥ A♦ K♣ Q♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.PAIR
    assert rank.primary_value == 11
    print(f"  Đôi J: {rank}")
    
    # Test High Card
    hand = Deck.parse_hand("A♠ K♥ Q♦ J♣ 9♠")
    rank = HandEvaluator.evaluate(hand)
    assert rank.hand_type == HandType.HIGH_CARD
    assert rank.primary_value == 14
    print(f"  Mậu thầu A: {rank}")
    
    print("✅ 5-card evaluation tests passed!")


def test_compare():
    """Test comparison"""
    print("Testing hand comparison...")
    from card import Deck
    
    # Test cùng loại, khác rank
    hand1 = Deck.parse_hand("A♠ A♥ K♦")
    hand2 = Deck.parse_hand("K♠ K♥ A♦")
    assert HandEvaluator.compare(hand1, hand2) == 1  # Đôi A > Đôi K
    
    # Test khác loại
    hand1 = Deck.parse_hand("5♠ 5♥ 5♦")  # Xám 5
    hand2 = Deck.parse_hand("A♠ A♥ K♦")  # Đôi A
    assert HandEvaluator.compare(hand1, hand2) == 1  # Xám > Đôi
    
    # Test bằng nhau
    hand1 = Deck.parse_hand("A♠ K♥ Q♦")
    hand2 = Deck.parse_hand("A♥ K♠ Q♣")
    assert HandEvaluator.compare(hand1, hand2) == 0
    
    # Test 5 lá
    hand1 = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠")  # Royal Flush
    hand2 = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♠")  # Tứ quý
    assert HandEvaluator.compare(hand1, hand2) == 1
    
    print("✅ Comparison tests passed!")


if __name__ == "__main__":
    test_evaluator_3_cards()
    print()
    test_evaluator_5_cards()
    print()
    test_compare()
    print("\n✅ All evaluator.py tests passed!")