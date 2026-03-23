"""
Module quản lý quân bài
"""
from enum import IntEnum
from typing import List, Set
from functools import total_ordering


class Suit(IntEnum):
    """Chất bài - theo thứ tự mạnh yếu"""
    CLUBS = 0     # ♣ Chuồn
    DIAMONDS = 1  # ♦ Rô
    HEARTS = 2    # ♥ Cơ
    SPADES = 3    # ♠ Bích
    
    def __str__(self):
        symbols = {
            Suit.CLUBS: '♣',
            Suit.DIAMONDS: '♦',
            Suit.HEARTS: '♥',
            Suit.SPADES: '♠'
        }
        return symbols[self]


class Rank(IntEnum):
    """Rank bài - theo thứ tự mạnh yếu"""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    
    def __str__(self):
        if self.value <= 10:
            return str(self.value)
        face_cards = {
            Rank.JACK: 'J',
            Rank.QUEEN: 'Q',
            Rank.KING: 'K',
            Rank.ACE: 'A'
        }
        return face_cards[self]


@total_ordering
class Card:
    """
    Đại diện cho một quân bài
    """
    
    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __repr__(self):
        return f"Card({self.rank}, {self.suit})"
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __lt__(self, other):
        """So sánh: rank trước, suit sau"""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    @property
    def value(self) -> int:
        """Giá trị số của rank"""
        return int(self.rank)
    
    def to_index(self) -> int:
        """
        Convert card to unique index (0-51)
        Dùng cho ML encoding
        """
        return (self.rank - 2) * 4 + int(self.suit)
    
    @staticmethod
    def from_index(index: int) -> 'Card':
        """Tạo card từ index"""
        rank = Rank(index // 4 + 2)
        suit = Suit(index % 4)
        return Card(rank, suit)
    
    @staticmethod
    def from_string(card_str: str) -> 'Card':
        """
        Parse string thành Card
        Ví dụ: "A♠", "10♥", "KD" (D = Diamond)
        """
        # Map symbols
        suit_map = {
            '♣': Suit.CLUBS, 'C': Suit.CLUBS, 'c': Suit.CLUBS,
            '♦': Suit.DIAMONDS, 'D': Suit.DIAMONDS, 'd': Suit.DIAMONDS,
            '♥': Suit.HEARTS, 'H': Suit.HEARTS, 'h': Suit.HEARTS,
            '♠': Suit.SPADES, 'S': Suit.SPADES, 's': Suit.SPADES,
        }
        
        rank_map = {
            'A': Rank.ACE, 'a': Rank.ACE,
            'K': Rank.KING, 'k': Rank.KING,
            'Q': Rank.QUEEN, 'q': Rank.QUEEN,
            'J': Rank.JACK, 'j': Rank.JACK,
        }
        
        # Parse suit (last character)
        suit_char = card_str[-1]
        if suit_char not in suit_map:
            raise ValueError(f"Invalid suit: {suit_char}")
        suit = suit_map[suit_char]
        
        # Parse rank (everything except last character)
        rank_str = card_str[:-1]
        
        if rank_str in rank_map:
            rank = rank_map[rank_str]
        else:
            try:
                rank_value = int(rank_str)
                if rank_value < 2 or rank_value > 14:
                    raise ValueError(f"Invalid rank: {rank_value}")
                rank = Rank(rank_value)
            except ValueError:
                raise ValueError(f"Invalid rank: {rank_str}")
        
        return Card(rank, suit)


class Deck:
    """Bộ bài 52 lá"""
    
    @staticmethod
    def full_deck() -> List[Card]:
        """Tạo bộ bài đầy đủ 52 lá"""
        return [
            Card(rank, suit)
            for rank in Rank
            for suit in Suit
        ]
    
    @staticmethod
    def parse_hand(hand_str: str) -> List[Card]:
        """
        Parse chuỗi thành danh sách cards
        Ví dụ: "A♠ K♥ Q♦ J♣ 10♠"
        """
        card_strings = hand_str.strip().split()
        return [Card.from_string(s) for s in card_strings]
    
    @staticmethod
    def cards_to_string(cards: List[Card]) -> str:
        """Convert list cards thành string"""
        return ' '.join(str(c) for c in cards)


# ==================== TESTS ====================

def test_card():
    """Test Card class"""
    print("Testing Card class...")
    
    # Test creation
    card1 = Card(Rank.ACE, Suit.SPADES)
    assert str(card1) == "A♠"
    
    # Test parsing
    card2 = Card.from_string("A♠")
    assert card1 == card2
    
    card3 = Card.from_string("10H")
    assert card3.rank == Rank.TEN
    assert card3.suit == Suit.HEARTS
    
    # Test comparison
    card4 = Card(Rank.KING, Suit.SPADES)
    assert card4 < card1  # K < A
    
    # Test index
    card5 = Card(Rank.TWO, Suit.CLUBS)
    assert card5.to_index() == 0
    
    card6 = Card(Rank.ACE, Suit.SPADES)
    assert card6.to_index() == 51
    
    # Test from_index
    card7 = Card.from_index(51)
    assert card7 == card6
    
    print("✅ Card tests passed!")


def test_deck():
    """Test Deck class"""
    print("Testing Deck class...")
    
    # Test full deck
    deck = Deck.full_deck()
    assert len(deck) == 52
    
    # Test unique cards
    assert len(set(deck)) == 52
    
    # Test parse hand
    hand_str = "A♠ K♥ Q♦ J♣ 10♠"
    hand = Deck.parse_hand(hand_str)
    assert len(hand) == 5
    assert hand[0] == Card(Rank.ACE, Suit.SPADES)
    
    # Test cards_to_string
    result = Deck.cards_to_string(hand)
    assert result == hand_str
    
    print("✅ Deck tests passed!")


if __name__ == "__main__":
    test_card()
    test_deck()
    print("\n✅ All card.py tests passed!")