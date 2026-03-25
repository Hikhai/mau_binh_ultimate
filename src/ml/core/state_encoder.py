"""
State Encoder V2 - Enhanced với metadata
"""
import sys
import os
import numpy as np
import torch
from typing import List
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card


class StateEncoderV2:
    """
    Enhanced state encoding:
    - 52-dim: one-hot của 13 lá
    - 13-dim: rank histogram
    - 4-dim: suit histogram
    - 8-dim: potential indicators (có đôi/xám/thú/...)
    
    Total: 77-dim (tăng từ 65-dim cũ)
    """
    
    STATE_SIZE = 77
    
    @staticmethod
    def encode(cards: List[Card]) -> np.ndarray:
        """
        Encode 13 cards → 77-dim vector
        
        Returns:
            np.ndarray shape (77,)
        """
        # Part 1: One-hot (52-dim)
        one_hot = np.zeros(52, dtype=np.float32)
        for card in cards:
            one_hot[card.to_index()] = 1.0
        
        # Part 2: Rank histogram (13-dim)
        rank_hist = np.zeros(13, dtype=np.float32)
        for card in cards:
            rank_idx = card.rank.value - 2
            rank_hist[rank_idx] += 1.0
        
        # Part 3: Suit histogram (4-dim)
        suit_hist = np.zeros(4, dtype=np.float32)
        for card in cards:
            suit_idx = card.suit.value
            suit_hist[suit_idx] += 1.0
        
        # Part 4: Potential indicators (8-dim)
        potential = StateEncoderV2._calculate_potential(cards)
        
        # Combine
        state = np.concatenate([one_hot, rank_hist, suit_hist, potential])
        
        return state
    
    @staticmethod
    def _calculate_potential(cards: List[Card]) -> np.ndarray:
        """
        Tính các chỉ số potential:
        [0] has_pair (có ít nhất 1 đôi)
        [1] has_two_pair (có 2 đôi)
        [2] has_trip (có xám)
        [3] has_quad (có tứ quý)
        [4] has_flush_potential (có >= 5 lá cùng chất)
        [5] has_straight_potential (có potential sảnh)
        [6] num_pairs (số đôi, normalized /6)
        [7] num_trips (số xám, normalized /4)
        """
        potential = np.zeros(8, dtype=np.float32)
        
        # Count ranks
        rank_counts = Counter(c.rank for c in cards)
        suit_counts = Counter(c.suit for c in cards)
        
        # Pairs & Trips & Quads
        pairs = [r for r, c in rank_counts.items() if c == 2]
        trips = [r for r, c in rank_counts.items() if c == 3]
        quads = [r for r, c in rank_counts.items() if c == 4]
        
        potential[0] = 1.0 if len(pairs) > 0 else 0.0
        potential[1] = 1.0 if len(pairs) >= 2 else 0.0
        potential[2] = 1.0 if len(trips) > 0 else 0.0
        potential[3] = 1.0 if len(quads) > 0 else 0.0
        
        # Flush potential
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        potential[4] = 1.0 if max_suit_count >= 5 else 0.0
        
        # Straight potential (simplified)
        sorted_ranks = sorted(set(c.rank.value for c in cards))
        has_straight_pot = False
        for i in range(len(sorted_ranks) - 4):
            if sorted_ranks[i+4] - sorted_ranks[i] <= 4:
                has_straight_pot = True
                break
        potential[5] = 1.0 if has_straight_pot else 0.0
        
        # Normalized counts
        potential[6] = min(len(pairs), 6) / 6.0
        potential[7] = min(len(trips), 4) / 4.0
        
        return potential
    
    @staticmethod
    def encode_batch(batch_cards: List[List[Card]]) -> torch.Tensor:
        """Batch encoding"""
        encoded = [StateEncoderV2.encode(cards) for cards in batch_cards]
        return torch.FloatTensor(np.array(encoded))
    
    @staticmethod
    def decode(state: np.ndarray) -> List[Card]:
        """Decode state → cards (chỉ dùng phần one-hot)"""
        cards = []
        for i in range(52):
            if state[i] > 0.5:
                card = Card.from_index(i)
                cards.append(card)
        return cards


# ==================== TESTS ====================

def test_state_encoder_v2():
    """Test StateEncoderV2"""
    print("Testing StateEncoderV2...")
    from card import Deck
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    
    state = StateEncoderV2.encode(cards)
    
    assert state.shape == (77,), f"Expected (77,), got {state.shape}"
    assert np.sum(state[:52]) == 13  # One-hot
    
    # Check potential indicators
    assert state[52+13+4+0] == 1.0  # has_pair
    assert state[52+13+4+1] == 1.0  # has_two_pair
    assert state[52+13+4+6] > 0     # num_pairs
    
    print(f"  State shape: {state.shape} ✓")
    print(f"  Has pair: {state[69]} ✓")
    print(f"  Num pairs: {state[75]:.2f} ✓")
    
    # Decode
    decoded = StateEncoderV2.decode(state)
    assert len(decoded) == 13
    
    print("✅ StateEncoderV2 tests passed!")


if __name__ == "__main__":
    test_state_encoder_v2()