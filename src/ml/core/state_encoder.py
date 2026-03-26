"""
State Encoder V3 - Production-Ready với Rich Features
FIXED: Correct dimension calculation
"""
import sys
import os
import numpy as np
import torch
from typing import List, Dict, Tuple
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card, Rank, Suit


class StateEncoderV3:
    """
    State Encoder V3 - Rich Feature Engineering
    
    Features (130 dims total):
    ┌─────────────────────────────────────────┬──────┐
    │ Feature Group                           │ Dims │
    ├─────────────────────────────────────────┼──────┤
    │ 1. One-hot cards                        │  52  │
    │ 2. Rank histogram (normalized)          │  13  │
    │ 3. Suit histogram (normalized)          │   4  │
    │ 4. Pair/Trip/Quad indicators            │  13  │
    │ 5. Straight potential (all 10 straights)│  10  │
    │ 6. Flush potential (per suit)           │   4  │
    │ 7. Special hand signals                 │  15  │
    │ 8. High card strength (top 5)           │   5  │
    │ 9. Card connectivity                    │   6  │
    │ 10. Hand balance metrics                │   5  │
    │ 11. Reserved                            │   3  │
    └─────────────────────────────────────────┴──────┘
    Total: 130 dims
    """
    
    STATE_SIZE = 130
    
    @staticmethod
    def encode(cards: List[Card]) -> np.ndarray:
        """
        Encode 13 cards → 130-dim feature vector
        """
        features = []
        
        # 1. One-hot (52)
        features.append(StateEncoderV3._encode_one_hot(cards))
        
        # 2. Rank histogram (13)
        features.append(StateEncoderV3._encode_rank_histogram(cards))
        
        # 3. Suit histogram (4)
        features.append(StateEncoderV3._encode_suit_histogram(cards))
        
        # 4. Pair/Trip/Quad indicators (13)
        features.append(StateEncoderV3._encode_pair_features(cards))
        
        # 5. Straight potential (10)
        features.append(StateEncoderV3._encode_straight_potential(cards))
        
        # 6. Flush potential (4)
        features.append(StateEncoderV3._encode_flush_potential(cards))
        
        # 7. Special hand signals (15)
        features.append(StateEncoderV3._encode_special_signals(cards))
        
        # 8. High card strength (5)
        features.append(StateEncoderV3._encode_high_cards(cards))
        
        # 9. Card connectivity (6)
        features.append(StateEncoderV3._encode_connectivity(cards))
        
        # 10. Balance metrics (5)
        features.append(StateEncoderV3._encode_balance(cards))
        
        # 11. Reserved (3)
        features.append(np.zeros(3, dtype=np.float32))
        
        state = np.concatenate(features)
        
        # Debug: Check actual size
        if state.shape[0] != StateEncoderV3.STATE_SIZE:
            sizes = [52, 13, 4, 13, 10, 4, 15, 5, 6, 5, 3]
            print(f"DEBUG - Expected sizes: {sizes}, sum={sum(sizes)}")
            print(f"DEBUG - Actual size: {state.shape[0]}")
        
        assert state.shape == (StateEncoderV3.STATE_SIZE,), \
            f"Expected {StateEncoderV3.STATE_SIZE}, got {state.shape}"
        
        return state
    
    @staticmethod
    def _encode_one_hot(cards: List[Card]) -> np.ndarray:
        """One-hot encoding 52 positions"""
        vec = np.zeros(52, dtype=np.float32)
        for card in cards:
            vec[card.to_index()] = 1.0
        return vec
    
    @staticmethod
    def _encode_rank_histogram(cards: List[Card]) -> np.ndarray:
        """Rank histogram normalized (13 dims)"""
        vec = np.zeros(13, dtype=np.float32)
        for card in cards:
            rank_idx = card.rank.value - 2
            vec[rank_idx] += 1.0
        vec /= 4.0
        return vec
    
    @staticmethod
    def _encode_suit_histogram(cards: List[Card]) -> np.ndarray:
        """Suit histogram (4 dims)"""
        vec = np.zeros(4, dtype=np.float32)
        for card in cards:
            suit_idx = card.suit.value
            vec[suit_idx] += 1.0
        vec /= 13.0
        return vec
    
    @staticmethod
    def _encode_pair_features(cards: List[Card]) -> np.ndarray:
        """
        Pair/Trip/Quad per rank (13 dims)
        0=none, 0.25=single, 0.5=pair, 0.75=trip, 1.0=quad
        """
        vec = np.zeros(13, dtype=np.float32)
        rank_counts = Counter(card.rank.value - 2 for card in cards)
        
        for rank_idx, count in rank_counts.items():
            vec[rank_idx] = count / 4.0
        
        return vec
    
    @staticmethod
    def _encode_straight_potential(cards: List[Card]) -> np.ndarray:
        """
        Check all 10 possible straights (10 dims)
        Value = (cards in straight) / 5
        """
        vec = np.zeros(10, dtype=np.float32)
        ranks = set(card.rank.value for card in cards)
        
        straights = [
            {14, 2, 3, 4, 5},      # Wheel
            {2, 3, 4, 5, 6},
            {3, 4, 5, 6, 7},
            {4, 5, 6, 7, 8},
            {5, 6, 7, 8, 9},
            {6, 7, 8, 9, 10},
            {7, 8, 9, 10, 11},
            {8, 9, 10, 11, 12},
            {9, 10, 11, 12, 13},
            {10, 11, 12, 13, 14},  # Broadway
        ]
        
        for i, straight_ranks in enumerate(straights):
            have_count = len(ranks & straight_ranks)
            vec[i] = have_count / 5.0
        
        return vec
    
    @staticmethod
    def _encode_flush_potential(cards: List[Card]) -> np.ndarray:
        """Flush potential per suit (4 dims)"""
        vec = np.zeros(4, dtype=np.float32)
        
        for card in cards:
            suit_idx = card.suit.value
            vec[suit_idx] += 1.0
        
        vec = np.minimum(vec / 5.0, 1.0)
        return vec
    
    @staticmethod
    def _encode_special_signals(cards: List[Card]) -> np.ndarray:
        """
        Special hand signals (15 dims)
        
        [0] Dragon potential (13 unique)
        [1] Consecutive ranks
        [2] Same suit max
        [3] 6 pairs potential
        [4] 5 pairs potential
        [5] Has triple
        [6] 3 flushes potential
        [7] 3 straights potential
        [8] Has quad
        [9] Full house potential
        [10] Straight flush potential
        [11] High pair count
        [12] Num pairs (normalized)
        [13] Num trips (normalized)
        [14] Total multiples
        """
        vec = np.zeros(15, dtype=np.float32)
        
        ranks = sorted(card.rank.value for card in cards)
        rank_counts = Counter(card.rank.value for card in cards)
        suit_counts = Counter(card.suit.value for card in cards)
        
        # [0] Dragon
        vec[0] = len(set(ranks)) / 13.0
        
        # [1] Consecutive
        consecutive = sum(1 for i in range(len(ranks)-1) if ranks[i+1] - ranks[i] == 1)
        vec[1] = consecutive / 12.0
        
        # [2] Same suit max
        max_suit = max(suit_counts.values()) if suit_counts else 0
        vec[2] = max_suit / 13.0
        
        # Count pairs/trips/quads
        num_pairs = sum(1 for c in rank_counts.values() if c == 2)
        num_trips = sum(1 for c in rank_counts.values() if c == 3)
        num_quads = sum(1 for c in rank_counts.values() if c == 4)
        
        # [3] 6 pairs
        vec[3] = min(num_pairs / 6.0, 1.0)
        
        # [4] 5 pairs
        vec[4] = min(num_pairs / 5.0, 1.0)
        
        # [5] Has triple
        vec[5] = 1.0 if num_trips > 0 else 0.0
        
        # [6] 3 flushes
        suits_ge3 = sum(1 for c in suit_counts.values() if c >= 3)
        vec[6] = min(suits_ge3 / 3.0, 1.0)
        
        # [7] 3 straights
        vec[7] = vec[1]
        
        # [8] Has quad
        vec[8] = 1.0 if num_quads > 0 else 0.0
        
        # [9] Full house
        vec[9] = 1.0 if (num_trips > 0 and num_pairs >= 1) else 0.0
        
        # [10] Straight flush potential
        for suit in suit_counts:
            if suit_counts[suit] >= 5:
                suit_cards = [c for c in cards if c.suit.value == suit]
                suit_ranks = sorted(c.rank.value for c in suit_cards)
                consec = sum(1 for i in range(len(suit_ranks)-1) if suit_ranks[i+1] - suit_ranks[i] == 1)
                if consec >= 4:
                    vec[10] = 1.0
                    break
        
        # [11] High pair count
        vec[11] = 1.0 if num_pairs >= 3 else 0.0
        
        # [12] Num pairs normalized
        vec[12] = min(num_pairs / 6.0, 1.0)
        
        # [13] Num trips normalized
        vec[13] = min(num_trips / 4.0, 1.0)
        
        # [14] Total multiples
        vec[14] = min((num_pairs + num_trips * 2 + num_quads * 3) / 10.0, 1.0)
        
        return vec
    
    @staticmethod
    def _encode_high_cards(cards: List[Card]) -> np.ndarray:
        """Top 5 highest cards (5 dims)"""
        vec = np.zeros(5, dtype=np.float32)
        sorted_ranks = sorted([c.rank.value for c in cards], reverse=True)
        
        for i in range(min(5, len(sorted_ranks))):
            vec[i] = sorted_ranks[i] / 14.0
        
        return vec
    
    @staticmethod
    def _encode_connectivity(cards: List[Card]) -> np.ndarray:
        """
        Card connectivity (6 dims)
        [0] Overall rank connectivity
        [1-4] Per suit connectivity
        [5] Max suit connectivity
        """
        vec = np.zeros(6, dtype=np.float32)
        
        # Overall
        all_ranks = sorted(c.rank.value for c in cards)
        consecutive = sum(1 for i in range(len(all_ranks)-1) if all_ranks[i+1] - all_ranks[i] == 1)
        vec[0] = consecutive / 12.0
        
        # Per suit
        suits_grouped = {}
        for card in cards:
            s = card.suit.value
            if s not in suits_grouped:
                suits_grouped[s] = []
            suits_grouped[s].append(card.rank.value)
        
        for suit_idx in range(4):
            if suit_idx in suits_grouped:
                suit_ranks = sorted(suits_grouped[suit_idx])
                if len(suit_ranks) >= 2:
                    consec = sum(1 for i in range(len(suit_ranks)-1) if suit_ranks[i+1] - suit_ranks[i] == 1)
                    vec[1 + suit_idx] = consec / max(len(suit_ranks) - 1, 1)
        
        # Max
        vec[5] = max(vec[1:5]) if any(vec[1:5]) else 0.0
        
        return vec
    
    @staticmethod
    def _encode_balance(cards: List[Card]) -> np.ndarray:
        """
        Balance metrics (5 dims)
        [0] Rank entropy
        [1] Suit entropy
        [2] High-low balance
        [3] Color balance
        [4] Rank spread
        """
        vec = np.zeros(5, dtype=np.float32)
        
        rank_counts = Counter(c.rank.value for c in cards)
        suit_counts = Counter(c.suit.value for c in cards)
        
        # [0] Rank entropy
        if rank_counts:
            probs = [c / 13.0 for c in rank_counts.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            vec[0] = entropy / np.log2(13)
        
        # [1] Suit entropy
        if suit_counts:
            probs = [c / 13.0 for c in suit_counts.values()]
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs if p > 0)
            vec[1] = entropy / np.log2(4)
        
        # [2] High-low balance
        high_cards = sum(1 for c in cards if c.rank.value >= 10)
        vec[2] = min(high_cards, 13 - high_cards) / 6.5
        
        # [3] Color balance
        red = sum(1 for c in cards if c.suit.value in [1, 2])  # Diamonds, Hearts
        vec[3] = min(red, 13 - red) / 6.5
        
        # [4] Rank spread
        ranks = [c.rank.value for c in cards]
        if ranks:
            vec[4] = (max(ranks) - min(ranks)) / 12.0
        
        return vec
    
    @staticmethod
    def encode_batch(batch_cards: List[List[Card]]) -> torch.Tensor:
        """Batch encoding"""
        encoded = [StateEncoderV3.encode(cards) for cards in batch_cards]
        return torch.FloatTensor(np.array(encoded))
    
    @staticmethod
    def decode(state: np.ndarray) -> List[Card]:
        """Decode state → cards"""
        cards = []
        for i in range(52):
            if state[i] > 0.5:
                cards.append(Card.from_index(i))
        return cards


# Backward compatibility
StateEncoderV2 = StateEncoderV3


# ==================== TEST ====================

def test_state_encoder_v3():
    """Test StateEncoderV3"""
    print("\n" + "="*60)
    print("Testing StateEncoderV3...")
    print("="*60)
    
    from card import Deck
    
    # Test 1: Basic
    print("\n[Test 1] Basic encoding")
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    state = StateEncoderV3.encode(cards)
    
    print(f"  State shape: {state.shape}")
    print(f"  One-hot sum: {np.sum(state[:52])}")
    
    assert state.shape == (130,), f"Expected (130,), got {state.shape}"
    assert np.sum(state[:52]) == 13
    print("  ✅ Basic encoding OK")
    
    # Test 2: Dragon
    print("\n[Test 2] Dragon detection")
    dragon = Deck.parse_hand("2♠ 3♥ 4♦ 5♣ 6♠ 7♥ 8♦ 9♣ 10♠ J♥ Q♦ K♣ A♠")
    state_dragon = StateEncoderV3.encode(dragon)
    
    # Special signals start at: 52+13+4+13+10+4 = 96
    special_start = 96
    dragon_signal = state_dragon[special_start]  # First special signal
    print(f"  Dragon potential: {dragon_signal:.3f}")
    assert dragon_signal > 0.9, "Dragon should be detected"
    print("  ✅ Dragon detection OK")
    
    # Test 3: 6 pairs
    print("\n[Test 3] 6 pairs detection")
    pairs = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ J♣ 10♠ 10♥ 9♦ 9♣ 8♠")
    state_pairs = StateEncoderV3.encode(pairs)
    pairs_signal = state_pairs[special_start + 3]  # [3] = 6 pairs
    print(f"  6 pairs potential: {pairs_signal:.3f}")
    assert pairs_signal > 0.9, "6 pairs should be detected"
    print("  ✅ 6 pairs detection OK")
    
    # Test 4: Straight
    print("\n[Test 4] Straight potential")
    straight = Deck.parse_hand("A♠ 2♥ 3♦ 4♣ 5♠ 6♥ 7♦ 8♣ 9♠ 10♥ J♦ Q♣ K♠")
    state_straight = StateEncoderV3.encode(straight)
    
    straight_start = 52 + 13 + 4 + 13  # After one_hot, rank, suit, pairs
    straight_pots = state_straight[straight_start:straight_start+10]
    print(f"  Straight potentials: {straight_pots[:3]}... (first 3)")
    assert straight_pots[0] == 1.0, "Wheel should be complete"
    print("  ✅ Straight potential OK")
    
    # Test 5: Flush
    print("\n[Test 5] Flush potential")
    flush = Deck.parse_hand("A♠ K♠ Q♠ J♠ 10♠ 9♠ 8♠ 2♥ 3♦ 4♣ 5♥ 6♦ 7♣")
    state_flush = StateEncoderV3.encode(flush)
    
    flush_start = straight_start + 10
    flush_pots = state_flush[flush_start:flush_start+4]
    print(f"  Flush potentials: {flush_pots}")
    assert max(flush_pots) >= 1.0, "Should have flush"
    print("  ✅ Flush potential OK")
    
    # Test 6: Batch
    print("\n[Test 6] Batch encoding")
    batch_tensor = StateEncoderV3.encode_batch([cards, dragon, pairs])
    assert batch_tensor.shape == (3, 130)
    print(f"  Batch shape: {batch_tensor.shape}")
    print("  ✅ Batch encoding OK")
    
    # Test 7: Decode
    print("\n[Test 7] Decode")
    decoded = StateEncoderV3.decode(state)
    assert len(decoded) == 13
    print(f"  Decoded {len(decoded)} cards")
    print("  ✅ Decode OK")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_state_encoder_v3()