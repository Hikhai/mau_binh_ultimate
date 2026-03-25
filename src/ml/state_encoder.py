"""
State & Action Encoder cho ML - FIXED VERSION
"""
import torch
import numpy as np
import sys
from typing import List, Tuple
from itertools import combinations

sys.path.insert(0, '../core')
from card import Card, Deck


class StateEncoder:
    """
    Encode 13 lá bài thành state vector
    """
    
    @staticmethod
    def encode(cards: List[Card]) -> np.ndarray:
        """
        Encode 13 cards với NHIỀU THÔNG TIN HƠN
        
        Format mới (65-dim):
        - 52-dim: one-hot của 13 lá
        - 13-dim: rank histogram (count của mỗi rank từ 2-A)
        
        Args:
            cards: List of 13 Card objects
            
        Returns:
            numpy array shape (65,)
        """
        # One-hot của 13 lá (52-dim)
        one_hot = np.zeros(52, dtype=np.float32)
        for card in cards:
            one_hot[card.to_index()] = 1.0
        
        # Rank histogram (13-dim: count của mỗi rank từ 2-A)
        rank_hist = np.zeros(13, dtype=np.float32)
        for card in cards:
            rank_idx = card.rank.value - 2  # 2→0, 3→1, ..., A→12
            rank_hist[rank_idx] += 1.0
        
        # Combine
        state = np.concatenate([one_hot, rank_hist])
        
        return state
    
    @staticmethod
    def encode_batch(batch_cards: List[List[Card]]) -> torch.Tensor:
        """
        Encode batch of hands
        
        Returns:
            torch.Tensor shape (batch_size, 65)
        """
        encoded = [StateEncoder.encode(cards) for cards in batch_cards]
        return torch.FloatTensor(np.array(encoded))
    
    @staticmethod
    def decode(state: np.ndarray) -> List[Card]:
        """
        Decode state vector về cards
        
        Args:
            state: numpy array shape (65,)
            
        Returns:
            List of Card objects
        """
        cards = []
        # Chỉ decode phần one-hot (52 chiều đầu)
        for i in range(52):
            if state[i] > 0.5:  # > 0.5 để handle floating point
                card = Card.from_index(i)
                cards.append(card)
        
        return cards


class ActionEncoder:
    """
    Encode/Decode actions ĐÚNG CÁCH
    
    Action = index của back trong C(13,5) = 1287 combinations
    """
    
    def __init__(self):
        """
        Pre-compute tất cả C(13,5) combinations
        """
        # Pre-compute tất cả combinations của back (5 lá từ 13 lá)
        self.all_back_combos = list(combinations(range(13), 5))
        self.action_space_size = len(self.all_back_combos)  # 1287
        
        # Create reverse mapping cho encode nhanh
        self.combo_to_idx = {combo: idx for idx, combo in enumerate(self.all_back_combos)}
        
        print(f"Action space size: {self.action_space_size}")
    
    def encode_action(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        all_cards: List[Card]
    ) -> int:
        """
        Encode arrangement → action index
        
        Action = index của back combination trong C(13,5)
        
        Args:
            arrangement: (back, middle, front)
            all_cards: 13 lá gốc
            
        Returns:
            action_index: 0-1286
        """
        back, middle, front = arrangement
        
        # Get indices của back cards trong all_cards
        back_indices = tuple(sorted(all_cards.index(c) for c in back))
        
        # Lookup trong pre-computed mapping
        if back_indices in self.combo_to_idx:
            return self.combo_to_idx[back_indices]
        else:
            # Fallback (không nên xảy ra)
            return 0
    
    def decode_action(
        self,
        action_index: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Decode action index → arrangement
        
        Args:
            action_index: 0-1286
            all_cards: 13 lá gốc
            
        Returns:
            (back, middle, front)
        """
        # Clamp action_index
        if action_index >= self.action_space_size or action_index < 0:
            action_index = 0
        
        # Get back indices
        back_indices = self.all_back_combos[action_index]
        back = [all_cards[i] for i in back_indices]
        
        # Remaining 8 cards
        remaining_indices = [i for i in range(13) if i not in back_indices]
        
        # Middle = first 5, Front = last 3
        middle = [all_cards[i] for i in remaining_indices[:5]]
        front = [all_cards[i] for i in remaining_indices[5:]]
        
        return (back, middle, front)
    
    def get_valid_actions_mask(
        self,
        all_cards: List[Card],
        valid_arrangements: List[Tuple]
    ) -> np.ndarray:
        """
        Tạo mask cho valid actions
        
        Returns:
            Binary mask (1287,) với 1 = valid, 0 = invalid
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        
        for arr in valid_arrangements:
            action_idx = self.encode_action(arr, all_cards)
            mask[action_idx] = 1.0
        
        return mask


# ==================== TESTS ====================

def test_state_encoder():
    """Test StateEncoder"""
    print("Testing StateEncoder...")
    
    # Create test hand
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    cards = Deck.parse_hand(hand_str)
    
    # Encode
    state = StateEncoder.encode(cards)
    
    # *** FIX: State giờ là 65-dim ***
    assert state.shape == (65,), f"Expected (65,), got {state.shape}"
    
    # First 52 elements = one-hot
    assert np.sum(state[:52]) == 13  # 13 lá = 13 giá trị 1
    
    # Last 13 elements = rank histogram
    assert np.sum(state[52:]) == 13  # Tổng count = 13 lá
    
    print(f"  State shape: {state.shape} ✓")
    print(f"  One-hot sum: {np.sum(state[:52])} ✓")
    print(f"  Rank hist sum: {np.sum(state[52:])} ✓")
    
    # Decode
    decoded_cards = StateEncoder.decode(state)
    assert len(decoded_cards) == 13
    assert set(decoded_cards) == set(cards)
    print(f"  Decode successful ✓")
    
    # Test batch encoding
    batch = [cards, cards]  # 2 hands giống nhau
    batch_tensor = StateEncoder.encode_batch(batch)
    assert batch_tensor.shape == (2, 65), f"Expected (2, 65), got {batch_tensor.shape}"
    print(f"  Batch encoding shape: {batch_tensor.shape} ✓")
    
    print("✅ StateEncoder tests passed!")


def test_action_encoder():
    """Test ActionEncoder"""
    print("\nTesting ActionEncoder...")
    
    encoder = ActionEncoder()
    
    # Test hand
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    cards = Deck.parse_hand(hand_str)
    
    # Test arrangement
    back = cards[:5]
    middle = cards[5:10]
    front = cards[10:13]
    arrangement = (back, middle, front)
    
    # Encode
    action_idx = encoder.encode_action(arrangement, cards)
    assert 0 <= action_idx < encoder.action_space_size
    print(f"  Action index: {action_idx} ✓")
    
    # Decode
    decoded = encoder.decode_action(action_idx, cards)
    assert len(decoded) == 3
    assert len(decoded[0]) == 5
    assert len(decoded[1]) == 5
    assert len(decoded[2]) == 3
    print(f"  Decode successful ✓")
    
    # *** TEST CONSISTENCY ***
    # Encode lại decoded arrangement → phải ra cùng action_idx
    action_idx2 = encoder.encode_action(decoded, cards)
    assert action_idx == action_idx2, f"Encode/decode not consistent! {action_idx} != {action_idx2}"
    print(f"  Encode/decode consistency ✓")
    
    # Test mask
    mask = encoder.get_valid_actions_mask(cards, [arrangement])
    assert mask.shape == (encoder.action_space_size,)
    assert np.sum(mask) >= 1  # Ít nhất 1 action valid
    print(f"  Valid actions mask: {np.sum(mask)} valid actions ✓")
    
    print("✅ ActionEncoder tests passed!")


if __name__ == "__main__":
    test_state_encoder()
    test_action_encoder()
    print("\n" + "="*60)
    print("✅ All state_encoder.py tests passed!")
    print("="*60)