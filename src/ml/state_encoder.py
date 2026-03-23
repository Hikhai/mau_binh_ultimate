"""
State & Action Encoder cho ML
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
        Encode 13 cards thành one-hot vector 52 chiều
        
        Args:
            cards: List of 13 Card objects
            
        Returns:
            numpy array shape (52,) với 13 giá trị = 1, còn lại = 0
        """
        state = np.zeros(52, dtype=np.float32)
        
        for card in cards:
            index = card.to_index()
            state[index] = 1.0
        
        return state
    
    @staticmethod
    def encode_batch(batch_cards: List[List[Card]]) -> torch.Tensor:
        """
        Encode batch of hands
        
        Returns:
            torch.Tensor shape (batch_size, 52)
        """
        encoded = [StateEncoder.encode(cards) for cards in batch_cards]
        return torch.FloatTensor(np.array(encoded))
    
    @staticmethod
    def decode(state: np.ndarray) -> List[Card]:
        """
        Decode state vector về cards
        
        Args:
            state: numpy array shape (52,)
            
        Returns:
            List of Card objects
        """
        cards = []
        for i in range(52):
            if state[i] > 0.5:  # > 0.5 để handle floating point
                card = Card.from_index(i)
                cards.append(card)
        
        return cards


class ActionEncoder:
    """
    Encode/Decode actions (cách xếp bài)
    
    Action space rất lớn: C(13,5) * C(8,5) = 72,072 cách
    Để ML khả thi, ta dùng simplified action space
    """
    
    def __init__(self, use_simplified: bool = True):
        """
        Args:
            use_simplified: Nếu True, dùng simplified action space (~1000 actions)
                           Nếu False, dùng full space (72k actions)
        """
        self.use_simplified = use_simplified
        
        if use_simplified:
            # Simplified: Chỉ xét các cách xếp "reasonable"
            self.action_space_size = 1000
        else:
            # Full: 72,072 cách
            self.action_space_size = 72072
    
    def encode_action(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        all_cards: List[Card]
    ) -> int:
        """
        Encode arrangement thành action index
        
        Simplified approach: Hash arrangement to fixed range
        """
        back, middle, front = arrangement
        
        # Tạo tuple of indices
        back_indices = tuple(sorted(all_cards.index(c) for c in back))
        middle_indices = tuple(sorted(all_cards.index(c) for c in middle))
        
        # Hash to action index
        action_hash = hash((back_indices, middle_indices))
        action_index = abs(action_hash) % self.action_space_size
        
        return action_index
    
    def decode_action(
        self,
        action_index: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Decode action index về arrangement
        
        Note: Vì dùng hash, không thể decode chính xác
        Thay vào đó, ta generate arrangement từ action_index như một seed
        """
        # Use action_index as seed for deterministic random
        np.random.seed(action_index)
        
        # Random shuffle và chia
        shuffled = all_cards.copy()
        np.random.shuffle(shuffled)
        
        back = shuffled[:5]
        middle = shuffled[5:10]
        front = shuffled[10:13]
        
        # Reset seed
        np.random.seed(None)
        
        return (back, middle, front)
    
    def get_valid_actions_mask(
        self,
        all_cards: List[Card],
        valid_arrangements: List[Tuple]
    ) -> np.ndarray:
        """
        Tạo mask cho valid actions
        
        Returns:
            Binary mask (action_space_size,) với 1 = valid, 0 = invalid
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
    
    assert state.shape == (52,)
    assert np.sum(state) == 13  # 13 lá = 13 giá trị 1
    print(f"  State shape: {state.shape} ✓")
    print(f"  Sum of state: {np.sum(state)} ✓")
    
    # Decode
    decoded_cards = StateEncoder.decode(state)
    assert len(decoded_cards) == 13
    assert set(decoded_cards) == set(cards)
    print(f"  Decode successful ✓")
    
    # Test batch encoding
    batch = [cards, cards[:10] + cards[:3]]  # 2 hands
    batch_tensor = StateEncoder.encode_batch(batch)
    assert batch_tensor.shape == (2, 52)
    print(f"  Batch encoding shape: {batch_tensor.shape} ✓")
    
    print("✅ StateEncoder tests passed!")


def test_action_encoder():
    """Test ActionEncoder"""
    print("\nTesting ActionEncoder...")
    
    encoder = ActionEncoder(use_simplified=True)
    
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
    
    # Decode (sẽ khác arrangement gốc vì dùng hash)
    decoded = encoder.decode_action(action_idx, cards)
    assert len(decoded) == 3
    assert len(decoded[0]) == 5
    assert len(decoded[1]) == 5
    assert len(decoded[2]) == 3
    print(f"  Decode successful ✓")
    
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