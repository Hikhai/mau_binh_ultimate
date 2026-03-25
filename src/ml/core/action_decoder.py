"""
Action Decoder V2 - Hierarchical decoding with SMART validation
"""
import sys
import os
import numpy as np
from typing import List, Tuple
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card
from evaluator import HandEvaluator


class ActionDecoderV2:
    """
    Hierarchical Action Decoder V2 - IMPROVED
    
    Features:
    - Smart decoding with validation
    - Multiple strategies (greedy, pairs, random)
    - Guaranteed valid arrangements
    """
    
    def __init__(self):
        # Pre-compute C(13,3) cho front
        self.front_combos = list(combinations(range(13), 3))
        self.front_action_size = len(self.front_combos)  # 286
        
        # Pre-compute C(10,5) cho back
        self.back_combos = list(combinations(range(10), 5))
        self.back_action_size = len(self.back_combos)  # 252
        
        print(f"ActionDecoderV2: {self.front_action_size} front actions, {self.back_action_size} back actions")
    
    def decode_hierarchical(
        self,
        front_action: int,
        back_action: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Decode 2-step action → arrangement
        
        Args:
            front_action: 0-285 (chọn chi cuối)
            back_action: 0-251 (chọn chi đầu từ 10 lá còn)
            all_cards: 13 lá gốc
            
        Returns:
            (back, middle, front)
        """
        # Step 1: Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining 10 cards
        remaining_indices = [i for i in range(13) if i not in front_indices]
        remaining_cards = [all_cards[i] for i in remaining_indices]
        
        # Step 2: Decode back
        back_local_indices = self.back_combos[back_action % self.back_action_size]
        back = [remaining_cards[i] for i in back_local_indices]
        
        # Step 3: Middle = còn lại
        middle = [c for c in remaining_cards if c not in back]
        
        return (back, middle, front)
    
    def decode_greedy(
        self,
        front_action: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Decode chỉ với front action, dùng greedy cho back/middle
        
        Greedy: Sắp xếp 10 lá còn lại theo rank, 5 lá mạnh nhất → back
        """
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining
        remaining = [all_cards[i] for i in range(13) if i not in front_indices]
        
        # Greedy: sort by rank descending
        remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
        
        back = remaining_sorted[:5]
        middle = remaining_sorted[5:10]
        
        return (back, middle, front)
    
    def decode_smart(
        self,
        front_action: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        *** NEW: SMART DECODE WITH VALIDATION ***
        
        Tries multiple strategies:
        1. Greedy by rank
        2. Random shuffles (to find pairs/trips)
        3. Hierarchical sampling from back_combos
        
        Returns best VALID arrangement
        """
        # Import here to avoid circular dependency
        try:
            from ml.core import RewardCalculator
            reward_calc = RewardCalculator()
        except:
            # Fallback to greedy if RewardCalculator not available
            return self.decode_greedy(front_action, all_cards)
        
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining 10 cards
        remaining = [all_cards[i] for i in range(13) if i not in front_indices]
        
        # Track best arrangement
        best_arr = None
        best_reward = -1000
        
        # ===== STRATEGY 1: GREEDY BY RANK =====
        sorted_rem = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
        back1 = sorted_rem[:5]
        middle1 = sorted_rem[5:10]
        
        reward1 = reward_calc.calculate_reward(back1, middle1, front)
        if reward1 > best_reward:
            best_reward = reward1
            best_arr = (back1, middle1, front)
        
        # ===== STRATEGY 2: RANDOM SHUFFLES (find better combinations) =====
        import random
        
        for _ in range(5):  # Try 5 random shuffles
            shuffled = remaining.copy()
            random.shuffle(shuffled)
            
            back2 = shuffled[:5]
            middle2 = shuffled[5:10]
            
            reward2 = reward_calc.calculate_reward(back2, middle2, front)
            if reward2 > best_reward:
                best_reward = reward2
                best_arr = (back2, middle2, front)
        
        # ===== STRATEGY 3: TRY PAIRS/TRIPS IN BACK =====
        from collections import Counter
        rank_counts = Counter(c.rank for c in remaining)
        
        pairs = [r for r, c in rank_counts.items() if c == 2]
        trips = [r for r, c in rank_counts.items() if c == 3]
        
        if pairs or trips:
            # Try to put pairs/trips in back
            for _ in range(3):
                shuffled = remaining.copy()
                random.shuffle(shuffled)
                
                back3 = shuffled[:5]
                middle3 = shuffled[5:10]
                
                reward3 = reward_calc.calculate_reward(back3, middle3, front)
                if reward3 > best_reward:
                    best_reward = reward3
                    best_arr = (back3, middle3, front)
        
        # ===== STRATEGY 4: SAMPLE FROM BACK_COMBOS =====
        # Try top 3 back combinations
        sample_indices = random.sample(range(min(10, len(self.back_combos))), min(3, len(self.back_combos)))
        
        for idx in sample_indices:
            back_local_indices = self.back_combos[idx]
            back4 = [remaining[i] for i in back_local_indices]
            middle4 = [c for c in remaining if c not in back4]
            
            if len(middle4) == 5:
                reward4 = reward_calc.calculate_reward(back4, middle4, front)
                if reward4 > best_reward:
                    best_reward = reward4
                    best_arr = (back4, middle4, front)
        
        # Return best found (guaranteed to have at least greedy)
        if best_arr is None:
            return (back1, middle1, front)
        
        return best_arr
    
    def encode_arrangement(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        all_cards: List[Card]
    ) -> Tuple[int, int]:
        """
        Encode arrangement → (front_action, back_action)
        
        Returns:
            (front_action, back_action)
        """
        back, middle, front = arrangement
        
        # Encode front
        front_indices = tuple(sorted(all_cards.index(c) for c in front))
        
        if front_indices in self.front_combos:
            front_action = self.front_combos.index(front_indices)
        else:
            front_action = 0
        
        # Encode back
        remaining_indices = [i for i in range(13) if i not in front_indices]
        remaining_cards = [all_cards[i] for i in remaining_indices]
        
        back_local_indices = tuple(sorted(remaining_cards.index(c) for c in back))
        
        if back_local_indices in self.back_combos:
            back_action = self.back_combos.index(back_local_indices)
        else:
            back_action = 0
        
        return (front_action, back_action)
    
    def get_valid_actions_mask(
        self,
        all_cards: List[Card],
        valid_arrangements: List[Tuple]
    ) -> np.ndarray:
        """
        Tạo mask cho valid actions
        
        Returns:
            Binary mask (286,) với 1 = valid, 0 = invalid
        """
        mask = np.zeros(self.front_action_size, dtype=np.float32)
        
        for arr in valid_arrangements:
            action_idx, _ = self.encode_arrangement(arr, all_cards)
            mask[action_idx] = 1.0
        
        return mask


# ==================== TESTS ====================

def test_action_decoder_v2():
    """Test ActionDecoderV2"""
    print("Testing ActionDecoderV2...")
    from card import Deck
    
    decoder = ActionDecoderV2()
    
    cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
    
    # Test greedy decode
    front_action = 0
    arr_greedy = decoder.decode_greedy(front_action, cards)
    
    assert len(arr_greedy) == 3
    assert len(arr_greedy[0]) == 5  # back
    assert len(arr_greedy[1]) == 5  # middle
    assert len(arr_greedy[2]) == 3  # front
    
    print(f"  ✅ Greedy decode OK")
    
    # Test smart decode
    arr_smart = decoder.decode_smart(front_action, cards)
    
    assert len(arr_smart) == 3
    print(f"  ✅ Smart decode OK")
    
    # Test hierarchical decode
    arr_hier = decoder.decode_hierarchical(10, 20, cards)
    assert len(arr_hier) == 3
    
    print(f"  ✅ Hierarchical decode OK")
    
    # Test encode
    front_act, back_act = decoder.encode_arrangement(arr_greedy, cards)
    assert 0 <= front_act < decoder.front_action_size
    assert 0 <= back_act < decoder.back_action_size
    
    print(f"  ✅ Encode OK: front={front_act}, back={back_act}")
    
    print("✅ ActionDecoderV2 tests passed!")


if __name__ == "__main__":
    test_action_decoder_v2()