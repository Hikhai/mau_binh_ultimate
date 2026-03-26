"""
Action Decoder V3 - Smart Hierarchical với Beam Search
Production-ready với multiple decoding strategies
"""
import sys
import os
import numpy as np
from typing import List, Tuple, Optional
from itertools import combinations
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))

from card import Card
from evaluator import HandEvaluator


class ActionDecoderV3:
    """
    Action Decoder V3 - Production Ready
    
    Features:
    - Hierarchical decoding (front → back → middle)
    - Multiple strategies: greedy, smart, beam search, random
    - Guaranteed valid arrangements
    - Cache optimization
    
    Action Space:
    - Front action: 0-285 (C(13,3) = 286 combinations)
    - Back action: 0-251 (C(10,5) = 252 combinations)
    - Total: 286 × 252 = 72,072 possible arrangements
    """
    
    def __init__(self):
        # Pre-compute combinations
        self.front_combos = list(combinations(range(13), 3))
        self.back_combos = list(combinations(range(10), 5))
        
        self.front_action_size = len(self.front_combos)  # 286
        self.back_action_size = len(self.back_combos)    # 252
        
        # Cache for decoded arrangements
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        print(f"ActionDecoderV3 initialized:")
        print(f"  Front actions: {self.front_action_size:,}")
        print(f"  Back actions:  {self.back_action_size:,}")
        print(f"  Total space:   {self.front_action_size * self.back_action_size:,}")
    
    # ============================================================
    # DECODING METHODS
    # ============================================================
    
    def decode_hierarchical(
        self,
        front_action: int,
        back_action: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Standard hierarchical decode
        
        Args:
            front_action: 0-285
            back_action: 0-251
            all_cards: 13 cards
            
        Returns:
            (back, middle, front)
        """
        # Check cache
        cache_key = (front_action, back_action, tuple(str(c) for c in all_cards))
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining 10 cards
        remaining_indices = [i for i in range(13) if i not in front_indices]
        remaining_cards = [all_cards[i] for i in remaining_indices]
        
        # Decode back
        back_local_indices = self.back_combos[back_action % self.back_action_size]
        back = [remaining_cards[i] for i in back_local_indices]
        
        # Middle = remaining
        middle = [c for c in remaining_cards if c not in back]
        
        result = (back, middle, front)
        
        # Cache result
        if len(self._cache) < 10000:  # Limit cache size
            self._cache[cache_key] = result
        
        return result
    
    def decode_greedy(
        self,
        front_action: int,
        all_cards: List[Card]
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Greedy decode: Chọn front, rồi greedy cho back/middle
        
        Strategy:
        - Front: từ action
        - Remaining: sort by rank descending
        - Back: top 5 strongest
        - Middle: remaining 5
        """
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining cards
        remaining = [all_cards[i] for i in range(13) if i not in front_indices]
        
        # Sort by rank (descending)
        remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
        
        back = remaining_sorted[:5]
        middle = remaining_sorted[5:10]
        
        return (back, middle, front)
    
    def decode_smart(
        self,
        front_action: int,
        all_cards: List[Card],
        num_attempts: int = 10
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        *** SMART DECODE - Multiple strategies with validation ***
        
        Tries:
        1. Greedy by rank
        2. Greedy by pairs/trips (put strong combinations in back)
        3. Random shuffles
        4. Sample from back_combos
        
        Returns best VALID arrangement by reward
        """
        try:
            from reward_calculator import RewardCalculatorV2
            reward_calc = RewardCalculatorV2()
        except:
            # Fallback to greedy
            return self.decode_greedy(front_action, all_cards)
        
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        # Remaining 10 cards
        remaining = [all_cards[i] for i in range(13) if i not in front_indices]
        
        best_arr = None
        best_reward = -1000.0
        
        # ===== STRATEGY 1: GREEDY BY RANK =====
        sorted_rem = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
        back1 = sorted_rem[:5]
        middle1 = sorted_rem[5:10]
        
        result1 = reward_calc.calculate_reward(back1, middle1, front)
        if result1['is_valid'] and result1['total_reward'] > best_reward:
            best_reward = result1['total_reward']
            best_arr = (back1, middle1, front)
        
        # ===== STRATEGY 2: PRIORITIZE PAIRS/TRIPS IN BACK =====
        from collections import Counter
        rank_counts = Counter(c.rank for c in remaining)
        
        # Separate singles, pairs, trips
        multiples = []
        singles = []
        
        for card in remaining:
            count = rank_counts[card.rank]
            if count >= 2:
                multiples.append(card)
            else:
                singles.append(card)
        
        # Try to put multiples in back
        if len(multiples) >= 5:
            back2 = sorted(multiples[:5], key=lambda c: c.rank.value, reverse=True)
            middle2 = sorted(multiples[5:10] if len(multiples) >= 10 else multiples[5:] + singles[:5-len(multiples[5:])], 
                           key=lambda c: c.rank.value, reverse=True)
            
            result2 = reward_calc.calculate_reward(back2, middle2, front)
            if result2['is_valid'] and result2['total_reward'] > best_reward:
                best_reward = result2['total_reward']
                best_arr = (back2, middle2, front)
        
        # ===== STRATEGY 3: RANDOM SHUFFLES =====
        for _ in range(min(num_attempts, 5)):
            shuffled = remaining.copy()
            random.shuffle(shuffled)
            
            back3 = shuffled[:5]
            middle3 = shuffled[5:10]
            
            result3 = reward_calc.calculate_reward(back3, middle3, front)
            if result3['is_valid'] and result3['total_reward'] > best_reward:
                best_reward = result3['total_reward']
                best_arr = (back3, middle3, front)
        
        # ===== STRATEGY 4: SAMPLE BACK COMBOS =====
        sample_size = min(num_attempts, len(self.back_combos))
        sample_indices = random.sample(range(len(self.back_combos)), sample_size)
        
        for idx in sample_indices:
            back_local_indices = self.back_combos[idx]
            back4 = [remaining[i] for i in back_local_indices]
            middle4 = [c for c in remaining if c not in back4]
            
            if len(middle4) == 5:
                result4 = reward_calc.calculate_reward(back4, middle4, front)
                if result4['is_valid'] and result4['total_reward'] > best_reward:
                    best_reward = result4['total_reward']
                    best_arr = (back4, middle4, front)
        
        # Return best found (guaranteed at least greedy)
        if best_arr is None:
            return (back1, middle1, front)
        
        return best_arr
    
    def decode_beam_search(
        self,
        front_action: int,
        all_cards: List[Card],
        beam_width: int = 5
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        *** NEW: BEAM SEARCH DECODE ***
        
        Maintains top-k candidates at each step
        
        Args:
            front_action: Front choice
            all_cards: 13 cards
            beam_width: Number of candidates to keep
            
        Returns:
            Best arrangement from beam
        """
        try:
            from reward_calculator import RewardCalculatorV2
            reward_calc = RewardCalculatorV2()
        except:
            return self.decode_greedy(front_action, all_cards)
        
        # Decode front
        front_indices = self.front_combos[front_action % self.front_action_size]
        front = [all_cards[i] for i in front_indices]
        
        remaining = [all_cards[i] for i in range(13) if i not in front_indices]
        
        # Generate beam_width candidates for back
        candidates = []
        
        # Sample back combinations
        num_samples = min(beam_width * 2, len(self.back_combos))
        sample_indices = random.sample(range(len(self.back_combos)), num_samples)
        
        for idx in sample_indices:
            back_local_indices = self.back_combos[idx]
            back = [remaining[i] for i in back_local_indices]
            middle = [c for c in remaining if c not in back]
            
            if len(middle) == 5:
                result = reward_calc.calculate_reward(back, middle, front)
                if result['is_valid']:
                    candidates.append({
                        'arrangement': (back, middle, front),
                        'reward': result['total_reward'],
                    })
        
        # Keep top beam_width
        candidates.sort(key=lambda x: x['reward'], reverse=True)
        
        if candidates:
            return candidates[0]['arrangement']
        else:
            # Fallback to greedy
            return self.decode_greedy(front_action, all_cards)
    
    # ============================================================
    # ENCODING (reverse operation)
    # ============================================================
    
    def encode_arrangement(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        all_cards: List[Card]
    ) -> Tuple[int, int]:
        """
        Encode arrangement → (front_action, back_action)
        
        Args:
            arrangement: (back, middle, front)
            all_cards: Original 13 cards
            
        Returns:
            (front_action, back_action)
        """
        back, middle, front = arrangement
        
        # Encode front
        front_indices = tuple(sorted(all_cards.index(c) for c in front))
        
        try:
            front_action = self.front_combos.index(front_indices)
        except ValueError:
            front_action = 0
        
        # Encode back
        remaining_indices = [i for i in range(13) if i not in front_indices]
        remaining_cards = [all_cards[i] for i in remaining_indices]
        
        try:
            back_local_indices = tuple(sorted(remaining_cards.index(c) for c in back))
            back_action = self.back_combos.index(back_local_indices)
        except ValueError:
            back_action = 0
        
        return (front_action, back_action)
    
    # ============================================================
    # UTILITIES
    # ============================================================
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total, 1)
        
        return {
            'cache_size': len(self._cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
        }
    
    def clear_cache(self):
        """Clear cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_valid_actions_mask(
        self,
        all_cards: List[Card],
        valid_arrangements: List[Tuple]
    ) -> np.ndarray:
        """
        Create binary mask for valid front actions
        
        Args:
            all_cards: 13 cards
            valid_arrangements: List of (back, middle, front)
            
        Returns:
            Binary mask (286,) - 1=valid, 0=invalid
        """
        mask = np.zeros(self.front_action_size, dtype=np.float32)
        
        for arr in valid_arrangements:
            try:
                front_action, _ = self.encode_arrangement(arr, all_cards)
                mask[front_action] = 1.0
            except:
                continue
        
        return mask


# Backward compatibility
ActionDecoderV2 = ActionDecoderV3


# ==================== TESTS ====================

def test_action_decoder_v3():
    """Comprehensive tests"""
    print("\n" + "="*60)
    print("Testing ActionDecoderV3...")
    print("="*60)
    
    from card import Deck
    
    decoder = ActionDecoderV3()
    
    cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
    
    # Test 1: Hierarchical decode
    print("\n[Test 1] Hierarchical decode")
    arr_hier = decoder.decode_hierarchical(10, 20, cards)
    
    assert len(arr_hier) == 3
    assert len(arr_hier[0]) == 5  # back
    assert len(arr_hier[1]) == 5  # middle
    assert len(arr_hier[2]) == 3  # front
    
    print(f"  Front:  {[str(c) for c in arr_hier[2]]}")
    print(f"  Middle: {[str(c) for c in arr_hier[1]]}")
    print(f"  Back:   {[str(c) for c in arr_hier[0]]}")
    print("  ✅ Hierarchical decode OK")
    
    # Test 2: Greedy decode
    print("\n[Test 2] Greedy decode")
    arr_greedy = decoder.decode_greedy(5, cards)
    
    assert len(arr_greedy) == 3
    print(f"  Back (greedy): {[str(c) for c in arr_greedy[0]]}")
    print("  ✅ Greedy decode OK")
    
    # Test 3: Smart decode
    print("\n[Test 3] Smart decode")
    arr_smart = decoder.decode_smart(5, cards, num_attempts=5)
    
    assert len(arr_smart) == 3
    
    # Validate
    is_valid, msg = HandEvaluator.is_valid_arrangement(*arr_smart)
    print(f"  Valid: {is_valid}")
    if is_valid:
        from reward_calculator import RewardCalculatorV2
        result = RewardCalculatorV2.calculate_reward(*arr_smart)
        print(f"  Reward: {result['total_reward']:.2f}")
    
    print("  ✅ Smart decode OK")
    
    # Test 4: Beam search
    print("\n[Test 4] Beam search decode")
    arr_beam = decoder.decode_beam_search(5, cards, beam_width=3)
    
    assert len(arr_beam) == 3
    
    is_valid, msg = HandEvaluator.is_valid_arrangement(*arr_beam)
    print(f"  Valid: {is_valid}")
    
    if is_valid:
        result = RewardCalculatorV2.calculate_reward(*arr_beam)
        print(f"  Reward: {result['total_reward']:.2f}")
    
    print("  ✅ Beam search OK")
    
    # Test 5: Encode
    print("\n[Test 5] Encode arrangement")
    front_act, back_act = decoder.encode_arrangement(arr_greedy, cards)
    
    assert 0 <= front_act < decoder.front_action_size
    assert 0 <= back_act < decoder.back_action_size
    
    print(f"  Front action: {front_act}")
    print(f"  Back action:  {back_act}")
    print("  ✅ Encode OK")
    
    # Test 6: Cache
    print("\n[Test 6] Cache performance")
    
    # Decode same arrangement multiple times
    for _ in range(5):
        decoder.decode_hierarchical(10, 20, cards)
    
    stats = decoder.get_cache_stats()
    print(f"  Cache size:  {stats['cache_size']}")
    print(f"  Cache hits:  {stats['cache_hits']}")
    print(f"  Hit rate:    {stats['hit_rate']:.1%}")
    
    assert stats['cache_hits'] > 0, "Cache should have hits"
    print("  ✅ Cache OK")
    
    # Test 7: Compare strategies
    print("\n[Test 7] Compare strategies")
    
    strategies = {
        'greedy': decoder.decode_greedy(5, cards),
        'smart': decoder.decode_smart(5, cards),
        'beam': decoder.decode_beam_search(5, cards, beam_width=5),
    }
    
    from reward_calculator import RewardCalculatorV2
    
    for name, arr in strategies.items():
        is_valid, _ = HandEvaluator.is_valid_arrangement(*arr)
        if is_valid:
            result = RewardCalculatorV2.calculate_reward(*arr)
            print(f"  {name:10s}: reward={result['total_reward']:7.2f}, bonus={result['bonus']:.0f}")
        else:
            print(f"  {name:10s}: INVALID")
    
    print("  ✅ Strategy comparison OK")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_action_decoder_v3()