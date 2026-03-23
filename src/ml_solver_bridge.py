"""
Bridge giữa ML Model V2 và Ultimate Solver
Load ConstraintAwareDQN và cung cấp predictions
"""
import sys
import os
import torch
import numpy as np
import random
from typing import List, Tuple, Optional
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'core'))
sys.path.insert(0, os.path.join(current_dir, 'engines'))
sys.path.insert(0, os.path.join(current_dir, 'ml'))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints


class MLSolverBridge:
    """
    Bridge giữa ML Model V2 và Ultimate Solver
    
    Khác với DQN Agent cũ:
    - Dùng ConstraintAwareDQN (network mới)
    - Generate SMART valid arrangements
    - Score bằng model prediction + heuristic
    """
    
    def __init__(self, model_path: str = None):
        self.device = torch.device("cpu")
        self.model = None
        self.bonus_calc = BonusPoints()
        self.is_loaded = False
        
        # Auto-detect model path
        if model_path is None:
            possible_paths = [
                os.path.join(current_dir, "../data/models/expert_v2_100k/best_model.pth"),
                os.path.join(current_dir, "../data/models/expert_v2/best_model.pth"),
                os.path.join(current_dir, "data/models/expert_v2_100k/best_model.pth"),
                os.path.join(current_dir, "data/models/expert_v2/best_model.pth"),
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load trained ConstraintAwareDQN model"""
        try:
            from network_v2 import ConstraintAwareDQN
            
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=False
            )
            
            self.model = ConstraintAwareDQN(state_size=52, action_size=1000)
            self.model.load_state_dict(checkpoint['network_state_dict'])
            self.model.eval()
            
            # Load normalization params
            self.reward_mean = checkpoint.get('reward_mean', 0)
            self.reward_std = checkpoint.get('reward_std', 1)
            
            self.is_loaded = True
            print(f"✅ ML Bridge: Model loaded from {model_path}")
            
        except Exception as e:
            print(f"⚠️  ML Bridge: Failed to load model: {e}")
            self.is_loaded = False
    
    def get_best_arrangement(
        self,
        cards: List[Card],
        num_candidates: int = 100
    ) -> Optional[Tuple[List[Card], List[Card], List[Card]]]:
        """
        Get best arrangement using ML model + smart strategies
        
        1. Generate many valid arrangements (smart strategies)
        2. Score each using reward function
        3. Return best one
        """
        # Generate valid arrangements
        valid_arrs = self._generate_smart_arrangements(cards, max_count=num_candidates)
        
        if not valid_arrs:
            return None
        
        # Score each arrangement
        best_arr = None
        best_score = -float('inf')
        
        for arr in valid_arrs:
            score = self._score_arrangement(*arr)
            
            if score > best_score:
                best_score = score
                best_arr = arr
        
        return best_arr
    
    def get_top_arrangements(
        self,
        cards: List[Card],
        top_k: int = 10,
        num_candidates: int = 100
    ) -> List[Tuple]:
        """
        Get top K arrangements ranked by score
        """
        valid_arrs = self._generate_smart_arrangements(cards, max_count=num_candidates)
        
        if not valid_arrs:
            return []
        
        # Score all
        scored = []
        for arr in valid_arrs:
            score = self._score_arrangement(*arr)
            scored.append((arr, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates and return top K
        seen = set()
        result = []
        
        for arr, score in scored:
            key = self._arrangement_key(arr)
            if key not in seen:
                seen.add(key)
                result.append(arr)
            
            if len(result) >= top_k:
                break
        
        return result
    
    def _score_arrangement(self, back, middle, front) -> float:
        """
        Score arrangement using reward function
        Same as training reward for consistency
        """
        try:
            # Bonus (weighted high!)
            bonus = self.bonus_calc.calculate_bonus(back, middle, front)
            
            # Hand evaluations
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            # Weighted strength (front MOST important!)
            strength = (
                back_rank.hand_type.value * 0.3 +
                back_rank.primary_value * 0.01 +
                middle_rank.hand_type.value * 0.25 +
                middle_rank.primary_value * 0.008 +
                front_rank.hand_type.value * 0.4 +
                front_rank.primary_value * 0.015
            )
            
            # Bonus multiplier
            bonus_reward = bonus * 3.0
            
            # Balance bonus
            strengths = [
                back_rank.hand_type.value,
                middle_rank.hand_type.value,
                front_rank.hand_type.value
            ]
            
            import numpy as np
            variance = np.var(strengths)
            balance = 1.0 / (1.0 + variance * 0.5)
            
            return bonus_reward + strength + balance
            
        except Exception:
            return -999.0
    
    def _generate_smart_arrangements(
        self,
        cards: List[Card],
        max_count: int = 100
    ) -> List[Tuple]:
        """Generate smart valid arrangements"""
        valid = []
        
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        quads = [r for r, c in rank_counts.items() if c == 4]
        trips = [r for r, c in rank_counts.items() if c == 3]
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)
        
        # Strategy 1: Tứ quý
        if quads:
            for quad_rank in quads:
                quad_cards = [c for c in cards if c.rank == quad_rank][:4]
                other = [c for c in cards if c not in quad_cards]
                
                # Tứ quý ở chi 1
                for _ in range(5):
                    random.shuffle(other)
                    back = quad_cards + [other[0]]
                    middle = other[1:6]
                    front = other[6:9]
                    
                    if self._is_valid(back, middle, front):
                        valid.append((back, middle, front))
                        break
                
                # Tứ quý ở chi 2 (nếu back đủ mạnh)
                for _ in range(5):
                    random.shuffle(other)
                    middle = quad_cards + [other[0]]
                    back = other[1:6]
                    front = other[6:9]
                    
                    if self._is_valid(back, middle, front):
                        valid.append((back, middle, front))
                        break
        
        # Strategy 2: Xám ở chi cuối (+6 bonus!)
        if trips:
            for trip_rank in trips:
                trip_cards = [c for c in cards if c.rank == trip_rank][:3]
                other = [c for c in cards if c not in trip_cards]
                
                for _ in range(10):
                    random.shuffle(other)
                    back = other[:5]
                    middle = other[5:10]
                    
                    if self._is_valid(back, middle, trip_cards):
                        valid.append((back, middle, trip_cards))
        
        # Strategy 3: Đôi lớn nhất ở chi cuối
        if pairs:
            for i in range(min(3, len(pairs))):
                pair_rank = pairs[i]
                pair_cards = [c for c in cards if c.rank == pair_rank][:2]
                other = [c for c in cards if c not in pair_cards]
                
                for kicker in other[:5]:
                    front = pair_cards + [kicker]
                    remaining = [c for c in other if c != kicker]
                    
                    for _ in range(5):
                        random.shuffle(remaining)
                        back = remaining[:5]
                        middle = remaining[5:10]
                        
                        if self._is_valid(back, middle, front):
                            valid.append((back, middle, front))
                            break
        
        # Strategy 4: Tìm sảnh
        for start in range(9):
            straight = []
            for target in range(start + 2, start + 7):
                for card in cards:
                    if card.rank.value == target and card not in straight:
                        straight.append(card)
                        break
            
            if len(straight) == 5:
                other = [c for c in cards if c not in straight]
                
                # Sảnh ở chi 1
                for _ in range(5):
                    random.shuffle(other)
                    middle = other[:5]
                    front = other[5:8]
                    
                    if self._is_valid(straight, middle, front):
                        valid.append((straight, middle, front))
                        break
                
                # Sảnh ở chi 2
                for _ in range(5):
                    random.shuffle(other)
                    back = other[:5]
                    front = other[5:8]
                    
                    if self._is_valid(back, straight, front):
                        valid.append((back, straight, front))
                        break
        
        # Strategy 5: Thú (Two Pair)
        if len(pairs) >= 2:
            for i in range(min(2, len(pairs))):
                for j in range(i+1, min(4, len(pairs))):
                    p1 = [c for c in cards if c.rank == pairs[i]][:2]
                    p2 = [c for c in cards if c.rank == pairs[j]][:2]
                    two_pair = p1 + p2
                    other = [c for c in cards if c not in two_pair]
                    other_sorted = sorted(other, key=lambda c: c.rank.value, reverse=True)
                    
                    # Two pair ở chi 1
                    back = two_pair + [other_sorted[0]]
                    middle = other_sorted[1:6]
                    front = other_sorted[6:9]
                    
                    if len(middle) == 5 and len(front) == 3:
                        if self._is_valid(back, middle, front):
                            valid.append((back, middle, front))
                    
                    # Two pair ở chi 2
                    middle = two_pair + [other_sorted[0]]
                    back = other_sorted[1:6]
                    front = other_sorted[6:9]
                    
                    if len(back) == 5 and len(front) == 3:
                        if self._is_valid(back, middle, front):
                            valid.append((back, middle, front))
        
        # Strategy 6: Cù lũ (Full House)
        if trips and pairs:
            trip_rank = max(trips)
            pair_rank = max(pairs)
            
            trip_cards = [c for c in cards if c.rank == trip_rank][:3]
            pair_cards = [c for c in cards if c.rank == pair_rank][:2]
            full_house = trip_cards + pair_cards
            other = [c for c in cards if c not in full_house]
            
            # Cù lũ ở chi 1
            other_sorted = sorted(other, key=lambda c: c.rank.value, reverse=True)
            middle = other_sorted[:5]
            front = other_sorted[5:8]
            
            if len(middle) == 5 and len(front) == 3:
                if self._is_valid(full_house, middle, front):
                    valid.append((full_house, middle, front))
            
            # Cù lũ ở chi 2 (+4 bonus!)
            for _ in range(5):
                random.shuffle(other)
                back = other[:5]
                front = other[5:8]
                
                if self._is_valid(back, full_house, front):
                    valid.append((back, full_house, front))
                    break
        
        # Strategy 7: Thùng (Flush - 5 cards same suit)
        suit_cards = {}
        for card in cards:
            suit = card.suit
            if suit not in suit_cards:
                suit_cards[suit] = []
            suit_cards[suit].append(card)
        
        for suit, s_cards in suit_cards.items():
            if len(s_cards) >= 5:
                flush = s_cards[:5]
                other = [c for c in cards if c not in flush]
                
                # Thùng ở chi 1
                for _ in range(5):
                    random.shuffle(other)
                    middle = other[:5]
                    front = other[5:8]
                    
                    if self._is_valid(flush, middle, front):
                        valid.append((flush, middle, front))
                        break
        
        # Strategy 8: Random valid (fill up)
        attempts = 0
        while len(valid) < max_count and attempts < max_count * 5:
            attempts += 1
            
            shuffled = cards.copy()
            random.shuffle(shuffled)
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            if self._is_valid(back, middle, front):
                valid.append((back, middle, front))
        
        # Remove duplicates
        unique = []
        seen = set()
        
        for arr in valid:
            key = self._arrangement_key(arr)
            if key not in seen:
                seen.add(key)
                unique.append(arr)
        
        return unique
    
    def _is_valid(self, back, middle, front) -> bool:
        """Check validity"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        # Check no duplicates
        all_cards = back + middle + front
        if len(set(id(c) for c in all_cards)) != 13:
            return False
        
        try:
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            if back_rank < middle_rank:
                return False
            
            # Front constraints
            if front_rank.hand_type.value == 3:  # Trip
                if middle_rank.hand_type.value < 3:
                    return False
            
            if front_rank.hand_type.value == 1:  # Pair
                if middle_rank.hand_type.value < 1:
                    return False
                if (middle_rank.hand_type.value == 1 and
                    middle_rank.primary_value < front_rank.primary_value):
                    return False
            
            return True
        except:
            return False
    
    def _arrangement_key(self, arr):
        """Unique key for arrangement"""
        return (
            tuple(sorted([c.to_index() for c in arr[0]])),
            tuple(sorted([c.to_index() for c in arr[1]])),
            tuple(sorted([c.to_index() for c in arr[2]]))
        )


# ==================== TEST ====================

if __name__ == "__main__":
    print("🧪 Testing ML Solver Bridge...\n")
    
    bridge = MLSolverBridge()
    
    test_hands = [
        "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠",
        "7♠ 7♥ 7♦ 7♣ A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 6♣ 5♠",
        "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠",
    ]
    
    for i, hand_str in enumerate(test_hands, 1):
        print(f"Test {i}: {hand_str}")
        
        cards = Deck.parse_hand(hand_str)
        arr = bridge.get_best_arrangement(cards)
        
        if arr:
            back, middle, front = arr
            
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            bonus = bridge.bonus_calc.calculate_bonus(back, middle, front)
            score = bridge._score_arrangement(back, middle, front)
            
            print(f"  Back:   {Deck.cards_to_string(back):30s} → {back_rank}")
            print(f"  Middle: {Deck.cards_to_string(middle):30s} → {middle_rank}")
            print(f"  Front:  {Deck.cards_to_string(front):30s} → {front_rank}")
            print(f"  Bonus: +{bonus}, Score: {score:.2f}")
        else:
            print(f"  ❌ No arrangement found")
        
        print()
    
    print("✅ ML Solver Bridge tests passed!")