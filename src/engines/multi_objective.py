"""
Multi-Objective Optimization Engine
Tối ưu hóa đồng thời nhiều mục tiêu
"""
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

sys.path.insert(0, '../core')
from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType

from game_theory import GameTheoryEngine, EVResult
from probability_engine import ProbabilityEngine


@dataclass
class ObjectiveWeights:
    """Trọng số cho các mục tiêu"""
    ev: float = 0.30              # Expected Value
    scoop: float = 0.25           # Xác suất thắng cả 3 chi
    bonus: float = 0.15           # Điểm thưởng
    front_strength: float = 0.20  # Độ mạnh chi cuối (quan trọng!)
    balance: float = 0.10         # Độ cân bằng 3 chi
    
    def normalize(self):
        """Chuẩn hóa tổng = 1.0"""
        total = self.ev + self.scoop + self.bonus + self.front_strength + self.balance
        if total > 0:
            self.ev /= total
            self.scoop /= total
            self.bonus /= total
            self.front_strength /= total
            self.balance /= total
    
    def __str__(self):
        return f"""
Objective Weights:
  • EV:              {self.ev:.1%}
  • Scoop chance:    {self.scoop:.1%}
  • Bonus:           {self.bonus:.1%}
  • Front strength:  {self.front_strength:.1%}
  • Balance:         {self.balance:.1%}
"""


@dataclass
class MultiObjectiveScore:
    """Điểm đa mục tiêu"""
    # Raw scores (0-1)
    ev_score: float
    scoop_score: float
    bonus_score: float
    front_score: float
    balance_score: float
    
    # Weighted total
    total_score: float
    
    # Original values
    ev: float
    p_scoop: float
    bonus: int
    front_strength: float
    balance: float
    
    def __str__(self):
        return f"""
Multi-Objective Scores:
──────────────────────────────────────────
Normalized Scores (0-1):
  • EV score:         {self.ev_score:.3f} (EV={self.ev:+.3f})
  • Scoop score:      {self.scoop_score:.3f} (P={self.p_scoop:.1%})
  • Bonus score:      {self.bonus_score:.3f} (bonus={self.bonus})
  • Front score:      {self.front_score:.3f} (strength={self.front_strength:.3f})
  • Balance score:    {self.balance_score:.3f} (balance={self.balance:.3f})

Total Weighted Score: {self.total_score:.3f}
"""


class MultiObjectiveOptimizer:
    """
    Multi-Objective Optimizer
    Tối ưu hóa đồng thời: EV, scoop chance, bonus, front strength, balance
    """
    
    def __init__(
        self,
        my_cards: List[Card],
        weights: Optional[ObjectiveWeights] = None,
        verbose: bool = False
    ):
        """
        Args:
            my_cards: 13 lá bài
            weights: Trọng số tùy chỉnh
            verbose: In log chi tiết
        """
        self.my_cards = my_cards
        self.weights = weights or ObjectiveWeights()
        self.weights.normalize()
        self.verbose = verbose
        
        # Engines
        self.prob_engine = ProbabilityEngine(my_cards, verbose=False)
        self.gt_engine = GameTheoryEngine(my_cards, verbose=False)
    
    def calculate_multi_objective_score(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        num_simulations: int = 5000
    ) -> MultiObjectiveScore:
        """
        Tính điểm đa mục tiêu cho một cách xếp
        
        Returns:
            MultiObjectiveScore
        """
        back, middle, front = arrangement
        
        # Calculate EV
        ev_result = self.gt_engine.calculate_ev(
            arrangement,
            num_simulations=num_simulations
        )
        
        # Objective 1: EV (normalize -3 to +3 -> 0 to 1)
        ev_normalized = self._normalize(ev_result.ev, -3, 3)
        
        # Objective 2: Scoop probability (already 0-1)
        scoop_normalized = ev_result.p_win_3_0
        
        # Objective 3: Bonus (normalize 0 to 20 -> 0 to 1)
        bonus_normalized = self._normalize(ev_result.bonus, 0, 20)
        
        # Objective 4: Front strength
        front_rank = HandEvaluator.evaluate(front)
        # Chi cuối có 3 lá: 0=high card, 1=pair, 3=trip
        # Normalize: trip=1.0, pair=0.5, high=0.0
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            front_strength = 1.0
        elif front_rank.hand_type == HandType.PAIR:
            # Pair càng cao càng tốt (2-14)
            front_strength = 0.3 + (front_rank.primary_value / 14) * 0.4
        else:
            # High card (mậu thầu)
            front_strength = (front_rank.primary_value / 14) * 0.3
        
        # Objective 5: Balance (độ cân bằng 3 chi)
        balance = self._calculate_balance(back, middle, front)
        
        # Calculate weighted total
        total_score = (
            self.weights.ev * ev_normalized +
            self.weights.scoop * scoop_normalized +
            self.weights.bonus * bonus_normalized +
            self.weights.front_strength * front_strength +
            self.weights.balance * balance
        )
        
        return MultiObjectiveScore(
            ev_score=ev_normalized,
            scoop_score=scoop_normalized,
            bonus_score=bonus_normalized,
            front_score=front_strength,
            balance_score=balance,
            total_score=total_score,
            ev=ev_result.ev,
            p_scoop=ev_result.p_win_3_0,
            bonus=ev_result.bonus,
            front_strength=front_strength,
            balance=balance
        )
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Chuẩn hóa giá trị về [0, 1]"""
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _calculate_balance(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> float:
        """
        Tính độ cân bằng của 3 chi
        Return: 0-1, càng cao càng cân bằng
        """
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        # Convert hand types to numerical strength
        back_strength = back_rank.hand_type.value
        middle_strength = middle_rank.hand_type.value
        # Front có 3 lá nên scale khác
        front_strength = front_rank.hand_type.value * 2  # x2 để tương đương
        
        # Tính variance (phương sai)
        strengths = [back_strength, middle_strength, front_strength]
        mean_strength = np.mean(strengths)
        variance = np.var(strengths)
        
        # Convert variance to balance score
        # Variance càng thấp = càng cân bằng = score càng cao
        # Assume max variance ~20 (rất không cân bằng)
        balance = 1.0 / (1.0 + variance / 10.0)
        
        return balance
    
    def find_pareto_optimal(
        self,
        valid_arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        num_simulations: int = 3000
    ) -> List[Tuple[Tuple, Dict]]:
        """
        Tìm Pareto Optimal solutions
        (Solutions mà không thể cải thiện 1 objective mà không làm xấu objective khác)
        
        Returns:
            List of (arrangement, objectives_dict)
        """
        if self.verbose:
            print(f"Finding Pareto optimal from {len(valid_arrangements)} arrangements...")
        
        # Evaluate all arrangements
        all_scores = []
        
        # Limit số lượng để tăng tốc
        sample_size = min(100, len(valid_arrangements))
        import random
        sampled = random.sample(valid_arrangements, sample_size)
        
        for i, arr in enumerate(sampled):
            if self.verbose and (i+1) % 20 == 0:
                print(f"  Evaluated {i+1}/{len(sampled)}...")
            
            score = self.calculate_multi_objective_score(arr, num_simulations)
            
            objectives = {
                'ev': score.ev,
                'scoop': score.p_scoop,
                'bonus': score.bonus,
                'front': score.front_strength,
                'balance': score.balance
            }
            
            all_scores.append((arr, objectives))
        
        # Find Pareto front
        pareto_front = []
        
        for i, (arr_i, obj_i) in enumerate(all_scores):
            is_dominated = False
            
            for j, (arr_j, obj_j) in enumerate(all_scores):
                if i == j:
                    continue
                
                # Check if obj_i is dominated by obj_j
                if self._dominates(obj_j, obj_i):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append((arr_i, obj_i))
        
        if self.verbose:
            print(f"  Found {len(pareto_front)} Pareto optimal solutions")
        
        return pareto_front
    
    def _dominates(self, obj1: Dict, obj2: Dict) -> bool:
        """
        Kiểm tra xem obj1 có dominate obj2 không
        obj1 dominates obj2 nếu:
        - obj1 >= obj2 trên TẤT CẢ objectives
        - obj1 > obj2 trên ÍT NHẤT MỘT objective
        """
        better_or_equal_all = all(obj1[k] >= obj2[k] for k in obj1.keys())
        better_at_least_one = any(obj1[k] > obj2[k] for k in obj1.keys())
        
        return better_or_equal_all and better_at_least_one
    
    def select_best(
        self,
        valid_arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        num_simulations: int = 5000,
        use_pareto: bool = True
    ) -> Tuple[Tuple[List[Card], List[Card], List[Card]], MultiObjectiveScore]:
        """
        Chọn arrangement tốt nhất
        
        Args:
            valid_arrangements: Danh sách các cách xếp hợp lệ
            num_simulations: Số lần simulation
            use_pareto: Nếu True, tìm Pareto optimal trước
            
        Returns:
            (best_arrangement, score)
        """
        if not valid_arrangements:
            return None, None
        
        if use_pareto and len(valid_arrangements) > 10:
            # Find Pareto optimal first
            pareto_front = self.find_pareto_optimal(
                valid_arrangements,
                num_simulations=num_simulations // 2
            )
            
            # Evaluate Pareto solutions với more simulations
            candidates = [arr for arr, _ in pareto_front]
        else:
            candidates = valid_arrangements
        
        # Detailed evaluation
        best_arr = None
        best_score = None
        best_total = -1
        
        for arr in candidates[:min(20, len(candidates))]:
            score = self.calculate_multi_objective_score(arr, num_simulations)
            
            if score.total_score > best_total:
                best_total = score.total_score
                best_arr = arr
                best_score = score
        
        return best_arr, best_score


# ==================== TESTS ====================

def test_objective_weights():
    """Test ObjectiveWeights"""
    print("Testing ObjectiveWeights...")
    
    weights = ObjectiveWeights(
        ev=0.3,
        scoop=0.3,
        bonus=0.2,
        front_strength=0.1,
        balance=0.1
    )
    
    weights.normalize()
    total = weights.ev + weights.scoop + weights.bonus + weights.front_strength + weights.balance
    assert abs(total - 1.0) < 0.01, f"Total should be 1.0, got {total}"
    
    print(weights)
    print("✅ ObjectiveWeights tests passed!")


def test_multi_objective_optimizer():
    """Test MultiObjectiveOptimizer"""
    print("\nTesting MultiObjectiveOptimizer...")
    
    # Test case
    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    my_cards = Deck.parse_hand(hand_str)
    
    optimizer = MultiObjectiveOptimizer(my_cards, verbose=True)
    
    # Test arrangement
    back = my_cards[:5]
    middle = my_cards[5:10]
    front = my_cards[10:13]
    
    print(f"\nArrangement:")
    print(f"  Back:   {Deck.cards_to_string(back)}")
    print(f"  Middle: {Deck.cards_to_string(middle)}")
    print(f"  Front:  {Deck.cards_to_string(front)}")
    
    # Calculate score
    score = optimizer.calculate_multi_objective_score(
        (back, middle, front),
        num_simulations=500
    )
    
    print(score)
    
    assert 0 <= score.total_score <= 1
    assert 0 <= score.ev_score <= 1
    assert 0 <= score.scoop_score <= 1
    
    print("✅ MultiObjectiveOptimizer tests passed!")


def test_pareto_optimal():
    """Test Pareto optimal finding"""
    print("\nTesting Pareto optimal finding...")
    
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    my_cards = Deck.parse_hand(hand_str)
    
    optimizer = MultiObjectiveOptimizer(my_cards, verbose=True)
    
    # FIX: Create valid arrangements (5-5-3)
    arr1 = (my_cards[:5], my_cards[5:10], my_cards[10:13])
    arr2 = (my_cards[8:13], my_cards[3:8], my_cards[:3])
    # arr3 cũ bị lỗi - sửa lại
    arr3 = (
        [my_cards[0], my_cards[2], my_cards[4], my_cards[6], my_cards[8]],   # 5 lá
        [my_cards[1], my_cards[3], my_cards[5], my_cards[7], my_cards[9]],   # 5 lá
        [my_cards[10], my_cards[11], my_cards[12]]                            # 3 lá
    )
    
    arrangements = [arr1, arr2, arr3]
    
    pareto_front = optimizer.find_pareto_optimal(
        arrangements,
        num_simulations=200
    )
    
    print(f"\nFound {len(pareto_front)} Pareto optimal solutions")
    assert len(pareto_front) > 0
    assert len(pareto_front) <= len(arrangements)
    
    print("✅ Pareto optimal tests passed!")


if __name__ == "__main__":
    test_objective_weights()
    test_multi_objective_optimizer()
    test_pareto_optimal()
    print("\n" + "="*60)
    print("✅ All multi_objective.py tests passed!")
    print("="*60)