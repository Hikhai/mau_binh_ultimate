"""
Ultimate Mau Binh Solver
Tích hợp: Basic + Probability + Game Theory + Multi-Objective + ML
"""
import sys
import time
import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, 'core')
sys.path.insert(0, 'engines')
sys.path.insert(0, 'ml')  # <-- THÊM DÒNG NÀY

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType

# Engines
from probability_engine import ProbabilityEngine
from game_theory import GameTheoryEngine
from multi_objective import MultiObjectiveOptimizer, ObjectiveWeights
from adaptive_strategy import AdaptiveStrategySelector, GameContext

# ML (optional) - THAY ĐỔI PHẦN NÀY
ML_AVAILABLE = False
try:
    from dqn_agent import DQNAgent
    ML_AVAILABLE = True
    print("✅ ML Agent available")
except ImportError as e:
    print(f"⚠️  ML module not available: {e}")
    DQNAgent = None


class SolverMode(Enum):
    """Chế độ solver"""
    FAST = "fast"                    # Nhanh nhất, độ chính xác thấp
    BALANCED = "balanced"            # Cân bằng tốc độ và độ chính xác
    ACCURATE = "accurate"            # Chính xác cao, chậm hơn
    ULTIMATE = "ultimate"            # Tất cả methods, chậm nhất nhưng tốt nhất
    ML_ONLY = "ml_only"             # Chỉ dùng ML (nếu có)


@dataclass
class SolverResult:
    """Kết quả từ Ultimate Solver"""
    # Best arrangement
    back: List[Card]
    middle: List[Card]
    front: List[Card]
    
    # Scores
    total_score: float
    ev: float
    bonus: int
    
    # Probabilities
    p_scoop: float
    p_win_2_of_3: float
    p_win_front: float
    p_win_middle: float
    p_win_back: float
    
    # Metadata
    mode: SolverMode
    computation_time: float
    num_arrangements_evaluated: int
    
    def __str__(self):
        return f"""
╔════════════════════════════════════════════════════════════╗
║  🏆 ULTIMATE MẬU BINH SOLVER - RESULT                      ║
╠════════════════════════════════════════════════════════════╣
║  BEST ARRANGEMENT:                                         ║
║                                                            ║
║  Chi 1 (Back):   {Deck.cards_to_string(self.back):30s}    ║
║  Chi 2 (Middle): {Deck.cards_to_string(self.middle):30s}    ║
║  Chi cuối:       {Deck.cards_to_string(self.front):30s}    ║
╠════════════════════════════════════════════════════════════╣
║  HAND EVALUATION:                                          ║
║  • Back:   {str(HandEvaluator.evaluate(self.back)):40s} ║
║  • Middle: {str(HandEvaluator.evaluate(self.middle)):40s} ║
║  • Front:  {str(HandEvaluator.evaluate(self.front)):40s} ║
╠════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS:                                      ║
║  • Total Score:      {self.total_score:6.3f}                           ║
║  • Expected Value:   {self.ev:+6.3f} units                        ║
║  • Bonus Points:     {self.bonus:+3d} points                          ║
╠════════════════════════════════════════════════════════════╣
║  WIN PROBABILITIES:                                        ║
║  • Scoop (3-0):      {self.p_scoop*100:5.1f}%                         ║
║  • Win 2/3:          {self.p_win_2_of_3*100:5.1f}%                         ║
║  • Front:            {self.p_win_front*100:5.1f}%                         ║
║  • Middle:           {self.p_win_middle*100:5.1f}%                         ║
║  • Back:             {self.p_win_back*100:5.1f}%                         ║
╠════════════════════════════════════════════════════════════╣
║  COMPUTATION INFO:                                         ║
║  • Mode:             {self.mode.value:10s}                        ║
║  • Time:             {self.computation_time:6.2f}s                        ║
║  • Arrangements:     {self.num_arrangements_evaluated:6d}                            ║
╚════════════════════════════════════════════════════════════╝
"""


class UltimateSolver:
    """
    Ultimate Mậu Binh Solver
    Kết hợp tất cả methods để tìm cách xếp bài tối ưu nhất
    """
    
    def __init__(
        self,
        cards: List[Card],
        mode: SolverMode = SolverMode.BALANCED,
        game_context: Optional[GameContext] = None,
        ml_model_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Args:
            cards: 13 lá bài
            mode: Chế độ solver
            game_context: Ngữ cảnh game (cho adaptive strategy)
            ml_model_path: Path đến trained ML model
            verbose: In log chi tiết
        """
        self.cards = cards
        self.mode = mode
        self.game_context = game_context
        self.verbose = verbose
        
        # Initialize engines
        self.prob_engine = ProbabilityEngine(cards, verbose=False)
        self.gt_engine = GameTheoryEngine(cards, verbose=False)
        
        # Multi-objective optimizer
        if game_context:
            weights = AdaptiveStrategySelector.select_weights(game_context)
        else:
            weights = ObjectiveWeights()
        
        self.mo_optimizer = MultiObjectiveOptimizer(cards, weights=weights, verbose=False)
        
        # ML Agent (NEW!)
        self.ml_agent = None
        if ML_AVAILABLE and DQNAgent is not None:
            # Auto-detect model path
            if ml_model_path is None:
                # Try common locations
                possible_paths = [
                    "data/models/pro_training_v1/best_model.pth",
                    "../data/models/pro_training_v1/best_model.pth",
                    "../../data/models/pro_training_v1/best_model.pth",
                ]
                
                for path in possible_paths:
                    if Path(path).exists():
                        ml_model_path = path
                        break
            
            # Load model
            if ml_model_path and Path(ml_model_path).exists():
                try:
                    self.ml_agent = DQNAgent(
                        state_size=52,
                        action_size=1000,
                        use_dueling=True
                    )
                    self.ml_agent.load(ml_model_path)
                    
                    if self.verbose:
                        print(f"✅ ML model loaded from {ml_model_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Failed to load ML model: {e}")
                    self.ml_agent = None
            else:
                if self.verbose:
                    print(f"⚠️  ML model not found at {ml_model_path}")
    
    def solve(self) -> SolverResult:
        """
        Giải bài toán xếp bài tối ưu
        
        Returns:
            SolverResult
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SOLVING with mode: {self.mode.value}")
            print(f"{'='*60}\n")
        
        # Chọn method theo mode
        if self.mode == SolverMode.FAST:
            result = self._solve_fast()
        elif self.mode == SolverMode.BALANCED:
            result = self._solve_balanced()
        elif self.mode == SolverMode.ACCURATE:
            result = self._solve_accurate()
        elif self.mode == SolverMode.ULTIMATE:
            result = self._solve_ultimate()
        elif self.mode == SolverMode.ML_ONLY:
            result = self._solve_ml_only()
        else:
            result = self._solve_balanced()
        
        computation_time = time.time() - start_time
        
        # Add metadata
        result.mode = self.mode
        result.computation_time = computation_time
        
        if self.verbose:
            print(result)
        
        return result
    
    def _solve_fast(self) -> SolverResult:
        """Fast mode: Greedy heuristic"""
        if self.verbose:
            print("⚡ Fast mode: Using greedy heuristic")
        
        # Simple greedy arrangement
        sorted_cards = sorted(self.cards, key=lambda c: c.rank, reverse=True)
        
        # Try a few simple patterns
        arrangements = [
            (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13]),
            (sorted_cards[8:13], sorted_cards[3:8], sorted_cards[:3]),
        ]
        
        valid_arrangements = [
            arr for arr in arrangements 
            if self._is_valid_arrangement(*arr)
        ]
        
        if not valid_arrangements:
            # Fallback
            valid_arrangements = [arrangements[0]]
        
        best_arr = valid_arrangements[0]
        
        # Quick evaluation
        back, middle, front = best_arr
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=500)
        
        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=ev_result.ev,
            ev=ev_result.ev,
            bonus=ev_result.bonus,
            p_scoop=ev_result.p_win_3_0,
            p_win_2_of_3=ev_result.p_win_2_1,
            p_win_front=0.5,
            p_win_middle=0.5,
            p_win_back=0.5,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(valid_arrangements)
        )
    
    def _solve_balanced(self) -> SolverResult:
        """Balanced mode: Multi-objective with moderate simulations"""
        if self.verbose:
            print("⚖️  Balanced mode: Multi-objective optimization")
        
        # Generate valid arrangements (sample)
        valid_arrangements = self._generate_sample_arrangements(max_count=50)
        
        # Multi-objective selection
        best_arr, mo_score = self.mo_optimizer.select_best(
            valid_arrangements,
            num_simulations=3000,
            use_pareto=False
        )
        
        # Detailed evaluation
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=5000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=5000)
        
        return SolverResult(
            back=best_arr[0],
            middle=best_arr[1],
            front=best_arr[2],
            total_score=mo_score.total_score,
            ev=ev_result.ev,
            bonus=ev_result.bonus,
            p_scoop=prob_result.p_scoop,
            p_win_2_of_3=prob_result.p_win_2_of_3,
            p_win_front=prob_result.p_win_front,
            p_win_middle=prob_result.p_win_middle,
            p_win_back=prob_result.p_win_back,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(valid_arrangements)
        )
    
    def _solve_accurate(self) -> SolverResult:
        """Accurate mode: Full multi-objective with high simulations"""
        if self.verbose:
            print("🎯 Accurate mode: Comprehensive evaluation")
        
        valid_arrangements = self._generate_sample_arrangements(max_count=100)
        
        best_arr, mo_score = self.mo_optimizer.select_best(
            valid_arrangements,
            num_simulations=10000,
            use_pareto=True
        )
        
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=10000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=10000)
        
        return SolverResult(
            back=best_arr[0],
            middle=best_arr[1],
            front=best_arr[2],
            total_score=mo_score.total_score,
            ev=ev_result.ev,
            bonus=ev_result.bonus,
            p_scoop=prob_result.p_scoop,
            p_win_2_of_3=prob_result.p_win_2_of_3,
            p_win_front=prob_result.p_win_front,
            p_win_middle=prob_result.p_win_middle,
            p_win_back=prob_result.p_win_back,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(valid_arrangements)
        )
    
    def _solve_ultimate(self) -> SolverResult:
        """Ultimate mode: Combine all methods including ML"""
        if self.verbose:
            print("🚀 Ultimate mode: Combining all methods")
        
        # Phase 1: Multi-objective screening (traditional)
        valid_arrangements = self._generate_sample_arrangements(max_count=100)
        
        pareto_front = self.mo_optimizer.find_pareto_optimal(
            valid_arrangements,
            num_simulations=5000
        )
        
        # Phase 2: ML suggestions
        ml_suggestions = self._get_ml_suggestions(num_suggestions=20)
        
        # Combine candidates
        candidates = [arr for arr, _ in pareto_front[:10]]
        
        if ml_suggestions:
            candidates.extend(ml_suggestions[:5])
            if self.verbose:
                print(f"🤖 Added {len(ml_suggestions[:5])} ML suggestions to candidates")
        
        # Phase 3: Detailed evaluation
        best_arr = None
        best_score = -float('inf')
        
        for arr in candidates:
            ev_result = self.gt_engine.calculate_ev(arr, num_simulations=10000)
            
            if ev_result.ev > best_score:
                best_score = ev_result.ev
                best_arr = arr
        
        # Final evaluation
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=15000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=15000)
        
        return SolverResult(
            back=best_arr[0],
            middle=best_arr[1],
            front=best_arr[2],
            total_score=ev_result.ev,
            ev=ev_result.ev,
            bonus=ev_result.bonus,
            p_scoop=prob_result.p_scoop,
            p_win_2_of_3=prob_result.p_win_2_of_3,
            p_win_front=prob_result.p_win_front,
            p_win_middle=prob_result.p_win_middle,
            p_win_back=prob_result.p_win_back,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(valid_arrangements) + len(ml_suggestions)
        )
    
    def _solve_ml_only(self) -> SolverResult:
        """ML only mode - fast ML predictions"""
        if self.verbose:
            print("🤖 ML mode: Using trained Deep Q-Network")
        
        if not self.ml_agent:
            if self.verbose:
                print("⚠️  ML model not available, falling back to balanced mode")
            return self._solve_balanced()
        
        # Get ML suggestions
        ml_suggestions = self._get_ml_suggestions(num_suggestions=20)
        
        if not ml_suggestions:
            if self.verbose:
                print("⚠️  ML failed to generate valid suggestions, using traditional")
            return self._solve_fast()
        
        # Evaluate all ML suggestions with quick scoring
        best_arr = None
        best_score = -float('inf')
        
        for arr in ml_suggestions:
            # Quick evaluation (no heavy simulation)
            back, middle, front = arr
            
            # Calculate simple score
            back_rank = HandEvaluator.evaluate(back)
            middle_rank = HandEvaluator.evaluate(middle)
            front_rank = HandEvaluator.evaluate(front)
            
            # Bonus
            from game_theory import BonusPoints
            bonus_calc = BonusPoints()
            bonus = bonus_calc.calculate_bonus(back, middle, front)
            
            # Simple score
            score = (
                bonus * 3.0 +
                back_rank.hand_type.value * 1.0 +
                middle_rank.hand_type.value * 0.8 +
                front_rank.hand_type.value * 0.5
            )
            
            if score > best_score:
                best_score = score
                best_arr = arr
        
        # Detailed evaluation of best
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=3000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=3000)
        
        return SolverResult(
            back=best_arr[0],
            middle=best_arr[1],
            front=best_arr[2],
            total_score=ev_result.ev,
            ev=ev_result.ev,
            bonus=ev_result.bonus,
            p_scoop=prob_result.p_scoop,
            p_win_2_of_3=prob_result.p_win_2_of_3,
            p_win_front=prob_result.p_win_front,
            p_win_middle=prob_result.p_win_middle,
            p_win_back=prob_result.p_win_back,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(ml_suggestions)
        )
    
    def _generate_sample_arrangements(self, max_count: int = 50) -> List[Tuple]:
        """Generate SMART valid arrangements - not random!"""
        from itertools import combinations
        from collections import Counter
        
        valid = []
        cards = self.cards
        
        # Analyze hand
        ranks = [c.rank for c in cards]
        suits = [c.suit for c in cards]
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Find special combinations
        quads = [r for r, c in rank_counts.items() if c == 4]  # Tứ quý
        trips = [r for r, c in rank_counts.items() if c == 3]  # Xám
        pairs = sorted([r for r, c in rank_counts.items() if c == 2], reverse=True)  # Đôi
        
        # Strategy 1: Put highest pair/trip in FRONT (chi cuối)
        if trips:
            # Xám ở chi cuối = +6 bonus!
            trip_rank = max(trips)
            trip_cards = [c for c in cards if c.rank == trip_rank][:3]
            other_cards = [c for c in cards if c not in trip_cards]
            
            # Try different middle/back combinations
            for _ in range(10):
                random.shuffle(other_cards)
                middle = other_cards[:5]
                back = other_cards[5:10]
                
                if self._is_valid_arrangement(back, middle, trip_cards):
                    valid.append((back, middle, trip_cards))
        
        if len(pairs) >= 1:
            # Đôi lớn nhất ở chi cuối
            best_pair_rank = pairs[0]
            pair_cards = [c for c in cards if c.rank == best_pair_rank][:2]
            other_cards = [c for c in cards if c not in pair_cards]
            
            # Need 1 more card for front
            for extra_card in other_cards:
                front = pair_cards + [extra_card]
                remaining = [c for c in other_cards if c != extra_card]
                
                # Try to make strong middle and back
                for _ in range(5):
                    random.shuffle(remaining)
                    middle = remaining[:5]
                    back = remaining[5:10]
                    
                    if self._is_valid_arrangement(back, middle, front):
                        valid.append((back, middle, front))
                        break
        
        # Strategy 2: Check for Sảnh (Straight)
        sorted_cards = sorted(cards, key=lambda c: c.rank.value)
        
        # Try to find 5-card straight
        for i in range(9):  # Check starting positions
            potential_straight = []
            target_ranks = list(range(i + 2, i + 7))  # e.g., 2-6, 3-7, ..., 10-A
            
            for target in target_ranks:
                for card in cards:
                    if card.rank.value == target and card not in potential_straight:
                        potential_straight.append(card)
                        break
            
            if len(potential_straight) == 5:
                # Found a straight!
                other_cards = [c for c in cards if c not in potential_straight]
                
                # Put straight in BACK (chi 1) - it's strong!
                for _ in range(5):
                    random.shuffle(other_cards)
                    middle = other_cards[:5]
                    front = other_cards[5:8]
                    
                    if self._is_valid_arrangement(potential_straight, middle, front):
                        valid.append((potential_straight, middle, front))
                        break
        
        # Strategy 3: Two pairs - Thú
        if len(pairs) >= 2:
            pair1_rank = pairs[0]
            pair2_rank = pairs[1]
            
            pair1_cards = [c for c in cards if c.rank == pair1_rank][:2]
            pair2_cards = [c for c in cards if c.rank == pair2_rank][:2]
            
            # Two pair in middle or back
            two_pair_cards = pair1_cards + pair2_cards
            other_cards = [c for c in cards if c not in two_pair_cards]
            
            # Need 1 more for middle (5 cards), use highest remaining
            other_cards_sorted = sorted(other_cards, key=lambda c: c.rank.value, reverse=True)
            
            # Middle = two pair + 1
            middle = two_pair_cards + [other_cards_sorted[0]]
            remaining = other_cards_sorted[1:]
            
            # Front = highest pair from remaining, or best 3
            front = remaining[:3]
            back = remaining[3:8]
            
            if len(back) == 5 and self._is_valid_arrangement(back, middle, front):
                valid.append((back, middle, front))
            
            # Alternative: Two pair in back
            back = two_pair_cards + [other_cards_sorted[0]]
            middle = other_cards_sorted[1:6]
            front = other_cards_sorted[6:9]
            
            if len(middle) == 5 and len(front) == 3:
                if self._is_valid_arrangement(back, middle, front):
                    valid.append((back, middle, front))
        
        # Strategy 4: Three pairs - put best in front!
        if len(pairs) >= 3:
            # Best pair in front
            best_pair = [c for c in cards if c.rank == pairs[0]][:2]
            mid_pair = [c for c in cards if c.rank == pairs[1]][:2]
            low_pair = [c for c in cards if c.rank == pairs[2]][:2]
            other_cards = [c for c in cards if c.rank not in pairs[:3]]
            
            # Front: best pair + 1
            front = best_pair + [sorted(other_cards, key=lambda c: c.rank.value, reverse=True)[0]]
            remaining = [c for c in cards if c not in front]
            
            # Sort remaining by rank
            remaining_sorted = sorted(remaining, key=lambda c: c.rank.value, reverse=True)
            middle = remaining_sorted[:5]
            back = remaining_sorted[5:10]
            
            if self._is_valid_arrangement(back, middle, front):
                valid.append((back, middle, front))
        
        # Strategy 5: Random valid (fallback)
        for _ in range(max_count - len(valid)):
            shuffled = cards.copy()
            random.shuffle(shuffled)
            
            back = shuffled[:5]
            middle = shuffled[5:10]
            front = shuffled[10:13]
            
            if self._is_valid_arrangement(back, middle, front):
                valid.append((back, middle, front))
            
            if len(valid) >= max_count:
                break
        
        # Remove duplicates
        seen = set()
        unique_valid = []
        for arr in valid:
            key = (
                tuple(sorted([c.to_index() for c in arr[0]])),
                tuple(sorted([c.to_index() for c in arr[1]])),
                tuple(sorted([c.to_index() for c in arr[2]]))
            )
            if key not in seen:
                seen.add(key)
                unique_valid.append(arr)
        
        return unique_valid if unique_valid else [(cards[:5], cards[5:10], cards[10:13])]

    def _get_ml_suggestions(self, num_suggestions: int = 10) -> List[Tuple]:
        """
        Get arrangement suggestions from ML model
        
        Returns:
            List of (back, middle, front) tuples
        """
        if not self.ml_agent:
            return []
        
        suggestions = []
        
        try:
            # Encode state
            state = np.zeros(52, dtype=np.float32)
            for card in self.cards:
                state[card.to_index()] = 1.0
            
            # Get multiple predictions with different epsilon
            for epsilon in [0.0, 0.05, 0.1, 0.15, 0.2]:
                action = self.ml_agent.select_action(state, epsilon=epsilon)
                arrangement = self.ml_agent.action_encoder.decode_action(action, self.cards)
                
                # Validate
                if self._is_valid_arrangement(*arrangement):
                    suggestions.append(arrangement)
                
                if len(suggestions) >= num_suggestions:
                    break
            
            if self.verbose and suggestions:
                print(f"🤖 ML model suggested {len(suggestions)} arrangements")
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  ML prediction failed: {e}")
        
        return suggestions
    
    def _is_valid_arrangement(self, back, middle, front) -> bool:
        """Check if arrangement is valid"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        
        return back_rank >= middle_rank


# ==================== CLI ====================

def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultimate Mau Binh Solver')
    parser.add_argument('cards', type=str, help='13 cards (e.g., "AS KH QD ...")')
    parser.add_argument('--mode', type=str, default='balanced',
                        choices=['fast', 'balanced', 'accurate', 'ultimate'],
                        help='Solver mode (default: balanced)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse cards
    try:
        cards = Deck.parse_hand(args.cards)
        if len(cards) != 13:
            print(f"❌ Error: Need exactly 13 cards, got {len(cards)}")
            return
    except Exception as e:
        print(f"❌ Error parsing cards: {e}")
        return
    
    # Solve
    mode = SolverMode(args.mode)
    solver = UltimateSolver(cards, mode=mode, verbose=args.verbose)
    result = solver.solve()
    
    print(result)


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("🃏 ULTIMATE MẬU BINH SOLVER - DEMO")
    print("="*60)
    
    # Test hand
    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    cards = Deck.parse_hand(hand_str)
    
    print(f"\n📇 Input: {hand_str}\n")
    
    # Test different modes
    for mode in [SolverMode.FAST, SolverMode.BALANCED]:
        solver = UltimateSolver(cards, mode=mode, verbose=True)
        result = solver.solve()
        
        print("\n" + "="*60 + "\n")