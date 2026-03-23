"""
Ultimate Mau Binh Solver - FINAL VERSION
Tích hợp: SmartSolver + Probability + Game Theory + Multi-Objective + ML V2
"""
import sys
import time
import random
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, 'core')
sys.path.insert(0, 'engines')
sys.path.insert(0, 'ml')

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType
from probability_engine import ProbabilityEngine
from game_theory import GameTheoryEngine
from multi_objective import MultiObjectiveOptimizer, ObjectiveWeights
from adaptive_strategy import AdaptiveStrategySelector, GameContext
from smart_solver import SmartSolver

# ML Bridge
ML_AVAILABLE = False
ml_bridge = None

try:
    from ml_solver_bridge import MLSolverBridge
    ml_bridge = MLSolverBridge()
    ML_AVAILABLE = ml_bridge.is_loaded

    if ML_AVAILABLE:
        print("✅ ML Model V2 loaded")
    else:
        print("⚠️  ML Model V2 not found, using traditional methods")
except Exception as e:
    print(f"⚠️  ML module not available: {e}")


class SolverMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    ULTIMATE = "ultimate"
    ML_ONLY = "ml_only"


@dataclass
class SolverResult:
    back: List[Card]
    middle: List[Card]
    front: List[Card]
    total_score: float
    ev: float
    bonus: int
    p_scoop: float
    p_win_2_of_3: float
    p_win_front: float
    p_win_middle: float
    p_win_back: float
    mode: SolverMode
    computation_time: float
    num_arrangements_evaluated: int

    def __str__(self):
        return f"""
╔════════════════════════════════════════════════════════════╗
║  🏆 ULTIMATE MẬU BINH SOLVER - RESULT                      ║
╠════════════════════════════════════════════════════════════╣
║  BEST ARRANGEMENT:                                         ║
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

    def __init__(
        self,
        cards: List[Card],
        mode: SolverMode = SolverMode.BALANCED,
        game_context: Optional[GameContext] = None,
        verbose: bool = False
    ):
        self.cards = cards
        self.mode = mode
        self.game_context = game_context
        self.verbose = verbose

        # CORE: Smart Solver (brute force deterministic)
        self.smart_solver = SmartSolver()

        # Engines (for probability analysis)
        self.prob_engine = ProbabilityEngine(cards, verbose=False)
        self.gt_engine = GameTheoryEngine(cards, verbose=False)

        if game_context:
            weights = AdaptiveStrategySelector.select_weights(game_context)
        else:
            weights = ObjectiveWeights()

        self.mo_optimizer = MultiObjectiveOptimizer(cards, weights=weights, verbose=False)

    def solve(self) -> SolverResult:
        start_time = time.time()

        # Set seed for Monte Carlo consistency
        seed = sum(c.to_index() for c in self.cards)
        random.seed(seed)
        np.random.seed(seed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SOLVING with mode: {self.mode.value}")
            print(f"{'='*60}\n")

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

        result.computation_time = time.time() - start_time
        result.mode = self.mode

        # Reset random
        random.seed(None)
        np.random.seed(None)

        if self.verbose:
            print(result)

        return result

    def _solve_fast(self) -> SolverResult:
        """Fast: SmartSolver top 1 + quick eval"""
        if self.verbose:
            print("⚡ Fast mode: SmartSolver")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=1)

        if not results:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        best_arr = (back, middle, front)

        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=500)

        return self._make_result(best_arr, ev_result, 1)

    def _solve_balanced(self) -> SolverResult:
        """Balanced: SmartSolver top 5 + Monte Carlo chọn best"""
        if self.verbose:
            print("⚖️  Balanced mode: SmartSolver + Monte Carlo")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=5)

        if not results:
            return self._fallback_solve()

        # Monte Carlo evaluate top 5
        best_arr = None
        best_ev = -float('inf')

        for back, middle, front, score in results:
            arr = (back, middle, front)
            ev_result = self.gt_engine.calculate_ev(arr, num_simulations=3000)

            if ev_result.ev > best_ev:
                best_ev = ev_result.ev
                best_arr = arr

        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=5000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=5000)

        return self._make_result_full(best_arr, ev_result, prob_result, best_ev, len(results))

    def _solve_accurate(self) -> SolverResult:
        """Accurate: SmartSolver top 10 + detailed eval"""
        if self.verbose:
            print("🎯 Accurate mode: SmartSolver + detailed evaluation")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)

        if not results:
            return self._fallback_solve()

        best_arr = None
        best_ev = -float('inf')

        for back, middle, front, score in results:
            arr = (back, middle, front)
            ev_result = self.gt_engine.calculate_ev(arr, num_simulations=8000)

            if ev_result.ev > best_ev:
                best_ev = ev_result.ev
                best_arr = arr

        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=10000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=10000)

        return self._make_result_full(best_arr, ev_result, prob_result, best_ev, len(results))

    def _solve_ultimate(self) -> SolverResult:
        """Ultimate: SmartSolver top 10 + ML + full eval"""
        if self.verbose:
            print("🚀 Ultimate mode: SmartSolver + ML + full evaluation")

        # Phase 1: SmartSolver top 10
        results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)
        candidates = [(back, middle, front) for back, middle, front, score in results]

        if self.verbose:
            print(f"  📊 SmartSolver: {len(candidates)} candidates")
            for i, (back, middle, front, score) in enumerate(results[:5]):
                back_r = HandEvaluator.evaluate(back)
                mid_r = HandEvaluator.evaluate(middle)
                front_r = HandEvaluator.evaluate(front)
                print(f"    #{i+1}: Score={score:.2f} | "
                      f"Back={back_r} | Mid={mid_r} | Front={front_r}")

        # Phase 2: Add ML suggestions
        if ML_AVAILABLE and ml_bridge:
            ml_arrs = ml_bridge.get_top_arrangements(self.cards, top_k=5)
            # Only add ML suggestions that are valid and not already in candidates
            added = 0
            for ml_arr in ml_arrs:
                if ml_arr not in candidates:
                    candidates.append(ml_arr)
                    added += 1
            if self.verbose:
                print(f"  🤖 ML added {added} unique suggestions")

        if self.verbose:
            print(f"  📊 Total candidates: {len(candidates)}")

        # Phase 3: Detailed EV evaluation of ALL candidates
        best_arr = None
        best_ev = -float('inf')

        for arr in candidates:
            ev_result = self.gt_engine.calculate_ev(arr, num_simulations=10000)
            if ev_result.ev > best_ev:
                best_ev = ev_result.ev
                best_arr = arr

        # Phase 4: Final evaluation with high simulations
        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=15000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=15000)

        return self._make_result_full(best_arr, ev_result, prob_result, best_ev, len(candidates))

    def _solve_ml_only(self) -> SolverResult:
        """ML mode: SmartSolver as primary, ML as supplement"""
        if self.verbose:
            print("🤖 ML mode: SmartSolver + ML Bridge")

        # SmartSolver is primary (always optimal)
        results = self.smart_solver.find_best_arrangement(self.cards, top_k=1)

        if results:
            back, middle, front, score = results[0]
            best_arr = (back, middle, front)
        elif ML_AVAILABLE and ml_bridge:
            best_arr = ml_bridge.get_best_arrangement(self.cards)
        else:
            return self._fallback_solve()

        if not best_arr:
            return self._fallback_solve()

        ev_result = self.gt_engine.calculate_ev(best_arr, num_simulations=3000)
        prob_result = self.prob_engine.calculate_win_probability(best_arr, num_simulations=3000)

        return self._make_result_full(best_arr, ev_result, prob_result, ev_result.ev, 1)

    def _fallback_solve(self) -> SolverResult:
        """Fallback khi không tìm được arrangement"""
        sorted_cards = sorted(self.cards, key=lambda c: c.rank.value, reverse=True)
        arr = (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])
        ev_result = self.gt_engine.calculate_ev(arr, num_simulations=500)
        return self._make_result(arr, ev_result, 1)

    def _is_valid_arrangement(self, back, middle, front) -> bool:
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        try:
            return HandEvaluator.evaluate(back) >= HandEvaluator.evaluate(middle)
        except:
            return False

    def _make_result(self, arr, ev_result, num_evaluated):
        return SolverResult(
            back=arr[0], middle=arr[1], front=arr[2],
            total_score=ev_result.ev, ev=ev_result.ev, bonus=ev_result.bonus,
            p_scoop=ev_result.p_win_3_0, p_win_2_of_3=ev_result.p_win_2_1,
            p_win_front=0.5, p_win_middle=0.5, p_win_back=0.5,
            mode=self.mode, computation_time=0, num_arrangements_evaluated=num_evaluated
        )

    def _make_result_full(self, arr, ev_result, prob_result, total_score, num_evaluated):
        return SolverResult(
            back=arr[0], middle=arr[1], front=arr[2],
            total_score=total_score, ev=ev_result.ev, bonus=ev_result.bonus,
            p_scoop=prob_result.p_scoop, p_win_2_of_3=prob_result.p_win_2_of_3,
            p_win_front=prob_result.p_win_front,
            p_win_middle=prob_result.p_win_middle,
            p_win_back=prob_result.p_win_back,
            mode=self.mode, computation_time=0, num_arrangements_evaluated=num_evaluated
        )


# ==================== DEMO ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🃏 ULTIMATE MẬU BINH SOLVER - FINAL VERSION")
    print("=" * 60)

    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    cards = Deck.parse_hand(hand_str)

    print(f"\n📇 Input: {hand_str}\n")

    for mode in [SolverMode.FAST, SolverMode.ML_ONLY, SolverMode.BALANCED, SolverMode.ULTIMATE]:
        solver = UltimateSolver(cards, mode=mode, verbose=True)
        result = solver.solve()
        print("\n" + "=" * 60 + "\n")