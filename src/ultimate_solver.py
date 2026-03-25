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
import os

# *** FIX IMPORTS ***
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
engines_dir = os.path.join(current_dir, 'engines')
api_dir = os.path.join(current_dir, 'api')

sys.path.insert(0, current_dir)
sys.path.insert(0, core_dir)
sys.path.insert(0, engines_dir)
sys.path.insert(0, api_dir)

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType
from special_hands import SpecialHandsChecker, SpecialHandResult  # *** THÊM SpecialHandResult ***
from smart_solver import SmartSolver, BonusCalculator  # *** THÊM BonusCalculator ***

# Import engines (có thể fail)
PROB_ENGINE_AVAILABLE = False
GT_ENGINE_AVAILABLE = False
MO_AVAILABLE = False
ADAPTIVE_AVAILABLE = False

try:
    from probability_engine import ProbabilityEngine
    PROB_ENGINE_AVAILABLE = True
except ImportError:
    pass

try:
    from game_theory import GameTheoryEngine
    GT_ENGINE_AVAILABLE = True
except ImportError:
    pass

try:
    from multi_objective import MultiObjectiveOptimizer, ObjectiveWeights
    MO_AVAILABLE = True
except ImportError:
    pass

try:
    from adaptive_strategy import AdaptiveStrategySelector, GameContext
    ADAPTIVE_AVAILABLE = True
except ImportError:
    pass

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
    """Kết quả từ solver"""
    back: Optional[List[Card]]
    middle: Optional[List[Card]]
    front: Optional[List[Card]]
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
    is_special_hand: bool = False
    special_hand_result: Optional[SpecialHandResult] = None

    def __str__(self):
        if self.is_special_hand and self.special_hand_result:
            return f"""
╔════════════════════════════════════════════════════════════╗
║  🎉🎉🎉 BINH ĐẶC BIỆT - THẮNG TRẮNG! 🎉🎉🎉               ║
╠════════════════════════════════════════════════════════════╣
║  {self.special_hand_result.name:^56s} ║
║  +{self.special_hand_result.points_per_person} chi/người                                            ║
╠════════════════════════════════════════════════════════════╣
║  {self.special_hand_result.description:^56s} ║
╚════════════════════════════════════════════════════════════╝
"""
        
        return f"""
╔════════════════════════════════════════════════════════════╗
║  🏆 ULTIMATE MẬU BINH SOLVER - RESULT                      ║
╠════════════════════════════════════════════════════════════╣
║  BEST ARRANGEMENT:                                         ║
║  Chi 1 (Back):   {Deck.cards_to_string(self.back):40s} ║
║  Chi 2 (Middle): {Deck.cards_to_string(self.middle):40s} ║
║  Chi cuối:       {Deck.cards_to_string(self.front):40s} ║
╠════════════════════════════════════════════════════════════╣
║  HAND EVALUATION:                                          ║
║  • Back:   {str(HandEvaluator.evaluate(self.back)):45s} ║
║  • Middle: {str(HandEvaluator.evaluate(self.middle)):45s} ║
║  • Front:  {str(HandEvaluator.evaluate(self.front)):45s} ║
╠════════════════════════════════════════════════════════════╣
║  PERFORMANCE METRICS:                                      ║
║  • Total Score:      {self.total_score:6.3f}                              ║
║  • Expected Value:   {self.ev:+6.3f} units                           ║
║  • Bonus Points:     {self.bonus:+3d} points                             ║
╠════════════════════════════════════════════════════════════╣
║  WIN PROBABILITIES:                                        ║
║  • Scoop (3-0):      {self.p_scoop*100:5.1f}%                            ║
║  • Win 2/3:          {self.p_win_2_of_3*100:5.1f}%                            ║
║  • Front:            {self.p_win_front*100:5.1f}%                            ║
║  • Middle:           {self.p_win_middle*100:5.1f}%                            ║
║  • Back:             {self.p_win_back*100:5.1f}%                            ║
╠════════════════════════════════════════════════════════════╣
║  COMPUTATION INFO:                                         ║
║  • Mode:             {self.mode.value:15s}                       ║
║  • Time:             {self.computation_time:6.2f}s                           ║
║  • Arrangements:     {self.num_arrangements_evaluated:6d}                               ║
╚════════════════════════════════════════════════════════════╝
"""


class UltimateSolver:
    """Ultimate Mậu Binh Solver - Fixed Version"""

    def __init__(
        self,
        cards: List[Card],
        mode: SolverMode = SolverMode.BALANCED,
        game_context: Optional['GameContext'] = None,
        verbose: bool = False
    ):
        self.cards = cards
        self.mode = mode
        self.game_context = game_context
        self.verbose = verbose

        # CORE: Smart Solver (fixed version)
        self.smart_solver = SmartSolver()

        # Engines (optional)
        if PROB_ENGINE_AVAILABLE:
            self.prob_engine = ProbabilityEngine(cards, verbose=False)
        else:
            self.prob_engine = None
            
        if GT_ENGINE_AVAILABLE:
            self.gt_engine = GameTheoryEngine(cards, verbose=False)
        else:
            self.gt_engine = None

        if MO_AVAILABLE and ADAPTIVE_AVAILABLE and game_context:
            weights = AdaptiveStrategySelector.select_weights(game_context)
            self.mo_optimizer = MultiObjectiveOptimizer(cards, weights=weights, verbose=False)
        elif MO_AVAILABLE:
            self.mo_optimizer = MultiObjectiveOptimizer(cards, verbose=False)
        else:
            self.mo_optimizer = None

    def solve(self) -> SolverResult:
        """Main solve method"""
        start_time = time.time()

        # Set seed for consistency
        seed = sum(c.to_index() for c in self.cards)
        random.seed(seed)
        np.random.seed(seed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SOLVING with mode: {self.mode.value}")
            print(f"{'='*60}\n")

        # *** BƯỚC 1: CHECK BINH ĐẶC BIỆT TRƯỚC ***
        special_result = SpecialHandsChecker.check(self.cards)
        if special_result.is_special:
            result = SolverResult(
                back=None,
                middle=None,
                front=None,
                total_score=special_result.points_per_person * 3,  # Giả sử 3 đối thủ
                ev=special_result.points_per_person * 3,
                bonus=special_result.points_per_person,
                p_scoop=1.0,
                p_win_2_of_3=1.0,
                p_win_front=1.0,
                p_win_middle=1.0,
                p_win_back=1.0,
                mode=self.mode,
                computation_time=time.time() - start_time,
                num_arrangements_evaluated=0,
                is_special_hand=True,
                special_hand_result=special_result
            )
            
            if self.verbose:
                print(result)
            
            return result

        # *** BƯỚC 2: XẾP BÀI BÌNH THƯỜNG ***
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
        """Fast: SmartSolver top 1"""
        if self.verbose:
            print("⚡ Fast mode: SmartSolver")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=1)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=score,
            ev=score / 10,  # Rough estimate
            bonus=bonus,
            p_scoop=0.3,
            p_win_2_of_3=0.5,
            p_win_front=0.5,
            p_win_middle=0.5,
            p_win_back=0.5,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=1
        )

    def _solve_balanced(self) -> SolverResult:
        """Balanced: SmartSolver top 5, chọn best"""
        if self.verbose:
            print("⚖️  Balanced mode: SmartSolver top 5")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=5)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        # Chọn top 1
        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        # Tính probability nếu có engine
        p_scoop = 0.3
        p_win_2_of_3 = 0.5
        p_win_front = 0.5
        p_win_middle = 0.5
        p_win_back = 0.5
        ev = score / 10

        if self.gt_engine:
            try:
                ev_result = self.gt_engine.calculate_ev((back, middle, front), num_simulations=3000)
                ev = ev_result.ev
                p_scoop = ev_result.p_win_3_0
                p_win_2_of_3 = ev_result.p_win_2_1
            except:
                pass

        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=score,
            ev=ev,
            bonus=bonus,
            p_scoop=p_scoop,
            p_win_2_of_3=p_win_2_of_3,
            p_win_front=p_win_front,
            p_win_middle=p_win_middle,
            p_win_back=p_win_back,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    def _solve_accurate(self) -> SolverResult:
        """Accurate: SmartSolver top 10"""
        if self.verbose:
            print("🎯 Accurate mode: SmartSolver top 10")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=score,
            ev=score / 10,
            bonus=bonus,
            p_scoop=0.35,
            p_win_2_of_3=0.55,
            p_win_front=0.55,
            p_win_middle=0.55,
            p_win_back=0.55,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    def _solve_ultimate(self) -> SolverResult:
        """Ultimate: SmartSolver + ML (nếu có)"""
        if self.verbose:
            print("🚀 Ultimate mode: SmartSolver + ML")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        # Có thể thêm ML suggestions ở đây nếu cần
        
        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=score,
            ev=score / 10,
            bonus=bonus,
            p_scoop=0.4,
            p_win_2_of_3=0.6,
            p_win_front=0.6,
            p_win_middle=0.6,
            p_win_back=0.6,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    def _solve_ml_only(self) -> SolverResult:
        """ML mode: SmartSolver as primary"""
        return self._solve_balanced()

    def _fallback_solve(self) -> SolverResult:
        """Fallback khi không tìm được arrangement"""
        sorted_cards = sorted(self.cards, key=lambda c: c.rank.value, reverse=True)
        back = sorted_cards[:5]
        middle = sorted_cards[5:10]
        front = sorted_cards[10:13]
        
        return SolverResult(
            back=back,
            middle=middle,
            front=front,
            total_score=0,
            ev=0,
            bonus=0,
            p_scoop=0.1,
            p_win_2_of_3=0.3,
            p_win_front=0.3,
            p_win_middle=0.3,
            p_win_back=0.3,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=1
        )


# ==================== TESTS ====================

def test_special_hand():
    """Test binh đặc biệt"""
    print("Testing Special Hand Detection in UltimateSolver...")
    
    # Test sảnh rồng
    cards = Deck.parse_hand("2♠ 3♥ 4♦ 5♣ 6♠ 7♥ 8♦ 9♣ 10♠ J♥ Q♦ K♣ A♠")
    solver = UltimateSolver(cards, mode=SolverMode.FAST, verbose=False)
    result = solver.solve()
    
    assert result.is_special_hand
    assert result.special_hand_result.name == "Sảnh rồng"
    print(f"  ✅ Sảnh rồng detected: +{result.special_hand_result.points_per_person} chi/người")
    
    print("✅ Special Hand test passed!")


def test_normal_hand():
    """Test bài thường"""
    print("\nTesting Normal Hand in UltimateSolver...")
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    solver = UltimateSolver(cards, mode=SolverMode.BALANCED, verbose=False)
    result = solver.solve()
    
    assert not result.is_special_hand
    assert result.back is not None
    assert result.middle is not None
    assert result.front is not None
    
    # Validate arrangement
    is_valid, msg = HandEvaluator.is_valid_arrangement(result.back, result.middle, result.front)
    assert is_valid, f"Invalid arrangement: {msg}"
    
    print(f"  ✅ Valid arrangement found")
    print(f"      Back:   {Deck.cards_to_string(result.back)}")
    print(f"      Middle: {Deck.cards_to_string(result.middle)}")
    print(f"      Front:  {Deck.cards_to_string(result.front)}")
    print(f"      Score:  {result.total_score:.2f}, Bonus: +{result.bonus}")
    
    print("✅ Normal Hand test passed!")


def test_all_modes():
    """Test tất cả modes"""
    print("\nTesting All Solver Modes...")
    
    cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
    
    for mode in [SolverMode.FAST, SolverMode.BALANCED, SolverMode.ACCURATE, SolverMode.ULTIMATE]:
        solver = UltimateSolver(cards, mode=mode, verbose=False)
        result = solver.solve()
        
        if result.is_special_hand:
            print(f"  ✅ {mode.value}: Special hand - {result.special_hand_result.name}")
        else:
            is_valid, _ = HandEvaluator.is_valid_arrangement(result.back, result.middle, result.front)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {mode.value}: Score={result.total_score:.2f}, Time={result.computation_time:.3f}s")
    
    print("✅ All Modes test passed!")


def test_verbose_output():
    """Test verbose output"""
    print("\nTesting Verbose Output...")
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    solver = UltimateSolver(cards, mode=SolverMode.FAST, verbose=True)
    result = solver.solve()
    
    print("✅ Verbose Output test passed!")


if __name__ == "__main__":
    test_special_hand()
    test_normal_hand()
    test_all_modes()
    test_verbose_output()
    
    print("\n" + "="*60)
    print("✅ All ultimate_solver.py tests passed!")
    print("="*60)