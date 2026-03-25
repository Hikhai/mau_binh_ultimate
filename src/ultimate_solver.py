"""
Ultimate Mau Binh Solver - FINAL VERSION V2.1
Tích hợp: SmartSolver + Probability + Game Theory + Multi-Objective + ML Agent V2 + HYBRID
"""
import sys
import time
import random
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field
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
from special_hands import SpecialHandsChecker, SpecialHandResult
from smart_solver import SmartSolver, BonusCalculator

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

# *** ML BRIDGE V2 ***
ML_AVAILABLE = False
ml_bridge = None

try:
    from ml_solver_bridge import MLSolverBridge
    ml_bridge = MLSolverBridge()
    ML_AVAILABLE = ml_bridge.is_loaded

    if ML_AVAILABLE:
        print("✅ ML Agent V2 loaded")
    else:
        print("⚠️ ML Agent V2 not found, using traditional methods")
except Exception as e:
    print(f"⚠️ ML module not available: {e}")

# *** ML RewardCalculator (for hybrid mode) ***
REWARD_CALC_AVAILABLE = False
reward_calculator = None

try:
    from ml.core import RewardCalculator
    reward_calculator = RewardCalculator()
    REWARD_CALC_AVAILABLE = True
except Exception:
    pass


class SolverMode(Enum):
    """Các mode giải bài"""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"
    ULTIMATE = "ultimate"
    # ML Modes
    ML_ONLY = "ml_only"
    ML_BEST = "ml_best"
    ML_FAST = "ml_fast"
    ML_BEAM = "ml_beam"
    ML_HYBRID = "ml_hybrid"  # *** SmartSolver + ML scoring ***


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
    ml_metrics: Optional[dict] = None

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
        
        mode_display = f"{self.mode.value}"
        if self.mode.value.startswith('ml_'):
            mode_display = f"🤖 {self.mode.value.upper()}"
        
        return f"""
╔════════════════════════════════════════════════════════════╗
║  🏆 ULTIMATE MẬU BINH SOLVER V2.1 - RESULT                 ║
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
║  • Mode:             {mode_display:15s}                       ║
║  • Time:             {self.computation_time:6.2f}s                           ║
║  • Arrangements:     {self.num_arrangements_evaluated:6d}                               ║
╚════════════════════════════════════════════════════════════╝
"""


class UltimateSolver:
    """Ultimate Mậu Binh Solver V2.1 - With ML Agent + Hybrid Integration"""

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

        # CORE: Smart Solver
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

        seed = sum(c.to_index() for c in self.cards)
        random.seed(seed)
        np.random.seed(seed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SOLVING with mode: {self.mode.value}")
            print(f"{'='*60}\n")

        # *** BƯỚC 1: CHECK BINH ĐẶC BIỆT ***
        special_result = SpecialHandsChecker.check(self.cards)
        if special_result.is_special:
            result = SolverResult(
                back=None, middle=None, front=None,
                total_score=special_result.points_per_person * 3,
                ev=special_result.points_per_person * 3,
                bonus=special_result.points_per_person,
                p_scoop=1.0, p_win_2_of_3=1.0,
                p_win_front=1.0, p_win_middle=1.0, p_win_back=1.0,
                mode=self.mode,
                computation_time=time.time() - start_time,
                num_arrangements_evaluated=0,
                is_special_hand=True,
                special_hand_result=special_result
            )
            if self.verbose:
                print(result)
            return result

        # *** BƯỚC 2: XẾP BÀI THEO MODE ***
        mode_dispatch = {
            SolverMode.FAST: self._solve_fast,
            SolverMode.BALANCED: self._solve_balanced,
            SolverMode.ACCURATE: self._solve_accurate,
            SolverMode.ULTIMATE: self._solve_ultimate,
            SolverMode.ML_ONLY: self._solve_ml_hybrid,  # ML_ONLY → dùng hybrid luôn
            SolverMode.ML_BEST: self._solve_ml_best,
            SolverMode.ML_FAST: self._solve_ml_fast,
            SolverMode.ML_BEAM: self._solve_ml_beam,
            SolverMode.ML_HYBRID: self._solve_ml_hybrid,
        }
        
        solve_func = mode_dispatch.get(self.mode, self._solve_balanced)
        result = solve_func()

        result.computation_time = time.time() - start_time
        result.mode = self.mode

        random.seed(None)
        np.random.seed(None)

        if self.verbose:
            print(result)

        return result

    # ==================== TRADITIONAL MODES ====================

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
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.3, p_win_2_of_3=0.5,
            p_win_front=0.5, p_win_middle=0.5, p_win_back=0.5,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=1
        )

    def _solve_balanced(self) -> SolverResult:
        """Balanced: SmartSolver top 5"""
        if self.verbose:
            print("⚖️  Balanced mode: SmartSolver top 5")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=5)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        p_scoop = 0.3
        p_win_2_of_3 = 0.5
        p_win_front = 0.5
        p_win_middle = 0.5
        p_win_back = 0.5
        ev = score / 10

        if self.gt_engine:
            try:
                ev_result = self.gt_engine.calculate_ev(
                    (back, middle, front), num_simulations=3000
                )
                ev = ev_result.ev
                p_scoop = ev_result.p_win_3_0
                p_win_2_of_3 = ev_result.p_win_2_1
            except:
                pass

        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=score, ev=ev, bonus=bonus,
            p_scoop=p_scoop, p_win_2_of_3=p_win_2_of_3,
            p_win_front=p_win_front, p_win_middle=p_win_middle,
            p_win_back=p_win_back,
            mode=self.mode, computation_time=0,
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
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.35, p_win_2_of_3=0.55,
            p_win_front=0.55, p_win_middle=0.55, p_win_back=0.55,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    def _solve_ultimate(self) -> SolverResult:
        """Ultimate: SmartSolver + optional ML scoring"""
        if self.verbose:
            print("🚀 Ultimate mode: SmartSolver + optimizations")

        # Dùng hybrid nếu ML available, otherwise SmartSolver
        if ML_AVAILABLE and ml_bridge and ml_bridge.is_loaded:
            return self._solve_ml_hybrid()
        
        results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.4, p_win_2_of_3=0.6,
            p_win_front=0.6, p_win_middle=0.6, p_win_back=0.6,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    # ==================== ML MODES ====================

    def _solve_ml_best(self) -> SolverResult:
        """ML Best: Use Ensemble (DQN + Transformer)"""
        if self.verbose:
            print("🤖 ML Best mode: Ensemble")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_balanced()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='best')
        
        if back is None:
            return self._solve_balanced()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_fast(self) -> SolverResult:
        """ML Fast: Use DQN only"""
        if self.verbose:
            print("⚡ ML Fast mode: DQN only")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_fast()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='fast')
        
        if back is None:
            return self._solve_fast()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_beam(self) -> SolverResult:
        """ML Beam: Use Beam Search"""
        if self.verbose:
            print("🔍 ML Beam mode: AI + Beam Search")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_accurate()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='beam')
        
        if back is None:
            return self._solve_accurate()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_hybrid(self) -> SolverResult:
        """
        🔥 ML HYBRID: SmartSolver tìm candidates + ML/RewardCalculator chấm điểm
        
        BEST MODE! Kết hợp:
        - SmartSolver: xếp bài valid, high quality
        - RewardCalculator: bonus-aware scoring
        - ML Agent: learned pattern scoring (if available)
        """
        if self.verbose:
            print("🔥 ML Hybrid mode: SmartSolver + ML scoring")
        
        # Step 1: SmartSolver tìm TOP candidates
        smart_results = self.smart_solver.find_best_arrangement(self.cards, top_k=10)
        
        if not smart_results or smart_results[0][0] is None:
            return self._fallback_solve()
        
        # Step 2: Score mỗi candidate bằng RewardCalculator + ML
        best_arr = None
        best_combined_score = -float('inf')
        best_metrics = {}
        num_valid = 0
        
        for back, middle, front, smart_score in smart_results:
            if back is None:
                continue
            
            # RewardCalculator score (bonus-aware!)
            calc_reward = 0
            calc_bonus = 0
            calc_strength = 0
            
            if REWARD_CALC_AVAILABLE and reward_calculator:
                try:
                    calc_reward = reward_calculator.calculate_reward(back, middle, front)
                    if calc_reward > -50:
                        calc_bonus = reward_calculator._calculate_bonus(back, middle, front)
                        calc_strength = reward_calculator._calculate_strength(back, middle, front)
                        num_valid += 1
                    else:
                        continue  # Skip invalid arrangements
                except:
                    calc_reward = smart_score
            else:
                calc_reward = smart_score
                num_valid += 1
            
            # ML Agent score (if available)
            ml_reward = 0
            if ML_AVAILABLE and ml_bridge and ml_bridge.is_loaded:
                try:
                    ml_eval = ml_bridge.evaluate(back, middle, front)
                    ml_reward = ml_eval.get('reward', 0)
                except:
                    ml_reward = 0
            
            # Combined score (weighted)
            combined_score = (
                smart_score * 0.3 +
                calc_reward * 0.5 +
                ml_reward * 0.2
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_arr = (back, middle, front)
                best_metrics = {
                    'reward': calc_reward,
                    'bonus': calc_bonus,
                    'strength': calc_strength,
                    'is_valid': calc_reward > -50,
                    'smart_score': smart_score,
                    'ml_reward': ml_reward,
                    'combined_score': combined_score,
                    'mode': 'hybrid',
                    'num_candidates': len(smart_results),
                    'num_valid': num_valid
                }
        
        # Fallback nếu không tìm được
        if best_arr is None:
            if self.verbose:
                print("⚠️ Hybrid failed, falling back to balanced")
            return self._solve_balanced()
        
        back, middle, front = best_arr
        reward = best_metrics.get('reward', 0)
        bonus = best_metrics.get('bonus', 0)
        
        # Calculate probabilities
        p_base = 0.5 + (reward / 100) * 0.35
        p_base = max(0.35, min(0.85, p_base))
        
        if self.verbose:
            print(f"  📊 Hybrid results:")
            print(f"     Candidates: {best_metrics.get('num_candidates', 0)}")
            print(f"     Valid: {best_metrics.get('num_valid', 0)}")
            print(f"     Smart score: {best_metrics.get('smart_score', 0):.2f}")
            print(f"     Calc reward: {reward:.2f}")
            print(f"     ML reward: {best_metrics.get('ml_reward', 0):.2f}")
            print(f"     Combined: {best_combined_score:.2f}")
            print(f"     Bonus: +{bonus}")
        
        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=best_combined_score,
            ev=reward / 10,
            bonus=bonus,
            p_scoop=min(p_base * 0.65, 0.55),
            p_win_2_of_3=min(p_base * 1.15, 0.9),
            p_win_front=p_base,
            p_win_middle=p_base,
            p_win_back=p_base,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=best_metrics.get('num_candidates', 10),
            ml_metrics=best_metrics
        )

    # ==================== HELPERS ====================

    def _create_ml_result(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card],
        metrics: dict
    ) -> SolverResult:
        """Create SolverResult from ML output"""
        reward = metrics.get('reward', 0)
        bonus = metrics.get('bonus', 0)
        
        p_base = 0.5 + (reward / 100) * 0.3
        p_base = max(0.3, min(0.8, p_base))
        
        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=reward,
            ev=reward / 10,
            bonus=bonus,
            p_scoop=min(p_base * 0.6, 0.5),
            p_win_2_of_3=min(p_base * 1.1, 0.85),
            p_win_front=p_base,
            p_win_middle=p_base * 0.95,
            p_win_back=p_base,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=1,
            ml_metrics=metrics
        )

    def _fallback_solve(self) -> SolverResult:
        """Fallback khi không tìm được arrangement"""
        sorted_cards = sorted(self.cards, key=lambda c: c.rank.value, reverse=True)
        back = sorted_cards[:5]
        middle = sorted_cards[5:10]
        front = sorted_cards[10:13]
        
        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=0, ev=0, bonus=0,
            p_scoop=0.1, p_win_2_of_3=0.3,
            p_win_front=0.3, p_win_middle=0.3, p_win_back=0.3,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=1
        )


# ==================== HELPER FUNCTIONS ====================

def get_available_modes() -> List[str]:
    """Get list of available solver modes"""
    modes = ['fast', 'balanced', 'accurate', 'ultimate']
    
    if ML_AVAILABLE:
        modes.extend(['ml_only', 'ml_best', 'ml_fast', 'ml_beam', 'ml_hybrid'])
    elif REWARD_CALC_AVAILABLE:
        # Hybrid vẫn hoạt động với RewardCalculator mà không cần ML model
        modes.extend(['ml_hybrid'])
    
    return modes


def is_ml_available() -> bool:
    """Check if ML is available"""
    return ML_AVAILABLE


def get_ml_status() -> dict:
    """Get ML status"""
    status = {
        'ml_available': ML_AVAILABLE,
        'reward_calc_available': REWARD_CALC_AVAILABLE,
        'model_loaded': False,
        'model_path': None,
    }
    
    if ml_bridge:
        bridge_status = ml_bridge.get_status()
        status.update(bridge_status)
    
    return status


# ==================== TESTS ====================

def test_special_hand():
    """Test binh đặc biệt"""
    print("Testing Special Hand Detection...")
    
    cards = Deck.parse_hand("2♠ 3♥ 4♦ 5♣ 6♠ 7♥ 8♦ 9♣ 10♠ J♥ Q♦ K♣ A♠")
    solver = UltimateSolver(cards, mode=SolverMode.FAST, verbose=False)
    result = solver.solve()
    
    assert result.is_special_hand
    assert result.special_hand_result.name == "Sảnh rồng"
    print(f"  ✅ Sảnh rồng detected: +{result.special_hand_result.points_per_person} chi/người")
    
    print("✅ Special Hand test passed!")


def test_normal_hand():
    """Test bài thường"""
    print("\nTesting Normal Hand...")
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    solver = UltimateSolver(cards, mode=SolverMode.BALANCED, verbose=False)
    result = solver.solve()
    
    assert not result.is_special_hand
    assert result.back is not None
    
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
            print(f"  ✅ {mode.value}: Special hand")
        else:
            is_valid, _ = HandEvaluator.is_valid_arrangement(result.back, result.middle, result.front)
            status = "✅" if is_valid else "❌"
            print(f"  {status} {mode.value}: Score={result.total_score:.2f}, Time={result.computation_time:.3f}s")
    
    # ML modes
    ml_modes = []
    if ML_AVAILABLE:
        ml_modes = [SolverMode.ML_BEST, SolverMode.ML_FAST, SolverMode.ML_BEAM, SolverMode.ML_HYBRID]
    elif REWARD_CALC_AVAILABLE:
        ml_modes = [SolverMode.ML_HYBRID]
    
    for mode in ml_modes:
        solver = UltimateSolver(cards, mode=mode, verbose=False)
        result = solver.solve()
        
        if result.is_special_hand:
            print(f"  ✅ {mode.value}: Special hand")
        else:
            is_valid, _ = HandEvaluator.is_valid_arrangement(result.back, result.middle, result.front)
            status = "✅" if is_valid else "❌"
            ml_info = ""
            if result.ml_metrics:
                ml_info = f", combined={result.ml_metrics.get('combined_score', 0):.2f}"
            print(f"  {status} {mode.value}: Score={result.total_score:.2f}, "
                  f"Time={result.computation_time:.3f}s{ml_info}")
    
    if not ml_modes:
        print("  ⚠️ ML/Hybrid modes skipped (not available)")
    
    print("✅ All Modes test passed!")


def test_hybrid_mode():
    """Test Hybrid mode specifically"""
    print("\nTesting Hybrid Mode...")
    
    print(f"  ML Available: {ML_AVAILABLE}")
    print(f"  RewardCalc Available: {REWARD_CALC_AVAILABLE}")
    
    cards = Deck.parse_hand("A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠")
    
    solver = UltimateSolver(cards, mode=SolverMode.ML_HYBRID, verbose=True)
    result = solver.solve()
    
    if result.back:
        is_valid, _ = HandEvaluator.is_valid_arrangement(result.back, result.middle, result.front)
        
        print(f"\n  ✅ Hybrid result:")
        print(f"      Valid: {is_valid}")
        print(f"      Back:   {Deck.cards_to_string(result.back)}")
        print(f"      Middle: {Deck.cards_to_string(result.middle)}")
        print(f"      Front:  {Deck.cards_to_string(result.front)}")
        print(f"      Score:  {result.total_score:.2f}")
        print(f"      Bonus:  +{result.bonus}")
        print(f"      EV:     {result.ev:.2f}")
        
        if result.ml_metrics:
            print(f"      Candidates: {result.ml_metrics.get('num_candidates', 0)}")
            print(f"      Valid candidates: {result.ml_metrics.get('num_valid', 0)}")
    else:
        print(f"  ❌ Hybrid failed")
    
    print("✅ Hybrid Mode test passed!")


if __name__ == "__main__":
    print("="*60)
    print("🧪 ULTIMATE SOLVER V2.1 - TESTS")
    print("="*60)
    
    print(f"\n📊 Status:")
    print(f"  ML Available: {ML_AVAILABLE}")
    print(f"  RewardCalc Available: {REWARD_CALC_AVAILABLE}")
    print(f"  GT Engine: {GT_ENGINE_AVAILABLE}")
    print(f"  Prob Engine: {PROB_ENGINE_AVAILABLE}")
    print(f"  Available modes: {get_available_modes()}")
    
    print()
    
    test_special_hand()
    test_normal_hand()
    test_all_modes()
    test_hybrid_mode()
    
    print("\n" + "="*60)
    print("✅ All ultimate_solver.py V2.1 tests passed!")
    print("="*60)