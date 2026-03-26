"""
Ultimate Mau Binh Solver - FINAL VERSION V3 - COMPLETELY FIXED
Tích hợp: SmartSolver + ML + HYBRID MODE (BEST!)
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

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
engines_dir = os.path.join(current_dir, 'engines')
sys.path.insert(0, current_dir)
sys.path.insert(0, core_dir)
sys.path.insert(0, engines_dir)

from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType
from special_hands import SpecialHandsChecker, SpecialHandResult
from smart_solver import SmartSolver, BonusCalculator

# ML Bridge
ML_AVAILABLE = False
ml_bridge = None

try:
    from ml_solver_bridge import MLSolverBridge
    ml_bridge = MLSolverBridge()
    ML_AVAILABLE = ml_bridge.is_loaded
    if ML_AVAILABLE:
        print("✅ ML Agent loaded")
except Exception as e:
    print(f"⚠️  ML not available: {e}")

# RewardCalculator
REWARD_CALC_AVAILABLE = False
reward_calculator = None

try:
    # Thử import path 1
    from ml.core.reward_calculator import RewardCalculator
    reward_calculator = RewardCalculator()
    REWARD_CALC_AVAILABLE = True
except ImportError:
    try:
        # Thử import path 2
        sys.path.insert(0, os.path.join(current_dir, 'ml', 'core'))
        from reward_calculator import RewardCalculator
        reward_calculator = RewardCalculator()
        REWARD_CALC_AVAILABLE = True
    except Exception as e:
        print(f"⚠️ RewardCalculator not available: {e}")
        pass


class SolverMode(Enum):
    """Các mode giải bài"""
    FAST = "fast"              # SmartSolver top 1
    BALANCED = "balanced"      # SmartSolver top 5
    ACCURATE = "accurate"      # SmartSolver top 10
    ULTIMATE = "ultimate"      # SmartSolver top 20
    ML_ONLY = "ml_only"        # ML Agent only (fallback to hybrid)
    ML_BEST = "ml_best"        # ML Ensemble
    ML_FAST = "ml_fast"        # ML DQN only
    ML_BEAM = "ml_beam"        # ML Beam Search
    ML_HYBRID = "ml_hybrid"    # 🔥 BEST MODE! SmartSolver + ML scoring


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
        
        bonus_display = f"+{self.bonus}" if self.bonus > 0 else "0"
        
        return f"""
╔════════════════════════════════════════════════════════════╗
║  🏆 ULTIMATE MẬU BINH SOLVER V3 - RESULT                   ║
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
║  • Total Score:      {self.total_score:6.2f}                              ║
║  • Expected Value:   {self.ev:+6.2f} units                           ║
║  • Bonus Points:     {bonus_display:>4s} chi                              ║
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
║  • Time:             {self.computation_time:6.3f}s                           ║
║  • Arrangements:     {self.num_arrangements_evaluated:6d}                               ║
╚════════════════════════════════════════════════════════════╝
"""


class UltimateSolver:
    """Ultimate Mậu Binh Solver V3 - FIXED"""

    def __init__(
        self,
        cards: List[Card],
        mode: SolverMode = SolverMode.BALANCED,
        verbose: bool = False
    ):
        self.cards = cards
        self.mode = mode
        self.verbose = verbose
        self.smart_solver = SmartSolver()

    def solve(self) -> SolverResult:
        """Main solve method"""
        start_time = time.time()

        # Seed for consistency
        seed = sum(c.to_index() for c in self.cards)
        random.seed(seed)
        np.random.seed(seed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🎯 SOLVING with mode: {self.mode.value}")
            print(f"{'='*60}\n")

        # Check binh đặc biệt
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

        # Dispatch to mode
        mode_dispatch = {
            SolverMode.FAST: self._solve_fast,
            SolverMode.BALANCED: self._solve_balanced,
            SolverMode.ACCURATE: self._solve_accurate,
            SolverMode.ULTIMATE: self._solve_ultimate,
            SolverMode.ML_ONLY: self._solve_ml_hybrid,
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
            print("⚡ Fast mode: SmartSolver top 1")

        results = self.smart_solver.find_best_arrangement(self.cards, top_k=1)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.30, p_win_2_of_3=0.50,
            p_win_front=0.50, p_win_middle=0.50, p_win_back=0.50,
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

        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.35, p_win_2_of_3=0.55,
            p_win_front=0.55, p_win_middle=0.55, p_win_back=0.55,
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
            p_scoop=0.40, p_win_2_of_3=0.60,
            p_win_front=0.60, p_win_middle=0.60, p_win_back=0.60,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    def _solve_ultimate(self) -> SolverResult:
        """Ultimate: SmartSolver top 20 hoặc hybrid nếu ML available"""
        if self.verbose:
            print("🚀 Ultimate mode")

        # Nếu có ML → dùng hybrid
        if ML_AVAILABLE and ml_bridge and ml_bridge.is_loaded:
            return self._solve_ml_hybrid()
        
        # Nếu không → SmartSolver top 20
        results = self.smart_solver.find_best_arrangement(self.cards, top_k=20)

        if not results or results[0][0] is None:
            return self._fallback_solve()

        back, middle, front, score = results[0]
        bonus = BonusCalculator.calculate(back, middle, front)

        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=score, ev=score / 10, bonus=bonus,
            p_scoop=0.45, p_win_2_of_3=0.65,
            p_win_front=0.65, p_win_middle=0.65, p_win_back=0.65,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=len(results)
        )

    # ==================== ML MODES ====================

    def _solve_ml_best(self) -> SolverResult:
        """ML Best: Ensemble"""
        if self.verbose:
            print("🤖 ML Best mode: Ensemble")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_balanced()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='best')
        
        if back is None:
            return self._solve_balanced()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_fast(self) -> SolverResult:
        """ML Fast: DQN only"""
        if self.verbose:
            print("⚡ ML Fast mode: DQN only")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_fast()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='fast')
        
        if back is None:
            return self._solve_fast()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_beam(self) -> SolverResult:
        """ML Beam: Beam Search"""
        if self.verbose:
            print("🔍 ML Beam mode")
        
        if not ML_AVAILABLE or ml_bridge is None or not ml_bridge.is_loaded:
            return self._solve_accurate()
        
        back, middle, front, metrics = ml_bridge.solve(self.cards, mode='beam')
        
        if back is None:
            return self._solve_accurate()
        
        return self._create_ml_result(back, middle, front, metrics)

    def _solve_ml_hybrid(self) -> SolverResult:
        """
        🔥 ML HYBRID - BEST MODE!
        
        Workflow:
        1. SmartSolver tìm TOP 10-20 candidates (valid + high quality)
        2. RewardCalculator chấm điểm chi tiết (bonus-aware)
        3. ML Agent scoring (nếu có)
        4. Weighted combination → chọn best
        """
        if self.verbose:
            print("🔥 ML Hybrid mode: SmartSolver + ML scoring")
        
        # Step 1: SmartSolver candidates
        smart_results = self.smart_solver.find_best_arrangement(self.cards, top_k=15)
        
        if not smart_results or smart_results[0][0] is None:
            return self._fallback_solve()
        
        # Step 2: Score mỗi candidate
        best_arr = None
        best_combined_score = -float('inf')
        best_metrics = {}
        num_valid = 0
        
        for back, middle, front, smart_score in smart_results:
            if back is None:
                continue
            
            # RewardCalculator score
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
                        continue  # Skip invalid
                except:
                    calc_reward = smart_score
                    num_valid += 1
            else:
                calc_reward = smart_score
                num_valid += 1
            
            # ML score (nếu có)
            ml_reward = 0
            if ML_AVAILABLE and ml_bridge and ml_bridge.is_loaded:
                try:
                    ml_eval = ml_bridge.evaluate(back, middle, front)
                    ml_reward = ml_eval.get('reward', 0)
                except:
                    ml_reward = 0
            
            # Combined score
            combined_score = (
                smart_score * 0.25 +
                calc_reward * 0.55 +
                ml_reward * 0.20
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
        
        # Fallback
        if best_arr is None:
            if self.verbose:
                print("⚠️  Hybrid failed, using balanced")
            return self._solve_balanced()
        
        back, middle, front = best_arr
        reward = best_metrics.get('reward', 0)
        bonus = best_metrics.get('bonus', 0)
        
        # Calculate probabilities
        p_base = 0.5 + (reward / 100) * 0.40
        p_base = max(0.40, min(0.90, p_base))
        
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
            p_scoop=min(p_base * 0.70, 0.60),
            p_win_2_of_3=min(p_base * 1.20, 0.95),
            p_win_front=p_base,
            p_win_middle=p_base * 0.98,
            p_win_back=p_base,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=best_metrics.get('num_candidates', 15),
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
        """Create result from ML output"""
        reward = metrics.get('reward', 0)
        bonus = metrics.get('bonus', 0)
        
        p_base = 0.5 + (reward / 100) * 0.35
        p_base = max(0.35, min(0.85, p_base))
        
        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=reward,
            ev=reward / 10,
            bonus=bonus,
            p_scoop=min(p_base * 0.65, 0.55),
            p_win_2_of_3=min(p_base * 1.15, 0.90),
            p_win_front=p_base,
            p_win_middle=p_base * 0.95,
            p_win_back=p_base,
            mode=self.mode,
            computation_time=0,
            num_arrangements_evaluated=1,
            ml_metrics=metrics
        )

    def _fallback_solve(self) -> SolverResult:
        """Fallback khi không tìm được"""
        sorted_cards = sorted(self.cards, key=lambda c: c.rank.value, reverse=True)
        back = sorted_cards[:5]
        middle = sorted_cards[5:10]
        front = sorted_cards[10:13]
        
        return SolverResult(
            back=back, middle=middle, front=front,
            total_score=0, ev=0, bonus=0,
            p_scoop=0.10, p_win_2_of_3=0.30,
            p_win_front=0.30, p_win_middle=0.30, p_win_back=0.30,
            mode=self.mode, computation_time=0,
            num_arrangements_evaluated=1
        )


# ==================== HELPER FUNCTIONS ====================

def get_available_modes() -> List[str]:
    """Get available modes"""
    modes = ['fast', 'balanced', 'accurate', 'ultimate']
    
    if ML_AVAILABLE:
        modes.extend(['ml_only', 'ml_best', 'ml_fast', 'ml_beam', 'ml_hybrid'])
    elif REWARD_CALC_AVAILABLE:
        modes.extend(['ml_hybrid'])
    
    return modes


def is_ml_available() -> bool:
    """Check if ML available"""
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


if __name__ == "__main__":
    print("ultimate_solver.py V3 - OK")