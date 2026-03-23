"""
Game Theory Engine - Tính toán EV và tối ưu hóa chiến lược
Đây là bộ não của hệ thống!
"""
import sys
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, '../core')
from card import Card, Deck
from evaluator import HandEvaluator
from hand_types import HandType

from probability_engine import ProbabilityEngine, ProbabilityResult


class PayoffStructure(Enum):
    """Cấu trúc trả thưởng"""
    STANDARD = "standard"      # 1-2-3 chi = +1/-1, 3-0 = +3/-3
    TOURNAMENT = "tournament"  # Có thưởng đặc biệt
    CASH_GAME = "cash_game"    # Tính theo chip


@dataclass
class BonusPoints:
    """Điểm thưởng cho các tổ hợp đặc biệt"""
    # Chi cuối (3 lá)
    front_trip: int = 6  # Xám chi cuối
    
    # Chi giữa (5 lá)
    middle_full_house: int = 4   # Cù lũ
    middle_four_kind: int = 16   # Tứ quý
    middle_straight_flush: int = 20  # Thùng phá sảnh
    middle_royal_flush: int = 20     # Royal flush
    
    # Chi 1 (5 lá)
    back_four_kind: int = 8      # Tứ quý
    back_straight_flush: int = 10    # Thùng phá sảnh
    back_royal_flush: int = 10       # Royal flush
    
    def calculate_bonus(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> int:
        """Tính tổng điểm thưởng"""
        total = 0
        
        # Evaluate hands
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        # Front bonuses
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            total += self.front_trip
        
        # Middle bonuses
        if middle_rank.hand_type == HandType.ROYAL_FLUSH:
            total += self.middle_royal_flush
        elif middle_rank.hand_type == HandType.STRAIGHT_FLUSH:
            total += self.middle_straight_flush
        elif middle_rank.hand_type == HandType.FOUR_OF_KIND:
            total += self.middle_four_kind
        elif middle_rank.hand_type == HandType.FULL_HOUSE:
            total += self.middle_full_house
        
        # Back bonuses
        if back_rank.hand_type == HandType.ROYAL_FLUSH:
            total += self.back_royal_flush
        elif back_rank.hand_type == HandType.STRAIGHT_FLUSH:
            total += self.back_straight_flush
        elif back_rank.hand_type == HandType.FOUR_OF_KIND:
            total += self.back_four_kind
        
        return total


@dataclass
class EVResult:
    """Kết quả tính EV"""
    ev: float                    # Expected Value
    ev_no_bonus: float          # EV không tính thưởng
    bonus: int                   # Điểm thưởng
    
    # Breakdown
    p_win_3_0: float            # P(thắng cả 3)
    p_win_2_1: float            # P(thắng 2/3)
    p_lose_1_2: float           # P(thua 2/3)
    p_lose_0_3: float           # P(thua cả 3)
    
    # Payoffs
    payoff_3_0: float           # Tiền thắng khi 3-0
    payoff_2_1: float           # Tiền thắng khi 2-1
    payoff_1_2: float           # Tiền thua khi 1-2
    payoff_0_3: float           # Tiền thua khi 0-3
    
    # Risk metrics
    risk: float                 # Xác suất thua cả 3 chi
    upside: float              # Xác suất thắng cả 3 chi
    sharpe_ratio: float        # EV / Risk
    
    def __str__(self):
        return f"""
╔════════════════════════════════════════════════════════════╗
║  EXPECTED VALUE ANALYSIS                                   ║
╠════════════════════════════════════════════════════════════╣
║  Expected Value (EV):                                      ║
║    • Total EV:        {self.ev:+7.3f} units                    ║
║    • EV (no bonus):   {self.ev_no_bonus:+7.3f} units              ║
║    • Bonus Points:    {self.bonus:+7d} points                  ║
╠════════════════════════════════════════════════════════════╣
║  Outcome Probabilities:                                    ║
║    • Win 3-0 (scoop): {self.p_win_3_0*100:5.1f}% → {self.payoff_3_0:+6.1f} units  ║
║    • Win 2-1:         {self.p_win_2_1*100:5.1f}% → {self.payoff_2_1:+6.1f} units  ║
║    • Lose 1-2:        {self.p_lose_1_2*100:5.1f}% → {self.payoff_1_2:+6.1f} units ║
║    • Lose 0-3:        {self.p_lose_0_3*100:5.1f}% → {self.payoff_0_3:+6.1f} units ║
╠════════════════════════════════════════════════════════════╣
║  Risk Metrics:                                             ║
║    • Risk (lose all): {self.risk*100:5.1f}%                        ║
║    • Upside (scoop):  {self.upside*100:5.1f}%                      ║
║    • Sharpe Ratio:    {self.sharpe_ratio:7.3f}                     ║
╚════════════════════════════════════════════════════════════╝
"""


class GameTheoryEngine:
    """
    Game Theory Engine - Tính EV và tối ưu hóa
    """
    
    def __init__(
        self,
        my_cards: List[Card],
        payoff_structure: PayoffStructure = PayoffStructure.STANDARD,
        bonus_points: Optional[BonusPoints] = None,
        verbose: bool = False
    ):
        """
        Args:
            my_cards: 13 lá bài của mình
            payoff_structure: Cấu trúc trả thưởng
            bonus_points: Điểm thưởng tùy chỉnh
            verbose: In log chi tiết
        """
        self.my_cards = my_cards
        self.payoff_structure = payoff_structure
        self.bonus_points = bonus_points or BonusPoints()
        self.verbose = verbose
        
        # Probability engine
        self.prob_engine = ProbabilityEngine(my_cards, verbose=verbose)
    
    def calculate_ev(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]],
        base_bet: float = 1.0,
        num_simulations: int = 10000
    ) -> EVResult:
        """
        Tính Expected Value cho một cách xếp bài
        
        Args:
            arrangement: (back, middle, front)
            base_bet: Tiền cược cơ bản
            num_simulations: Số lần simulation
            
        Returns:
            EVResult
        """
        back, middle, front = arrangement
        
        # Tính xác suất thắng
        prob_result = self.prob_engine.calculate_win_probability(
            arrangement,
            num_simulations=num_simulations
        )
        
        p_win_front = prob_result.p_win_front
        p_win_middle = prob_result.p_win_middle
        p_win_back = prob_result.p_win_back
        
        # Tính các outcome probabilities
        # Win 3-0 (scoop)
        p_win_3_0 = p_win_front * p_win_middle * p_win_back
        
        # Win 2-1
        p_win_2_1_fmb = p_win_front * p_win_middle * (1 - p_win_back)
        p_win_2_1_fbm = p_win_front * (1 - p_win_middle) * p_win_back
        p_win_2_1_mfb = (1 - p_win_front) * p_win_middle * p_win_back
        p_win_2_1 = p_win_2_1_fmb + p_win_2_1_fbm + p_win_2_1_mfb
        
        # Lose 1-2
        p_lose_1_2_f = p_win_front * (1 - p_win_middle) * (1 - p_win_back)
        p_lose_1_2_m = (1 - p_win_front) * p_win_middle * (1 - p_win_back)
        p_lose_1_2_b = (1 - p_win_front) * (1 - p_win_middle) * p_win_back
        p_lose_1_2 = p_lose_1_2_f + p_lose_1_2_m + p_lose_1_2_b
        
        # Lose 0-3
        p_lose_0_3 = (1 - p_win_front) * (1 - p_win_middle) * (1 - p_win_back)
        
        # Tính payoffs theo cấu trúc
        if self.payoff_structure == PayoffStructure.STANDARD:
            payoff_3_0 = 3 * base_bet
            payoff_2_1 = 1 * base_bet
            payoff_1_2 = -1 * base_bet
            payoff_0_3 = -3 * base_bet
        else:
            # Có thể customize cho tournament, cash game
            payoff_3_0 = 3 * base_bet
            payoff_2_1 = 1 * base_bet
            payoff_1_2 = -1 * base_bet
            payoff_0_3 = -3 * base_bet
        
        # Tính EV không tính thưởng
        ev_no_bonus = (
            p_win_3_0 * payoff_3_0 +
            p_win_2_1 * payoff_2_1 +
            p_lose_1_2 * payoff_1_2 +
            p_lose_0_3 * payoff_0_3
        )
        
        # Tính điểm thưởng
        bonus = self.bonus_points.calculate_bonus(back, middle, front)
        
        # EV tổng = EV cơ bản + thưởng
        # (Thưởng chỉ nhận khi thắng hoặc hòa, giả định đơn giản là cộng vào)
        ev_total = ev_no_bonus + bonus * base_bet
        
        # Risk metrics
        risk = p_lose_0_3
        upside = p_win_3_0
        
        # Sharpe ratio = EV / Risk (higher is better)
        sharpe_ratio = ev_total / max(risk, 0.01)
        
        return EVResult(
            ev=ev_total,
            ev_no_bonus=ev_no_bonus,
            bonus=bonus,
            p_win_3_0=p_win_3_0,
            p_win_2_1=p_win_2_1,
            p_lose_1_2=p_lose_1_2,
            p_lose_0_3=p_lose_0_3,
            payoff_3_0=payoff_3_0,
            payoff_2_1=payoff_2_1,
            payoff_1_2=payoff_1_2,
            payoff_0_3=payoff_0_3,
            risk=risk,
            upside=upside,
            sharpe_ratio=sharpe_ratio
        )
    
    def compare_arrangements(
        self,
        arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        num_simulations: int = 5000
    ) -> List[Tuple[int, EVResult]]:
        """
        So sánh nhiều cách xếp bài
        
        Returns:
            List of (arrangement_index, EVResult) sorted by EV descending
        """
        results = []
        
        for i, arr in enumerate(arrangements):
            if self.verbose:
                print(f"Evaluating arrangement {i+1}/{len(arrangements)}...")
            
            ev_result = self.calculate_ev(arr, num_simulations=num_simulations)
            results.append((i, ev_result))
        
        # Sort by EV descending
        results.sort(key=lambda x: x[1].ev, reverse=True)
        
        return results
    
    def find_max_ev_arrangement(
        self,
        valid_arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        num_simulations: int = 5000,
        top_k: int = 10
    ) -> Tuple[Tuple[List[Card], List[Card], List[Card]], EVResult]:
        """
        Tìm cách xếp có EV cao nhất
        
        Args:
            valid_arrangements: Danh sách các cách xếp hợp lệ
            num_simulations: Số lần simulation
            top_k: Đánh giá chi tiết top K candidates
            
        Returns:
            (best_arrangement, ev_result)
        """
        if not valid_arrangements:
            return None, None
        
        if self.verbose:
            print(f"Finding max EV from {len(valid_arrangements)} arrangements...")
        
        # Phase 1: Quick screening với ít simulations
        if len(valid_arrangements) > top_k:
            if self.verbose:
                print(f"Phase 1: Quick screening with {num_simulations//5} sims...")
            
            quick_results = []
            for arr in valid_arrangements:
                ev_result = self.calculate_ev(arr, num_simulations=num_simulations//5)
                quick_results.append((arr, ev_result.ev))
            
            # Lấy top K
            quick_results.sort(key=lambda x: x[1], reverse=True)
            candidates = [arr for arr, _ in quick_results[:top_k]]
        else:
            candidates = valid_arrangements
        
        # Phase 2: Detailed evaluation
        if self.verbose:
            print(f"Phase 2: Detailed evaluation of top {len(candidates)}...")
        
        best_arr = None
        best_ev_result = None
        best_ev = float('-inf')
        
        for arr in candidates:
            ev_result = self.calculate_ev(arr, num_simulations=num_simulations)
            
            if ev_result.ev > best_ev:
                best_ev = ev_result.ev
                best_arr = arr
                best_ev_result = ev_result
        
        return best_arr, best_ev_result
    
    def risk_adjusted_selection(
        self,
        valid_arrangements: List[Tuple[List[Card], List[Card], List[Card]]],
        risk_tolerance: float = 0.3,
        num_simulations: int = 5000
    ) -> Tuple[Tuple[List[Card], List[Card], List[Card]], EVResult]:
        """
        Chọn cách xếp cân bằng giữa EV và Risk
        
        Args:
            valid_arrangements: Danh sách cách xếp hợp lệ
            risk_tolerance: Ngưỡng risk chấp nhận được (0.0 - 1.0)
            num_simulations: Số lần simulation
            
        Returns:
            (best_arrangement, ev_result)
        """
        if not valid_arrangements:
            return None, None
        
        if self.verbose:
            print(f"Risk-adjusted selection (tolerance={risk_tolerance})...")
        
        # Evaluate tất cả arrangements
        candidates = []
        
        for arr in valid_arrangements[:min(50, len(valid_arrangements))]:  # Limit để tăng tốc
            ev_result = self.calculate_ev(arr, num_simulations=num_simulations)
            
            # Chỉ xét các arrangement có risk chấp nhận được
            if ev_result.risk <= risk_tolerance:
                candidates.append((arr, ev_result))
        
        if not candidates:
            # Nếu không có arrangement nào đủ an toàn, chọn theo max EV
            if self.verbose:
                print("  No arrangement meets risk tolerance, selecting max EV...")
            return self.find_max_ev_arrangement(valid_arrangements, num_simulations)
        
        # Sắp xếp theo Sharpe ratio (EV/Risk)
        candidates.sort(key=lambda x: x[1].sharpe_ratio, reverse=True)
        
        best_arr, best_ev_result = candidates[0]
        
        if self.verbose:
            print(f"  Selected arrangement with Sharpe={best_ev_result.sharpe_ratio:.2f}")
        
        return best_arr, best_ev_result


# ==================== TESTS ====================

def test_bonus_calculation():
    """Test tính điểm thưởng"""
    print("Testing bonus calculation...")
    
    bonus_calc = BonusPoints()
    
    # Test 1: Xám chi cuối
    front = Deck.parse_hand("A♠ A♥ A♦")
    middle = Deck.parse_hand("K♠ Q♥ J♦ 10♣ 9♠")
    back = Deck.parse_hand("8♥ 7♦ 6♣ 5♠ 4♥")
    
    bonus = bonus_calc.calculate_bonus(back, middle, front)
    assert bonus == 6, f"Expected 6, got {bonus}"
    print(f"  Xám chi cuối: {bonus} points ✓")
    
    # Test 2: Tứ quý chi 2
    front = Deck.parse_hand("K♠ Q♥ J♦")
    middle = Deck.parse_hand("7♠ 7♥ 7♦ 7♣ A♠")
    back = Deck.parse_hand("K♥ K♦ Q♠ Q♣ J♠")
    
    bonus = bonus_calc.calculate_bonus(back, middle, front)
    assert bonus == 16, f"Expected 16, got {bonus}"
    print(f"  Tứ quý chi 2: {bonus} points ✓")
    
    # Test 3: Thùng phá sảnh chi 1
    front = Deck.parse_hand("K♠ Q♥ J♦")
    middle = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠")
    back = Deck.parse_hand("9♥ 8♥ 7♥ 6♥ 5♥")
    
    bonus = bonus_calc.calculate_bonus(back, middle, front)
    assert bonus == 10, f"Expected 10, got {bonus}"
    print(f"  Thùng phá sảnh chi 1: {bonus} points ✓")
    
    print("✅ Bonus calculation tests passed!")


def test_ev_calculation():
    """Test tính EV"""
    print("\nTesting EV calculation...")
    
    # Bài test
    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    my_cards = Deck.parse_hand(hand_str)
    
    engine = GameTheoryEngine(my_cards, verbose=True)
    
    # Cách xếp test
    back = my_cards[:5]    # A A K K Q
    middle = my_cards[5:10]  # Q J 10 9 8
    front = my_cards[10:13]  # 7 6 5
    
    print(f"\nArrangement:")
    print(f"  Back:   {Deck.cards_to_string(back)}")
    print(f"  Middle: {Deck.cards_to_string(middle)}")
    print(f"  Front:  {Deck.cards_to_string(front)}")
    
    # Tính EV với ít simulations để test nhanh
    ev_result = engine.calculate_ev(
        (back, middle, front),
        num_simulations=500
    )
    
    print(ev_result)
    
    # Kiểm tra kết quả hợp lý
    assert -3 <= ev_result.ev <= 3, f"EV out of range: {ev_result.ev}"
    assert 0 <= ev_result.risk <= 1, f"Risk out of range: {ev_result.risk}"
    assert ev_result.sharpe_ratio != 0, "Sharpe ratio should not be 0"
    
    print("✅ EV calculation tests passed!")


def test_risk_adjusted_selection():
    """Test risk-adjusted selection"""
    print("\nTesting risk-adjusted selection...")
    
    # Bài test
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    my_cards = Deck.parse_hand(hand_str)
    
    engine = GameTheoryEngine(my_cards, verbose=False)
    
    # Tạo vài cách xếp để test
    arr1 = (my_cards[:5], my_cards[5:10], my_cards[10:13])
    arr2 = (my_cards[8:13], my_cards[3:8], my_cards[:3])
    
    arrangements = [arr1, arr2]
    
    # Test với risk tolerance khác nhau
    for risk_tol in [0.2, 0.3, 0.5]:
        print(f"\nRisk tolerance: {risk_tol}")
        best_arr, ev_result = engine.risk_adjusted_selection(
            arrangements,
            risk_tolerance=risk_tol,
            num_simulations=200  # Ít để test nhanh
        )
        
        if best_arr:
            print(f"  Selected arrangement with risk={ev_result.risk:.2%}")
            assert ev_result.risk <= risk_tol or len(arrangements) == 1
    
    print("\n✅ Risk-adjusted selection tests passed!")


if __name__ == "__main__":
    test_bonus_calculation()
    test_ev_calculation()
    test_risk_adjusted_selection()
    print("\n" + "="*60)
    print("✅ All game_theory.py tests passed!")
    print("="*60)