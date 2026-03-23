"""
Probability Engine - Tính xác suất thắng dựa trên Monte Carlo simulation
Đây là trái tim của việc đánh giá bài!
"""
import random
import sys
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import time

# Cache manager
from cache_manager import get_cache_manager

# Import từ core
sys.path.insert(0, '../core')
from card import Card, Deck, Rank, Suit
from evaluator import HandEvaluator
from hand_types import HandRank, HandType


@dataclass
class ProbabilityResult:
    """Kết quả phân tích xác suất"""
    # Xác suất thắng từng chi
    p_win_front: float
    p_win_middle: float
    p_win_back: float
    
    # Xác suất tổng hợp
    p_win_2_of_3: float  # Thắng ít nhất 2/3 chi
    p_scoop: float        # Thắng cả 3 chi
    p_lose_all: float     # Thua cả 3 chi
    
    # Phân phối độ mạnh đối thủ
    front_distribution: Counter
    middle_distribution: Counter
    back_distribution: Counter
    
    # Thống kê
    num_simulations: int
    simulation_time: float
    
    def __str__(self):
        return f"""
Probability Analysis ({self.num_simulations} simulations in {self.simulation_time:.2f}s):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Win Probabilities:
  • Front (Chi cuối):  {self.p_win_front*100:5.1f}%
  • Middle (Chi 2):     {self.p_win_middle*100:5.1f}%
  • Back (Chi 1):       {self.p_win_back*100:5.1f}%

Combined Probabilities:
  • Win 2/3 chi:        {self.p_win_2_of_3*100:5.1f}%
  • Scoop (3/3):        {self.p_scoop*100:5.1f}%
  • Lose all:           {self.p_lose_all*100:5.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


class ProbabilityEngine:
    """
    Engine tính toán xác suất bằng Monte Carlo simulation
    """
    
    def __init__(self, my_cards: List[Card], verbose: bool = False):
        """
        Args:
            my_cards: 13 lá bài của mình
            verbose: In log chi tiết
        """
        self.my_cards = my_cards
        self.verbose = verbose
        
        # Tạo danh sách 39 lá còn lại
        full_deck = Deck.full_deck()
        self.remaining_cards = [c for c in full_deck if c not in my_cards]
        
        # Cache để tăng tốc
        self._eval_cache = {}

        # Cache manager
        self.cache = get_cache_manager()
    
    def simulate_opponents(
        self,
        num_simulations: int = 10000,
        num_opponents: int = 1
    ) -> Dict[str, any]:
        """
        Mô phỏng bài của đối thủ
        
        Args:
            num_simulations: Số lần mô phỏng
            num_opponents: Số đối thủ (1-3)
            
        Returns:
            Dict chứa thống kê phân phối bài đối thủ
        """
        if self.verbose:
            print(f"🎲 Simulating {num_simulations} opponent hands...")
        
        start_time = time.time()
        
        # Lưu phân phối độ mạnh của đối thủ
        front_types = []
        middle_types = []
        back_types = []
        
        front_ranks = []
        middle_ranks = []
        back_ranks = []
        
        for i in range(num_simulations):
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/{num_simulations}")
            
            # Random 13 lá cho đối thủ
            opp_cards = random.sample(self.remaining_cards, 13)
            
            # Xếp bài tối ưu cho đối thủ (dùng greedy - nhanh hơn)
            opp_arrangement = self._greedy_arrange(opp_cards)
            
            if opp_arrangement:
                back, middle, front = opp_arrangement
                
                # Đánh giá từng chi
                back_rank = self._cached_evaluate(back)
                middle_rank = self._cached_evaluate(middle)
                front_rank = self._cached_evaluate(front)
                
                # Lưu loại bài
                front_types.append(front_rank.hand_type)
                middle_types.append(middle_rank.hand_type)
                back_types.append(back_rank.hand_type)
                
                # Lưu toàn bộ rank để phân tích chi tiết
                front_ranks.append(front_rank)
                middle_ranks.append(middle_rank)
                back_ranks.append(back_rank)
        
        elapsed = time.time() - start_time
        
        if self.verbose:
            print(f"✅ Completed in {elapsed:.2f}s")
        
        return {
            'front_types': Counter(front_types),
            'middle_types': Counter(middle_types),
            'back_types': Counter(back_types),
            'front_ranks': front_ranks,
            'middle_ranks': middle_ranks,
            'back_ranks': back_ranks,
            'num_simulations': num_simulations,
            'simulation_time': elapsed
        }

    def simulate_opponents_cached(
        self,
        num_simulations: int = 10000,
        cache_key_suffix: str = ""
    ) -> Dict[str, any]:
        """
        Simulate với cache
        Cache key dựa trên my_cards và num_simulations
        """
        # Tạo cache key từ bài của mình
        cards_key = ''.join(str(c) for c in sorted(self.my_cards))
        cache_key = f"sim_{cards_key}_{num_simulations}_{cache_key_suffix}"

        # Try get from cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            if self.verbose:
                print(f"✅ Using cached simulation results")
            return cached

        # Compute
        result = self.simulate_opponents(num_simulations)

        # Cache it
        try:
            self.cache.set(cache_key, result, memory_only=True)
        except Exception:
            # If cache API differs or fails, ignore caching silently
            if self.verbose:
                print("⚠️ Failed to write simulation results to cache")

        return result
    
    def calculate_win_probability(
        self,
        my_arrangement: Tuple[List[Card], List[Card], List[Card]],
        num_simulations: int = 10000
    ) -> ProbabilityResult:
        """
        Tính xác suất thắng cho cách xếp cụ thể
        
        Args:
            my_arrangement: (back, middle, front)
            num_simulations: Số lần mô phỏng
            
        Returns:
            ProbabilityResult
        """
        my_back, my_middle, my_front = my_arrangement
        
        # Đánh giá bài của mình
        my_back_rank = self._cached_evaluate(my_back)
        my_middle_rank = self._cached_evaluate(my_middle)
        my_front_rank = self._cached_evaluate(my_front)
        
        if self.verbose:
            print(f"My arrangement:")
            print(f"  Front:  {my_front_rank}")
            print(f"  Middle: {my_middle_rank}")
            print(f"  Back:   {my_back_rank}")
        
        # Simulate đối thủ
        opp_stats = self.simulate_opponents(num_simulations)
        
        # Tính xác suất thắng từng chi
        p_win_front = self._calculate_single_chi_win_prob(
            my_front_rank,
            opp_stats['front_ranks']
        )
        
        p_win_middle = self._calculate_single_chi_win_prob(
            my_middle_rank,
            opp_stats['middle_ranks']
        )
        
        p_win_back = self._calculate_single_chi_win_prob(
            my_back_rank,
            opp_stats['back_ranks']
        )
        
        # Tính xác suất tổng hợp
        # P(thắng cả 3 chi)
        p_scoop = p_win_front * p_win_middle * p_win_back
        
        # P(thắng ít nhất 2/3 chi)
        # = P(WWW) + P(WWL) + P(WLW) + P(LWW)
        p_win_2_of_3 = (
            p_scoop +
            p_win_front * p_win_middle * (1 - p_win_back) +
            p_win_front * (1 - p_win_middle) * p_win_back +
            (1 - p_win_front) * p_win_middle * p_win_back
        )
        
        # P(thua cả 3 chi)
        p_lose_all = (1 - p_win_front) * (1 - p_win_middle) * (1 - p_win_back)
        
        return ProbabilityResult(
            p_win_front=p_win_front,
            p_win_middle=p_win_middle,
            p_win_back=p_win_back,
            p_win_2_of_3=p_win_2_of_3,
            p_scoop=p_scoop,
            p_lose_all=p_lose_all,
            front_distribution=opp_stats['front_types'],
            middle_distribution=opp_stats['middle_types'],
            back_distribution=opp_stats['back_types'],
            num_simulations=num_simulations,
            simulation_time=opp_stats['simulation_time']
        )
    
    def _calculate_single_chi_win_prob(
        self,
        my_rank: HandRank,
        opp_ranks: List[HandRank]
    ) -> float:
        """
        Tính xác suất thắng 1 chi dựa trên phân phối đối thủ
        
        Args:
            my_rank: HandRank của mình
            opp_ranks: List HandRank của đối thủ từ simulation
            
        Returns:
            Xác suất thắng (0.0 - 1.0)
        """
        if not opp_ranks:
            return 0.5
        
        wins = 0
        ties = 0
        
        for opp_rank in opp_ranks:
            if my_rank > opp_rank:
                wins += 1
            elif my_rank == opp_rank:
                ties += 1
        
        # Khi bằng nhau, tính là 50% thắng
        total = len(opp_ranks)
        win_prob = (wins + ties * 0.5) / total
        
        return win_prob
    
    def _greedy_arrange(self, cards: List[Card]) -> Optional[Tuple]:
        """
        Xếp bài nhanh bằng thuật toán tham lam
        Dùng cho simulation (không cần tối ưu hoàn hảo)
        
        Returns:
            (back, middle, front) hoặc None nếu không xếp được
        """
        from collections import Counter
        
        # Phân tích bài
        ranks = [c.rank for c in cards]
        rank_counts = Counter(ranks)
        
        # Tìm các tổ hợp
        quads = [r for r, c in rank_counts.items() if c == 4]
        trips = [r for r, c in rank_counts.items() if c == 3]
        pairs = [r for r, c in rank_counts.items() if c == 2]
        
        # Chiến thuật đơn giản:
        # 1. Nếu có tứ quý -> để chi 1
        # 2. Nếu có 2 xám -> 1 xám chi cuối, 1 xám + đôi = cù lũ chi 2
        # 3. Nếu có xám + 2 đôi -> xám chi cuối, thú chi 2
        # 4. Còn lại -> đôi lớn nhất cho chi cuối
        
        sorted_cards = sorted(cards, key=lambda c: c.rank, reverse=True)
        
        # Simple greedy: chia 3-5-5
        front = sorted_cards[:3]
        middle = sorted_cards[3:8]
        back = sorted_cards[8:13]
        
        # Kiểm tra hợp lệ
        if self._is_valid_arrangement(back, middle, front):
            return (back, middle, front)
        
        # Nếu không hợp lệ, thử đổi chỗ
        # Đưa quân mạnh xuống chi sau
        front = sorted_cards[10:13]
        middle = sorted_cards[5:10]
        back = sorted_cards[:5]
        
        if self._is_valid_arrangement(back, middle, front):
            return (back, middle, front)
        
        # Nếu vẫn không được, return None
        return None
    
    def _is_valid_arrangement(
        self,
        back: List[Card],
        middle: List[Card],
        front: List[Card]
    ) -> bool:
        """Kiểm tra xếp bài có hợp lệ không"""
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        back_rank = self._cached_evaluate(back)
        middle_rank = self._cached_evaluate(middle)
        front_rank = self._cached_evaluate(front)
        
        # back >= middle
        if back_rank < middle_rank:
            return False
        
        # middle >= front (tricky vì khác số lá)
        # Quy tắc đơn giản: nếu front là xám, middle phải >= xám
        if front_rank.hand_type == HandType.THREE_OF_KIND:
            if middle_rank.hand_type < HandType.THREE_OF_KIND:
                return False
            if (middle_rank.hand_type == HandType.THREE_OF_KIND and
                middle_rank.primary_value < front_rank.primary_value):
                return False
        
        # Nếu front là đôi, middle phải >= đôi
        elif front_rank.hand_type == HandType.PAIR:
            if middle_rank.hand_type < HandType.PAIR:
                return False
            if (middle_rank.hand_type == HandType.PAIR and
                middle_rank.primary_value < front_rank.primary_value):
                return False
        
        return True
    
    def _cached_evaluate(self, cards: List[Card]) -> HandRank:
        """Evaluate với cache để tăng tốc"""
        # Tạo key từ cards
        key = tuple(sorted(cards))
        
        if key not in self._eval_cache:
            self._eval_cache[key] = HandEvaluator.evaluate(cards)
        
        return self._eval_cache[key]
    
    def analyze_opponent_distribution(
        self,
        num_simulations: int = 10000
    ) -> str:
        """
        Phân tích chi tiết phân phối bài đối thủ
        """
        stats = self.simulate_opponents(num_simulations)
        
        result = f"""
╔════════════════════════════════════════════════════════════╗
║  OPPONENT HAND DISTRIBUTION ANALYSIS                       ║
║  ({num_simulations} simulations)                           ║
╠════════════════════════════════════════════════════════════╣
"""
        
        # Front (Chi cuối)
        result += "║  FRONT (Chi cuối - 3 lá):                                ║\n"
        front_total = sum(stats['front_types'].values())
        for hand_type in sorted(stats['front_types'].keys(), reverse=True):
            count = stats['front_types'][hand_type]
            pct = count / front_total * 100
            result += f"║    {str(hand_type):15s} : {pct:5.1f}% ({count:5d})        ║\n"
        
        result += "╠════════════════════════════════════════════════════════════╣\n"
        
        # Middle (Chi 2)
        result += "║  MIDDLE (Chi 2 - 5 lá):                                   ║\n"
        middle_total = sum(stats['middle_types'].values())
        for hand_type in sorted(stats['middle_types'].keys(), reverse=True):
            count = stats['middle_types'][hand_type]
            pct = count / middle_total * 100
            result += f"║    {str(hand_type):15s} : {pct:5.1f}% ({count:5d})        ║\n"
        
        result += "╠════════════════════════════════════════════════════════════╣\n"
        
        # Back (Chi 1)
        result += "║  BACK (Chi 1 - 5 lá):                                     ║\n"
        back_total = sum(stats['back_types'].values())
        for hand_type in sorted(stats['back_types'].keys(), reverse=True):
            count = stats['back_types'][hand_type]
            pct = count / back_total * 100
            result += f"║    {str(hand_type):15s} : {pct:5.1f}% ({count:5d})        ║\n"
        
        result += "╚════════════════════════════════════════════════════════════╝\n"
        
        return result


# ==================== TESTS ====================

def test_probability_engine():
    """Test ProbabilityEngine"""
    print("Testing ProbabilityEngine...")
    
    # Test case: Bài khá mạnh
    hand_str = "A♠ A♥ K♦ K♣ Q♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠"
    my_cards = Deck.parse_hand(hand_str)
    
    print(f"My 13 cards: {Deck.cards_to_string(my_cards)}")
    
    # Tạo engine
    engine = ProbabilityEngine(my_cards, verbose=True)
    
    # Test simulate opponents (ít lần để test nhanh)
    print("\n--- Test 1: Simulate opponents ---")
    stats = engine.simulate_opponents(num_simulations=100)
    print(f"Front types: {stats['front_types']}")
    print(f"Middle types: {stats['middle_types']}")
    print(f"Back types: {stats['back_types']}")
    
    # Test arrangement
    print("\n--- Test 2: Calculate win probability ---")
    back = Deck.parse_hand("A♠ K♦ Q♠ J♦ 10♣")   # Sảnh A
    middle = Deck.parse_hand("A♥ K♣ Q♥ 9♠ 8♥")  # Đôi... không, mậu thầu A
    front = Deck.parse_hand("7♦ 6♣ 5♠")          # Mậu thầu 7
    
    # Fix middle để hợp lệ
    middle = Deck.parse_hand("K♣ Q♥ 9♠ 8♥ 7♦")
    front = Deck.parse_hand("A♥ K♦ 6♣")
    
    # Actually, let's use proper arrangement
    back = [my_cards[0], my_cards[1], my_cards[2], my_cards[3], my_cards[4]]  # A A K K Q
    middle = [my_cards[5], my_cards[6], my_cards[7], my_cards[8], my_cards[9]]  # Q J 10 9 8
    front = [my_cards[10], my_cards[11], my_cards[12]]  # 7 6 5
    
    prob_result = engine.calculate_win_probability(
        (back, middle, front),
        num_simulations=500  # Ít để test nhanh
    )
    
    print(prob_result)
    
    # Kiểm tra kết quả hợp lý
    assert 0 <= prob_result.p_win_front <= 1
    assert 0 <= prob_result.p_win_middle <= 1
    assert 0 <= prob_result.p_win_back <= 1
    assert 0 <= prob_result.p_scoop <= 1
    
    print("✅ ProbabilityEngine tests passed!")


def test_distribution_analysis():
    """Test phân tích phân phối"""
    print("\nTesting distribution analysis...")
    
    # Bài trung bình
    hand_str = "K♠ Q♥ J♦ 10♣ 9♠ 8♥ 7♦ 6♣ 5♠ 4♥ 3♦ 2♣ A♠"
    my_cards = Deck.parse_hand(hand_str)
    
    engine = ProbabilityEngine(my_cards, verbose=False)
    
    # Phân tích (ít simulations để test nhanh)
    analysis = engine.analyze_opponent_distribution(num_simulations=500)
    print(analysis)
    
    print("✅ Distribution analysis test passed!")


if __name__ == "__main__":
    test_probability_engine()
    print()
    test_distribution_analysis()
    print("\n✅ All probability_engine.py tests passed!")