"""
Benchmark - Đo performance của các engines
"""
import time
import sys
from typing import Callable, List
import statistics

sys.path.insert(0, '../core')
from card import Deck, Card
from probability_engine import ProbabilityEngine


class Benchmark:
    """Benchmark framework"""
    
    @staticmethod
    def measure_time(func: Callable, *args, **kwargs) -> tuple:
        """
        Đo thời gian thực thi
        
        Returns:
            (result, elapsed_time)
        """
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    
    @staticmethod
    def run_multiple(func: Callable, n: int, *args, **kwargs) -> dict:
        """
        Chạy function nhiều lần và tính statistics
        
        Returns:
            Dict với mean, median, std, min, max
        """
        times = []
        results = []
        
        for _ in range(n):
            result, elapsed = Benchmark.measure_time(func, *args, **kwargs)
            times.append(elapsed)
            results.append(result)
        
        return {
            'times': times,
            'results': results,
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }


def benchmark_probability_engine():
    """Benchmark ProbabilityEngine"""
    print("=" * 60)
    print("BENCHMARKING PROBABILITY ENGINE")
    print("=" * 60)
    
    # Test case
    hand_str = "A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠"
    my_cards = Deck.parse_hand(hand_str)
    
    engine = ProbabilityEngine(my_cards, verbose=False)
    
    # Benchmark 1: Simulate opponents với số lượng khác nhau
    print("\n1. Simulate Opponents Performance:")
    print("-" * 60)
    
    for num_sims in [100, 500, 1000, 5000, 10000]:
        _, elapsed = Benchmark.measure_time(
            engine.simulate_opponents,
            num_simulations=num_sims
        )
        
        sims_per_sec = num_sims / elapsed
        print(f"  {num_sims:5d} sims: {elapsed:6.2f}s ({sims_per_sec:7.1f} sims/sec)")
    
    # Benchmark 2: Cache impact
    print("\n2. Cache Impact:")
    print("-" * 60)
    
    # Without cache
    _, time_no_cache = Benchmark.measure_time(
        engine.simulate_opponents,
        num_simulations=5000
    )
    print(f"  Without cache: {time_no_cache:.2f}s")
    
    # With cache (lần 2)
    _, time_with_cache = Benchmark.measure_time(
        engine.simulate_opponents_cached,
        num_simulations=5000
    )
    print(f"  With cache (1st): {time_with_cache:.2f}s")
    
    # With cache (lần 3 - should be instant)
    _, time_cached = Benchmark.measure_time(
        engine.simulate_opponents_cached,
        num_simulations=5000
    )
    print(f"  With cache (2nd): {time_cached:.2f}s")
    print(f"  Speedup: {time_no_cache / max(time_cached, 0.001):.1f}x")
    
    # Benchmark 3: Win probability calculation
    print("\n3. Win Probability Calculation:")
    print("-" * 60)
    
    back = my_cards[:5]
    middle = my_cards[5:10]
    front = my_cards[10:13]
    
    stats = Benchmark.run_multiple(
        engine.calculate_win_probability,
        n=5,
        my_arrangement=(back, middle, front),
        num_simulations=1000
    )
    
    print(f"  Mean time: {stats['mean']:.2f}s")
    print(f"  Std dev:   {stats['std']:.2f}s")
    print(f"  Min/Max:   {stats['min']:.2f}s / {stats['max']:.2f}s")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    benchmark_probability_engine()