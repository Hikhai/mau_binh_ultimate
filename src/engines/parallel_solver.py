"""
Parallel Solver using multiprocessing
"""
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
from card import Card
from evaluator import HandEvaluator


def evaluate_arrangement_worker(args):
    """Worker function for parallel evaluation"""
    arrangement, cards = args
    back, middle, front = arrangement
    
    try:
        back_cards = [cards[i] for i in back]
        middle_cards = [cards[i] for i in middle]
        front_cards = [cards[i] for i in front]
        
        back_rank = HandEvaluator.evaluate(back_cards)
        middle_rank = HandEvaluator.evaluate(middle_cards)
        front_rank = HandEvaluator.evaluate(front_cards)
        
        # Calculate score
        score = (
            back_rank.hand_type.value * 1000 + back_rank.primary_value * 10 +
            middle_rank.hand_type.value * 2000 + middle_rank.primary_value * 20 +
            front_rank.hand_type.value * 3000 + front_rank.primary_value * 30
        )
        
        return (arrangement, score, True)
    except:
        return (arrangement, -1, False)


class ParallelSolver:
    """
    Parallel solver using all CPU cores
    """
    
    def __init__(self, num_workers: int = None):
        """
        Args:
            num_workers: Number of worker processes (default: CPU count)
        """
        if num_workers is None:
            num_workers = cpu_count()
        
        self.num_workers = num_workers
        print(f"🚀 Parallel Solver with {self.num_workers} workers")
    
    def solve_parallel(
        self,
        cards: List[Card],
        valid_arrangements: List[Tuple]
    ) -> Tuple:
        """
        Evaluate arrangements in parallel
        
        Args:
            cards: 13 cards
            valid_arrangements: List of (back_indices, middle_indices, front_indices)
            
        Returns:
            Best arrangement
        """
        # Prepare work items
        work_items = [(arr, cards) for arr in valid_arrangements]
        
        # Parallel evaluation
        with Pool(self.num_workers) as pool:
            results = pool.map(evaluate_arrangement_worker, work_items)
        
        # Find best
        valid_results = [r for r in results if r[2]]
        
        if not valid_results:
            return None
        
        best = max(valid_results, key=lambda x: x[1])
        
        return best[0]
    
    def solve_batch(
        self,
        batch_cards: List[List[Card]],
        batch_arrangements: List[List[Tuple]]
    ) -> List[Tuple]:
        """
        Solve multiple hands in parallel
        
        Args:
            batch_cards: List of 13-card hands
            batch_arrangements: List of arrangement lists
            
        Returns:
            List of best arrangements
        """
        # Flatten work
        work_items = []
        work_indices = []
        
        for i, (cards, arrangements) in enumerate(zip(batch_cards, batch_arrangements)):
            for arr in arrangements:
                work_items.append((arr, cards))
                work_indices.append(i)
        
        # Parallel evaluation
        with Pool(self.num_workers) as pool:
            results = pool.map(evaluate_arrangement_worker, work_items)
        
        # Group by hand
        hand_results = {}
        for idx, result in zip(work_indices, results):
            if idx not in hand_results:
                hand_results[idx] = []
            hand_results[idx].append(result)
        
        # Find best for each hand
        best_arrangements = []
        for i in range(len(batch_cards)):
            if i in hand_results:
                valid = [r for r in hand_results[i] if r[2]]
                if valid:
                    best = max(valid, key=lambda x: x[1])
                    best_arrangements.append(best[0])
                else:
                    best_arrangements.append(None)
            else:
                best_arrangements.append(None)
        
        return best_arrangements


# ==================== BENCHMARK ====================

def benchmark_parallel():
    """Benchmark parallel vs sequential"""
    from card import Deck
    import time
    import random
    
    print("="*60)
    print("PARALLEL SOLVER BENCHMARK")
    print("="*60)
    
    # Generate test data
    num_hands = 100
    arrangements_per_hand = 100
    
    batch_cards = []
    batch_arrangements = []
    
    for _ in range(num_hands):
        deck = Deck.full_deck()
        cards = random.sample(deck, 13)
        batch_cards.append(cards)
        
        # Generate random arrangements
        arrangements = []
        for _ in range(arrangements_per_hand):
            indices = list(range(13))
            random.shuffle(indices)
            back = tuple(indices[:5])
            middle = tuple(indices[5:10])
            front = tuple(indices[10:13])
            arrangements.append((back, middle, front))
        
        batch_arrangements.append(arrangements)
    
    total_evaluations = num_hands * arrangements_per_hand
    
    print(f"Test: {num_hands} hands × {arrangements_per_hand} arrangements")
    print(f"Total: {total_evaluations:,} evaluations\n")
    
    # Sequential
    print("Sequential processing...")
    start = time.time()
    
    for cards, arrangements in zip(batch_cards, batch_arrangements):
        best = None
        best_score = -1
        
        for arr in arrangements:
            result = evaluate_arrangement_worker((arr, cards))
            if result[2] and result[1] > best_score:
                best_score = result[1]
                best = result[0]
    
    seq_time = time.time() - start
    print(f"  Time: {seq_time:.2f}s")
    print(f"  Speed: {total_evaluations/seq_time:,.0f} eval/s\n")
    
    # Parallel
    for num_workers in [2, 4, 8]:
        print(f"Parallel processing ({num_workers} workers)...")
        solver = ParallelSolver(num_workers=num_workers)
        
        start = time.time()
        results = solver.solve_batch(batch_cards, batch_arrangements)
        par_time = time.time() - start
        
        speedup = seq_time / par_time
        
        print(f"  Time: {par_time:.2f}s")
        print(f"  Speed: {total_evaluations/par_time:,.0f} eval/s")
        print(f"  Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    benchmark_parallel()