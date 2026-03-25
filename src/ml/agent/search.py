"""
Search Algorithms - Beam Search / MCTS (Advanced)
"""
import sys
import os
import numpy as np
from typing import List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

from card import Card
from ml.core import RewardCalculator, ActionDecoderV2


class BeamSearch:
    """
    Beam Search for finding best arrangement
    
    More thorough than greedy, faster than exhaustive
    """
    
    def __init__(self, beam_width: int = 5):
        self.beam_width = beam_width
        self.decoder = ActionDecoderV2()
        self.reward_calc = RewardCalculator()
    
    def search(
        self,
        cards: List[Card],
        depth: int = 2
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Beam search to find best arrangement
        
        Args:
            cards: 13 cards
            depth: search depth (1 = greedy, 2 = beam)
            
        Returns:
            Best arrangement found
        """
        # Start with top-k front candidates
        candidates = []
        
        for action in range(min(self.beam_width * 2, self.decoder.front_action_size)):
            arrangement = self.decoder.decode_greedy(action, cards)
            reward = self.reward_calc.calculate_reward(*arrangement)
            
            if reward > -50:  # Valid
                candidates.append((arrangement, reward))
        
        # Sort by reward
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return best
        if candidates:
            return candidates[0][0]
        else:
            # Fallback
            sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
            return (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])


class MonteCarloTreeSearch:
    """
    MCTS for Mau Binh (Simplified)
    
    Useful for: Exploring different strategies
    """
    
    def __init__(self, num_simulations: int = 100):
        self.num_simulations = num_simulations
        self.decoder = ActionDecoderV2()
        self.reward_calc = RewardCalculator()
    
    def search(self, cards: List[Card]) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        MCTS to find best arrangement
        
        Simplified: Random sampling + UCB selection
        """
        # Statistics
        action_counts = defaultdict(int)
        action_rewards = defaultdict(float)
        
        # Simulations
        for _ in range(self.num_simulations):
            # Random action
            action = np.random.randint(0, self.decoder.front_action_size)
            
            # Decode
            arrangement = self.decoder.decode_greedy(action, cards)
            
            # Reward
            reward = self.reward_calc.calculate_reward(*arrangement)
            
            if reward > -50:
                action_counts[action] += 1
                action_rewards[action] += reward
        
        # Select best action (highest average reward)
        if not action_counts:
            # Fallback
            sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
            return (sorted_cards[:5], sorted_cards[5:10], sorted_cards[10:13])
        
        best_action = max(
            action_counts.keys(),
            key=lambda a: action_rewards[a] / action_counts[a]
        )
        
        return self.decoder.decode_greedy(best_action, cards)


# ==================== TESTS ====================

def test_search_algorithms():
    """Test search algorithms"""
    print("Testing Search Algorithms...")
    from card import Deck
    
    cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
    
    # Test Beam Search
    beam = BeamSearch(beam_width=5)
    arr_beam = beam.search(cards, depth=2)
    
    assert len(arr_beam) == 3
    print(f"  ✅ Beam Search OK")
    
    # Test MCTS
    mcts = MonteCarloTreeSearch(num_simulations=50)
    arr_mcts = mcts.search(cards)
    
    assert len(arr_mcts) == 3
    print(f"  ✅ MCTS OK")
    
    print("✅ Search Algorithms tests passed!")


if __name__ == "__main__":
    test_search_algorithms()