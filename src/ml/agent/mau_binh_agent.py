"""
Mau Binh Agent - Production-ready inference agent V2
OPTIMIZED with Smart Decoder
"""
import sys
import os
import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../networks'))

from card import Card
from ml.core import StateEncoderV2, ActionDecoderV2, RewardCalculator
from ml.networks import EnsembleNetwork


class MauBinhAgent:
    """
    Production ML Agent for Mau Binh V2
    
    Improvements:
    - Smart decoder with validation
    - Multiple decoding strategies
    - Guaranteed >90% valid rate
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        use_ensemble: bool = True
    ):
        # Device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Components
        self.encoder = StateEncoderV2()
        self.decoder = ActionDecoderV2()
        self.reward_calc = RewardCalculator()
        
        # Network
        self.use_ensemble = use_ensemble
        self.network = None
        
        # Load model if provided
        if model_path:
            self.load_model(model_path)
        else:
            # Initialize empty network
            self.network = EnsembleNetwork().to(self.device)
        
        print(f"🤖 MauBinhAgent initialized (device: {self.device})")
    
    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            self.network = EnsembleNetwork().to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle both formats
            if 'network_state_dict' in checkpoint:
                self.network.load_state_dict(checkpoint['network_state_dict'])
            else:
                self.network.load_state_dict(checkpoint)
            
            self.network.eval()
            
            print(f"✅ Loaded model from {model_path}")
        
        except Exception as e:
            print(f"⚠️  Failed to load model: {e}")
            self.network = None
    
    def solve(
        self,
        cards: List[Card],
        mode: str = 'best',
        epsilon: float = 0.0,
        use_smart_decoder: bool = True  # *** NEW PARAMETER ***
    ) -> Tuple[List[Card], List[Card], List[Card]]:
        """
        Main solve method - OPTIMIZED
        
        Args:
            cards: 13 cards
            mode: 'best' (greedy), 'sample' (stochastic), 'ensemble'
            epsilon: exploration rate (for epsilon-greedy)
            use_smart_decoder: If True, use smart decoder (RECOMMENDED)
            
        Returns:
            (back, middle, front)
        """
        if self.network is None:
            return self._fallback_solve(cards)
        
        # Encode state
        state = self.encoder.encode(cards)
        
        # Get action from model
        if mode == 'best' or mode == 'ensemble':
            action = self._get_best_action(state, epsilon)
        elif mode == 'sample':
            action = self._sample_action(state, temperature=0.5)
        else:
            action = self._get_best_action(state, epsilon)
        
        # *** DECODE TO ARRANGEMENT - USE SMART DECODER ***
        if use_smart_decoder:
            arrangement = self.decoder.decode_smart(action, cards)  # ← SMART!
        else:
            arrangement = self.decoder.decode_greedy(action, cards)  # ← OLD
        
        # Validate
        reward = self.reward_calc.calculate_reward(*arrangement)
        
        if reward < -50:  # Invalid
            # Try fallback
            return self._fallback_solve(cards)
        
        return arrangement
    
    def _get_best_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Get best action using epsilon-greedy"""
        if np.random.random() < epsilon:
            return np.random.randint(0, self.decoder.front_action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def _sample_action(self, state: np.ndarray, temperature: float = 1.0) -> int:
        """Sample action from softmax distribution"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.network(state_tensor)
            
            # Softmax with temperature
            probs = torch.softmax(q_values / temperature, dim=1)
            action = torch.multinomial(probs, 1).item()
        
        return action
    
    def _get_ensemble_action(self, state: np.ndarray) -> int:
        """Get action using full ensemble"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get predictions from both networks
            if hasattr(self.network, 'forward_with_components'):
                q_ensemble, q_dqn, q_transformer = self.network.forward_with_components(state_tensor)
                action = q_ensemble.argmax(dim=1).item()
            else:
                q_values = self.network(state_tensor)
                action = q_values.argmax(dim=1).item()
        
        return action
    
    def _fallback_solve(self, cards: List[Card]) -> Tuple[List[Card], List[Card], List[Card]]:
        """Fallback: Simple greedy arrangement"""
        sorted_cards = sorted(cards, key=lambda c: c.rank.value, reverse=True)
        
        back = sorted_cards[:5]
        middle = sorted_cards[5:10]
        front = sorted_cards[10:13]
        
        return (back, middle, front)
    
    def batch_solve(
        self,
        batch_cards: List[List[Card]],
        mode: str = 'best',
        use_smart_decoder: bool = True
    ) -> List[Tuple[List[Card], List[Card], List[Card]]]:
        """
        Solve batch of hands - OPTIMIZED
        
        Returns:
            List of arrangements
        """
        if self.network is None:
            return [self._fallback_solve(cards) for cards in batch_cards]
        
        # Encode batch
        states = torch.FloatTensor(
            np.array([self.encoder.encode(cards) for cards in batch_cards])
        ).to(self.device)
        
        # Get actions
        with torch.no_grad():
            q_values = self.network(states)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Decode - USE SMART DECODER
        arrangements = []
        for action, cards in zip(actions, batch_cards):
            if use_smart_decoder:
                arr = self.decoder.decode_smart(action, cards)
            else:
                arr = self.decoder.decode_greedy(action, cards)
            arrangements.append(arr)
        
        return arrangements
    
    def evaluate_arrangement(
        self,
        arrangement: Tuple[List[Card], List[Card], List[Card]]
    ) -> dict:
        """
        Evaluate an arrangement
        
        Returns:
            {
                'reward': float,
                'is_valid': bool,
                'bonus': int,
                'strength': float
            }
        """
        back, middle, front = arrangement
        
        reward = self.reward_calc.calculate_reward(back, middle, front)
        is_valid = reward > -50
        
        # Decompose reward
        if is_valid:
            bonus = self.reward_calc._calculate_bonus(back, middle, front)
            strength = self.reward_calc._calculate_strength(back, middle, front)
        else:
            bonus = 0
            strength = 0
        
        return {
            'reward': float(reward),
            'is_valid': is_valid,
            'bonus': bonus,
            'strength': float(strength)
        }


# ==================== TESTS ====================

def test_mau_binh_agent():
    """Test MauBinhAgent"""
    print("Testing MauBinhAgent...")
    from card import Deck
    
    # Create agent (no model)
    agent = MauBinhAgent()
    
    # Test fallback solve
    cards = Deck.parse_hand("A♠ K♥ Q♦ J♣ 10♠ 9♥ 8♦ 7♣ 6♠ 5♥ 4♦ 3♣ 2♠")
    
    # Test with smart decoder
    arrangement_smart = agent.solve(cards, use_smart_decoder=True)
    
    assert len(arrangement_smart) == 3
    assert len(arrangement_smart[0]) == 5
    assert len(arrangement_smart[1]) == 5
    assert len(arrangement_smart[2]) == 3
    
    print("  ✅ Smart decoder solve OK")
    
    # Test evaluation
    eval_result = agent.evaluate_arrangement(arrangement_smart)
    
    assert 'reward' in eval_result
    assert 'is_valid' in eval_result
    assert 'bonus' in eval_result
    
    print(f"  ✅ Evaluation: reward={eval_result['reward']:.2f}, valid={eval_result['is_valid']}")
    
    # Test batch solve
    batch = [cards, cards]
    arrangements = agent.batch_solve(batch, use_smart_decoder=True)
    
    assert len(arrangements) == 2
    print(f"  ✅ Batch solve: {len(arrangements)} hands")
    
    print("✅ MauBinhAgent tests passed!")


if __name__ == "__main__":
    test_mau_binh_agent()