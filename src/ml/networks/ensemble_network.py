"""
Ensemble Network - Kết hợp DQN + Transformer
"""
import torch
import torch.nn as nn

try:
    from .dqn_network import DQNNetwork
    from .transformer_network import TransformerNetwork
except ImportError:
    from dqn_network import DQNNetwork
    from transformer_network import TransformerNetwork


class EnsembleNetwork(nn.Module):
    """
    Ensemble của DQN + Transformer
    
    Strategy:
    - DQN: Fast, stable baseline
    - Transformer: Powerful, learns complex patterns
    - Ensemble: Take weighted average
    
    Best of both worlds!
    """
    
    def __init__(
        self,
        state_size: int = 77,
        action_size: int = 286,
        dqn_weight: float = 0.4,
        transformer_weight: float = 0.6
    ):
        super(EnsembleNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Weights
        self.dqn_weight = dqn_weight
        self.transformer_weight = transformer_weight
        
        # Networks
        self.dqn = DQNNetwork(state_size, action_size)
        self.transformer = TransformerNetwork(state_size, action_size)
        
        print(f"EnsembleNetwork: DQN weight={dqn_weight}, Transformer weight={transformer_weight}")
    
    def forward(self, state):
        """
        Forward pass - ensemble prediction
        
        Args:
            state: (batch_size, 77)
            
        Returns:
            q_values: (batch_size, 286)
        """
        # Get predictions from both networks
        q_dqn = self.dqn(state)
        q_transformer = self.transformer(state)
        
        # Weighted average
        q_ensemble = self.dqn_weight * q_dqn + self.transformer_weight * q_transformer
        
        return q_ensemble
    
    def forward_with_components(self, state):
        """
        Forward pass with individual components
        
        Returns:
            (q_ensemble, q_dqn, q_transformer)
        """
        q_dqn = self.dqn(state)
        q_transformer = self.transformer(state)
        q_ensemble = self.dqn_weight * q_dqn + self.transformer_weight * q_transformer
        
        return q_ensemble, q_dqn, q_transformer
    
    def get_action(self, state, epsilon=0.0, use_ensemble=True):
        """
        Get action
        
        Args:
            state: (77,) numpy array
            epsilon: exploration rate
            use_ensemble: if True, use ensemble; else use DQN only (faster)
        """
        import numpy as np
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            if use_ensemble:
                q_values = self.forward(state_tensor)
            else:
                q_values = self.dqn(state_tensor)
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def set_weights(self, dqn_weight: float, transformer_weight: float):
        """Dynamically adjust ensemble weights"""
        self.dqn_weight = dqn_weight
        self.transformer_weight = transformer_weight


# ==================== TESTS ====================

def test_ensemble_network():
    """Test EnsembleNetwork"""
    print("Testing EnsembleNetwork...")
    
    # Create network
    net = EnsembleNetwork(state_size=77, action_size=286)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  Network created")
    print(f"  Total parameters: {total_params:,}")
    
    # Test forward
    batch_size = 4
    state = torch.randn(batch_size, 77)
    
    q_values = net(state)
    assert q_values.shape == (batch_size, 286)
    print(f"  ✅ Forward pass: {q_values.shape}")
    
    # Test forward with components
    q_ensemble, q_dqn, q_transformer = net.forward_with_components(state)
    
    assert q_ensemble.shape == (batch_size, 286)
    assert q_dqn.shape == (batch_size, 286)
    assert q_transformer.shape == (batch_size, 286)
    print(f"  ✅ Component forward pass OK")
    
    # Verify ensemble is weighted average
    expected = 0.4 * q_dqn + 0.6 * q_transformer
    assert torch.allclose(q_ensemble, expected, atol=1e-5)
    print(f"  ✅ Ensemble weights verified")
    
    # Test get_action
    import numpy as np
    state_np = np.random.randn(77)
    
    action_ensemble = net.get_action(state_np, epsilon=0.0, use_ensemble=True)
    assert 0 <= action_ensemble < 286
    print(f"  ✅ Get action (ensemble): {action_ensemble}")
    
    action_dqn = net.get_action(state_np, epsilon=0.0, use_ensemble=False)
    assert 0 <= action_dqn < 286
    print(f"  ✅ Get action (DQN only): {action_dqn}")
    
    # Test weight adjustment
    net.set_weights(0.7, 0.3)
    assert net.dqn_weight == 0.7
    assert net.transformer_weight == 0.3
    print(f"  ✅ Dynamic weight adjustment OK")
    
    print("✅ EnsembleNetwork tests passed!")


if __name__ == "__main__":
    test_ensemble_network()