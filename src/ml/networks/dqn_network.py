"""
DQN Network - Baseline Deep Q-Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network cho Mậu Binh
    
    Architecture:
    - Input: 77-dim state
    - Hidden: 3 layers (512 → 256 → 128)
    - Output: 286-dim Q-values (cho front actions)
    
    Simple but effective baseline
    """
    
    def __init__(
        self,
        state_size: int = 77,
        action_size: int = 286,
        hidden_sizes: list = [512, 256, 128]
    ):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Feature extraction layers
        layers = []
        in_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_size = hidden_size
        
        self.feature_net = nn.Sequential(*layers)
        
        # Q-value head
        self.q_head = nn.Linear(hidden_sizes[-1], action_size)
        
        # Value head (for Dueling DQN)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state: (batch_size, 77)
            
        Returns:
            q_values: (batch_size, 286)
        """
        # Feature extraction
        features = self.feature_net(state)
        
        # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(features)
        advantages = self.q_head(features)
        
        # Combine
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """
        Get action using epsilon-greedy
        
        Args:
            state: (77,) numpy array
            epsilon: exploration rate
            
        Returns:
            action: int (0-285)
        """
        import numpy as np
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action


# ==================== TESTS ====================

def test_dqn_network():
    """Test DQNNetwork"""
    print("Testing DQNNetwork...")
    
    # Create network
    net = DQNNetwork(state_size=77, action_size=286)
    
    print(f"  Network created")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, 77)
    
    q_values = net(state)
    
    assert q_values.shape == (batch_size, 286), f"Expected (4, 286), got {q_values.shape}"
    print(f"  ✅ Forward pass: {q_values.shape}")
    
    # Test get_action
    import numpy as np
    state_np = np.random.randn(77)
    
    action = net.get_action(state_np, epsilon=0.0)
    assert 0 <= action < 286
    print(f"  ✅ Get action (greedy): {action}")
    
    action_random = net.get_action(state_np, epsilon=1.0)
    assert 0 <= action_random < 286
    print(f"  ✅ Get action (random): {action_random}")
    
    # Test gradient flow
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    target = torch.randn(batch_size, 286)
    loss = F.mse_loss(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  ✅ Gradient flow OK, loss: {loss.item():.4f}")
    
    print("✅ DQNNetwork tests passed!")


if __name__ == "__main__":
    test_dqn_network()