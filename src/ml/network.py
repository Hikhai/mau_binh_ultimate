"""
Deep Q-Network Architecture for Mau Binh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import List, Tuple

sys.path.insert(0, '../core')
from card import Card


class MauBinhDQN(nn.Module):
    """
    Deep Q-Network cho Mậu Binh
    
    Architecture:
    Input (52) → FC(512) → FC(256) → FC(128) → Output(action_size)
    """
    
    def __init__(self, state_size: int = 52, action_size: int = 1000):
        """
        Args:
            state_size: Kích thước state (52 = one-hot encoding 52 lá)
            action_size: Số lượng actions khả thi
        """
        super(MauBinhDQN, self).__init__()
        
        # Fully connected layers
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(128, action_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: State tensor (batch_size, state_size)
            
        Returns:
            Q-values tensor (batch_size, action_size)
        """
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        
        return x


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (advanced)
    Tách riêng Value stream và Advantage stream
    """
    
    def __init__(self, state_size: int = 52, action_size: int = 1000):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        # Value stream
        self.value_fc1 = nn.Linear(256, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(256, 128)
        self.advantage_fc2 = nn.Linear(128, action_size)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# ==================== TESTS ====================

def test_dqn():
    """Test DQN architecture"""
    print("Testing MauBinhDQN...")
    
    # Create network
    network = MauBinhDQN(state_size=52, action_size=100)
    
    # Test forward pass
    batch_size = 32
    dummy_state = torch.randn(batch_size, 52)
    
    output = network(dummy_state)
    
    assert output.shape == (batch_size, 100)
    print(f"  Output shape: {output.shape} ✓")
    
    # Test parameter count
    total_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("✅ MauBinhDQN tests passed!")


def test_dueling_dqn():
    """Test Dueling DQN"""
    print("\nTesting DuelingDQN...")
    
    network = DuelingDQN(state_size=52, action_size=100)
    
    batch_size = 32
    dummy_state = torch.randn(batch_size, 52)
    
    output = network(dummy_state)
    
    assert output.shape == (batch_size, 100)
    print(f"  Output shape: {output.shape} ✓")
    
    total_params = sum(p.numel() for p in network.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    print("✅ DuelingDQN tests passed!")


if __name__ == "__main__":
    test_dqn()
    test_dueling_dqn()
    print("\n" + "="*60)
    print("✅ All network.py tests passed!")
    print("="*60)