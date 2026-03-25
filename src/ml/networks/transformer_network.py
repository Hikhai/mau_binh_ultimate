"""
Transformer Network - SOTA architecture cho Mậu Binh
"""
import torch
import torch.nn as nn
import math


class TransformerNetwork(nn.Module):
    """
    Transformer-based Q-Network
    
    Ý tưởng:
    - Treat 13 cards như 13 tokens
    - Self-attention để học mối quan hệ giữa các lá
    - Output Q-values cho actions
    
    More powerful than DQN, but slower
    """
    
    def __init__(
        self,
        state_size: int = 77,
        action_size: int = 286,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super(TransformerNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(state_size, d_model)
        
        # Positional encoding (for card sequence)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=13)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.fc = nn.Linear(d_model, 256)
        self.q_head = nn.Linear(256, action_size)
        self.value_head = nn.Linear(256, 1)
        
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
        batch_size = state.size(0)
        
        # Project to d_model
        x = self.input_proj(state)  # (batch, d_model)
        
        # Add batch dimension for transformer (treat as sequence of length 1)
        # Hoặc reshape state thành (batch, 13, features_per_card) nếu cần
        # Simplified: treat whole state as 1 token
        x = x.unsqueeze(1)  # (batch, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)  # (batch, 1, d_model)
        
        # Global pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        # FC
        features = torch.relu(self.fc(x))  # (batch, 256)
        
        # Dueling architecture
        value = self.value_head(features)
        advantages = self.q_head(features)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state, epsilon=0.0):
        """Get action using epsilon-greedy"""
        import numpy as np
        
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        return action


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=13):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== TESTS ====================

def test_transformer_network():
    """Test TransformerNetwork"""
    print("Testing TransformerNetwork...")
    
    # Create network
    net = TransformerNetwork(state_size=77, action_size=286)
    
    print(f"  Network created")
    print(f"  Parameters: {sum(p.numel() for p in net.parameters()):,}")
    
    # Test forward
    batch_size = 4
    state = torch.randn(batch_size, 77)
    
    q_values = net(state)
    
    assert q_values.shape == (batch_size, 286)
    print(f"  ✅ Forward pass: {q_values.shape}")
    
    # Test get_action
    import numpy as np
    state_np = np.random.randn(77)
    
    action = net.get_action(state_np, epsilon=0.0)
    assert 0 <= action < 286
    print(f"  ✅ Get action: {action}")
    
    # Test gradient
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    target = torch.randn(batch_size, 286)
    loss = torch.nn.functional.mse_loss(q_values, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  ✅ Gradient flow OK, loss: {loss.item():.4f}")
    
    print("✅ TransformerNetwork tests passed!")


if __name__ == "__main__":
    test_transformer_network()