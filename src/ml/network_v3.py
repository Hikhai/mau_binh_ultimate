"""
Simple & Effective Network for Mau Binh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MauBinhNetworkV3(nn.Module):
    """
    Đơn giản nhưng hiệu quả:
    - No residual, no batch norm (gây overfitting với small dataset)
    - Dropout cho regularization
    - Smaller network
    """
    
    def __init__(self, state_dim=65, action_dim=1287):
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(256, action_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = self.fc4(x)
        
        return x


if __name__ == "__main__":
    # Test
    model = MauBinhNetworkV3()
    
    batch_size = 16
    state = torch.randn(batch_size, 65)
    
    output = model(state)
    
    print(f"Input shape:  {state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    
    assert output.shape == (batch_size, 1287)
    print("✅ Network test passed!")