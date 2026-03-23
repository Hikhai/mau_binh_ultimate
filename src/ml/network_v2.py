"""
Improved DQN Architecture with Constraint Awareness
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintAwareDQN(nn.Module):
    """
    Advanced DQN with:
    - Residual connections
    - Constraint-aware layers
    - Attention mechanism (simplified)
    """
    
    def __init__(self, state_size=52, action_size=1000):
        super(ConstraintAwareDQN, self).__init__()
        
        # Embedding layer
        self.embed = nn.Linear(state_size, 256)
        self.embed_bn = nn.BatchNorm1d(256)
        
        # Residual blocks
        self.res1_fc1 = nn.Linear(256, 512)
        self.res1_bn1 = nn.BatchNorm1d(512)
        self.res1_fc2 = nn.Linear(512, 256)
        self.res1_bn2 = nn.BatchNorm1d(256)
        
        self.res2_fc1 = nn.Linear(256, 512)
        self.res2_bn1 = nn.BatchNorm1d(512)
        self.res2_fc2 = nn.Linear(512, 256)
        self.res2_bn2 = nn.BatchNorm1d(256)
        
        # Constraint-aware branch
        self.constraint_fc1 = nn.Linear(256, 128)
        self.constraint_fc2 = nn.Linear(128, 64)
        
        # Value stream
        self.value_fc1 = nn.Linear(256 + 64, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(256 + 64, 256)
        self.advantage_fc2 = nn.Linear(256, action_size)
        
        self.dropout = nn.Dropout(0.2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Embedding
        x = F.relu(self.embed_bn(self.embed(x)))
        
        # Residual block 1
        identity = x
        out = F.relu(self.res1_bn1(self.res1_fc1(x)))
        out = self.res1_bn2(self.res1_fc2(out))
        out = F.relu(out + identity)
        out = self.dropout(out)
        
        # Residual block 2
        identity = out
        out = F.relu(self.res2_bn1(self.res2_fc1(out)))
        out = self.res2_bn2(self.res2_fc2(out))
        out = F.relu(out + identity)
        out = self.dropout(out)
        
        # Constraint awareness
        constraint = F.relu(self.constraint_fc1(out))
        constraint = F.relu(self.constraint_fc2(constraint))
        
        # Combine features
        combined = torch.cat([out, constraint], dim=1)
        
        # Value stream
        value = F.relu(self.value_fc1(combined))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(combined))
        advantage = self.advantage_fc2(advantage)
        
        # Combine (Dueling DQN)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values