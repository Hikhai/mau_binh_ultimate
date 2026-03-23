"""
Deep Q-Network Agent cho Mậu Binh
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../core'))
sys.path.insert(0, os.path.join(current_dir, '../engines'))

from card import Card, Deck
from network import MauBinhDQN, DuelingDQN
from state_encoder import StateEncoder, ActionEncoder
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent với các features:
    - Double DQN
    - Target Network
    - Experience Replay
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_size: int = 52,
        action_size: int = 1000,
        use_dueling: bool = False,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Networks
        NetworkClass = DuelingDQN if use_dueling else MauBinhDQN
        
        self.policy_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net = NetworkClass(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=buffer_size, prioritized=False)
        
        # Encoders
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder(use_simplified=True)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        self.losses = []
    
    def select_action(
        self,
        state: np.ndarray,
        valid_actions_mask: Optional[np.ndarray] = None,
        epsilon: Optional[float] = None
    ) -> int:
        """Select action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
        
        if random.random() < epsilon:
            # Explore
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask > 0)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit
            # FIX: Set to eval mode for single sample inference
            self.policy_net.eval()
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                # Avoid requiring numpy at runtime by converting to native python list
                q_values = self.policy_net(state_tensor).cpu().detach().tolist()[0]
                
                if valid_actions_mask is not None:
                    q_values = q_values * valid_actions_mask
                    q_values[valid_actions_mask == 0] = -float('inf')
            
            # Set back to train mode
            self.policy_net.train()
            
            return int(np.argmax(q_values))
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # FIX: Ensure train mode
        self.policy_net.train()
        
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            # Double DQN
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        self.steps += 1
        
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath: str):
        """Save model checkpoint"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'losses': self.losses
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)
        self.losses = checkpoint.get('losses', [])
        
        print(f"📂 Model loaded from {filepath}")
        print(f"   Episodes: {self.episodes}, Steps: {self.steps}, Epsilon: {self.epsilon:.3f}")


if __name__ == "__main__":
    print("Testing DQN Agent...")
    
    agent = DQNAgent(
        state_size=52,
        action_size=100,
        use_dueling=False,
        buffer_size=1000,
        batch_size=32
    )
    
    print(f"  Device: {agent.device} ✓")
    
    state = np.random.rand(52)
    action = agent.select_action(state)
    assert 0 <= action < 100
    print(f"  Selected action: {action} ✓")
    
    next_state = np.random.rand(52)
    agent.store_transition(state, action, 1.0, next_state, False)
    
    for i in range(100):
        s = np.random.rand(52)
        a = np.random.randint(0, 100)
        r = np.random.randn()
        ns = np.random.rand(52)
        d = (i % 10 == 0)
        agent.store_transition(s, a, r, ns, d)
    
    loss = agent.train_step()
    print(f"  Training loss: {loss:.4f} ✓")
    
    print("✅ DQN Agent tests passed!")