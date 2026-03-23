"""
Experience Replay Buffer
"""
import random
import numpy as np
import torch
from collections import deque, namedtuple
from typing import List, Tuple


# Experience tuple
Experience = namedtuple('Experience', [
    'state',       # Current state
    'action',      # Action taken
    'reward',      # Reward received
    'next_state',  # Next state
    'done'         # Episode done flag
])


class ReplayBuffer:
    """
    Experience Replay Buffer với prioritization option
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        prioritized: bool = False
    ):
        """
        Args:
            capacity: Maximum buffer size
            prioritized: Use prioritized replay (advanced)
        """
        self.buffer = deque(maxlen=capacity)
        self.prioritized = prioritized
        
        if prioritized:
            self.priorities = deque(maxlen=capacity)
            self.alpha = 0.6  # Priority exponent
            self.beta = 0.4   # Importance sampling weight
            self.beta_increment = 0.001
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: float = None
    ):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
        
        if self.prioritized:
            # Max priority for new experience
            max_priority = max(self.priorities) if self.priorities else 1.0
            self.priorities.append(priority or max_priority)
    
    def sample(self, batch_size: int) -> Tuple:
        """
        Sample batch from buffer
        
        Returns:
            (states, actions, rewards, next_states, dones, [weights, indices])
        """
        if self.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def _sample_uniform(self, batch_size: int) -> Tuple:
        """Uniform random sampling"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def _sample_prioritized(self, batch_size: int) -> Tuple:
        """Prioritized sampling"""
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            replace=False,
            p=probs
        )
        
        # Get experiences
        batch = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])
        
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
            torch.FloatTensor(weights),
            indices
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences"""
        if not self.prioritized:
            return
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant to avoid 0
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        if self.prioritized:
            self.priorities.clear()


# ==================== TESTS ====================

def test_replay_buffer():
    """Test ReplayBuffer"""
    print("Testing ReplayBuffer...")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000, prioritized=False)
    
    # Add experiences
    for i in range(100):
        state = np.random.rand(52)
        action = np.random.randint(0, 100)
        reward = np.random.randn()
        next_state = np.random.rand(52)
        done = (i % 10 == 0)
        
        buffer.push(state, action, reward, next_state, done)
    
    assert len(buffer) == 100
    print(f"  Buffer size: {len(buffer)} ✓")
    
    # Sample batch
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones = batch
    
    assert states.shape == (32, 52)
    assert actions.shape == (32,)
    assert rewards.shape == (32,)
    assert next_states.shape == (32, 52)
    assert dones.shape == (32,)
    
    print(f"  Sample batch shapes: {states.shape} ✓")
    
    print("✅ ReplayBuffer tests passed!")


def test_prioritized_buffer():
    """Test prioritized replay"""
    print("\nTesting Prioritized ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=1000, prioritized=True)
    
    # Add experiences with different priorities
    for i in range(100):
        state = np.random.rand(52)
        action = np.random.randint(0, 100)
        reward = np.random.randn()
        next_state = np.random.rand(52)
        done = False
        priority = np.random.rand()  # Random priority
        
        buffer.push(state, action, reward, next_state, done, priority)
    
    # Sample
    batch = buffer.sample(32)
    assert len(batch) == 7  # Includes weights and indices
    
    states, actions, rewards, next_states, dones, weights, indices = batch
    
    assert weights.shape == (32,)
    assert len(indices) == 32
    
    print(f"  Prioritized sample successful ✓")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}] ✓")
    
    # Update priorities
    new_priorities = np.random.rand(32)
    buffer.update_priorities(indices, new_priorities)
    
    print("✅ Prioritized ReplayBuffer tests passed!")


if __name__ == "__main__":
    test_replay_buffer()
    test_prioritized_buffer()
    print("\n" + "="*60)
    print("✅ All replay_buffer.py tests passed!")
    print("="*60)