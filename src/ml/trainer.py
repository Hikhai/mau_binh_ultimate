"""
Training Pipeline for Mau Binh DQN
"""
import sys
import os
import time
import numpy as np
from typing import List, Tuple
from pathlib import Path
import random

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../core'))
sys.path.insert(0, os.path.join(current_dir, '../engines'))

from card import Card, Deck
from evaluator import HandEvaluator
from game_theory import BonusPoints
from dqn_agent import DQNAgent
from state_encoder import StateEncoder


class MauBinhEnvironment:
    def __init__(self):
        self.bonus_calc = BonusPoints()
        self.state_encoder = StateEncoder()
    
    def reset(self) -> Tuple[List[Card], np.ndarray]:
        deck = Deck.full_deck()
        self.my_cards = random.sample(deck, 13)
        state = self.state_encoder.encode(self.my_cards)
        return self.my_cards, state
    
    def step(self, arrangement: Tuple[List[Card], List[Card], List[Card]]) -> Tuple[float, bool]:
        back, middle, front = arrangement
        
        if not self._is_valid(back, middle, front):
            return -10.0, True
        
        bonus = self.bonus_calc.calculate_bonus(back, middle, front)
        
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        front_rank = HandEvaluator.evaluate(front)
        
        strength_reward = (
            back_rank.hand_type.value * 0.5 +
            middle_rank.hand_type.value * 0.3 +
            front_rank.hand_type.value * 0.2
        )
        
        reward = bonus + strength_reward
        done = True
        
        return reward, done
    
    def _is_valid(self, back: List[Card], middle: List[Card], front: List[Card]) -> bool:
        if len(back) != 5 or len(middle) != 5 or len(front) != 3:
            return False
        
        all_cards = back + middle + front
        if set(all_cards) != set(self.my_cards):
            return False
        
        back_rank = HandEvaluator.evaluate(back)
        middle_rank = HandEvaluator.evaluate(middle)
        
        if back_rank < middle_rank:
            return False
        
        return True


class Trainer:
    def __init__(self, agent: DQNAgent, save_dir: str = "../../data/models"):
        self.agent = agent
        self.env = MauBinhEnvironment()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
    
    def train(self, num_episodes: int = 10000, save_every: int = 1000, verbose: bool = True):
        print("="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Episodes: {num_episodes}")
        print(f"Device: {self.agent.device}")
        print()
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            cards, state = self.env.reset()
            action = self.agent.select_action(state)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            reward, done = self.env.step(arrangement)
            _, next_state = self.env.reset()
            
            self.agent.store_transition(state, action, reward, next_state, done)
            loss = self.agent.train_step()
            
            self.episode_rewards.append(reward)
            self.episode_lengths.append(1)
            self.agent.episodes += 1
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_loss = np.mean(self.agent.losses[-100:]) if self.agent.losses else 0
                elapsed = time.time() - start_time
                eps_per_sec = (episode + 1) / elapsed
                
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Reward: {avg_reward:6.2f} | "
                      f"Loss: {avg_loss:7.4f} | "
                      f"ε: {self.agent.epsilon:.3f} | "
                      f"Speed: {eps_per_sec:.1f} eps/s")
            
            if (episode + 1) % save_every == 0:
                checkpoint_path = self.save_dir / f"checkpoint_ep{episode+1}.pth"
                self.agent.save(checkpoint_path)
        
        final_path = self.save_dir / "final_model.pth"
        self.agent.save(final_path)
        
        print("\n✅ TRAINING COMPLETED")
    
    def evaluate(self, num_episodes: int = 100, epsilon: float = 0.0) -> dict:
        print(f"🎯 Evaluating ({num_episodes} episodes)...")
        
        rewards = []
        for _ in range(num_episodes):
            cards, state = self.env.reset()
            action = self.agent.select_action(state, epsilon=epsilon)
            arrangement = self.agent.action_encoder.decode_action(action, cards)
            reward, _ = self.env.step(arrangement)
            rewards.append(reward)
        
        stats = {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
        }
        
        print(f"  Mean reward: {stats['mean_reward']:.2f}")
        return stats


if __name__ == "__main__":
    print("Testing Trainer...")
    
    agent = DQNAgent(state_size=52, action_size=100, buffer_size=1000, batch_size=32)
    trainer = Trainer(agent)
    
    trainer.train(num_episodes=10, save_every=5, verbose=True)
    stats = trainer.evaluate(num_episodes=5)
    
    print("✅ Trainer tests passed!")