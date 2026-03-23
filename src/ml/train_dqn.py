"""
Main training script
"""
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dqn_agent import DQNAgent
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    print("🃏 MAU BINH DQN TRAINING")
    print(f"Episodes: {args.episodes}")
    
    agent = DQNAgent(state_size=52, action_size=100, batch_size=args.batch_size)
    trainer = Trainer(agent)
    trainer.train(num_episodes=args.episodes, save_every=50)
    
    print("✅ Training complete!")


if __name__ == "__main__":
    main()