"""
Self-Play Data Generator
"""
import sys
import os
import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

from card import Deck
from ml.core import StateEncoderV2, ActionDecoderV2, RewardCalculator


class SelfPlayGenerator:
    def __init__(self, model_path=None, output_dir="data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = StateEncoderV2()
        self.decoder = ActionDecoderV2()
        self.reward_calc = RewardCalculator()
        self.model = None

    def load_model(self, model_path):
        pass

    def generate_with_model(self, num_samples=1000, epsilon=0.1, output_name="self_play.pkl"):
        return str(self.output_dir / output_name)


if __name__ == "__main__":
    print("SelfPlayGenerator OK")