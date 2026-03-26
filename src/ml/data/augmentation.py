"""
Data Augmentation
"""
import sys
import os
import random
import numpy as np
from typing import List, Tuple
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))

from card import Card, Suit
from ml.core import StateEncoderV2, RewardCalculator


class DataAugmentation:
    def __init__(self):
        self.encoder = StateEncoderV2()
        self.reward_calc = RewardCalculator()

    @staticmethod
    def permute_suits(arrangement):
        back, middle, front = arrangement
        suits = list(Suit)
        shuffled_suits = suits.copy()
        random.shuffle(shuffled_suits)
        suit_map = {s: shuffled_suits[i] for i, s in enumerate(suits)}

        def map_card(card):
            return Card(card.rank, suit_map[card.suit])

        return ([map_card(c) for c in back],
                [map_card(c) for c in middle],
                [map_card(c) for c in front])

    @staticmethod
    def add_noise_to_state(state, dropout_rate=0.1):
        noisy = state.copy()
        mask = np.random.random(52) > dropout_rate
        noisy[:52] *= mask
        return noisy

    def augment_sample(self, sample, num_augmentations=1):
        augmented = [sample]
        for _ in range(num_augmentations):
            if random.random() < 0.7:
                new_arr = self.permute_suits(sample['arrangement'])
                all_cards = new_arr[0] + new_arr[1] + new_arr[2]
                new_state = self.encoder.encode(all_cards)
                augmented.append({
                    'state': new_state,
                    'arrangement': new_arr,
                    'reward': sample['reward'],
                })
        return augmented

    def augment_dataset(self, dataset, augmentation_factor=2):
        print(f"🔄 Augmenting dataset (factor={augmentation_factor})...")
        result = []
        for sample in dataset:
            result.extend(self.augment_sample(sample, augmentation_factor))
        print(f"✅ Augmented: {len(dataset)} → {len(result)} samples")
        return result


if __name__ == "__main__":
    print("DataAugmentation OK")