"""
Generate Expert Data V3 - PARALLEL VERSION
Sử dụng multiprocessing để tăng tốc 4-8x
"""
import sys
import os
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm
import json
import random
from multiprocessing import Pool, cpu_count
from functools import partial

# Fix imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, os.path.join(parent_dir, 'core'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from card import Deck
from smart_solver import SmartSolver
from state_encoder import StateEncoder, ActionEncoder


# Global objects for worker processes
_solver = None
_state_encoder = None
_action_encoder = None


def init_worker():
    """Initialize worker process"""
    global _solver, _state_encoder, _action_encoder
    _solver = SmartSolver()
    _state_encoder = StateEncoder()
    _action_encoder = ActionEncoder()


def generate_single_sample(seed: int):
    """
    Generate 1 sample - called by worker process
    """
    global _solver, _state_encoder, _action_encoder
    
    # Set seed for reproducibility
    random.seed(seed)
    
    try:
        # Random hand
        all_cards = Deck.full_deck()
        random.shuffle(all_cards)
        cards = all_cards[:13]
        
        # Get expert solution
        results = _solver.find_best_arrangement(cards, top_k=1)
        
        if not results or results[0][0] is None:
            return None
        
        back, middle, front, score = results[0]
        arrangement = (back, middle, front)
        
        # Encode
        state = _state_encoder.encode(cards)
        action = _action_encoder.encode_action(arrangement, cards)
        reward = float(score)
        cards_str = " ".join([str(c) for c in cards])
        
        return {
            'state': state,
            'action': action,
            'reward': reward,
            'cards_str': cards_str
        }
        
    except Exception as e:
        return None


def generate_expert_dataset_parallel(
    num_samples: int = 10000,
    output_dir: str = "../../data/training",
    num_workers: int = None
):
    """
    Generate expert dataset với multiprocessing
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Để lại 1 core cho system
    
    print("="*60)
    print(f"🎓 GENERATING EXPERT DATASET V3 - PARALLEL")
    print("="*60)
    print(f"Target samples: {num_samples:,}")
    print(f"Workers: {num_workers}")
    print(f"Estimated time: ~{num_samples * 0.5 / num_workers / 60:.1f} minutes")
    print()
    
    # Generate seeds
    seeds = list(range(num_samples * 2))  # Extra seeds in case of failures
    random.shuffle(seeds)
    
    print("🔄 Generating samples in parallel...")
    
    # Create pool and run
    dataset = []
    failed_count = 0
    
    with Pool(processes=num_workers, initializer=init_worker) as pool:
        # Use imap for progress tracking
        results = pool.imap(generate_single_sample, seeds, chunksize=10)
        
        pbar = tqdm(total=num_samples, desc="Progress")
        
        for result in results:
            if result is not None:
                dataset.append(result)
                pbar.update(1)
            else:
                failed_count += 1
            
            # Stop when we have enough
            if len(dataset) >= num_samples:
                break
        
        pbar.close()
        pool.terminate()
    
    # Trim to exact size
    dataset = dataset[:num_samples]
    
    print(f"\n✅ Generated {len(dataset):,} samples")
    print(f"⚠️  Failed: {failed_count} samples")
    
    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"expert_dataset_v3_{len(dataset)}.pkl"
    filepath = output_path / filename
    
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n💾 Saved to: {filepath}")
    
    # Statistics
    rewards = [s['reward'] for s in dataset]
    actions = [s['action'] for s in dataset]
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples:   {len(dataset):,}")
    print(f"   Reward mean:     {np.mean(rewards):.2f}")
    print(f"   Reward std:      {np.std(rewards):.2f}")
    print(f"   Reward min:      {np.min(rewards):.2f}")
    print(f"   Reward max:      {np.max(rewards):.2f}")
    print(f"   Unique actions:  {len(set(actions))}")
    print(f"   Action coverage: {len(set(actions))/1287*100:.1f}%")
    
    # Save metadata
    metadata = {
        'num_samples': len(dataset),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'reward_min': float(np.min(rewards)),
        'reward_max': float(np.max(rewards)),
        'unique_actions': len(set(actions)),
        'action_coverage': len(set(actions))/1287,
        'failed_count': failed_count,
        'num_workers': num_workers
    }
    
    metadata_file = output_path / filename.replace('.pkl', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved to: {metadata_file}")
    
    return str(filepath)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Expert Data V3 - Parallel')
    parser.add_argument('--samples', type=int, default=10000, 
                       help='Number of samples')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes')
    parser.add_argument('--output-dir', type=str, default='../../data/training',
                       help='Output directory')
    
    args = parser.parse_args()
    
    dataset_path = generate_expert_dataset_parallel(
        num_samples=args.samples,
        output_dir=args.output_dir,
        num_workers=args.workers
    )
    
    print(f"\n{'='*60}")
    print(f"✅ DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDataset: {dataset_path}")
    print(f"\nNext: python train_v3.py --dataset {dataset_path}")