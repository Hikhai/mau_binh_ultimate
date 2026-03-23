"""
Complete ML Pipeline: Generate → Train → Validate → Report
One command to do everything!
"""
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../engines'))
sys.path.insert(0, os.path.dirname(__file__))


def run_full_pipeline(
    num_samples: int = 50000,
    num_epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    patience: int = 20,
    num_workers: int = 4,
    num_tests: int = 1000,
    experiment_name: str = None
):
    """Run complete pipeline"""
    
    if experiment_name is None:
        experiment_name = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🔥" * 30)
    print("🚀 COMPLETE ML PIPELINE")
    print("🔥" * 30)
    print(f"\nExperiment: {experiment_name}")
    print(f"Samples: {num_samples}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch: {batch_size}")
    print(f"LR: {learning_rate}")
    print()
    
    # ==========================================
    # PHASE 1: GENERATE DATA
    # ==========================================
    print("\n" + "="*60)
    print("📊 PHASE 1: GENERATE EXPERT DATA")
    print("="*60)
    
    from expert_data_generator_v2 import ExpertDataGeneratorV2
    
    generator = ExpertDataGeneratorV2()
    dataset_path = generator.generate_dataset(
        num_samples=num_samples,
        num_workers=num_workers
    )
    
    # ==========================================
    # PHASE 2: TRAIN MODEL
    # ==========================================
    print("\n" + "="*60)
    print("🧠 PHASE 2: TRAIN MODEL")
    print("="*60)
    
    from train_expert_v2 import ExpertTrainerV2
    
    trainer = ExpertTrainerV2(
        dataset_path=dataset_path,
        experiment_name=experiment_name
    )
    
    trainer.train(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patience=patience
    )
    
    # ==========================================
    # PHASE 3: VALIDATE MODEL
    # ==========================================
    print("\n" + "="*60)
    print("🧪 PHASE 3: VALIDATE MODEL")
    print("="*60)
    
    model_path = f"../../data/models/{experiment_name}/best_model.pth"
    
    from validate_model_v2 import ModelValidator
    
    validator = ModelValidator(model_path)
    results = validator.run_full_validation(num_tests=num_tests)
    
    # ==========================================
    # FINAL REPORT
    # ==========================================
    print("\n" + "🏆" * 20)
    print("📋 FINAL PIPELINE REPORT")
    print("🏆" * 20)
    
    print(f"\nExperiment: {experiment_name}")
    print(f"Training samples: {num_samples}")
    print(f"Training epochs: {len(trainer.train_losses)}")
    print(f"Best val loss: {trainer.best_val_loss:.4f}")
    print()
    
    print("Validation Results:")
    print(f"  Validity:   {results['validity']:.1f}%")
    print(f"  Win Rate:   {results['win_rate']:.1f}%")
    print(f"  Bonus Rate: {results['bonus_rate']:.1f}%")
    print(f"  Speed:      {results['speed_ms']:.1f}ms")
    
    overall = (
        results['validity'] * 0.30 +
        results['win_rate'] * 0.40 +
        results['bonus_rate'] * 0.20 +
        (100 if results['speed_ms'] < 100 else 50) * 0.10
    )
    
    print(f"\n  OVERALL SCORE: {overall:.1f}%")
    
    if overall >= 75:
        print("  🏆 PRODUCTION READY! Deploy this model!")
    elif overall >= 65:
        print("  👍 GOOD! Consider training with more data")
    else:
        print("  ⚠️  Needs improvement. Try different hyperparameters")
    
    # Suggestions
    print("\n💡 Suggestions:")
    
    if results['validity'] < 95:
        print("  - Generate more data with stricter validation")
    
    if results['win_rate'] < 60:
        print("  - Increase training data (try 100k samples)")
        print("  - Increase epochs")
    
    if results['bonus_rate'] < 5:
        print("  - Add more expert strategies (trips, straights)")
        print("  - Increase bonus weight in reward function")
    
    print(f"\nModel saved at: ../../data/models/{experiment_name}/best_model.pth")
    print()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Complete ML Pipeline')
    
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of training samples (default: 50000)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs (default: 200)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Data generation workers (default: 4)')
    parser.add_argument('--tests', type=int, default=1000,
                        help='Validation tests (default: 1000)')
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        num_samples=args.samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        num_workers=args.workers,
        num_tests=args.tests,
        experiment_name=args.name
    )