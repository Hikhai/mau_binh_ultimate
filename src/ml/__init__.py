"""
Mau Binh ML Module V2.0 - Production Ready
Complete ML pipeline for Mau Binh card game

Components:
- Core: State encoding, action decoding, reward calculation
- Networks: DQN, Transformer, Ensemble
- Data: Expert generation, self-play, augmentation
- Training: Advanced trainer, curriculum learning, callbacks
- Agent: Production inference agent
- Evaluation: Validation, benchmarking, metrics

Author: Ultimate Team
Version: 2.0.0
"""

# Core components
from .core import (
    RewardCalculator,
    StateEncoderV2,
    ActionDecoderV2,
    ArrangementValidator,
)

# Networks
from .networks import (
    DQNNetwork,
    TransformerNetwork,
    EnsembleNetwork,
)

# Data generation
from .data import (
    ExpertDataGeneratorV3,
    SelfPlayGenerator,
    DataAugmentation,
)

# Training
from .training import (
    TrainerV3,
    CurriculumScheduler,
    ProgressLogger,
    MetricTracker,
    EarlyStoppingCallback,
    ModelCheckpoint,
)

# Agent
from .agent import (
    MauBinhAgent,
    BeamSearch,
    MonteCarloTreeSearch,
)

# Evaluation
from .evaluation import (
    ModelValidator,
    Benchmark,
    MetricsVisualizer,
)

__version__ = '2.0.0'

__all__ = [
    # Core
    'RewardCalculator',
    'StateEncoderV2',
    'ActionDecoderV2',
    'ArrangementValidator',
    
    # Networks
    'DQNNetwork',
    'TransformerNetwork',
    'EnsembleNetwork',
    
    # Data
    'ExpertDataGeneratorV3',
    'SelfPlayGenerator',
    'DataAugmentation',
    
    # Training
    'TrainerV3',
    'CurriculumScheduler',
    'ProgressLogger',
    'MetricTracker',
    'EarlyStoppingCallback',
    'ModelCheckpoint',
    
    # Agent
    'MauBinhAgent',
    'BeamSearch',
    'MonteCarloTreeSearch',
    
    # Evaluation
    'ModelValidator',
    'Benchmark',
    'MetricsVisualizer',
]

print(f"""
╔════════════════════════════════════════════════════════════╗
║  🎯 MAU BINH ML MODULE V{__version__}                            ║
║  Production-Ready Machine Learning Pipeline                ║
╠════════════════════════════════════════════════════════════╣
║  ✅ Core components loaded                                 ║
║  ✅ Neural networks ready                                  ║
║  ✅ Data generators ready                                  ║
║  ✅ Training pipeline ready                                ║
║  ✅ Agent ready for inference                              ║
║  ✅ Evaluation tools ready                                 ║
╚════════════════════════════════════════════════════════════╝
""")