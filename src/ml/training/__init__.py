"""
Training Module
"""

from .trainer import TrainerV3
from .curriculum import CurriculumScheduler
from .callbacks import (
    TrainingCallback,
    ProgressLogger,
    MetricTracker,
    EarlyStoppingCallback,
    ModelCheckpoint
)

__all__ = [
    'TrainerV3',
    'CurriculumScheduler',
    'TrainingCallback',
    'ProgressLogger',
    'MetricTracker',
    'EarlyStoppingCallback',
    'ModelCheckpoint',
]

__version__ = '2.0.0'

print(f"✅ ML Training v{__version__} loaded")