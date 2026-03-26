"""
Data Generation Module
"""

from .expert_generator import ExpertDataGeneratorV3
from .self_play_generator import SelfPlayGenerator
from .augmentation import DataAugmentation

__all__ = [
    'ExpertDataGeneratorV3',
    'SelfPlayGenerator',
    'DataAugmentation',
]

__version__ = '2.0.0'

print(f"✅ ML Data v{__version__} loaded")