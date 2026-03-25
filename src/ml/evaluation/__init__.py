"""
Evaluation Module
"""

from .validator import ModelValidator
from .benchmark import Benchmark
from .metrics import MetricsVisualizer

__all__ = [
    'ModelValidator',
    'Benchmark',
    'MetricsVisualizer',
]

__version__ = '2.0.0'

print(f"✅ ML Evaluation v{__version__} loaded")