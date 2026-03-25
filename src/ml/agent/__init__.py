"""
Agent Module
"""

from .mau_binh_agent import MauBinhAgent
from .search import BeamSearch, MonteCarloTreeSearch

__all__ = [
    'MauBinhAgent',
    'BeamSearch',
    'MonteCarloTreeSearch',
]

__version__ = '2.0.0'

print(f"✅ ML Agent v{__version__} loaded")