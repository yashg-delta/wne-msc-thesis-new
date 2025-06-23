"""
Enhanced Strategy Evaluation Module

Provides comprehensive strategy analysis capabilities including:
- Individual trade tracking and analysis
- Advanced risk metrics
- Detailed performance analytics
- Professional reporting and visualization

This module extends the original strategy evaluation framework
without modifying existing code.
"""

from .core.enhanced_evaluator import EnhancedEvaluator
from .core.trade_analyzer import TradeAnalyzer
from .core.advanced_metrics import AdvancedMetrics

__version__ = "1.0.0"
__author__ = "Enhanced Evaluation Framework"

__all__ = [
    "EnhancedEvaluator",
    "TradeAnalyzer", 
    "AdvancedMetrics"
]