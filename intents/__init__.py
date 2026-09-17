"""
Intent classification module for customer support requests.

Exports:
- IntentCategory, RiskLevel, IntentDefinition, SUPPORT_TAXONOMY
- BaseIntentClassifier, CalibratedHeuristicClassifier, LLMIntentClassifier, HybridIntentClassifier, IntentPrediction
"""

from intents.classifier import (
    BaseIntentClassifier,
    CalibratedHeuristicClassifier,
    HybridIntentClassifier,
    IntentPrediction,
    LLMIntentClassifier,
)
from intents.taxonomy import (
    IntentCategory,
    IntentDefinition,
    RiskLevel,
    SUPPORT_TAXONOMY,
)

__all__ = [
    "IntentCategory",
    "RiskLevel",
    "IntentDefinition",
    "SUPPORT_TAXONOMY",
    "BaseIntentClassifier",
    "CalibratedHeuristicClassifier",
    "LLMIntentClassifier",
    "HybridIntentClassifier",
    "IntentPrediction",
]
