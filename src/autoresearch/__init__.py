"""LLM-driven autoresearch loop for Decision Transformer hyperparameter search."""

from .agent import AutoresearchAgent
from .ledger import ExperimentLedger, LedgerEntry
from .runner import AutoresearchRunner
from .stage_a import StageAScreen
from .stage_b import StageBEvaluator
from .llm_backend import LlamaCppBackend, OllamaBackend, OpenAIBackend

__all__ = [
    "AutoresearchAgent",
    "ExperimentLedger",
    "LedgerEntry",
    "AutoresearchRunner",
    "StageAScreen",
    "StageBEvaluator",
    "LlamaCppBackend",
    "OllamaBackend",
    "OpenAIBackend",
]
