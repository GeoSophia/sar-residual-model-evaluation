"""Core statistics for SAR residual-model evaluation."""

from .analysis import paired_block_bootstrap, recovery_metrics, summarize_residuals

__version__ = "0.1.0"
__all__ = ["paired_block_bootstrap", "recovery_metrics", "summarize_residuals"]
