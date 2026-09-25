"""haplophase: exact cross-fragment diploid haplotype phasing.

Standard-library-only weighted minimum-error-correction solver using branch
and bound. See ``haplophase.core`` for the full contract.
"""

from .core import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    MAX_TOP_K,
    Candidate,
    InvalidPhasingInputError,
    Observation,
    PhasingError,
    PhasingResult,
    ReadAssignment,
    SearchBudgetExhaustedError,
    SearchStats,
    TopKResult,
    phase_haplotypes,
    top_k_haplotypes,
)

__all__ = [
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
    "MAX_TOP_K",
    "Candidate",
    "InvalidPhasingInputError",
    "Observation",
    "PhasingError",
    "PhasingResult",
    "ReadAssignment",
    "SearchBudgetExhaustedError",
    "SearchStats",
    "TopKResult",
    "phase_haplotypes",
    "top_k_haplotypes",
]

__version__ = "1.0.0"
