"""haplophase: exact cross-fragment diploid haplotype phasing.

Standard-library-only weighted minimum-error-correction solver using branch
and bound. See ``haplophase.core`` for the full contract.
"""

from .core import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    MAX_TOP_K,
    InvalidPhasingInputError,
    Observation,
    PhasingError,
    PhasingResult,
    ReadAssignment,
    SearchBudgetExhaustedError,
    SearchStats,
    TopKCandidate,
    TopKResult,
    phase_haplotypes,
    phase_haplotypes_top_k,
)

__all__ = [
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
    "MAX_TOP_K",
    "InvalidPhasingInputError",
    "Observation",
    "PhasingError",
    "PhasingResult",
    "ReadAssignment",
    "SearchBudgetExhaustedError",
    "SearchStats",
    "TopKCandidate",
    "TopKResult",
    "phase_haplotypes",
    "phase_haplotypes_top_k",
]

__version__ = "1.0.0"
