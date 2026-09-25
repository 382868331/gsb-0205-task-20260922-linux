"""haplophase: exact cross-fragment diploid haplotype phasing.

Standard-library-only weighted minimum-error-correction solver using branch
and bound. See ``haplophase.core`` for the full contract.
"""

from .core import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    InvalidPhasingInputError,
    Observation,
    PhasingError,
    PhasingResult,
    PosteriorResult,
    ReadAssignment,
    SearchBudgetExhaustedError,
    SearchStats,
    phase_haplotypes,
    phase_posterior,
)

__all__ = [
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
    "InvalidPhasingInputError",
    "Observation",
    "PhasingError",
    "PhasingResult",
    "PosteriorResult",
    "ReadAssignment",
    "SearchBudgetExhaustedError",
    "SearchStats",
    "phase_haplotypes",
    "phase_posterior",
]

__version__ = "1.0.0"
