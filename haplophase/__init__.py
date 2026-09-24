"""haplophase: exact cross-fragment haplotype phasing.

Standard-library-only weighted minimum-error-correction solvers: a diploid
branch-and-bound entry (``phase_haplotypes``) and an explicit-copy-number
entry for ploidy 2 or 3 (``phase_polyploid``). See ``haplophase.core`` for
the full contract.
"""

from .core import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    POLYPLOID_MAX_LOCI,
    POLYPLOID_MAX_OBSERVATIONS,
    POLYPLOID_PLOIDIES,
    CopyAssignment,
    InvalidPhasingInputError,
    Observation,
    PhasingError,
    PhasingResult,
    PolyploidPhasingResult,
    ReadAssignment,
    SearchBudgetExhaustedError,
    SearchStats,
    phase_haplotypes,
    phase_polyploid,
)

__all__ = [
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
    "POLYPLOID_MAX_LOCI",
    "POLYPLOID_MAX_OBSERVATIONS",
    "POLYPLOID_PLOIDIES",
    "CopyAssignment",
    "InvalidPhasingInputError",
    "Observation",
    "PhasingError",
    "PhasingResult",
    "PolyploidPhasingResult",
    "ReadAssignment",
    "SearchBudgetExhaustedError",
    "SearchStats",
    "phase_haplotypes",
    "phase_polyploid",
]

__version__ = "1.1.0"
