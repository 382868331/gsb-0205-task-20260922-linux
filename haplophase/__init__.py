"""haplophase: exact cross-fragment haplotype phasing.

Standard-library-only weighted minimum-error-correction solvers:
``phase_haplotypes`` (diploid, branch and bound) and ``phase_polyploid``
(explicit ploidy 2 or 3, exact enumeration).  See ``haplophase.core`` and
``haplophase.polyploid`` for the full contracts.
"""

from .core import (
    MAX_LOCI,
    MAX_OBSERVATIONS,
    InvalidPhasingInputError,
    Observation,
    PhasingError,
    PhasingResult,
    ReadAssignment,
    SearchBudgetExhaustedError,
    SearchStats,
    phase_haplotypes,
)
from .polyploid import (
    MAX_PLOIDY_LOCI,
    MAX_PLOIDY_OBSERVATIONS,
    SUPPORTED_PLOIDIES,
    EnumerationStats,
    PolyploidPhasingResult,
    PolyploidReadAssignment,
    phase_polyploid,
)

__all__ = [
    "MAX_LOCI",
    "MAX_OBSERVATIONS",
    "MAX_PLOIDY_LOCI",
    "MAX_PLOIDY_OBSERVATIONS",
    "SUPPORTED_PLOIDIES",
    "EnumerationStats",
    "InvalidPhasingInputError",
    "Observation",
    "PhasingError",
    "PhasingResult",
    "PolyploidPhasingResult",
    "PolyploidReadAssignment",
    "ReadAssignment",
    "SearchBudgetExhaustedError",
    "SearchStats",
    "phase_haplotypes",
    "phase_polyploid",
]

__version__ = "1.1.0"
