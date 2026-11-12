"""
Combines Type A and Type B standard uncertainties into a combined standard
uncertainty and an expanded uncertainty, per GUM Section 5 (combination) and
Section 6 (expanded uncertainty / coverage factor).

Pure arithmetic, no I/O and no dependency on the rest of this project, so it
can be unit-tested or reused on its own.
"""
import math
from dataclasses import dataclass

from .type_a import TypeAResult


@dataclass
class CombinedUncertainty:
    channel: str
    type_a_standard_uncertainty: float
    type_b_standard_uncertainty: float
    combined_standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float
    n_samples: int  # how many repeated readings the Type A estimate is based on


def combine(channel: str, type_a: TypeAResult, type_b_standard_uncertainty: float,
            coverage_factor: float = 2.0) -> CombinedUncertainty:
    """
    Combined standard uncertainty for uncorrelated inputs (GUM 5.1.2):
        u_c = sqrt(u_A^2 + u_B^2)
    Expanded uncertainty (GUM 6.2):
        U = k * u_c
    k=2 is used as a conventional approximation of ~95% coverage assuming an
    approximately normal distribution of the combined uncertainty; it is not
    a rigorous Welch-Satterthwaite effective-degrees-of-freedom calculation,
    which would require more information (e.g. degrees of freedom per input)
    than is available here. This is called out explicitly rather than
    presented as more rigorous than it is.
    """
    u_a = type_a.standard_uncertainty
    u_b = type_b_standard_uncertainty
    u_c = math.sqrt(u_a ** 2 + u_b ** 2)
    return CombinedUncertainty(
        channel=channel,
        type_a_standard_uncertainty=u_a,
        type_b_standard_uncertainty=u_b,
        combined_standard_uncertainty=u_c,
        expanded_uncertainty=coverage_factor * u_c,
        coverage_factor=coverage_factor,
        n_samples=type_a.n,
    )
