"""Mathematical helpers: Hermite functions, polynomials, and densities."""

from __future__ import annotations

from math import factorial, pi, sqrt

import numpy as np
from scipy.misc import derivative
from scipy.special import hermite


def hermite_function(n: int, x: np.ndarray) -> np.ndarray:
    """Computes probabilists' oscillator Hermite function psi_n(x).

    psi_n(x) = (2^n n! sqrt(pi))^(-1/2) (-1)^n d^n/dx^n [exp(-x^2/2)].

    Args:
        n: Non-negative integer order.
        x: Evaluation points.

    Returns:
        Values of psi_n at ``x``, same shape as ``x``.
    """

    def base_function(t: np.ndarray | float) -> np.ndarray | float:
        return np.exp(-(t ** 2) / 2)

    dn_base = derivative(base_function, x, dx=1e-6, n=n, order=2 * n + 1)
    normalization = (2**n * factorial(n) * sqrt(pi)) ** (-0.5)
    sign = (-1) ** n
    return normalization * sign * dn_base


def standard_normal_density(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density phi(x).

    Args:
        x: Points where the density is evaluated.

    Returns:
        Density values (1 / sqrt(2 pi)) exp(-x^2 / 2).
    """
    return (1.0 / sqrt(2 * pi)) * np.exp(-(x ** 2) / 2)


def hermite_poly(n: int):
    """Returns the physicists' Hermite polynomial H_n as a callable.

    Args:
        n: Polynomial degree (non-negative).

    Returns:
        A polynomial object from ``scipy.special.hermite`` that can be evaluated
        on arrays.
    """
    return hermite(n)
