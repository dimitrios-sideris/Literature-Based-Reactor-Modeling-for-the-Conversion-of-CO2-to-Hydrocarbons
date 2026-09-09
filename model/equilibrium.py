"""
Temperature-dependent equilibrium constants used in the kinetic model.

Equilibrium correlations from:
Cordero-Lanzac et al. (2023),
"A CO2 valorization plant to produce light hydrocarbons:
kinetic model, process design and life cycle assessment",
Journal of CO2 Utilization, 67, 102337.
DOI: 10.1016/j.jcou.2022.102337

Conventions:
- Temperature T in K
- Partial pressures in bar
- K_CO2_to_MeOH and K_CO_to_MeOH in bar^-2
- K_rWGS dimensionless
"""

import numpy as np


def K_CO2_to_MeOH(T):
    """
    Equilibrium constant for methanol synthesis from CO2:

        CO2 + 3 H2 <-> CH3OH + H2O

    Parameters
    ----------
    T : float or array-like
        Temperature [K].

    Returns
    -------
    K : float or ndarray
        Equilibrium constant [bar^-2].
    """

    # Literature correlation for ln(K)
    ln_K = (
        4213.0 / T
        - 5.752 * np.log(T)
        - 1.707e-3 * T
        + 2.682e-6 * T**2
        - 7.232e-10 * T**3
        + 17.6
    )

    return np.exp(ln_K)


def K_rWGS(T):
    """
    Equilibrium constant for the reverse water-gas shift reaction:

        CO2 + H2 <-> CO + H2O

    Parameters
    ----------
    T : float or array-like
        Temperature [K].

    Returns
    -------
    K : float or ndarray
        Equilibrium constant [-].
    """

    # Correlation is expressed for log10(K) in the opposite
    # (water-gas shift) reaction direction.
    log10_K_WGS = (
        2167.0 / T
        - 0.5194 * np.log10(T)
        + 1.037e-3 * T
        - 2.331e-7 * T**2
        - 1.2777
    )

    # Inversion gives the equilibrium constant for rWGS.
    return 10.0 ** (-log10_K_WGS)


def K_CO_to_MeOH(T):
    """
    Equilibrium constant for methanol synthesis from CO:

        CO + 2 H2 <-> CH3OH

    Obtained by combining the CO2-to-MeOH and rWGS equilibria:

        K_CO_to_MeOH = K_CO2_to_MeOH / K_rWGS

    Parameters
    ----------
    T : float or array-like
        Temperature [K].

    Returns
    -------
    K : float or ndarray
        Equilibrium constant [bar^-2].
    """

    return K_CO2_to_MeOH(T) / K_rWGS(T)