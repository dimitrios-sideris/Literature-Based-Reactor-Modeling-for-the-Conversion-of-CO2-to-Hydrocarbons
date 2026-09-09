"""
Kinetic model for the seven-reaction network used for direct CO2
hydrogenation to light paraffins.

Kinetic parameters and rate expressions from:
Cordero-Lanzac et al. (2023),
"A CO2 valorization plant to produce light hydrocarbons:
kinetic model, process design and life cycle assessment",
Journal of CO2 Utilization, 67, 102337.
DOI: 10.1016/j.jcou.2022.102337

Species order used throughout the model:
CO2, H2, CH3OH, H2O, CO, C2H6, C3H8, C4H10, CH4, He

Conventions:
- Temperature in K
- Pressure and partial pressures in bar
- Molar flows in mol/s
- Reaction rates returned in mol/(kg_cat s)
"""

import numpy as np

from model.equilibrium import K_CO2_to_MeOH, K_rWGS, K_CO_to_MeOH


# Species indices used throughout the reactor model
SPECIES = [
    "CO2",
    "H2",
    "CH3OH",
    "H2O",
    "CO",
    "C2H6",
    "C3H8",
    "C4H10",
    "CH4",
    "He",
]


# Reaction network
REACTIONS = [
    "CO2 + 3 H2 <-> CH3OH + H2O",       # R1: CO2 hydrogenation to methanol
    "CO2 + H2 <-> CO + H2O",             # R2: reverse water-gas shift
    "CO + 2 H2 <-> CH3OH",               # R3: CO hydrogenation to methanol
    "CO + 3 H2 -> CH4 + H2O",             # R4: CO methanation
    "2 CH3OH + H2 -> C2H6 + 2 H2O",       # R5: formation of C2 paraffin
    "3 CH3OH + H2 -> C3H8 + 3 H2O",       # R6: formation of C3 paraffin
    "4 CH3OH + H2 -> C4H10 + 4 H2O",      # R7: formation of C4 paraffin
]


R = 8.314       # Ideal gas constant [J/(mol K)]
T_REF = 623.0   # Reference temperature for kinetic parameters [K]


# Kinetic constants at T_REF.
# Units depend on the pressure order of the respective rate expression;
# pressure is expressed in bar and rates in mol/(g_cat h).
K_REF = np.array([
    8.35e-6,    # R1
    4.49e-4,    # R2
    1.32e-5,    # R3
    9.11e-6,    # R4
    3.14e-2,    # R5
    4.21e-2,    # R6
    9.11e-3,    # R7
])


# Apparent activation energies [J/mol]
E_A = np.array([
    1.04e2,     # R1
    4.91e1,     # R2
    5.97e1,     # R3
    4.55e1,     # R4
    8.38e1,     # R5
    8.38e1,     # R6
    8.38e1,     # R7
]) * 1000.0


# Adsorption / inhibition constants [bar^-1]
K_CO2_ADS = 6.34e-2
K_H2_ADS = 1.30e-2
K_H2O_ADS = 9.11e-1


def calculate_rate_constants(T):
    """
    Calculate the seven kinetic constants at reactor temperature T.

    Parameters
    ----------
    T : float
        Reactor temperature [K].

    Returns
    -------
    k : ndarray
        Temperature-dependent kinetic constants.
    """

    # Arrhenius relation referenced to T_REF
    return K_REF * np.exp(
        (E_A / R) * ((1.0 / T_REF) - (1.0 / T))
    )


def calculate_reaction_rates(F, T, P_bar):
    """
    Calculate the reaction rates for reactions R1-R7.

    Parameters
    ----------
    F : array-like
        Component molar flows [mol/s] in the order defined by SPECIES.
    T : float
        Reactor temperature [K].
    P_bar : float
        Total reactor pressure [bar].

    Returns
    -------
    r : ndarray
        Reaction rates [mol/(kg_cat s)].
    """

    F = np.asarray(F, dtype=float)

    # Mole fractions and partial pressures
    F_total = np.sum(F)
    y = F / F_total
    p = P_bar * y                         # Partial pressures [bar]

    # Temperature-dependent kinetic constants
    k = calculate_rate_constants(T)

    # Thermodynamic equilibrium constants
    K1 = K_CO2_to_MeOH(T)                 # R1
    K2 = K_rWGS(T)                        # R2
    K3 = K_CO_to_MeOH(T)                  # R3

    # Adsorption term for reactions on the metal/oxide catalyst
    denominator_metal = (
        1.0
        + K_CO2_ADS * p[0]
        + np.sqrt(K_H2_ADS * p[1])
    )**2

    # Water inhibition term for hydrocarbon formation
    denominator_zeolite = 1.0 + K_H2O_ADS * p[3]

    r = np.zeros(7)

    # R1: CO2 + 3 H2 <-> CH3OH + H2O
    r[0] = (
        k[0]
        * (p[0] * p[1]**3 - (p[2] * p[3]) / K1)
        / denominator_metal
    )

    # R2: CO2 + H2 <-> CO + H2O
    r[1] = (
        k[1]
        * (p[0] * p[1] - (p[4] * p[3]) / K2)
        / denominator_metal
    )

    # R3: CO + 2 H2 <-> CH3OH
    r[2] = (
        k[2]
        * (p[4] * p[1]**2 - p[2] / K3)
        / denominator_metal
    )

    # R4: CO + 3 H2 -> CH4 + H2O
    r[3] = (
        k[3]
        * (p[4] * p[1])
        / denominator_metal
    )

    # R5-R7: methanol conversion to C2-C4 paraffins
    r[4] = k[4] * p[2] / denominator_zeolite
    r[5] = k[5] * p[2] / denominator_zeolite
    r[6] = k[6] * p[2] / denominator_zeolite

    # Literature rates: mol/(g_cat h)
    # Reactor model:    mol/(kg_cat s)
    return r * (1000.0 / 3600.0)


def calculate_species_rates(F, T, P_bar):
    """
    Calculate the net formation or consumption rate of each species.

    Parameters
    ----------
    F : array-like
        Component molar flows [mol/s].
    T : float
        Reactor temperature [K].
    P_bar : float
        Reactor pressure [bar].

    Returns
    -------
    dF_dW : ndarray
        Net species rates [mol/(kg_cat s)].
        These are used directly in the packed-bed balance dF_i/dW = r_i.
    """

    r = calculate_reaction_rates(F, T, P_bar)

    dF_dW = np.zeros_like(np.asarray(F, dtype=float))

    # Species rates obtained from the reaction stoichiometry
    dF_dW[0] = -r[0] - r[1]                                            # CO2
    dF_dW[1] = -3*r[0] - r[1] - 2*r[2] - 3*r[3] - r[4] - r[5] - r[6] # H2
    dF_dW[2] = r[0] + r[2] - 2*r[4] - 3*r[5] - 4*r[6]                # CH3OH
    dF_dW[3] = r[0] + r[1] + r[3] + 2*r[4] + 3*r[5] + 4*r[6]         # H2O
    dF_dW[4] = r[1] - r[2] - r[3]                                    # CO
    dF_dW[5] = r[4]                                                    # C2H6
    dF_dW[6] = r[5]                                                    # C3H8
    dF_dW[7] = r[6]                                                    # C4H10
    dF_dW[8] = r[3]                                                    # CH4
    dF_dW[9] = 0.0                                                     # He, inert

    return dF_dW
