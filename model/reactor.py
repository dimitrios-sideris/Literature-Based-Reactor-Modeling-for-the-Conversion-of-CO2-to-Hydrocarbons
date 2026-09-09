"""
Packed-bed reactor model and feed/result calculations.

The reactor is modelled as an isothermal and isobaric packed-bed reactor
using the design equation:

    dF_i / dW = r_i

with catalyst mass W as the independent variable.

Conventions:
- Catalyst mass in kg
- Molar flows in mol/s
- Temperature in K
- Pressure in bar
- Space time in g_cat h / mol_COx
"""

import numpy as np
from scipy.integrate import solve_ivp

from model.kinetics import SPECIES, calculate_species_rates


def reactor_ode(W, F, T, P_bar):
    """
    Packed-bed reactor balance:

        dF_i / dW = r_i

    where F_i is the molar flow of species i and W is catalyst mass.
    """

    return calculate_species_rates(F, T, P_bar)


def solve_reactor(
    catalyst_mass_kg,
    inlet_molar_flows,
    T,
    P_bar,
    evaluation_points=None,
):
    """
    Solve the isothermal, isobaric packed-bed reactor model.

    Parameters
    ----------
    catalyst_mass_kg : float
        Total catalyst mass [kg].
    inlet_molar_flows : array-like
        Inlet molar flows [mol/s] in the order defined by SPECIES.
    T : float
        Reactor temperature [K].
    P_bar : float
        Reactor pressure [bar].
    evaluation_points : array-like, optional
        Catalyst-mass positions at which the solution is returned [kg].

    Returns
    -------
    W : ndarray
        Catalyst-mass coordinate [kg].
    F : ndarray
        Species molar-flow profiles [mol/s].
        Rows correspond to species and columns to catalyst-mass positions.
    """

    W_span = (0.0, catalyst_mass_kg)

    # Maximum integration step: 0.1% of the total catalyst mass
    max_step = catalyst_mass_kg * 1e-3

    result = solve_ivp(
        reactor_ode,
        W_span,
        np.asarray(inlet_molar_flows, dtype=float),
        args=(T, P_bar),
        method="BDF",
        dense_output=True,
        max_step=max_step,
        rtol=1e-9,
        atol=1e-9,
        t_eval=evaluation_points,
    )

    if not result.success:
        raise RuntimeError(result.message)

    W = result.t
    F = result.y

    # Remove very small numerical values produced by the ODE solver
    F[np.abs(F) < 1e-10] = 0.0

    # Negative molar flows are non-physical and only occur from
    # numerical integration close to zero.
    F[F < 0.0] = 0.0

    return W, F


def create_inlet_from_space_time(
    space_time,
    catalyst_mass_g=1.0,
    H2_over_COx=3.0,
    CO2_fraction_in_COx=1.0,
    CO_fraction_in_COx=None,
    helium_fraction=0.2,
):
    """
    Calculate the inlet molar flows from the specified space time.

    Space time is defined as:

        tau = W / F_COx

    where W is catalyst mass and F_COx is the combined inlet molar flow
    of CO2 and CO.

    Parameters
    ----------
    space_time : float
        Space time [g_cat h / mol_COx].
    catalyst_mass_g : float
        Catalyst mass [g].
    H2_over_COx : float
        Inlet molar ratio H2 / (CO2 + CO) [-].
    CO2_fraction_in_COx : float
        Fraction of CO2 in the total COx feed [-].
    CO_fraction_in_COx : float, optional
        Fraction of CO in the total COx feed [-].
        If omitted, it is calculated as 1 - CO2_fraction_in_COx.
    helium_fraction : float
        Helium mole fraction in the total inlet gas [-].

    Returns
    -------
    F : ndarray
        Inlet molar flows [mol/s] in the order defined by SPECIES.
    """

    if CO_fraction_in_COx is None:
        CO_fraction_in_COx = 1.0 - CO2_fraction_in_COx

    # COx molar flow from tau = W / F_COx
    # Division by 3600 converts mol/h to mol/s.
    F_COx = catalyst_mass_g / space_time / 3600.0

    F = np.zeros(len(SPECIES))

    # Reactive inlet flows
    F[0] = CO2_fraction_in_COx * F_COx
    F[4] = CO_fraction_in_COx * F_COx
    F[1] = H2_over_COx * F_COx

    # Helium is included as an inert gas.
    # The following relation gives the required He flow so that
    # helium_fraction refers to the total inlet gas.
    reactive_flow = F[0] + F[1] + F[4]
    F[9] = (
        helium_fraction
        / (1.0 - helium_fraction)
        * reactive_flow
    )

    return F


def carbon_basis_fractions(F):
    """
    Calculate CO2, CO and C1-C4 fractions on a carbon basis.

    Each species flow is weighted by its number of carbon atoms before
    normalization. Methanol is not included because it is treated as an
    intermediate in the reaction network.

    Parameters
    ----------
    F : array-like
        Species molar flows [mol/s].

    Returns
    -------
    dict
        Carbon-basis fractions of CO2, CO and C1-C4 [-].
    """

    F = np.asarray(F, dtype=float)

    # Carbon molar flows [mol_C/s]
    carbon_flows = {
        "CO2": F[0],
        "CO":  F[4],
        "C1":  F[8],
        "C2":  2.0 * F[5],
        "C3":  3.0 * F[6],
        "C4":  4.0 * F[7],
    }

    total_carbon = sum(carbon_flows.values())

    return {
        species: flow / total_carbon
        for species, flow in carbon_flows.items()
    }