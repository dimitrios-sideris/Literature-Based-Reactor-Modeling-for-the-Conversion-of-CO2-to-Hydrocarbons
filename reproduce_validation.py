"""Validation and literature-comparison plots for the kinetic reactor model.

S4 and S5 compare the calculated carbon-basis product distributions with
digitized data from Cordero-Lanzac et al. (2023). Figure 5a reproduces the
water buildup along the catalyst bed, while Figure 5b shows the forward and
reverse reaction-rate contributions for different COx feeds.

Digitized data are read from ./data and generated figures are saved in ./figures.
"""

from pathlib import Path
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.equilibrium import K_CO2_to_MeOH, K_rWGS, K_CO_to_MeOH
from model.kinetics import (
    K_REF,
    E_A,
    R,
    T_REF,
    K_CO2_ADS,
    K_H2_ADS,
    K_H2O_ADS,
)
from model.reactor import solve_reactor, carbon_basis_fractions


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

# Display figures after saving (False: save only).
SHOW_FIGURES = True

VALIDATION_SPECIES = ["CO2", "CO", "C1", "C2", "C3", "C4"]

# S4/S5 operating conditions and corresponding digitized data files.
S4_CASES = [
    ("S4a", 325, 40, 1.0, 0.0, 33, "S4a_325C_40bar_CO2.csv"),
    ("S4b", 350, 30, 1.0, 0.0, 17, "S4b_350C_30bar_CO2.csv"),
    ("S4c", 350, 50, 1.0, 0.0, 65, "S4c_350C_50bar_CO2.csv"),
    ("S4d", 375, 40, 1.0, 0.0, 33, "S4d_375C_40bar_CO2.csv"),
    ("S4e", 400, 40, 1.0, 0.0, 33, "S4e_400C_40bar_CO2.csv"),
    ("S4f", 350, 20, 1.0, 0.0, 33, "S4f_350C_20bar_CO2.csv"),
]

S5_CASES = [
    ("S5a", 350, 20, 0.0, 1.0, 33, "S5a_350C_20bar_CO.csv"),
    ("S5b", 350, 20, 0.5, 0.5, 33, "S5b_350C_20bar_CO2_CO_50_50.csv"),
    ("S5c", 350, 30, 0.5, 0.5, 33, "S5c_350C_30bar_CO2_CO_50_50.csv"),
    ("S5d", 350, 30, 0.0, 1.0, 33, "S5d_350C_30bar_CO.csv"),
    ("S5e", 400, 40, 0.5, 0.5, 33, "S5e_400C_40bar_CO2_CO_50_50.csv"),
    ("S5f", 400, 40, 0.0, 1.0, 33, "S5f_400C_40bar_CO.csv"),
]


def load_validation_csv(filename):
    """Read digitized data with species in the first header row and X/Y in the second."""
    df = pd.read_csv(DATA_DIR / filename, header=[0, 1])

    top = df.columns.get_level_values(0).astype(str)
    top = pd.Series(top).where(~pd.Series(top).str.match(r"^Unnamed"), other=np.nan)
    top = top.replace("nan", np.nan).ffill().fillna("").astype(str).str.strip().values
    sub = df.columns.get_level_values(1).astype(str).str.strip().values
    df.columns = pd.MultiIndex.from_arrays([top, sub])

    return df


def label_with_subscript(text):
    return re.sub(r"(\d+)", r"$_{\1}$", text)


def simulate_space_time_curve(
    T_C,
    P_bar,
    CO2_fraction_in_COx,
    CO_fraction_in_COx,
    tau_max,
    n_points,
):
    """Calculate the carbon-basis composition over a space-time range.

    The rate expressions depend on gas composition and partial pressures rather
    than the absolute flow scale. A normalized inlet with F_COx = 1 mol/s is
    therefore used to calculate the complete space-time profile in one reactor
    integration.
    """
    H2_over_COx = 3.0
    helium_fraction = 0.2

    tau = np.linspace(0.0, tau_max, n_points)

    # Normalize the total COx inlet flow to 1 mol/s.
    F_in = np.zeros(10)
    F_in[0] = CO2_fraction_in_COx
    F_in[4] = CO_fraction_in_COx
    F_in[1] = H2_over_COx

    reactive_flow = F_in[0] + F_in[4] + F_in[1]
    F_in[9] = helium_fraction / (1.0 - helium_fraction) * reactive_flow

    # From tau = W_g / F_COx,h, F_COx = 1 mol/s gives W [kg] = 3.6 * tau.
    W_points = 3.6 * tau

    _, F_profile = solve_reactor(
        W_points[-1],
        F_in,
        T_C + 273.15,
        P_bar,
        evaluation_points=W_points,
    )

    curves = {species: np.zeros_like(tau) for species in VALIDATION_SPECIES}
    for i in range(F_profile.shape[1]):
        y_carbon = carbon_basis_fractions(F_profile[:, i])
        for species in VALIDATION_SPECIES:
            curves[species][i] = y_carbon[species]

    return tau, curves


def plot_validation_panel(ax, T_C, P_bar, CO2_fraction, CO_fraction, tau_max, n_points, csv_file):
    tau, curves = simulate_space_time_curve(
        T_C,
        P_bar,
        CO2_fraction,
        CO_fraction,
        tau_max,
        n_points,
    )

    # Calculated carbon-basis composition profiles
    for species in VALIDATION_SPECIES:
        ax.plot(tau, curves[species], label=f"Model {label_with_subscript(species)}")

    # Digitized literature data
    data = load_validation_csv(csv_file)
    for species in VALIDATION_SPECIES:
        x = data[(species, "X")]
        y = data[(species, "Y")]
        valid = x.notna() & y.notna()
        ax.scatter(
            x[valid].to_numpy(dtype=float),
            y[valid].to_numpy(dtype=float),
            s=30,
            zorder=5,
            label=f"Exp {label_with_subscript(species)}",
        )

    ax.set_xlim(0, tau_max)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Space time (g h mol$^{-1}$)")
    ax.set_ylabel("y (carbon basis)")
    ax.legend(ncol=2, frameon=False)


def plot_S4_validation():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

    for ax, (tag, T_C, P_bar, CO2_fraction, CO_fraction, tau_max, csv_file) in zip(axes.flat, S4_CASES):
        plot_validation_panel(
            ax,
            T_C,
            P_bar,
            CO2_fraction,
            CO_fraction,
            tau_max,
            60,
            csv_file,
        )
        ax.set_title(f"{tag}: {T_C}°C, {P_bar} bar")

    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")

    fig.tight_layout()
    output = FIGURE_DIR / "S4_validation.png"
    fig.savefig(output, dpi=100, bbox_inches="tight")
    return fig, output


def plot_S5_validation():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)

    for ax, (tag, T_C, P_bar, CO2_fraction, CO_fraction, tau_max, csv_file) in zip(axes.flat, S5_CASES):
        plot_validation_panel(
            ax,
            T_C,
            P_bar,
            CO2_fraction,
            CO_fraction,
            tau_max,
            30,
            csv_file,
        )
        ax.set_title(
            f"{tag}: {T_C}°C, {P_bar} bar, "
            f"CO2/COx={CO2_fraction:.2f}, CO/COx={CO_fraction:.2f}"
        )

    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")

    fig.tight_layout()
    output = FIGURE_DIR / "S5_validation.png"
    fig.savefig(output, dpi=100, bbox_inches="tight")
    return fig, output


# -----------------------------------------------------------------------------
# Figure 5a: water buildup along the catalyst bed
# -----------------------------------------------------------------------------
def create_GHSV_inlet(CO2_fraction_in_COx, T_C, P_bar, H2_over_COx=3.0, GHSV=3000.0, catalyst_mass_g=0.1, helium_fraction=0.2):
    """Calculate inlet molar flows from GHSV using the original model formulation."""
    normal_molar_volume = 22414.0
    y_rest = 1.0 - helium_fraction
    denominator = H2_over_COx + 1.0 + (helium_fraction / y_rest) * (1.0 + H2_over_COx)

    y_CO2 = CO2_fraction_in_COx / denominator
    y_H2 = H2_over_COx / denominator
    y_CO = 1.0 - y_CO2 - y_H2 - helium_fraction

    F_total = catalyst_mass_g * GHSV / (normal_molar_volume * 3600.0)

    F = np.zeros(10)
    F[0] = y_CO2 * F_total
    F[1] = y_H2 * F_total
    F[4] = y_CO * F_total
    F[9] = helium_fraction * F_total
    return F


def water_profile(T_C, P_bar, CO2_fraction_in_COx):
    catalyst_mass_g = 0.1
    F_in = create_GHSV_inlet(CO2_fraction_in_COx, T_C, P_bar)
    W, F = solve_reactor(catalyst_mass_g * 1e-3, F_in, T_C + 273.15, P_bar)

    relative_bed_length = W / W[-1]
    carbon_flow = F[0] + F[4] + F[8] + 2.0*F[5] + 3.0*F[6] + 4.0*F[7]
    water_per_carbon = np.where(carbon_flow > 0.0, F[3] / carbon_flow, 0.0)

    return relative_bed_length, water_per_carbon


def plot_Figure5a_water_profile():
    colors = {325: "#4C9BE8", 350: "#E74C3C", 375: "#2ECC71", 400: "#F39C12"}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    pure_CO2_cases = [
        (400, 30, "-"),
        (375, 30, "-"),
        (350, 30, "--"),
        (350, 40, "-"),
        (325, 30, "-"),
    ]

    for T_C, P_bar, line_style in pure_CO2_cases:
        x, y = water_profile(T_C, P_bar, 1.0)
        ax.plot(x, y, color=colors[T_C], lw=2.0, ls=line_style, label=f"CO$_2$: {T_C} °C, {P_bar} bar")

    x, y = water_profile(350, 30, 0.5)
    ax.plot(x, y, color="k", lw=2.0, ls="--", alpha=0.6, label="CO$_2$/CO: 350 °C, 30 bar")

    x, y = water_profile(350, 30, 0.0)
    ax.plot(x, y, color="k", lw=2.0, ls=":", alpha=0.85, label="CO: 350 °C, 30 bar")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis="y", which="minor", length=3)
    ax.tick_params(axis="y", which="major", length=6)
    ax.set_xlabel("Relative bed length (z/L)")
    ax.set_ylabel(r"$y_{\mathrm{H_2O}}$ (mol mol$_C^{-1}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    fig.subplots_adjust(right=0.72)
    fig.tight_layout()

    output = FIGURE_DIR / "Figure5a_water_profile.png"
    fig.savefig(output, dpi=100, bbox_inches="tight")
    return fig, output


# -----------------------------------------------------------------------------
# Figure 5b: forward and reverse reaction-rate contributions at the outlet
# -----------------------------------------------------------------------------
def calculate_forward_reverse_rate_components(F, T_K, P_bar):
    F_total = np.sum(F)
    p = P_bar * F / F_total

    k = K_REF * np.exp((E_A / R) * ((1.0 / T_REF) - (1.0 / T_K)))

    K1 = K_CO2_to_MeOH(T_K)
    K2 = K_rWGS(T_K)
    K3 = K_CO_to_MeOH(T_K)

    denominator_metal = (1.0 + K_CO2_ADS * p[0] + np.sqrt(K_H2_ADS * p[1]))**2
    denominator_zeolite = 1.0 + K_H2O_ADS * p[3]

    r0_forward = k[0] * p[0] * p[1]**3 / denominator_metal
    r0_reverse = k[0] * ((p[2] * p[3]) / K1) / denominator_metal

    r1_forward = k[1] * p[0] * p[1] / denominator_metal
    r1_reverse = k[1] * ((p[4] * p[3]) / K2) / denominator_metal

    r2_forward = k[2] * p[4] * p[1]**2 / denominator_metal
    r2_reverse = k[2] * (p[2] / K3) / denominator_metal

    r3 = k[3] * p[4] * p[1] / denominator_metal
    r4 = k[4] * p[2] / denominator_zeolite
    r5 = k[5] * p[2] / denominator_zeolite
    r6 = k[6] * p[2] / denominator_zeolite

    # Convert reaction rates from mol/(g_cat h) to mol/(kg_cat h).
    return {
        "CO2_MeOH_forward": 1000.0 * r0_forward,
        "CO2_MeOH_reverse": 1000.0 * r0_reverse,
        "CO2_CO_forward": 1000.0 * r1_forward,
        "CO2_CO_reverse": 1000.0 * r1_reverse,
        "CO_MeOH_forward": 1000.0 * r2_forward,
        "CO_MeOH_reverse": 1000.0 * r2_reverse,
        "CO_to_C1": 1000.0 * r3,
        "MeOH_to_LPG": 1000.0 * (r4 + r5 + r6),
    }


def outlet_rate_components(CO2_fraction_in_COx):
    T_C = 350.0
    P_bar = 30.0
    catalyst_mass_g = 0.1

    F_in = create_GHSV_inlet(CO2_fraction_in_COx, T_C, P_bar)
    _, F = solve_reactor(catalyst_mass_g * 1e-3, F_in, T_C + 273.15, P_bar)

    return calculate_forward_reverse_rate_components(F[:, -1], T_C + 273.15, P_bar)


def plot_Figure5b_reaction_rates():
    groups = [("CO$_2$", 1.0), ("CO$_2$/CO", 0.5), ("CO", 0.0)]
    rates = [outlet_rate_components(CO2_fraction) for _, CO2_fraction in groups]

    blue = "#1f77b4"
    red = "#d62728"
    green = "#2ca02c"
    purple = "#9467bd"
    orange = "#ff7f0e"

    bar_width = 0.10
    offsets = (np.arange(8) - 3.5) * bar_width * 1.05
    x_groups = np.arange(len(groups))

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(8.2, 4.6),
        gridspec_kw={"height_ratios": [1, 2]},
    )

    ax_bottom.set_ylim(0, 15)
    ax_top.set_ylim(35, 50)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()

    d = 0.007
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1.0)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    for group_index, rate in enumerate(rates):
        bars = [
            (rate["CO2_MeOH_forward"], blue, None),
            (rate["CO2_MeOH_reverse"], blue, "///"),
            (rate["CO2_CO_forward"], red, None),
            (rate["CO2_CO_reverse"], red, "///"),
            (rate["CO_MeOH_forward"], green, None),
            (rate["CO_MeOH_reverse"], green, "///"),
            (rate["CO_to_C1"], purple, None),
            (rate["MeOH_to_LPG"], orange, None),
        ]

        for ax in [ax_top, ax_bottom]:
            for bar_index, (height, color, hatch) in enumerate(bars):
                ax.bar(
                    x_groups[group_index] + offsets[bar_index],
                    height,
                    width=bar_width,
                    color=color,
                    edgecolor="k",
                    linewidth=0.5,
                    hatch=hatch,
                    alpha=0.75 if hatch else 1.0,
                )

    for ax in [ax_top, ax_bottom]:
        ax.axvline(0.5, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.axvline(1.5, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.25)

    ax_bottom.set_xticks(x_groups)
    ax_bottom.set_xticklabels([group[0] for group in groups])
    ax_bottom.set_ylabel(r"Rate (mol kg$^{-1}$ h$^{-1}$)")
    ax_top.set_title("T=350 °C, P=30 bar, H$_2$/COx=3, GHSV=3000")

    legend_handles = [
        mpatches.Patch(facecolor=blue, edgecolor="k", label=r"CO$_2\leftrightarrow$MeOH"),
        mpatches.Patch(facecolor=red, edgecolor="k", label=r"CO$_2\leftrightarrow$CO"),
        mpatches.Patch(facecolor=green, edgecolor="k", label=r"CO$\leftrightarrow$MeOH"),
        mpatches.Patch(facecolor=purple, edgecolor="k", label=r"CO$\rightarrow$C$_1$"),
        mpatches.Patch(facecolor=orange, edgecolor="k", label=r"MeOH$\rightarrow$LPG"),
        mpatches.Patch(facecolor="white", edgecolor="k", label="Forward"),
        mpatches.Patch(facecolor="white", edgecolor="k", hatch="///", label="Reverse"),
    ]

    ax_top.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.subplots_adjust(right=0.76)
    fig.tight_layout()

    output = FIGURE_DIR / "Figure5b_reaction_rates.png"
    fig.savefig(output, dpi=100, bbox_inches="tight")
    return fig, output


def main():
    plot_functions = [
        plot_S4_validation,
        plot_S5_validation,
        plot_Figure5a_water_profile,
        plot_Figure5b_reaction_rates,
    ]

    print("Reproducing validation figures...\n")
    for make_plot in plot_functions:
        fig, path = make_plot()
        print(f"Saved: {path.relative_to(ROOT)}")

        # Figures are displayed sequentially after being saved.
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
