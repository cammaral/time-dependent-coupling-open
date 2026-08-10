"""Generate the literal outputs requested in the simulation-list PDF.

Outputs:
  * six standalone coherence density PNGs (three modulations x two channels);
  * two-column DAT files for <sigma_z> and <n> at the four requested points;
  * eight standalone expected-value PNGs read back exclusively from those DATs.

Raw/canonical datasets are read-only.  All derived products are written below
``requested_outputs`` in the consolidated dataset.
"""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


def locate_dataset_root():
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent,
        Path.cwd() / "paper_dataset_consolidated1",
        Path.cwd() / "results" / "paper_dataset_consolidated1",
    ]
    for candidate in candidates:
        if (candidate / "specific_both_losses").is_dir() and (candidate / "only_dephasing").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate paper_dataset_consolidated1.")


ROOT = locate_dataset_root()
OUTPUT_ROOT = ROOT / "requested_outputs"
DAT_ROOT = OUTPUT_ROOT / "specific_parameters" / "dat"
DENSITY_FIGURE_ROOT = OUTPUT_ROOT / "figures" / "coherence_density"
EXPECTED_FIGURE_ROOT = OUTPUT_ROOT / "figures" / "specific_expected_values"
for directory in [DAT_ROOT, DENSITY_FIGURE_ROOT, EXPECTED_FIGURE_ROOT]:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (7.2, 5.6),
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 16,
    "axes.linewidth": 1.3,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "savefig.dpi": 220,
})

MATHEMATICA_DENSITY = LinearSegmentedColormap.from_list(
    "mathematica_density",
    ["#6f1d32", "#a94f58", "#d39a7c", "#e5d9b8", "#9bc0c9", "#4e91b5", "#0d4f85"],
)

REGIMES = {
    "only_dephasing": {
        "title": r"Only atomic dephasing: $\gamma_{\phi}=10^{-2},\;\kappa=0$",
        "filename": "only_dephasing",
    },
    "only_cavity_damping": {
        "title": r"Only cavity damping: $\gamma_{\phi}=0,\;\kappa=10^{-1}$",
        "filename": "only_cavity_damping",
    },
}

COHERENCE_CASES = {
    "gaussian_zeta": {
        "parameter": r"\zeta",
        "subtitle": r"Gaussian modulation, varying $\zeta$ ($T=25$, $\sigma=-1$)",
    },
    "gaussian_T": {
        "parameter": "T",
        "subtitle": r"Gaussian modulation, varying $T$ ($\zeta=8$, $\sigma=-1$)",
    },
    "cosine_omega": {
        "parameter": r"\omega",
        "subtitle": r"Cosine modulation, varying $\omega$ ($\phi=0$)",
    },
}

SPECIFIC_CASES = [
    {
        "coupling": "gaussian_zeta",
        "slug": "zeta_2p4",
        "parameter_label": r"$\zeta=2.4$",
        "variable_csv": "var_aberto_0000_zeta_specific_2p4_observables.csv",
    },
    {
        "coupling": "gaussian_T",
        "slug": "T_7p5",
        "parameter_label": r"$T=7.5$",
        "variable_csv": "var_aberto_0000_T_specific_7p5_observables.csv",
    },
    {
        "coupling": "cosine_omega",
        "slug": "omega_pi_6",
        "parameter_label": r"$\omega_{\min}=\pi/6$",
        "variable_csv": "var_aberto_0000_omega_specific_0p523598775598_observables.csv",
    },
    {
        "coupling": "cosine_omega",
        "slug": "omega_pi_3",
        "parameter_label": r"$\omega_{\mathrm{av}}=\pi/3$",
        "variable_csv": "var_aberto_0001_omega_specific_1p0471975512_observables.csv",
    },
]

OBSERVABLES = {
    "Z": {
        "csv_candidates": ["expect_Z_qubit", "expect_Z", "expect_Z_global"],
        "ylabel": r"$\langle\sigma_z\rangle$",
        "dat_header": "t    <sigma_z>",
    },
    "N": {
        "csv_candidates": ["expect_N"],
        "ylabel": r"$\langle\hat{n}\rangle$",
        "dat_header": "t    <n>",
    },
}


def numeric_array(value):
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return np.asarray(np.real_if_close(array), dtype=float)


def cell_edges(centers):
    centers = numeric_array(centers)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("At least two sample centers are required.")
    middle = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - (middle[0] - centers[0])], middle, [centers[-1] + (centers[-1] - middle[-1])]))


def pi_tick(value, _position=None):
    if abs(value) < 1e-12:
        return r"$0$"
    fraction = Fraction(float(value / np.pi)).limit_denominator(20)
    numerator, denominator = fraction.numerator, fraction.denominator
    if denominator == 1:
        coefficient = "" if numerator == 1 else str(numerator)
        return rf"${coefficient}\pi$"
    coefficient = "" if numerator == 1 else str(numerator)
    return rf"$\frac{{{coefficient}\pi}}{{{denominator}}}$"


def omega_ticks(maximum):
    if maximum <= np.pi / 5 + 1e-10:
        return [0.0, np.pi / 20, np.pi / 10, 3 * np.pi / 20, np.pi / 5]
    return [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3]


def style_axes(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.15)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def pick_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return frame[candidate].to_numpy(dtype=float)
    raise KeyError(f"None of the columns {candidates!r} is available.")


def print_sources(paths):
    print("\nPastas-fonte da figura:")
    seen = set()
    for path in paths:
        folder = path if path.is_dir() else path.parent
        relative = folder.relative_to(ROOT)
        if relative not in seen:
            print(f"  - {relative}")
            seen.add(relative)


def generate_specific_dat_files():
    generated = []
    for case in SPECIFIC_CASES:
        source = ROOT / "specific_both_losses" / "wigner" / case["coupling"] / "exact_runs"
        variable = pd.read_csv(source / case["variable_csv"])
        constant = pd.read_csv(source / "const_aberto_observables.csv")
        destination = DAT_ROOT / case["coupling"]
        destination.mkdir(parents=True, exist_ok=True)

        for observable, metadata in OBSERVABLES.items():
            variable_path = destination / f"{case['slug']}_expected_{observable}.dat"
            variable_values = pick_column(variable, metadata["csv_candidates"])
            np.savetxt(
                variable_path,
                np.column_stack([variable["time"].to_numpy(dtype=float), variable_values]),
                fmt="%.16e",
                header=metadata["dat_header"],
            )
            generated.append(variable_path)

            constant_path = destination / f"constant_expected_{observable}.dat"
            if not constant_path.exists():
                constant_values = pick_column(constant, metadata["csv_candidates"])
                np.savetxt(
                    constant_path,
                    np.column_stack([constant["time"].to_numpy(dtype=float), constant_values]),
                    fmt="%.16e",
                    header=metadata["dat_header"],
                )
                generated.append(constant_path)
    print(f"Arquivos .dat disponíveis em: {DAT_ROOT.relative_to(ROOT)}")
    return generated


def plot_standalone_coherence_densities():
    outputs = []
    for regime, regime_info in REGIMES.items():
        for coupling, case in COHERENCE_CASES.items():
            case_root = ROOT / regime / "coherence" / coupling
            raw = case_root / "full_scan"
            corrected = case_root / "corrected_coherence"
            t = numeric_array(np.load(raw / "t.npy", allow_pickle=True))
            scan = numeric_array(np.load(raw / "scan_values.npy", allow_pickle=True))
            coherence = numeric_array(np.load(corrected / "var_aberto.npy", allow_pickle=True))
            if coherence.shape != (len(scan), len(t)):
                raise ValueError(f"Unexpected density shape in {corrected}: {coherence.shape}")

            print_sources([raw, corrected])
            fig, ax = plt.subplots(constrained_layout=True)
            mesh = ax.pcolormesh(
                cell_edges(t),
                cell_edges(scan),
                coherence,
                shading="flat",
                cmap=MATHEMATICA_DENSITY,
                vmin=0.0,
                vmax=1.0,
                rasterized=True,
            )
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(scan[0], scan[-1])
            ax.set_title(regime_info["title"] + "\n" + case["subtitle"])
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(rf"${case['parameter']}$")
            if case["parameter"] == r"\omega":
                ax.set_yticks(omega_ticks(float(scan[-1])))
                ax.yaxis.set_major_formatter(FuncFormatter(pi_tick))
            style_axes(ax)
            colorbar = fig.colorbar(mesh, ax=ax)
            colorbar.set_label(r"$\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$")
            colorbar.ax.tick_params(direction="in")

            output = DENSITY_FIGURE_ROOT / f"{regime_info['filename']}_coherence_{coupling}.png"
            fig.savefig(output, bbox_inches="tight", facecolor="white")
            plt.show()
            plt.close(fig)
            outputs.append(output)
            print(f"Figura salva em: {output.relative_to(ROOT)}")
    return outputs


def plot_specific_values_from_dat():
    outputs = []
    for case in SPECIFIC_CASES:
        dat_folder = DAT_ROOT / case["coupling"]
        for observable, metadata in OBSERVABLES.items():
            variable_path = dat_folder / f"{case['slug']}_expected_{observable}.dat"
            constant_path = dat_folder / f"constant_expected_{observable}.dat"

            # The plots below intentionally read only the requested two-column DATs.
            variable = np.loadtxt(variable_path, comments="#")
            constant = np.loadtxt(constant_path, comments="#")
            if variable.ndim != 2 or variable.shape[1] != 2 or constant.ndim != 2 or constant.shape[1] != 2:
                raise ValueError("Every DAT used for plotting must have exactly two columns.")

            print_sources([variable_path, constant_path])
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(constant[:, 0], constant[:, 1], color="#808080", ls=(0, (1.2, 1.2)), lw=1.9, label=r"$g(t)=g_0$")
            ax.plot(variable[:, 0], variable[:, 1], color="#2f6fdd", lw=1.7, label=case["parameter_label"])
            ax.set_xlim(0.0, 15.0)
            ax.set_title(
                metadata["ylabel"]
                + " - "
                + case["parameter_label"]
                + r"; $\kappa=10^{-1}$, $\gamma_{\phi}=10^{-2}$"
            )
            ax.set_xlabel(r"$t$")
            ax.set_ylabel(metadata["ylabel"])
            ax.legend(loc="best", fontsize=11)
            style_axes(ax)

            output = EXPECTED_FIGURE_ROOT / f"{case['slug']}_expected_{observable}.png"
            fig.savefig(output, bbox_inches="tight", facecolor="white")
            plt.show()
            plt.close(fig)
            outputs.append(output)
            print(f"Figura salva em: {output.relative_to(ROOT)}")
    return outputs


if __name__ == "__main__":
    dat_files = generate_specific_dat_files()
    density_figures = plot_standalone_coherence_densities()
    expected_figures = plot_specific_values_from_dat()

    print("\nResumo:")
    print(f"  DATs: {len(dat_files)} novos nesta execução; {len(list(DAT_ROOT.rglob('*.dat')))} disponíveis")
    print(f"  Mapas densos individuais: {len(density_figures)}")
    print(f"  Gráficos específicos lidos dos DATs: {len(expected_figures)}")
