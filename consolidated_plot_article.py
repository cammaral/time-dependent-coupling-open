"""Plot all article datasets from the consolidated package.

The visual language follows the Mathematica figures already used in the
article: serif/STIX typography, framed axes, inward ticks, no grid, the same
line hierarchy, and a maroon--cream--blue density palette.  Density plots use
the physical, nonuniform time coordinates instead of treating array columns
as equally spaced.
"""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
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
        if (candidate / "article_both_losses").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate article_both_losses/.")


ROOT = locate_dataset_root()
FIGURE_DIR = ROOT / "figures" / "article"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (15.5, 9.0),
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 15,
    "axes.linewidth": 1.25,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "savefig.dpi": 200,
})

MATHEMATICA_DENSITY = LinearSegmentedColormap.from_list(
    "mathematica_density",
    ["#6f1d32", "#a94f58", "#d39a7c", "#e5d9b8", "#9bc0c9", "#4e91b5", "#0d4f85"],
)

CURVE_STYLE = {
    "const": dict(color="#808080", linestyle=(0, (1.2, 1.2)), linewidth=1.8),
    "min": dict(color="#e34a33", linestyle=(0, (1.0, 1.0)), linewidth=1.5),
    "av": dict(color="#2f6fdd", linestyle=(0, (3.0, 2.0, 1.0, 2.0)), linewidth=1.6),
    "max": dict(color="#14913c", linestyle="-", linewidth=1.6),
}

OBSERVABLE_COLORS = {
    "expect_X": "#2f6fdd",
    "expect_Y": "#e34a33",
    "expect_Z": "#14913c",
}


CASES = [
    dict(resource="wigner", coupling="gaussian_zeta", title=r"Wigner negativity $\mathcal{N}(\rho)$ - Gaussian modulation, varying $\zeta$", axis_file="ep_list.npy", parameter=r"\zeta", selected=[1.8, 2.4, 3.0]),
    dict(resource="wigner", coupling="gaussian_T", title=r"Wigner negativity $\mathcal{N}(\rho)$ - Gaussian modulation, varying $T$", axis_file="T_list.npy", parameter="T", selected=[4.5, 7.5, 10.5]),
    dict(resource="wigner", coupling="cosine_omega", title=r"Wigner negativity $\mathcal{N}(\rho)$ - cosine modulation, varying $\omega$", axis_file="w_list.npy", parameter=r"\omega", selected=[np.pi / 6, np.pi / 3, 2 * np.pi / 3]),
    dict(resource="coherence", coupling="gaussian_zeta", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $\zeta$", axis_file="scan_values.npy", parameter=r"\zeta", selected=[6.0, 8.0, 10.0]),
    dict(resource="coherence", coupling="gaussian_T", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $T$", axis_file="scan_values.npy", parameter="T", selected=[15.0, 25.0, 35.0]),
    dict(resource="coherence", coupling="cosine_omega", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - cosine modulation, varying $\omega$", axis_file="scan_values.npy", parameter=r"\omega", selected=[np.pi / 20, np.pi / 10, np.pi / 5]),
    dict(resource="magic_W_half", coupling="gaussian_zeta", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $\zeta$", parameter=r"\zeta", selected=[6.0, 8.0, 10.0]),
    dict(resource="magic_W_half", coupling="gaussian_T", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $T$", parameter="T", selected=[15.0, 25.0, 35.0]),
    dict(resource="magic_W_half", coupling="cosine_omega", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - cosine modulation, varying $\omega$", parameter=r"\omega", selected=[np.pi / 20, np.pi / 10, np.pi / 5]),
    dict(resource="entanglement", coupling="gaussian_zeta", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - Gaussian modulation, varying $\zeta$", axis_file="ep_list.npy", parameter=r"\zeta", selected=[6.0, 8.0, 10.0]),
    dict(resource="entanglement", coupling="gaussian_T", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - Gaussian modulation, varying $T$", axis_file="T_list.npy", parameter="T", selected=[15.0, 25.0, 35.0]),
    dict(resource="entanglement", coupling="cosine_omega", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - cosine modulation, varying $\omega$", axis_file="w_list.npy", parameter=r"\omega", selected=[np.pi / 20, np.pi / 10, np.pi / 5]),
]

METRIC_LABEL = {
    "wigner": r"$\mathcal{N}(\rho)$",
    "coherence": r"$\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$",
    "magic_W_half": r"$W_{1/2}(\rho_{\mathrm{q}})$",
    "entanglement": r"$E_{\mathcal{N}}(\rho)$",
}


def numeric_array(value):
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(array.tolist())
    return np.asarray(np.real_if_close(array), dtype=float)


def load_npy(path):
    return numeric_array(np.load(path, allow_pickle=True))


def binary_entropy(probability):
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    result = np.zeros_like(probability)
    mask = (probability > 0.0) & (probability < 1.0)
    result[mask] = -probability[mask] * np.log2(probability[mask]) - (1.0 - probability[mask]) * np.log2(1.0 - probability[mask])
    return result


def coherence_from_bloch(x, y, z):
    radius = np.clip(np.sqrt(x * x + y * y + z * z), 0.0, 1.0)
    return binary_entropy((1.0 + z) / 2.0) - binary_entropy((1.0 + radius) / 2.0)


def expected_npz(case_root, label="av"):
    if label == "const":
        path = case_root / "selected_expected_values" / "const_aberto_observables.npz"
    else:
        path = case_root / "selected_expected_values" / f"var_aberto_{label}_observables.npz"
    return np.load(path)


def cell_edges(centers):
    """Convert monotonic sample centers to cell edges for nonuniform pcolormesh."""
    centers = numeric_array(centers)
    if centers.ndim != 1 or len(centers) < 2:
        raise ValueError("At least two one-dimensional centers are required.")
    middle = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate(([centers[0] - (middle[0] - centers[0])], middle, [centers[-1] + (centers[-1] - middle[-1])]))


def pi_tick(value, _position=None):
    if abs(value) < 1e-12:
        return r"$0$"
    fraction = Fraction(float(value / np.pi)).limit_denominator(20)
    numerator, denominator = fraction.numerator, fraction.denominator
    sign = "-" if numerator < 0 else ""
    numerator = abs(numerator)
    if denominator == 1:
        coefficient = "" if numerator == 1 else str(numerator)
        return rf"${sign}{coefficient}\pi$"
    coefficient = "" if numerator == 1 else str(numerator)
    return rf"${sign}\frac{{{coefficient}\pi}}{{{denominator}}}$"


def style_axes(ax):
    ax.tick_params(which="both", direction="in", top=True, right=True, width=1.15)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)


def omega_ticks(maximum):
    if maximum <= np.pi / 5 + 1e-10:
        return [0.0, np.pi / 20, np.pi / 10, 3 * np.pi / 20, np.pi / 5]
    return [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3]


def apply_parameter_axis(ax, case, scan):
    ax.set_ylabel(rf"${case['parameter']}$")
    if case["parameter"] == r"\omega":
        ax.set_yticks(omega_ticks(float(scan[-1])))
        ax.yaxis.set_major_formatter(FuncFormatter(pi_tick))


def format_parameter(value, case):
    if case["parameter"] == r"\omega":
        return pi_tick(value).strip("$")
    return f"{value:g}"


def load_full_case(case):
    case_root = ROOT / "article_both_losses" / case["resource"] / case["coupling"]
    resource = case["resource"]

    if resource == "coherence":
        folder = case_root / "full_scan_corrected"
        t = load_npy(folder / "t.npy")
        scan = load_npy(folder / case["axis_file"])
        variable = load_npy(folder / "var_aberto_correct.npy")
        constant = load_npy(folder / "const_aberto_correct.npy")
    elif resource == "magic_W_half":
        folder = case_root / "full_scan_W_half"
        with np.load(folder / "var_aberto_details_mixed_magic.npz") as full:
            t = numeric_array(full["time"])
            scan = numeric_array(full["scan_values"])
            variable = numeric_array(full["W_half"])
        with np.load(folder / "const_aberto_details_mixed_magic.npz") as const:
            constant = numeric_array(const["W_half"])
    else:
        folder = case_root / "full_scan"
        t = load_npy(folder / "t.npy")
        scan = load_npy(folder / case["axis_file"])
        variable = load_npy(folder / "var_aberto.npy")
        constant = load_npy(folder / "const_aberto.npy")

    if variable.ndim != 2 or variable.shape != (len(scan), len(t)):
        raise ValueError(f"Unexpected shape for {resource}/{case['coupling']}: {variable.shape}")

    if resource == "wigner":
        scale = max(float(np.max(variable)), float(np.max(constant)), 1e-15)
        variable = variable / scale
        constant = constant / scale
    return case_root, folder, t, scan, variable, constant


def selected_metric_curves(case, case_root, scan, variable):
    curves = []
    labels = ["min", "av", "max"]
    if case["resource"] == "magic_W_half":
        for label, target in zip(labels, case["selected"]):
            with np.load(case_root / "selected_W_half" / f"var_aberto_{label}_observables_mixed_magic.npz") as data:
                curves.append((target, numeric_array(data["W_half"]), label))
        return curves
    if case["resource"] == "coherence":
        for label, target in zip(labels, case["selected"]):
            with expected_npz(case_root, label) as data:
                curve = coherence_from_bloch(data["expect_X"], data["expect_Y"], data["expect_Z"])
            curves.append((target, curve, label))
        return curves
    for label, target in zip(labels, case["selected"]):
        index = int(np.argmin(np.abs(scan - target)))
        curves.append((float(scan[index]), variable[index], label))
    return curves


def print_source_folders(case_root, full_folder, resource):
    folders = [full_folder, case_root / "selected_expected_values"]
    if resource == "magic_W_half":
        folders.append(case_root / "selected_W_half")
    print("\nPastas-fonte da figura:")
    for folder in folders:
        print(f"  - {folder.relative_to(ROOT)}")


def plot_expected_pair(ax, variable_data, constant_data, key, color, ylabel, title):
    ax.plot(variable_data["t"], variable_data[key], color=color, lw=1.6, label=r"$g(t)$")
    ax.plot(constant_data["t"], constant_data[key], color="#808080", ls=(0, (1.2, 1.2)), lw=1.7, label=r"$g(t)=g_0$")
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="best")
    style_axes(ax)


def plot_case(case):
    case_root, full_folder, t, scan, variable, constant = load_full_case(case)
    curves = selected_metric_curves(case, case_root, scan, variable)
    variable_expected = expected_npz(case_root, "av")
    constant_expected = expected_npz(case_root, "const")
    print_source_folders(case_root, full_folder, case["resource"])

    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.suptitle(case["title"], fontsize=16)

    ax = axes[0, 0]
    norm = None
    if case["resource"] == "magic_W_half" and np.min(variable) < 0 < np.max(variable):
        norm = TwoSlopeNorm(vcenter=0.0, vmin=float(np.min(variable)), vmax=float(np.max(variable)))
    mesh = ax.pcolormesh(cell_edges(t), cell_edges(scan), variable, shading="flat", cmap=MATHEMATICA_DENSITY, norm=norm, rasterized=True)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(scan[0], scan[-1])
    ax.set_title(r"Complete parameter scan")
    ax.set_xlabel(r"$t$")
    apply_parameter_axis(ax, case, scan)
    style_axes(ax)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(METRIC_LABEL[case["resource"]])
    colorbar.ax.tick_params(direction="in")

    ax = axes[0, 1]
    ax.plot(t, constant, label=r"$\mathrm{const.}$", **CURVE_STYLE["const"])
    for value, curve, label in curves:
        parameter_value = format_parameter(value, case)
        ax.plot(t, curve, label=rf"${case['parameter']}_{{\mathrm{{{label}}}}}={parameter_value}$", **CURVE_STYLE[label])
    if case["resource"] == "magic_W_half":
        ax.axhline(0.0, color="black", lw=0.9)
    ax.set_title(r"Selected parameter values")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(METRIC_LABEL[case["resource"]])
    ax.legend(fontsize=9, loc="best")
    style_axes(ax)

    plot_expected_pair(axes[0, 2], variable_expected, constant_expected, "g_t", "#7a4eb2", r"$g(t)$", r"Coupling function")

    ax = axes[1, 0]
    observable_labels = {
        "expect_X": r"$\langle\sigma_x\rangle$",
        "expect_Y": r"$\langle\sigma_y\rangle$",
        "expect_Z": r"$\langle\sigma_z\rangle$",
    }
    for key, label in observable_labels.items():
        color = OBSERVABLE_COLORS[key]
        ax.plot(variable_expected["t"], variable_expected[key], color=color, lw=1.5, label=label)
        ax.plot(constant_expected["t"], constant_expected[key], color=color, ls=(0, (1.2, 1.2)), lw=1.4, label=label + r" $[g_0]$")
    ax.set_title(r"Atomic expectation values")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\langle\sigma_j\rangle$")
    ax.legend(fontsize=8, ncol=2, loc="best")
    style_axes(ax)

    plot_expected_pair(axes[1, 1], variable_expected, constant_expected, "expect_N", "#14913c", r"$\langle\hat{n}\rangle$", r"Mean photon number")
    plot_expected_pair(axes[1, 2], variable_expected, constant_expected, "var_N", "#e34a33", r"$\operatorname{Var}(\hat{n})$", r"Photon-number variance")

    output = FIGURE_DIR / f"article_{case['resource']}_{case['coupling']}.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    variable_expected.close()
    constant_expected.close()
    print(f"Figura salva em: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    for article_case in CASES:
        plot_case(article_case)
    print(f"\nForam geradas {len(CASES)} figuras em {FIGURE_DIR.relative_to(ROOT)}")
