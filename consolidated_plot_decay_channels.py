"""Compare isolated dephasing and isolated cavity damping.

The plots follow the Mathematica style used by the article, use the physical
nonuniform time coordinates in density maps, include the constant-coupling
expectation values, and load magic exclusively from W_half post-processing.
"""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        if (candidate / "only_dephasing").is_dir() and (candidate / "only_cavity_damping").is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate the consolidated loss-channel folders.")


ROOT = locate_dataset_root()
FIGURE_DIR = ROOT / "figures" / "separate_decay_channels"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (15.5, 13.0),
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "axes.linewidth": 1.25,
    "axes.grid": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "savefig.dpi": 200,
})

MATHEMATICA_DENSITY = LinearSegmentedColormap.from_list(
    "mathematica_density",
    ["#6f1d32", "#a94f58", "#d39a7c", "#e5d9b8", "#9bc0c9", "#4e91b5", "#0d4f85"],
)

CHANNEL_STYLE = {
    "only_dephasing": dict(color="#2f6fdd", label=r"$\gamma_{\phi}=10^{-2},\;\kappa=0$"),
    "only_cavity_damping": dict(color="#e34a33", label=r"$\gamma_{\phi}=0,\;\kappa=10^{-1}$"),
}

CASES = [
    dict(resource="wigner", coupling="gaussian_zeta", title=r"Wigner negativity $\mathcal{N}(\rho)$ - Gaussian modulation, varying $\zeta$", parameter=r"\zeta", selected=2.4),
    dict(resource="wigner", coupling="gaussian_T", title=r"Wigner negativity $\mathcal{N}(\rho)$ - Gaussian modulation, varying $T$", parameter="T", selected=7.5),
    dict(resource="wigner", coupling="cosine_omega", title=r"Wigner negativity $\mathcal{N}(\rho)$ - cosine modulation, varying $\omega$", parameter=r"\omega", selected=np.pi / 3),
    dict(resource="coherence", coupling="gaussian_zeta", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $\zeta$", parameter=r"\zeta", selected=8.0),
    dict(resource="coherence", coupling="gaussian_T", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $T$", parameter="T", selected=25.0),
    dict(resource="coherence", coupling="cosine_omega", title=r"Relative entropy of coherence $\mathcal{C}_{\mathrm{rel}}(\rho_{\mathrm{q}})$ - cosine modulation, varying $\omega$", parameter=r"\omega", selected=np.pi / 10),
    dict(resource="magic_W_half", coupling="gaussian_zeta", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $\zeta$", parameter=r"\zeta", selected=8.0),
    dict(resource="magic_W_half", coupling="gaussian_T", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - Gaussian modulation, varying $T$", parameter="T", selected=25.0),
    dict(resource="magic_W_half", coupling="cosine_omega", title=r"Atomic magic witness $W_{1/2}(\rho_{\mathrm{q}})$ - cosine modulation, varying $\omega$", parameter=r"\omega", selected=np.pi / 10),
    dict(resource="entanglement", coupling="gaussian_zeta", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - Gaussian modulation, varying $\zeta$", parameter=r"\zeta", selected=8.0),
    dict(resource="entanglement", coupling="gaussian_T", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - Gaussian modulation, varying $T$", parameter="T", selected=25.0),
    dict(resource="entanglement", coupling="cosine_omega", title=r"Atom-field logarithmic negativity $E_{\mathcal{N}}(\rho)$ - cosine modulation, varying $\omega$", parameter=r"\omega", selected=np.pi / 10),
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


def cell_edges(centers):
    centers = numeric_array(centers)
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


def assemble_magic(folder, scan_length):
    files = sorted(folder.glob("var_aberto_[0-9][0-9][0-9][0-9]_*_observables_mixed_magic.npz"))
    if len(files) != scan_length:
        raise ValueError(f"Expected {scan_length} W_half files in {folder}, found {len(files)}")
    rows = []
    for path in files:
        with np.load(path) as data:
            rows.append(numeric_array(data["W_half"]))
    with np.load(folder / "const_aberto_observables_mixed_magic.npz") as data:
        constant = numeric_array(data["W_half"])
    return np.vstack(rows), constant


def load_channel(case, regime):
    root = ROOT / regime / case["resource"] / case["coupling"]
    raw = root / "full_scan"
    t = load_npy(raw / "t.npy")
    scan = load_npy(raw / "scan_values.npy")
    derived_folder = None

    if case["resource"] == "coherence":
        derived_folder = root / "corrected_coherence"
        variable = load_npy(derived_folder / "var_aberto.npy")
        constant = load_npy(derived_folder / "const_aberto.npy")
    elif case["resource"] == "magic_W_half":
        derived_folder = root / "W_half"
        variable, constant = assemble_magic(derived_folder, len(scan))
    else:
        variable = load_npy(raw / "var_aberto.npy")
        constant = load_npy(raw / "const_aberto.npy")

    if variable.shape != (len(scan), len(t)):
        raise ValueError(f"Unexpected shape for {regime}/{case['resource']}/{case['coupling']}: {variable.shape}")
    if case["resource"] == "wigner":
        scale = max(float(np.max(variable)), float(np.max(constant)), 1e-15)
        variable = variable / scale
        constant = constant / scale

    selected_index = int(np.argmin(np.abs(scan - case["selected"])))
    files = sorted(raw.glob(f"var_aberto_{selected_index:04d}_*_observables.csv"))
    if len(files) != 1:
        raise ValueError(f"Could not uniquely resolve observables for index {selected_index} in {raw}")
    constant_file = raw / "const_aberto_observables.csv"
    if not constant_file.is_file():
        raise FileNotFoundError(constant_file)
    return dict(
        root=root,
        raw=raw,
        derived_folder=derived_folder,
        t=t,
        scan=scan,
        variable=variable,
        constant=constant,
        selected_index=selected_index,
        selected_value=float(scan[selected_index]),
        observables=pd.read_csv(files[0]),
        constant_observables=pd.read_csv(constant_file),
    )


def observable_column(frame, short_name, long_name=None):
    if short_name in frame.columns:
        return frame[short_name].to_numpy(dtype=float)
    if long_name and long_name in frame.columns:
        return frame[long_name].to_numpy(dtype=float)
    raise KeyError(f"Neither {short_name!r} nor {long_name!r} is present.")


def print_source_folders(data):
    print("\nPastas-fonte da figura:")
    for regime in CHANNEL_STYLE:
        channel = data[regime]
        print(f"  - {channel['raw'].relative_to(ROOT)}")
        if channel["derived_folder"] is not None:
            print(f"  - {channel['derived_folder'].relative_to(ROOT)}")


def heatmap(ax, data, t, scan, case, title):
    norm = None
    if case["resource"] == "magic_W_half" and np.min(data) < 0 < np.max(data):
        norm = TwoSlopeNorm(vcenter=0.0, vmin=float(np.min(data)), vmax=float(np.max(data)))
    mesh = ax.pcolormesh(cell_edges(t), cell_edges(scan), data, shading="flat", cmap=MATHEMATICA_DENSITY, norm=norm, rasterized=True)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(scan[0], scan[-1])
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(rf"${case['parameter']}$")
    if case["parameter"] == r"\omega":
        ax.set_yticks(omega_ticks(float(scan[-1])))
        ax.yaxis.set_major_formatter(FuncFormatter(pi_tick))
    style_axes(ax)
    return mesh


def plot_channel_expectation(ax, data, key, ylabel, title, alternate_key=None, show_legend=False):
    for regime, style in CHANNEL_STYLE.items():
        channel = data[regime]
        variable_frame = channel["observables"]
        constant_frame = channel["constant_observables"]
        variable = observable_column(variable_frame, key, alternate_key)
        constant = observable_column(constant_frame, key, alternate_key)
        ax.plot(variable_frame["time"], variable, color=style["color"], lw=1.5, label=style["label"] + r", $g(t)$")
        ax.plot(constant_frame["time"], constant, color=style["color"], ls=(0, (1.2, 1.2)), lw=1.5, label=style["label"] + r", $g_0$")
    ax.set_title(title)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(fontsize=7, loc="best")
    style_axes(ax)


def plot_case(case):
    data = {regime: load_channel(case, regime) for regime in CHANNEL_STYLE}
    print_source_folders(data)
    fig, axes = plt.subplots(3, 3, constrained_layout=True)
    fig.suptitle(case["title"], fontsize=16)

    for column, regime in enumerate(CHANNEL_STYLE):
        channel = data[regime]
        image = heatmap(axes[0, column], channel["variable"], channel["t"], channel["scan"], case, CHANNEL_STYLE[regime]["label"])
        colorbar = fig.colorbar(image, ax=axes[0, column])
        colorbar.set_label(METRIC_LABEL[case["resource"]])
        colorbar.ax.tick_params(direction="in")

    ax = axes[0, 2]
    for regime, style in CHANNEL_STYLE.items():
        channel = data[regime]
        parameter_value = pi_tick(channel["selected_value"]).strip("$") if case["parameter"] == r"\omega" else f"{channel['selected_value']:g}"
        ax.plot(channel["t"], channel["constant"], color=style["color"], ls=(0, (1.2, 1.2)), lw=1.5, label=style["label"] + r", $\mathrm{const.}$")
        ax.plot(channel["t"], channel["variable"][channel["selected_index"]], color=style["color"], lw=1.6, label=style["label"] + rf", ${case['parameter']}={parameter_value}$")
    if case["resource"] == "magic_W_half":
        ax.axhline(0.0, color="black", lw=0.9)
    ax.set_title(r"Selected parameter value")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(METRIC_LABEL[case["resource"]])
    ax.legend(fontsize=7, loc="best")
    style_axes(ax)

    plot_channel_expectation(axes[1, 0], data, "g_t", r"$g(t)$", r"Coupling function", show_legend=True)
    plot_channel_expectation(axes[1, 1], data, "expect_X", r"$\langle\sigma_x\rangle$", r"Atomic expectation value $\langle\sigma_x\rangle$", "expect_X_qubit")
    plot_channel_expectation(axes[1, 2], data, "expect_Y", r"$\langle\sigma_y\rangle$", r"Atomic expectation value $\langle\sigma_y\rangle$", "expect_Y_qubit")
    plot_channel_expectation(axes[2, 0], data, "expect_Z", r"$\langle\sigma_z\rangle$", r"Atomic expectation value $\langle\sigma_z\rangle$", "expect_Z_qubit", show_legend=True)
    plot_channel_expectation(axes[2, 1], data, "expect_N", r"$\langle\hat{n}\rangle$", r"Mean photon number")
    plot_channel_expectation(axes[2, 2], data, "var_N", r"$\operatorname{Var}(\hat{n})$", r"Photon-number variance")

    output = FIGURE_DIR / f"channels_{case['resource']}_{case['coupling']}.png"
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close(fig)
    print(f"Figura salva em: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    for channel_case in CASES:
        plot_case(channel_case)
    print(f"\nForam geradas {len(CASES)} figuras em {FIGURE_DIR.relative_to(ROOT)}")
