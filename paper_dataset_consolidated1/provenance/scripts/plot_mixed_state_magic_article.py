"""Generate article-style plots for the corrected mixed-state magic analysis.

The density maps use the complete parameter scans from ``results/magic``.
The selected curves use the exact min/average/max reruns from
``results/expected_values_all_cases_xy``.  No dynamics are rerun here.

Primary article figures show W_{1/2}.  For the reduced atomic qubit,
W_{1/2} > 0 is exactly equivalent to being outside the stabilizer octahedron.
A separate comparison PDF shows the previous M_2 together with W_{1/2}, W_2,
and W_3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE = PROJECT_ROOT / "tmp" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


CASE_CONFIGS = (
    {
        "case_id": "magic_gauss_epsilon",
        "article_name": "gauss1",
        "full_scan_folder": "magic_gauss1",
        "scan_symbol": r"\zeta",
        "scan_label": r"Gaussian width $\zeta$",
        "scan_limits": (6.0, 10.0),
        "curve_labels": {
            "min": r"$\zeta_{\min}=6$",
            "av": r"$\zeta_{\mathrm{av}}=8$",
            "max": r"$\zeta_{\max}=10$",
        },
        "title": r"Gaussian coupling: variable width ($T=25$)",
    },
    {
        "case_id": "magic_gauss_T",
        "article_name": "gauss2",
        "full_scan_folder": "magic_gauss2",
        "scan_symbol": "T",
        "scan_label": r"Gaussian peak time $T$",
        "scan_limits": (15.0, 35.0),
        "curve_labels": {
            "min": r"$T_{\min}=15$",
            "av": r"$T_{\mathrm{av}}=25$",
            "max": r"$T_{\max}=35$",
        },
        "title": r"Gaussian coupling: variable peak time ($\zeta=8$)",
    },
    {
        "case_id": "magic_cos_omega",
        "article_name": "cos1",
        "full_scan_folder": "magic_cos1",
        "scan_symbol": r"\omega",
        "scan_label": r"Modulation frequency $\omega$",
        "scan_limits": (np.pi / 20.0, np.pi / 5.0),
        "curve_labels": {
            "min": r"$\omega_{\min}=\pi/20$",
            "av": r"$\omega_{\mathrm{av}}=\pi/10$",
            "max": r"$\omega_{\max}=\pi/5$",
        },
        "title": r"Cosine coupling ($\phi=0$)",
    },
)

CURVE_ORDER = ("const", "min", "av", "max")
CURVE_STYLES = {
    "const": {"color": "0.45", "linestyle": "--", "linewidth": 1.8},
    "min": {"color": "#D55E00", "linestyle": ":", "linewidth": 1.8},
    "av": {"color": "#0072B2", "linestyle": "-.", "linewidth": 1.8},
    "max": {"color": "#009E73", "linestyle": "-", "linewidth": 1.8},
}
METRIC_LABELS = {
    "M_2": r"$M_2(\rho)$ (not a mixed-state certificate)",
    "W_half": r"$\mathcal{W}_{1/2}(\rho)$",
    "W_2": r"$\mathcal{W}_{2}(\rho)$",
    "W_3": r"$\mathcal{W}_{3}(\rho)$",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera as figuras corrigidas de magia para os tres casos do artigo."
    )
    parser.add_argument(
        "--mixed-run",
        type=Path,
        default=None,
        help="Pasta mixed_magicN. Se omitida, usa a mais recente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Pasta de saida. Padrao: output/pdf/mixed_state_magic_article/<run>.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def numbered_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_numbered_folder(parent: Path, prefix: str) -> Path:
    folders = [path for path in parent.glob(f"{prefix}*") if path.is_dir()]
    if not folders:
        raise FileNotFoundError(f"Nenhuma pasta {prefix}* encontrada em {parent}")
    return max(folders, key=numbered_suffix)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    mixed_parent = PROJECT_ROOT / "results" / "mixed_state_magic_witnesses"
    mixed_run = (
        args.mixed_run.resolve()
        if args.mixed_run is not None
        else latest_numbered_folder(mixed_parent, "mixed_magic")
    )
    if not mixed_run.is_dir():
        raise NotADirectoryError(f"Pasta de witnesses inexistente: {mixed_run}")

    expected_parent = PROJECT_ROOT / "results" / "expected_values_all_cases_xy"
    expected_run = latest_numbered_folder(expected_parent, "expected_values_all_cases_xy")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else PROJECT_ROOT
        / "output"
        / "pdf"
        / "mixed_state_magic_article"
        / mixed_run.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return mixed_run, expected_run, output_dir


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_case_data(
    config: dict[str, object], mixed_run: Path, expected_run: Path
) -> dict[str, object]:
    derived = mixed_run / "derived"
    full_path = (
        derived
        / "magic"
        / str(config["full_scan_folder"])
        / "var_aberto_details_mixed_magic.npz"
    )
    full = load_npz(full_path)

    expected_relative = expected_run.relative_to(PROJECT_ROOT / "results")
    selected_derived_dir = derived / expected_relative / str(config["case_id"])
    selected_source_dir = expected_run / str(config["case_id"])

    curves: dict[str, dict[str, np.ndarray]] = {}
    for key in CURVE_ORDER:
        source_label = "const_aberto" if key == "const" else f"var_aberto_{key}"
        metric_path = selected_derived_dir / f"{source_label}_observables_mixed_magic.npz"
        source_path = selected_source_dir / f"{source_label}_observables.npz"
        metric_data = load_npz(metric_path)
        source_data = load_npz(source_path)
        metric_data["g_t"] = np.asarray(source_data["g_t"], dtype=float)
        curves[key] = metric_data

    return {
        "config": config,
        "full": full,
        "curves": curves,
        "sources": {
            "full": str(full_path),
            "selected_derived": str(selected_derived_dir),
            "selected_source": str(selected_source_dir),
        },
    }


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 10.5,
            "axes.linewidth": 1.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
        }
    )


def witness_norm(values: np.ndarray) -> TwoSlopeNorm:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin >= 0.0:
        vmin = -max(vmax * 0.05, 1e-6)
    if vmax <= 0.0:
        vmax = max(abs(vmin) * 0.05, 1e-6)
    return TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)


def curve_label(config: dict[str, object], key: str) -> str:
    if key == "const":
        return "constant"
    return str(config["curve_labels"][key])


def add_coupling_inset(ax: plt.Axes, case_data: dict[str, object]) -> None:
    curves = case_data["curves"]
    inset = ax.inset_axes([0.64, 0.10, 0.31, 0.29])
    for key in CURVE_ORDER:
        curve = curves[key]
        inset.plot(curve["time"], curve["g_t"], **CURVE_STYLES[key])
    inset.set_xlim(0.0, 50.0)
    inset.set_ylabel(r"$g(t)$", labelpad=0)
    inset.set_xlabel(r"$t$", labelpad=-1)
    inset.tick_params(labelsize=7, pad=1)
    inset.grid(alpha=0.15, linewidth=0.5)


def plot_primary_case(
    case_data: dict[str, object], output_dir: Path, dpi: int
) -> tuple[Path, Path]:
    config = case_data["config"]
    full = case_data["full"]
    curves = case_data["curves"]
    metric = "W_half"

    t = np.asarray(full["time"], dtype=float)
    scan = np.asarray(full["scan_values"], dtype=float)
    values = np.asarray(full[metric], dtype=float)
    scan_min, scan_max = config["scan_limits"]
    scan_mask = (scan >= float(scan_min) - 1e-12) & (
        scan <= float(scan_max) + 1e-12
    )
    if not np.any(scan_mask):
        raise ValueError(f"Scan fora dos limites para {config['case_id']}")

    visible_values = values[scan_mask]
    norm = witness_norm(visible_values)

    fig = plt.figure(figsize=(7.2, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=(1.08, 0.92))
    ax_density = fig.add_subplot(grid[0])
    ax_curves = fig.add_subplot(grid[1])

    image = ax_density.pcolormesh(
        t,
        scan,
        values,
        shading="auto",
        cmap="RdBu",
        norm=norm,
        rasterized=True,
    )
    ax_density.contour(
        t,
        scan,
        values,
        levels=[0.0],
        colors="black",
        linewidths=0.65,
        alpha=0.8,
    )
    ax_density.set_xlim(float(np.min(t)), float(np.max(t)))
    ax_density.set_ylim(float(scan_min), float(scan_max))
    ax_density.set_xlabel(r"$t$")
    ax_density.set_ylabel(str(config["scan_label"]))
    ax_density.text(
        0.975,
        0.955,
        "(a)",
        transform=ax_density.transAxes,
        ha="right",
        va="top",
        color="black",
        fontsize=12,
    )
    colorbar = fig.colorbar(image, ax=ax_density, pad=0.02)
    colorbar.set_label(METRIC_LABELS[metric])
    colorbar.ax.axhline(0.0, color="black", linewidth=0.8)

    all_curve_values = []
    for key in CURVE_ORDER:
        curve = curves[key]
        tt = np.asarray(curve["time"], dtype=float)
        yy = np.asarray(curve[metric], dtype=float)
        all_curve_values.append(yy)
        ax_curves.plot(
            tt,
            yy,
            label=curve_label(config, key),
            **CURVE_STYLES[key],
        )
    ax_curves.axhline(0.0, color="black", linewidth=0.9, alpha=0.85)
    ax_curves.set_xlim(0.0, 50.0)
    ymin = min(float(np.min(values)), *(float(np.min(v)) for v in all_curve_values))
    ymax = max(float(np.max(values)), *(float(np.max(v)) for v in all_curve_values))
    padding = 0.06 * max(ymax - ymin, 1e-6)
    ax_curves.set_ylim(ymin - padding, ymax + padding)
    ax_curves.set_xlabel(r"$t$")
    ax_curves.set_ylabel(METRIC_LABELS[metric])
    ax_curves.grid(alpha=0.18, linewidth=0.6)
    ax_curves.legend(
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=1.0,
        fontsize=8.5,
    )
    ax_curves.text(
        0.975,
        0.955,
        "(b)",
        transform=ax_curves.transAxes,
        ha="right",
        va="top",
        fontsize=12,
    )
    add_coupling_inset(ax_curves, case_data)
    fig.suptitle(str(config["title"]), fontsize=12.5)

    base = output_dir / f"article_{config['article_name']}_mixed_magic_W_half"
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)
    return pdf_path, png_path


def comparison_page(case_data: dict[str, object]) -> plt.Figure:
    config = case_data["config"]
    curves = case_data["curves"]
    metrics = ("M_2", "W_half", "W_2", "W_3")

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.7), sharex=True)
    for ax, metric in zip(axes.ravel(), metrics):
        for key in CURVE_ORDER:
            curve = curves[key]
            ax.plot(
                curve["time"],
                curve[metric],
                label=curve_label(config, key),
                **CURVE_STYLES[key],
            )
        if metric != "M_2":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.85)
        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.set_ylabel("value")
        ax.grid(alpha=0.18, linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel(r"$t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(str(config["title"]), y=0.99, fontsize=12.5)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    return fig


def positive_duration(t: np.ndarray, values: np.ndarray, tol: float = 0.0) -> float:
    """Duration of values > tol, linearly interpolating threshold crossings."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(values, dtype=float) - tol
    duration = 0.0
    for left_t, right_t, left_y, right_y in zip(t[:-1], t[1:], y[:-1], y[1:]):
        dt = float(right_t - left_t)
        left_positive = left_y > 0.0
        right_positive = right_y > 0.0
        if left_positive and right_positive:
            duration += dt
        elif left_positive != right_positive and right_y != left_y:
            crossing_fraction = float(-left_y / (right_y - left_y))
            crossing_fraction = min(1.0, max(0.0, crossing_fraction))
            duration += dt * (crossing_fraction if left_positive else 1.0 - crossing_fraction)
    return duration


def first_positive_to_nonpositive_crossing(
    t: np.ndarray, values: np.ndarray
) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(values, dtype=float)
    for index in range(len(y) - 1):
        if y[index] > 0.0 and y[index + 1] <= 0.0:
            fraction = -y[index] / (y[index + 1] - y[index])
            return float(t[index] + fraction * (t[index + 1] - t[index]))
    return np.nan


def count_revivals(values: np.ndarray) -> int:
    positive = np.asarray(values, dtype=float) > 0.0
    transitions = np.flatnonzero((~positive[:-1]) & positive[1:])
    return int(len(transitions))


def summarize_cases(case_datasets: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for case_data in case_datasets:
        config = case_data["config"]
        for key in CURVE_ORDER:
            curve = case_data["curves"][key]
            t = np.asarray(curve["time"], dtype=float)
            row = {
                "case_id": config["case_id"],
                "curve": key,
                "curve_label": curve_label(config, key),
                "time_min": float(t[0]),
                "time_max": float(t[-1]),
            }
            for metric in ("W_half", "W_2", "W_3"):
                values = np.asarray(curve[metric], dtype=float)
                duration = positive_duration(t, values)
                row[f"{metric}_initial"] = float(values[0])
                row[f"{metric}_min"] = float(np.min(values))
                row[f"{metric}_max"] = float(np.max(values))
                row[f"{metric}_final"] = float(values[-1])
                row[f"{metric}_positive_duration"] = duration
                row[f"{metric}_positive_time_fraction"] = duration / float(t[-1] - t[0])
                row[f"{metric}_first_loss_time"] = first_positive_to_nonpositive_crossing(
                    t, values
                )
                row[f"{metric}_revivals"] = count_revivals(values)
            row["M_2_final"] = float(np.asarray(curve["M_2"])[-1])
            row["purity_final"] = float(np.asarray(curve["purity"])[-1])
            row["bloch_l1_final"] = float(np.asarray(curve["bloch_l1"])[-1])
            row["is_magic_exact_final"] = bool(
                np.asarray(curve["is_magic_exact"])[-1]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    configure_matplotlib()
    mixed_run, expected_run, output_dir = resolve_paths(args)

    case_datasets = [
        load_case_data(config, mixed_run, expected_run) for config in CASE_CONFIGS
    ]

    generated = []
    for case_data in case_datasets:
        pdf_path, png_path = plot_primary_case(case_data, output_dir, args.dpi)
        generated.extend((str(pdf_path), str(png_path)))

    comparison_pdf = output_dir / "mixed_magic_M_and_W_comparison.pdf"
    with PdfPages(comparison_pdf) as pdf:
        for case_data in case_datasets:
            fig = comparison_page(case_data)
            pdf.savefig(fig)
            comparison_png = (
                output_dir
                / f"comparison_{case_data['config']['article_name']}_M_and_W.png"
            )
            fig.savefig(comparison_png, dpi=args.dpi)
            generated.append(str(comparison_png))
            plt.close(fig)
    generated.append(str(comparison_pdf))

    summary = summarize_cases(case_datasets)
    summary_path = output_dir / "selected_curves_quantitative_summary.csv"
    summary.to_csv(summary_path, index=False)
    generated.append(str(summary_path))

    manifest = {
        "mixed_run": str(mixed_run),
        "expected_values_run": str(expected_run),
        "output_dir": str(output_dir),
        "primary_metric": "W_half",
        "primary_interpretation": (
            "For the reduced one-qubit atomic state, W_half > 0 if and only if "
            "the state is outside the stabilizer octahedron."
        ),
        "cases": [case_data["sources"] for case_data in case_datasets],
        "generated": generated,
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Mixed-state run: {mixed_run}")
    print(f"Exact selected-curve run: {expected_run}")
    print(f"Figures and summaries: {output_dir}")
    print(f"Primary article figures: {len(CASE_CONFIGS)}")
    print(f"Comparison PDF: {comparison_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
