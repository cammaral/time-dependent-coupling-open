#!/usr/bin/env python3
"""Build a non-destructive, provenance-tracked copy of the paper datasets.

The source directories are never modified.  The script fails if the target
already exists, which prevents accidental merging or overwriting of an older
consolidation.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parent
TARGET = REPO / "results" / "paper_dataset_consolidated1"
PAPER_TEX = Path("/Users/cesar/Downloads/paper_draft-2.tex")


ARTICLE_CASES = {
    "wigner": {
        "gaussian_zeta": {
            "full_scan": "results/wigner_gauss4",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig4_wigner_gauss_epsilon",
            "all_time_wigner": "results/wigner_gauss_var_ep",
        },
        "gaussian_T": {
            "full_scan": "results/wigner_gauss6",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig5_wigner_gauss_T",
            "all_time_wigner": "results/wigner_gauss_gauss_var_T",
        },
        "cosine_omega": {
            "full_scan": "results/wigner_cos1",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig6_wigner_cos_omega",
            "all_time_wigner": "results/wigner_cos_paper_omegas_all_times1",
        },
    },
    "coherence": {
        "gaussian_zeta": {
            "full_scan_corrected": "results/coherence_article_corrected/article_coherence_gauss_zeta",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig1_coherence_gauss_epsilon",
        },
        "gaussian_T": {
            "full_scan_corrected": "results/coherence_article_corrected/article_coherence_gauss_T",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig2_coherence_gauss_T",
        },
        "cosine_omega": {
            "full_scan_corrected": "results/coherence_article_corrected/article_coherence_cos_omega",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig3_coherence_cos_omega",
        },
    },
    "magic_W_half": {
        "gaussian_zeta": {
            "raw_dynamics_legacy_M2": "results/magic/magic_gauss1",
            "full_scan_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/magic/magic_gauss1",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_gauss_epsilon",
            "selected_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_gauss_epsilon",
        },
        "gaussian_T": {
            "raw_dynamics_legacy_M2": "results/magic/magic_gauss2",
            "full_scan_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/magic/magic_gauss2",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_gauss_T",
            "selected_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_gauss_T",
        },
        "cosine_omega": {
            "raw_dynamics_legacy_M2": "results/magic/magic_cos1",
            "full_scan_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/magic/magic_cos1",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_cos_omega",
            "selected_W_half": "results/mixed_state_magic_witnesses/mixed_magic1/derived/expected_values_all_cases_xy/expected_values_all_cases_xy1/magic_cos_omega",
        },
    },
    "entanglement": {
        "gaussian_zeta": {
            "full_scan": "results/entanglement_gauss1",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig7_entanglement_gauss_epsilon",
        },
        "gaussian_T": {
            "full_scan": "results/entanglement_gauss3",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig8_entanglement_gauss_T",
        },
        "cosine_omega": {
            "full_scan": "results/entanglement_cos6",
            "selected_expected_values": "results/expected_values_all_cases_xy/expected_values_all_cases_xy1/fig9_entanglement_cos_omega",
        },
    },
}


CASE_SUFFIX = {
    ("wigner", "gaussian_zeta"): "wigner_gauss_zeta",
    ("wigner", "gaussian_T"): "wigner_gauss_T",
    ("wigner", "cosine_omega"): "wigner_cos_omega",
    ("coherence", "gaussian_zeta"): "coherence_gauss_zeta",
    ("coherence", "gaussian_T"): "coherence_gauss_T",
    ("coherence", "cosine_omega"): "coherence_cos_omega",
    ("magic_W_half", "gaussian_zeta"): "magic_gauss_zeta",
    ("magic_W_half", "gaussian_T"): "magic_gauss_T",
    ("magic_W_half", "cosine_omega"): "magic_cos_omega",
    ("entanglement", "gaussian_zeta"): "entanglement_gauss_zeta",
    ("entanglement", "gaussian_T"): "entanglement_gauss_T",
    ("entanglement", "cosine_omega"): "entanglement_cos_omega",
}


ARTICLE_FIGURES = {
    ("wigner", "gaussian_zeta"): [
        "wolfram_plots/wigner_plots/densityplot_gauss4_wigner.pdf",
        "wolfram_plots/wigner_plots/plottimeev_gauss4_wigner.pdf",
    ],
    ("wigner", "gaussian_T"): [
        "wolfram_plots/wigner_plots/densityplot_gauss6_wigner.pdf",
        "wolfram_plots/wigner_plots/plottimeev_gauss6_wigner.pdf",
    ],
    ("wigner", "cosine_omega"): [
        "wolfram_plots/wigner_plots/densityplot_cos1_wigner.pdf",
        "wolfram_plots/wigner_plots/plottimeev_cos1_wigner.pdf",
    ],
    ("coherence", "gaussian_zeta"): [
        "wolfram_plots/coerence_plots/densityplotcombinedcoer_gauss4.pdf",
        "wolfram_plots/coerence_plots/plottimeev_gauss4.pdf",
    ],
    ("coherence", "gaussian_T"): [
        "wolfram_plots/coerence_plots/densityplotcombinedcoer_gauss6.pdf",
        "wolfram_plots/coerence_plots/plottimeev_gauss6.pdf",
    ],
    ("coherence", "cosine_omega"): [
        "wolfram_plots/coerence_plots/densityplot_cos5_coer.pdf",
        "wolfram_plots/coerence_plots/plottimeev_cos5_coer.pdf",
    ],
    ("magic_W_half", "gaussian_zeta"): [
        "wolfram_plots/magic_plots/densityplot_gauss1_magic.pdf",
        "wolfram_plots/magic_plots/plottimeev_gauss1_magic.pdf",
    ],
    ("magic_W_half", "gaussian_T"): [
        "wolfram_plots/magic_plots/densityplot_gauss2_magic.pdf",
        "wolfram_plots/magic_plots/plottimeev_gauss2_magic.pdf",
    ],
    ("magic_W_half", "cosine_omega"): [
        "wolfram_plots/magic_plots/densityplot_cos1_magic.pdf",
        "wolfram_plots/magic_plots/plottimeev_cos1_magic.pdf",
    ],
    ("entanglement", "gaussian_zeta"): [
        "wolfram_plots/entanglement_plots/densityplot_gauss1_ent.pdf",
        "wolfram_plots/entanglement_plots/plottimeev_gauss1_ent.pdf",
    ],
    ("entanglement", "gaussian_T"): [
        "wolfram_plots/entanglement_plots/densityplot_gauss3_ent.pdf",
        "wolfram_plots/entanglement_plots/plottimeev_gauss3_ent.pdf",
    ],
    ("entanglement", "cosine_omega"): [
        "wolfram_plots/entanglement_plots/densityplot_cos6_ent.pdf",
        "wolfram_plots/entanglement_plots/plottimeev_cos6_ent.pdf",
    ],
}


def ignore_system_files(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == ".DS_Store" or name == "__pycache__"}


def directory_stats(path: Path) -> tuple[int, int]:
    count = 0
    size = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            size += item.stat().st_size
    return count, size


def copy_directory(source: Path, destination: Path, row: dict) -> dict:
    if not source.is_dir():
        raise FileNotFoundError(f"Missing source directory: {source}")
    print(f"COPY {source.relative_to(REPO)} -> {destination.relative_to(TARGET)}", flush=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2, ignore=ignore_system_files)
    count, size = directory_stats(destination)
    return {
        **row,
        "source": str(source),
        "destination": str(destination),
        "file_count": count,
        "bytes": size,
    }


def copy_file(source: Path, destination: Path, row: dict) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        **row,
        "source": str(source),
        "destination": str(destination),
        "file_count": 1,
        "bytes": destination.stat().st_size,
    }


def make_notebook(title: str, description: str, script_name: str, output: Path) -> None:
    script = (REPO / script_name).read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", description + "\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in script.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")


def preflight_sources() -> None:
    """Validate every source before creating the destination directory."""
    required_directories: set[Path] = set()
    required_files: set[Path] = set()

    for resource, couplings in ARTICLE_CASES.items():
        for coupling, roles in couplings.items():
            required_directories.update(REPO / relative for relative in roles.values())
            required_files.update(
                REPO / "sistema_aberto_acoplamento_variavel" / relative
                for relative in ARTICLE_FIGURES[(resource, coupling)]
            )

    raw_open = REPO / "results/lista_simulacoes_open_system/open_system_sims1"
    derived_magic = REPO / "results/mixed_state_magic_witnesses/mixed_magic1/derived/lista_simulacoes_open_system/open_system_sims1"
    corrected_coherence = REPO / "results/correct_coerence"
    for regime in ["only_dephasing", "only_cavity_damping"]:
        for (resource, _coupling), suffix in CASE_SUFFIX.items():
            required_directories.add(raw_open / f"{regime}_{suffix}")
            if resource == "coherence":
                required_directories.add(corrected_coherence / f"{regime}_{suffix}")
            if resource == "magic_W_half":
                required_directories.add(derived_magic / f"{regime}_{suffix}")

    for (resource, _coupling), suffix in CASE_SUFFIX.items():
        source_name = f"specific_parameters_{suffix}_specific"
        required_directories.add(raw_open / source_name)
        if resource == "coherence":
            required_directories.add(corrected_coherence / source_name)
        if resource == "magic_W_half":
            required_directories.add(derived_magic / source_name)

    required_files.update(
        REPO / name
        for name in [
            "wigner.py",
            "run_coherence_article_corrected.py",
            "quantum_magic_gauss_varep.py",
            "quantum_magic_gauss_varT.py",
            "quantum_magic_cos_var_w.py",
            "entanglment.py",
            "expected_values_all_article_cases_with_xy.py",
            "run_lista_simulacoes_open_system.py",
            "recompute_correct_coherence.py",
            "recompute_mixed_state_magic_witnesses.py",
            "plot_mixed_state_magic_article.py",
            "consolidated_plot_article.py",
            "consolidated_plot_decay_channels.py",
        ]
    )
    required_files.add(PAPER_TEX)

    missing = sorted(
        [path for path in required_directories if not path.is_dir()]
        + [path for path in required_files if not path.is_file()],
        key=str,
    )
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Preflight failed; missing sources:\n{details}")
    print(
        f"PREFLIGHT OK: {len(required_directories)} directories and "
        f"{len(required_files)} files",
        flush=True,
    )


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Target already exists; refusing to overwrite: {TARGET}")
    preflight_sources()
    TARGET.mkdir(parents=True)
    manifest: list[dict] = []

    # 1) Article data, organized by nonclassicality and modulation profile.
    for resource, couplings in ARTICLE_CASES.items():
        for coupling, roles in couplings.items():
            case_root = TARGET / "article_both_losses" / resource / coupling
            for role, relative_source in roles.items():
                manifest.append(
                    copy_directory(
                        REPO / relative_source,
                        case_root / role,
                        {
                            "regime": "article_both_losses",
                            "resource": resource,
                            "coupling": coupling,
                            "role": role,
                        },
                    )
                )

            # Copy the PDFs currently included in the article for visual provenance.
            for relative_figure in ARTICLE_FIGURES[(resource, coupling)]:
                source = REPO / "sistema_aberto_acoplamento_variavel" / relative_figure
                manifest.append(
                    copy_file(
                        source,
                        case_root / "current_article_figures" / source.name,
                        {
                            "regime": "article_both_losses",
                            "resource": resource,
                            "coupling": coupling,
                            "role": "current_article_figure",
                        },
                    )
                )

    # 2) Only one loss channel at a time.
    raw_open = REPO / "results/lista_simulacoes_open_system/open_system_sims1"
    derived_magic = REPO / "results/mixed_state_magic_witnesses/mixed_magic1/derived/lista_simulacoes_open_system/open_system_sims1"
    corrected_coherence = REPO / "results/correct_coerence"

    for regime in ["only_dephasing", "only_cavity_damping"]:
        for (resource, coupling), suffix in CASE_SUFFIX.items():
            raw_case = raw_open / f"{regime}_{suffix}"
            destination = TARGET / regime / resource / coupling
            manifest.append(
                copy_directory(
                    raw_case,
                    destination / "full_scan",
                    {"regime": regime, "resource": resource, "coupling": coupling, "role": "full_scan"},
                )
            )
            if resource == "coherence":
                manifest.append(
                    copy_directory(
                        corrected_coherence / f"{regime}_{suffix}",
                        destination / "corrected_coherence",
                        {"regime": regime, "resource": resource, "coupling": coupling, "role": "corrected_coherence"},
                    )
                )
            if resource == "magic_W_half":
                manifest.append(
                    copy_directory(
                        derived_magic / f"{regime}_{suffix}",
                        destination / "W_half",
                        {"regime": regime, "resource": resource, "coupling": coupling, "role": "W_half"},
                    )
                )

    # 3) Exact/specific points with both losses active.
    regime = "specific_both_losses"
    for (resource, coupling), suffix in CASE_SUFFIX.items():
        source_name = f"specific_parameters_{suffix}_specific"
        destination = TARGET / regime / resource / coupling
        manifest.append(
            copy_directory(
                raw_open / source_name,
                destination / "exact_runs",
                {"regime": regime, "resource": resource, "coupling": coupling, "role": "exact_runs"},
            )
        )
        if resource == "coherence":
            manifest.append(
                copy_directory(
                    corrected_coherence / source_name,
                    destination / "corrected_coherence",
                    {"regime": regime, "resource": resource, "coupling": coupling, "role": "corrected_coherence"},
                )
            )
        if resource == "magic_W_half":
            manifest.append(
                copy_directory(
                    derived_magic / source_name,
                    destination / "W_half",
                    {"regime": regime, "resource": resource, "coupling": coupling, "role": "W_half"},
                )
            )

    # 4) Provenance files and plotting notebooks.
    provenance = TARGET / "provenance"
    provenance.mkdir()
    if PAPER_TEX.is_file():
        manifest.append(
            copy_file(
                PAPER_TEX,
                provenance / "paper_draft-2.tex",
                {"regime": "provenance", "resource": "article", "coupling": "all", "role": "paper_source"},
            )
        )
    scripts = [
        "wigner.py",
        "run_coherence_article_corrected.py",
        "quantum_magic_gauss_varep.py",
        "quantum_magic_gauss_varT.py",
        "quantum_magic_cos_var_w.py",
        "entanglment.py",
        "expected_values_all_article_cases_with_xy.py",
        "run_lista_simulacoes_open_system.py",
        "recompute_correct_coherence.py",
        "recompute_mixed_state_magic_witnesses.py",
        "plot_mixed_state_magic_article.py",
    ]
    for script_name in scripts:
        manifest.append(
            copy_file(
                REPO / script_name,
                provenance / "scripts" / script_name,
                {"regime": "provenance", "resource": "all", "coupling": "all", "role": "source_script"},
            )
        )

    notebooks = TARGET / "notebooks"
    notebooks.mkdir()
    shutil.copy2(REPO / "consolidated_plot_article.py", notebooks / "plot_article_results.py")
    shutil.copy2(REPO / "consolidated_plot_decay_channels.py", notebooks / "plot_separate_decay_channels.py")
    make_notebook(
        "Resultados do artigo — dados consolidados",
        "Plota as 12 varreduras do artigo. A magia é representada por $W_{1/2}$, não por $M_2$.",
        "consolidated_plot_article.py",
        notebooks / "plot_article_results.ipynb",
    )
    make_notebook(
        "Canais dissipativos separados",
        "Compara apenas dephasing atômico e apenas damping da cavidade para todos os recursos e modulações.",
        "consolidated_plot_decay_channels.py",
        notebooks / "plot_separate_decay_channels.ipynb",
    )

    # 5) Manifest and human-readable guide.
    manifest_path = TARGET / "MANIFEST.csv"
    fields = ["regime", "resource", "coupling", "role", "source", "destination", "file_count", "bytes"]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    total_files = sum(int(row["file_count"]) for row in manifest)
    total_bytes = sum(int(row["bytes"]) for row in manifest)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": str(TARGET),
        "copy_mode": "physical copy; no source was moved or modified",
        "article_open_system_rates": {"kappa": 0.1, "gamma_phi": 0.01, "gamma": 0.0},
        "only_dephasing_rates": {"kappa": 0.0, "gamma_phi": 0.01, "gamma": 0.0},
        "only_cavity_damping_rates": {"kappa": 0.1, "gamma_phi": 0.0, "gamma": 0.0},
        "alpha": "sqrt(5)",
        "Nb": 45,
        "magic_primary_quantity": "W_half",
        "magic_rule": "W_half > 0 iff |<X>|+|<Y>|+|<Z>| > 1 for the reduced one-qubit state",
        "legacy_magic_note": "M2 files are retained only to preserve the original run; notebooks never use them as the magic result.",
        "manifest_entries": len(manifest),
        "copied_files": total_files,
        "copied_bytes": total_bytes,
    }
    (TARGET / "METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    readme = f"""# Consolidated Open-TDJC paper dataset

This directory is a non-destructive physical copy of the datasets mapped to
`paper_draft-2.tex` plus the one-channel dissipation studies.

## Main structure

- `article_both_losses/`: article cases with kappa=0.1 and gamma_phi=0.01.
- `only_dephasing/`: kappa=0 and gamma_phi=0.01.
- `only_cavity_damping/`: kappa=0.1 and gamma_phi=0.
- `specific_both_losses/`: exact parameter runs with both losses active.
- `notebooks/`: plotting notebooks and equivalent Python scripts.
- `provenance/`: paper source and simulation/post-processing scripts.

Within each regime, data are separated by resource and coupling profile:
`wigner`, `coherence`, `magic_W_half`, `entanglement`, followed by
`gaussian_zeta`, `gaussian_T`, or `cosine_omega`.

## Magic

The scientific magic result is `W_half`.  The original `M2` arrays are copied
inside `raw_dynamics_legacy_M2` solely to preserve the original simulations.
They are not used by either notebook.

## Expected values

Where available, the package includes g(t), <X>, <Y>, <Z>, <N>, <N^2>, and
Var(N), in both NPY/NPZ and CSV formats.

## Copy summary

- Manifest entries: {len(manifest)}
- Files copied: {total_files}
- Bytes copied: {total_bytes}

See `MANIFEST.csv` for the original path of every copied dataset group.
"""
    (TARGET / "README.md").write_text(readme, encoding="utf-8")
    print(f"DONE {TARGET}", flush=True)
    print(f"FILES {total_files}", flush=True)
    print(f"BYTES {total_bytes}", flush=True)


if __name__ == "__main__":
    main()
