"""Recalculate mixed-state magic diagnostics from existing simulation data.

The script does not rerun the Jaynes-Cummings/Lindblad dynamics.  It searches
the existing result tree for magic datasets that contain the atomic-qubit
expectation values <X>, <Y>, and <Z>, and calculates

    M_{1/2}, M_2, M_3,
    W_{1/2}, W_2, W_3,
    S_2 = -ln Tr(rho^2), purity,
    |x| + |y| + |z|,
    the exact one-qubit stabilizer-octahedron test.

The mixed-state witnesses follow Haug and Tarabunga,
"Efficient witnessing and testing of magic in mixed quantum states",
npj Quantum Information 12, 40 (2026).

Examples
--------
Process every existing magic dataset under results/:

    .venv/bin/python recompute_mixed_state_magic_witnesses.py

Only inspect and validate the inputs, without writing derived files:

    .venv/bin/python recompute_mixed_state_magic_witnesses.py --dry-run

Write flattened CSV files also for multidimensional parameter scans:

    .venv/bin/python recompute_mixed_state_magic_witnesses.py --full-csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantum.non_classicality import (
    mixed_state_magic_witness_qubit_from_expectations,
)


ALPHAS = ((0.5, "half"), (2.0, "2"), (3.0, "3"))
PAULI_KEY_SETS = (
    ("px", "py", "pz"),
    ("expect_X", "expect_Y", "expect_Z"),
    ("expect_X_qubit", "expect_Y_qubit", "expect_Z_qubit"),
)
SCAN_AXIS_NAMES = (
    "scan_values.npy",
    "w_list.npy",
    "T_list.npy",
    "ep_list.npy",
    "epsilon_list.npy",
    "epsilon_list_selected.npy",
    "T_list_selected.npy",
    "w_list_selected.npy",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Recalcula witnesses de magia de estados mistos usando resultados "
            "ja existentes; nao executa novamente a dinamica."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project_root / "results",
        help="Raiz contendo os resultados existentes (padrao: results).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "results" / "mixed_state_magic_witnesses",
        help="Raiz das saidas derivadas.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Nome exato da pasta de saida; se omitido, usa mixed_magic1, 2, ...",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Tolerancia numerica para classificar W>0 e |x|+|y|+|z|>1.",
    )
    parser.add_argument(
        "--full-csv",
        action="store_true",
        help="Tambem salva CSV achatado para scans multidimensionais.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula e valida tudo, mas nao escreve arquivos.",
    )
    return parser.parse_args()


def is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_sources(results_root: Path, output_root: Path) -> list[Path]:
    """Find every magic result that may contain all three Pauli components."""
    sources: list[Path] = []
    for path in results_root.rglob("*"):
        if not path.is_file() or is_inside(path, output_root):
            continue
        if "magic" not in "/".join(path.parts).lower():
            continue

        name = path.name.lower()
        if path.suffix.lower() == ".npz" and (
            name.endswith("_details.npz") or name.endswith("_observables.npz")
        ):
            sources.append(path)
        elif path.suffix.lower() == ".csv" and name.endswith("_observables.csv"):
            sources.append(path)

    return sorted(sources)


def as_real_array(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    arr = np.real_if_close(arr, tol=1000)
    if np.iscomplexobj(arr):
        max_imag = float(np.max(np.abs(np.imag(arr)))) if arr.size else 0.0
        if max_imag > 1e-8:
            raise ValueError(
                f"{name} possui parte imaginaria inesperada: max={max_imag:.3e}"
            )
        arr = np.real(arr)
    return np.asarray(arr, dtype=float)


def find_pauli_keys(keys: set[str]) -> tuple[str, str, str] | None:
    for key_set in PAULI_KEY_SETS:
        if all(key in keys for key in key_set):
            return key_set
    return None


def sibling_npz_for_csv(source: Path) -> Path:
    return source.with_suffix(".npz")


def sibling_metric_path(source: Path) -> Path:
    name = source.name
    for suffix in ("_observables.npz", "_observables.csv", "_details.npz"):
        if name.endswith(suffix):
            return source.with_name(name[: -len(suffix)] + ".npy")
    return source.with_suffix(".npy")


def load_source(source: Path) -> dict[str, Any]:
    """Load Pauli expectations and optional axes/legacy M2 from NPZ or CSV."""
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            keys = set(data.files)
            pauli_keys = find_pauli_keys(keys)
            if pauli_keys is None:
                raise KeyError(
                    "faltam as tres componentes de Pauli X, Y e Z; "
                    f"chaves presentes: {sorted(keys)}"
                )
            kx, ky, kz = pauli_keys
            loaded = {
                "px": as_real_array(data[kx], kx),
                "py": as_real_array(data[ky], ky),
                "pz": as_real_array(data[kz], kz),
                "time": as_real_array(data["t"], "t") if "t" in keys else None,
                "legacy_M2": (
                    as_real_array(data["M2"], "M2") if "M2" in keys else None
                ),
                "pauli_keys": pauli_keys,
            }
    else:
        frame = pd.read_csv(source)
        keys = set(frame.columns)
        pauli_keys = find_pauli_keys(keys)
        if pauli_keys is None:
            raise KeyError(
                "faltam as tres componentes de Pauli X, Y e Z; "
                f"colunas presentes: {sorted(keys)}"
            )
        kx, ky, kz = pauli_keys
        time_key = "time" if "time" in keys else "t" if "t" in keys else None
        loaded = {
            "px": as_real_array(frame[kx].to_numpy(), kx),
            "py": as_real_array(frame[ky].to_numpy(), ky),
            "pz": as_real_array(frame[kz].to_numpy(), kz),
            "time": (
                as_real_array(frame[time_key].to_numpy(), time_key)
                if time_key is not None
                else None
            ),
            "legacy_M2": (
                as_real_array(frame["M2"].to_numpy(), "M2")
                if "M2" in keys
                else None
            ),
            "pauli_keys": pauli_keys,
        }

    px, py, pz = np.broadcast_arrays(loaded["px"], loaded["py"], loaded["pz"])
    loaded["px"] = np.asarray(px, dtype=float)
    loaded["py"] = np.asarray(py, dtype=float)
    loaded["pz"] = np.asarray(pz, dtype=float)

    if loaded["legacy_M2"] is None:
        metric_path = sibling_metric_path(source)
        if metric_path.exists():
            metric = as_real_array(np.load(metric_path, allow_pickle=False), "legacy_M2")
            if metric.shape == loaded["px"].shape:
                loaded["legacy_M2"] = metric

    return loaded


def load_axes(source: Path, shape: tuple[int, ...], loaded_time: Any) -> dict[str, Any]:
    time = loaded_time
    if time is None:
        time_path = source.parent / "t.npy"
        if time_path.exists():
            candidate = as_real_array(np.load(time_path, allow_pickle=False), "time")
            if shape and candidate.ndim == 1 and candidate.size == shape[-1]:
                time = candidate

    scan_values = None
    scan_axis_file = None
    if len(shape) >= 2:
        for filename in SCAN_AXIS_NAMES:
            candidate_path = source.parent / filename
            if not candidate_path.exists():
                continue
            candidate = as_real_array(
                np.load(candidate_path, allow_pickle=False), filename
            )
            if candidate.ndim == 1 and candidate.size == shape[0]:
                scan_values = candidate
                scan_axis_file = filename
                break

    return {
        "time": time,
        "scan_values": scan_values,
        "scan_axis_file": scan_axis_file,
    }


def calculate_all(px: np.ndarray, py: np.ndarray, pz: np.ndarray, tol: float) -> dict[str, np.ndarray]:
    results = {
        label: mixed_state_magic_witness_qubit_from_expectations(
            px, py, pz, alpha=alpha, tol=tol
        )
        for alpha, label in ALPHAS
    }

    base = results["half"]
    output: dict[str, np.ndarray] = {
        "px": np.asarray(px, dtype=float),
        "py": np.asarray(py, dtype=float),
        "pz": np.asarray(pz, dtype=float),
        "purity": np.asarray(base["purity"], dtype=float),
        "renyi2_entropy": np.asarray(base["renyi2_entropy"], dtype=float),
        "bloch_l1": np.asarray(base["bloch_l1"], dtype=float),
        "is_magic_exact": np.asarray(base["is_magic_exact"], dtype=bool),
    }

    for _, label in ALPHAS:
        output[f"A_{label}"] = np.asarray(results[label]["A_alpha"], dtype=float)
        output[f"M_{label}"] = np.asarray(results[label]["M_alpha"], dtype=float)
        output[f"W_{label}"] = np.asarray(results[label]["W_alpha"], dtype=float)
        output[f"is_magic_witnessed_{label}"] = np.asarray(
            results[label]["is_magic_witnessed"], dtype=bool
        )

    return output


def make_run_dir(output_root: Path, run_name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if run_name:
        run_dir = output_root / run_name
        if run_dir.exists():
            raise FileExistsError(
                f"A pasta de saida ja existe: {run_dir}. Escolha outro --run-name."
            )
        run_dir.mkdir()
        return run_dir

    index = 1
    while True:
        run_dir = output_root / f"mixed_magic{index}"
        if not run_dir.exists():
            run_dir.mkdir()
            return run_dir
        index += 1


def output_stem(run_dir: Path, results_root: Path, source: Path) -> Path:
    relative = source.relative_to(results_root)
    destination = run_dir / "derived" / relative.parent
    destination.mkdir(parents=True, exist_ok=True)
    return destination / f"{source.stem}_mixed_magic"


def one_dimensional_frame(metrics: dict[str, np.ndarray], time: Any) -> pd.DataFrame:
    size = metrics["px"].size
    columns: dict[str, Any] = {"point_index": np.arange(size, dtype=int)}
    if time is not None and np.asarray(time).size == size:
        columns["time"] = np.asarray(time, dtype=float)
    columns.update({name: np.ravel(values) for name, values in metrics.items()})
    return pd.DataFrame(columns)


def flattened_frame(
    metrics: dict[str, np.ndarray], time: Any, scan_values: Any
) -> pd.DataFrame:
    shape = metrics["px"].shape
    flat_indices = np.indices(shape).reshape(len(shape), -1).T
    columns: dict[str, Any] = {"flat_index": np.arange(metrics["px"].size)}
    for axis_index in range(len(shape)):
        columns[f"axis_{axis_index}_index"] = flat_indices[:, axis_index]
    if scan_values is not None and len(shape) >= 2:
        columns["scan_value"] = np.asarray(scan_values)[flat_indices[:, 0]]
    if time is not None:
        time = np.asarray(time)
        if time.ndim == 1 and time.size == shape[-1]:
            columns["time"] = time[flat_indices[:, -1]]
    columns.update({name: np.ravel(values) for name, values in metrics.items()})
    return pd.DataFrame(columns)


def summary_row(
    source: Path,
    output_path: Path | None,
    metrics: dict[str, np.ndarray],
    legacy_M2: Any,
    physical_excess: float,
    scan_axis_file: str | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source": str(source),
        "output_npz": str(output_path) if output_path is not None else "",
        "shape": "x".join(str(value) for value in metrics["px"].shape),
        "n_points": int(metrics["px"].size),
        "scan_axis_file": scan_axis_file or "",
        "purity_min": float(np.min(metrics["purity"])),
        "purity_max": float(np.max(metrics["purity"])),
        "physical_bloch_radius_excess": physical_excess,
        "fraction_M2_positive": float(np.mean(metrics["M_2"] > 1e-10)),
        "fraction_magic_exact": float(np.mean(metrics["is_magic_exact"])),
        "fraction_W_half_positive": float(
            np.mean(metrics["is_magic_witnessed_half"])
        ),
        "fraction_W2_positive": float(np.mean(metrics["is_magic_witnessed_2"])),
        "fraction_W3_positive": float(np.mean(metrics["is_magic_witnessed_3"])),
        "M2_max": float(np.max(metrics["M_2"])),
        "W_half_max": float(np.max(metrics["W_half"])),
        "W2_max": float(np.max(metrics["W_2"])),
        "W3_max": float(np.max(metrics["W_3"])),
    }
    if legacy_M2 is not None:
        legacy_M2 = np.asarray(legacy_M2, dtype=float)
        if legacy_M2.shape == metrics["M_2"].shape:
            row["legacy_M2_max_abs_error"] = float(
                np.max(np.abs(legacy_M2 - metrics["M_2"]))
            )
        else:
            row["legacy_M2_max_abs_error"] = np.nan
    else:
        row["legacy_M2_max_abs_error"] = np.nan
    return row


def per_curve_summary(metrics: dict[str, np.ndarray], scan_values: Any) -> pd.DataFrame:
    rows = []
    for index in range(metrics["px"].shape[0]):
        exact = np.ravel(metrics["is_magic_exact"][index])
        w_half = np.ravel(metrics["W_half"][index])
        w2 = np.ravel(metrics["W_2"][index])
        w3 = np.ravel(metrics["W_3"][index])
        m2 = np.ravel(metrics["M_2"][index])
        rows.append(
            {
                "curve_index": index,
                "scan_value": (
                    float(np.asarray(scan_values)[index])
                    if scan_values is not None
                    else np.nan
                ),
                "n_points": int(exact.size),
                "fraction_M2_positive": float(np.mean(m2 > 1e-10)),
                "fraction_magic_exact": float(np.mean(exact)),
                "fraction_W_half_positive": float(np.mean(w_half > 1e-10)),
                "fraction_W2_positive": float(np.mean(w2 > 1e-10)),
                "fraction_W3_positive": float(np.mean(w3 > 1e-10)),
                "M2_max": float(np.max(m2)),
                "W_half_max": float(np.max(w_half)),
                "W2_max": float(np.max(w2)),
                "W3_max": float(np.max(w3)),
                "M2_final": float(m2[-1]),
                "W_half_final": float(w_half[-1]),
                "W2_final": float(w2[-1]),
                "W3_final": float(w3[-1]),
            }
        )
    return pd.DataFrame(rows)


def save_dataset(
    stem: Path,
    metrics: dict[str, np.ndarray],
    axes: dict[str, Any],
    full_csv: bool,
) -> Path:
    payload = dict(metrics)
    if axes["time"] is not None:
        payload["time"] = np.asarray(axes["time"], dtype=float)
    if axes["scan_values"] is not None:
        payload["scan_values"] = np.asarray(axes["scan_values"], dtype=float)

    npz_path = stem.with_suffix(".npz")
    np.savez_compressed(npz_path, **payload)

    if metrics["px"].ndim == 1:
        one_dimensional_frame(metrics, axes["time"]).to_csv(
            stem.with_suffix(".csv"), index=False
        )
    else:
        per_curve_summary(metrics, axes["scan_values"]).to_csv(
            stem.with_name(stem.name + "_summary_per_curve.csv"), index=False
        )
        if full_csv:
            flattened_frame(metrics, axes["time"], axes["scan_values"]).to_csv(
                stem.with_suffix(".csv"), index=False
            )

    return npz_path


def write_readme(run_dir: Path) -> None:
    text = """# Mixed-state magic post-processing

Every `*_mixed_magic.npz` file contains the original atomic-qubit Pauli
expectation values and the derived quantities for alpha = 1/2, 2, and 3.

Main fields:

- `M_half`, `M_2`, `M_3`: mixed-state SRE extensions. They are retained for
  comparison, but positivity alone is not a mixed-state magic certificate.
- `W_half`, `W_2`, `W_3`: Haug--Tarabunga mixed-state magic witnesses.
  Positive values certify magic.
- `bloch_l1`: |<X>| + |<Y>| + |<Z>|.
- `is_magic_exact`: exact one-qubit stabilizer-octahedron membership test.
- `purity`: Tr(rho^2).
- `renyi2_entropy`: -ln Tr(rho^2).

For the one-qubit atomic subsystem, `W_half > 0`, `bloch_l1 > 1`, and
`is_magic_exact == True` are equivalent (up to numerical tolerance).

`W_alpha` is a witness, not a magic monotone. Genuine mixed-state resource
monotones discussed in the reference include log-free robustness of magic and
mixed-state stabilizer fidelity; they are not computed by this script.
"""
    (run_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_root = args.output_root.resolve()

    if not results_root.is_dir():
        raise NotADirectoryError(f"Raiz de resultados inexistente: {results_root}")

    sources = discover_sources(results_root, output_root)
    run_dir = None if args.dry_run else make_run_dir(output_root, args.run_name)

    inventory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    duplicates = 0

    for source in sources:
        if source.suffix.lower() == ".csv" and sibling_npz_for_csv(source).exists():
            duplicates += 1
            inventory_rows.append(
                {
                    "source": str(source),
                    "status": "duplicate_prefer_npz",
                    "reason": str(sibling_npz_for_csv(source)),
                }
            )
            continue

        try:
            loaded = load_source(source)
            metrics = calculate_all(
                loaded["px"], loaded["py"], loaded["pz"], tol=args.tol
            )
            axes = load_axes(source, metrics["px"].shape, loaded["time"])

            bloch_radius = np.sqrt(
                metrics["px"] ** 2 + metrics["py"] ** 2 + metrics["pz"] ** 2
            )
            physical_excess = max(0.0, float(np.max(bloch_radius)) - 1.0)

            witness_identity_error = float(
                np.max(np.abs(metrics["W_half"] - 2.0 * np.log(
                    np.maximum((1.0 + metrics["bloch_l1"]) / 2.0, 1e-15)
                )))
            )
            if witness_identity_error > 1e-11:
                raise ValueError(
                    "falha na identidade W_half = 2 ln[(1+|r|_1)/2]: "
                    f"erro={witness_identity_error:.3e}"
                )

            npz_output = None
            if run_dir is not None:
                stem = output_stem(run_dir, results_root, source)
                npz_output = save_dataset(stem, metrics, axes, args.full_csv)

            summary_rows.append(
                summary_row(
                    source,
                    npz_output,
                    metrics,
                    loaded["legacy_M2"],
                    physical_excess,
                    axes["scan_axis_file"],
                )
            )
            inventory_rows.append(
                {
                    "source": str(source),
                    "status": "processed",
                    "reason": "",
                }
            )
            processed += 1
        except (KeyError, ValueError, OSError) as exc:
            inventory_rows.append(
                {
                    "source": str(source),
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            skipped += 1

    summary = pd.DataFrame(summary_rows)
    inventory = pd.DataFrame(inventory_rows)

    if run_dir is not None:
        summary.to_csv(run_dir / "dataset_summary.csv", index=False)
        inventory.to_csv(run_dir / "input_inventory.csv", index=False)
        write_readme(run_dir)
        run_info = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "results_root": str(results_root),
            "output_root": str(output_root),
            "run_dir": str(run_dir),
            "tolerance": args.tol,
            "full_csv": bool(args.full_csv),
            "sources_discovered": len(sources),
            "processed": processed,
            "skipped": skipped,
            "duplicate_csv_prefer_npz": duplicates,
            "reference": {
                "authors": "Tobias Haug and Poetri Sonya Tarabunga",
                "title": "Efficient witnessing and testing of magic in mixed quantum states",
                "journal": "npj Quantum Information 12, 40 (2026)",
                "doi": "10.1038/s41534-026-01189-z",
            },
        }
        (run_dir / "run_info.json").write_text(
            json.dumps(run_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Fontes encontradas: {len(sources)}")
    print(f"Processadas: {processed}")
    print(f"Ignoradas por falta de X/Y/Z ou formato invalido: {skipped}")
    print(f"CSVs duplicados (NPZ preferido): {duplicates}")
    if not summary.empty:
        total_points = int(summary["n_points"].sum())
        print(f"Pontos recalculados: {total_points}")
        max_legacy_error = summary["legacy_M2_max_abs_error"].dropna()
        if not max_legacy_error.empty:
            print(
                "Maior erro ao reproduzir o M2 antigo: "
                f"{float(max_legacy_error.max()):.3e}"
            )
    if run_dir is not None:
        print(f"Saida: {run_dir}")
    else:
        print("Dry-run concluido; nenhum arquivo foi escrito.")

    return 0 if processed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
