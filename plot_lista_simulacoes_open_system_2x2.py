"""
plot_lista_simulacoes_open_system_2x2.py

Script simples para plotar os resultados gerados por:

    python run_lista_simulacoes_open_system.py

Ele foi feito no mesmo estilo do notebook plot_article_all_cases_2x2.ipynb,
mas adaptado para a estrutura nova:

    results/lista_simulacoes_open_system/open_system_sims*/

Para cada caso, gera uma figura 2x2 com:
  1) nao-classicalidade;
  2) acoplamento g(t);
  3) valor esperado <Z>;
  4) valor esperado <N>.

Por padrao, o script NAO plota todas as 100/200/250 curvas. Ele usa apenas os
indices representativos pedidos no PDF:
  - zeta_av, T_av;
  - omega_min e omega_av;
  - todos os valores dos casos specific_parameters.

Rode na raiz do projeto, no mesmo nivel da pasta results/:

    python plot_lista_simulacoes_open_system_2x2.py
"""

import os
import glob
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACAO
# ============================================================
BASE = "results/lista_simulacoes_open_system"
ROOT = None

# Se True, usa apenas os indices pedidos no PDF.
# Se False, tenta plotar todos os scans de cada caso. Para 100/200/250 curvas,
# isso pode ficar pesado e visualmente ruim.
SELECTED_ONLY = True

# Salva figuras em ROOT/figures_2x2.
SAVE_FIGURES = True
SAVE_PNG = True
SAVE_PDF = True

# Mostra as figuras na tela. Em servidor/headless, deixe False.
SHOW_FIGURES = False

DPI = 220

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


# ============================================================
# FUNCOES DE LEITURA
# ============================================================
def latest_root(base=BASE):
    folders = sorted(glob.glob(os.path.join(base, "open_system_sims*")))
    if not folders:
        raise FileNotFoundError(
            "Nao encontrei resultados. Rode primeiro: "
            "python run_lista_simulacoes_open_system.py"
        )
    return folders[-1]


if ROOT is None:
    ROOT = latest_root()


def case_dir(case_id):
    return os.path.join(ROOT, case_id)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_case_map():
    path = os.path.join(ROOT, "case_map.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nao achei: {path}")
    return pd.read_csv(path)


def load_case(case_id):
    folder = case_dir(case_id)
    meta_path = os.path.join(folder, "case_metadata.json")
    t_path = os.path.join(folder, "t.npy")
    args_path = os.path.join(folder, "args_per_scan.csv")
    summary_path = os.path.join(folder, "summary.csv")

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Nao achei: {meta_path}")
    if not os.path.exists(t_path):
        raise FileNotFoundError(f"Nao achei: {t_path}")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Nao achei: {args_path}")

    out = {
        "folder": folder,
        "meta": read_json(meta_path),
        "t": np.load(t_path),
        "args": pd.read_csv(args_path),
    }

    if os.path.exists(summary_path):
        out["summary"] = pd.read_csv(summary_path)
    else:
        out["summary"] = None

    return out


def load_metric(folder, label):
    path = os.path.join(folder, f"{label}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nao achei: {path}")
    return np.load(path)


def load_observables(folder, label):
    path = os.path.join(folder, f"{label}_observables.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Nao achei: {path}")
    return pd.read_csv(path)


# ============================================================
# ESCOLHA DOS INDICES DO PDF
# ============================================================
def clip_indices(indices, n):
    out = []
    for idx in indices:
        idx = int(idx)
        if idx < 0:
            continue
        if idx >= n:
            idx = n - 1
        if idx not in out:
            out.append(idx)
    return out


def selected_scan_indices(meta, args_df):
    """
    Retorna indices Python para plotagem.

    O PDF usa notacao do tipo "125+1th parameter". Aqui isso vira indice
    Python 125, pois a contagem do texto e 1-based.
    """
    n = len(args_df)
    if n == 0:
        return []

    block = str(meta.get("block", ""))
    resource = str(meta.get("resource", ""))
    coupling = str(meta.get("coupling", ""))
    scan_name = str(meta.get("scan_name", ""))
    scan_display = str(meta.get("scan_display_name", ""))

    # Nos casos specific_parameters, o script de simulacao ja rodou exatamente
    # os parametros pedidos no PDF. Entao plota todos.
    if block == "specific_parameters":
        return list(range(n))

    # Wigner: t em [0, 15]
    if resource == "wigner":
        if coupling == "gauss" and scan_name == "epsilon":
            return clip_indices([125], n)          # zeta_av
        if coupling == "gauss" and scan_name == "T":
            return clip_indices([50], n)           # T_av
        if coupling == "cos" or scan_name == "w":
            return clip_indices([25, 50], n)       # omega_min, omega_av

    # Coherence, Magic, Entanglement: t em [0, 50]
    if resource in {"coherence", "magic", "entanglement"}:
        if coupling == "gauss" and scan_name == "epsilon":
            return clip_indices([50], n)           # zeta_av
        if coupling == "gauss" and scan_name == "T":
            return clip_indices([50], n)           # T_av
        if coupling == "cos" or scan_name == "w":
            return clip_indices([50, 100], n)      # omega_min, omega_av

    # Fallback simples.
    if "specific" in scan_display:
        return list(range(n))
    return clip_indices([n // 2], n)


def rows_to_plot(data):
    args_df = data["args"].copy()
    if not SELECTED_ONLY:
        indices = list(range(len(args_df)))
    else:
        indices = selected_scan_indices(data["meta"], args_df)

    rows = []

    # Curva constante aberta, se existir.
    const_path = os.path.join(data["folder"], "const_aberto.npy")
    const_obs = os.path.join(data["folder"], "const_aberto_observables.csv")
    if os.path.exists(const_path) and os.path.exists(const_obs):
        rows.append({
            "label": "const_aberto",
            "pretty": "constant open",
            "kind": "constant",
            "scan_index_python": -1,
            "scan_value": np.nan,
        })

    for idx in indices:
        if idx < 0 or idx >= len(args_df):
            continue
        row = args_df.iloc[int(idx)]
        label = str(row["label"])
        rows.append({
            "label": label,
            "pretty": pretty_scan_label(row),
            "kind": "variable",
            "scan_index_python": int(row.get("scan_index_python", idx)),
            "scan_value": float(row.get("scan_value", np.nan)),
        })

    return rows


# ============================================================
# FORMATACAO
# ============================================================
def safe_log_limits(arrays, floor=1e-12):
    vals = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        vals.extend(arr[np.isfinite(arr) & (arr > 0)].tolist())
    if not vals:
        return floor, 1.0
    return max(floor, min(vals) * 0.7), max(vals) * 1.3


def use_log_for_metric(resource):
    return resource in {"coherence", "entanglement"}


def metric_name(meta):
    label = meta.get("metric_label")
    if isinstance(label, str) and label.strip():
        return label

    resource = meta.get("resource", "")
    if resource == "wigner":
        return r"$N_W(t)$"
    if resource == "coherence":
        return r"$C_q(t)$"
    if resource == "magic":
        return r"$M_2(t)$"
    if resource == "entanglement":
        return r"$E_N(t)$"
    return "non-classicality"


def format_pi_value(x, tol=1e-8):
    if not np.isfinite(x):
        return "nan"

    candidates = [
        (1 / 20, r"\pi/20"),
        (1 / 10, r"\pi/10"),
        (1 / 6, r"\pi/6"),
        (1 / 5, r"\pi/5"),
        (1 / 3, r"\pi/3"),
        (2 / 3, r"2\pi/3"),
    ]
    for frac, text in candidates:
        if abs(x - frac * np.pi) < tol:
            return f"${text}$"
    return f"{x:.4g}"


def pretty_scan_label(row):
    name = str(row.get("scan_display_name", row.get("scan_name", "param")))
    value = float(row.get("scan_value", np.nan))
    idx = int(row.get("scan_index_python", -1))

    if "omega" in name or str(row.get("scan_name", "")) == "w":
        val = format_pi_value(value)
    else:
        val = f"{value:.4g}"

    return f"{name} = {val}  [i={idx}]"


def get_z_column(df):
    for col in ["expect_Z_qubit", "expect_Z", "expect_Z_global"]:
        if col in df.columns:
            return col
    raise KeyError(
        "Nao achei coluna de Z. Esperava uma destas: "
        "expect_Z_qubit, expect_Z, expect_Z_global"
    )


def safe_filename(name):
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


# ============================================================
# PLOT
# ============================================================
def plot_case_2x2(case_id, save=SAVE_FIGURES, show=SHOW_FIGURES):
    data = load_case(case_id)
    meta = data["meta"]
    t = data["t"]
    folder = data["folder"]
    resource = meta.get("resource", "")

    rows = rows_to_plot(data)
    if not rows:
        print(f"[skip] {case_id}: nenhuma curva encontrada")
        return None

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.0),
        sharex="col",
        constrained_layout=True,
    )

    ax_metric = axes[0, 0]
    ax_g = axes[0, 1]
    ax_z = axes[1, 0]
    ax_n = axes[1, 1]

    metric_arrays = []
    legend_handles = []
    legend_labels = []
    plotted_rows = []

    for row in rows:
        label = row["label"]
        pretty = row["pretty"]

        try:
            y_metric = load_metric(folder, label)
            obs = load_observables(folder, label)
        except FileNotFoundError as exc:
            print(f"[missing] {case_id} | {label}: {exc}")
            continue

        tt = obs["time"].to_numpy(dtype=float) if "time" in obs.columns else t
        z_col = get_z_column(obs)

        style = {"lw": 2.2, "linestyle": "--"} if row["kind"] == "constant" else {"lw": 1.9}

        line_metric, = ax_metric.plot(t, y_metric, label=pretty, **style)
        ax_g.plot(tt, obs["g_t"].to_numpy(dtype=float), **style)
        ax_z.plot(tt, obs[z_col].to_numpy(dtype=float), **style)
        ax_n.plot(tt, obs["expect_N"].to_numpy(dtype=float), **style)

        metric_arrays.append(y_metric)
        legend_handles.append(line_metric)
        legend_labels.append(pretty)
        plotted_rows.append(row)

    if not plotted_rows:
        plt.close(fig)
        print(f"[skip] {case_id}: arquivos ainda incompletos")
        return None

    if use_log_for_metric(resource):
        ax_metric.set_yscale("log")
        ax_metric.set_ylim(*safe_log_limits(metric_arrays))

    ax_metric.set_ylabel(metric_name(meta))
    ax_metric.set_title("Non-classicality")

    ax_g.set_ylabel(r"$g(t)$")
    ax_g.set_title("Coupling")

    ax_z.set_ylabel(r"$\langle Z \rangle$")
    ax_z.set_title(r"Expected value of $Z$")
    ax_z.set_xlabel("t")

    ax_n.set_ylabel(r"$\langle N \rangle$")
    ax_n.set_title(r"Expected value of $N$")
    ax_n.set_xlabel("t")

    title = (
        f"{meta.get('block', '')} | {meta.get('resource', '')} | "
        f"{meta.get('coupling', '')} | {meta.get('scan_display_name', '')}"
    )
    fig.suptitle(title, fontsize=13)
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, fontsize=9)

    out_paths = []
    if save:
        fig_dir = os.path.join(ROOT, "figures_2x2")
        os.makedirs(fig_dir, exist_ok=True)
        base = os.path.join(fig_dir, f"plot_2x2_{safe_filename(case_id)}")

        if SAVE_PNG:
            png = base + ".png"
            fig.savefig(png, dpi=DPI, bbox_inches="tight")
            out_paths.append(png)

        if SAVE_PDF:
            pdf = base + ".pdf"
            fig.savefig(pdf, bbox_inches="tight")
            out_paths.append(pdf)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[ok] {case_id}: {len(plotted_rows)} curva(s)")
    for path in out_paths:
        print(f"     salvo: {path}")

    return {
        "case_id": case_id,
        "n_curves": len(plotted_rows),
        "files": ";".join(out_paths),
    }


def main():
    print("ROOT =", ROOT)
    case_map = load_case_map()

    records = []
    for case_id in case_map["case_id"].tolist():
        try:
            rec = plot_case_2x2(case_id)
            if rec is not None:
                records.append(rec)
        except Exception as exc:
            print(f"[error] {case_id}: {exc}")

    if records:
        out = os.path.join(ROOT, "figures_2x2", "figures_2x2_index.csv")
        pd.DataFrame(records).to_csv(out, index=False)
        print("\nIndice salvo em:", out)

    print("\nFinalizado.")


if __name__ == "__main__":
    main()
