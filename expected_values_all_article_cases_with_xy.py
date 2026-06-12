"""
expected_values_all_article_cases_with_xy.py

Roda os valores esperados dos observaveis do sistema Jaynes-Cummings
para todos os casos do artigo/notebook `plot_article_all_cases_2x2.ipynb`,
agora incluindo tambem <X> e <Y>.

O script calcula e salva, para cada caso e para cada curva:
    <X>, <Y>, <Z>, <N>, <N^2>, Var(N) = <N^2> - <N>^2, g(t)

Casos salvos por padrao:
    - const_aberto
    - const
    - var_aberto_min / var_aberto_av / var_aberto_max
    - var_min / var_av / var_max

Se quiser somente os casos abertos, mude RUN_CLOSED_CASES = False.

Como usar:
    Coloque este arquivo na raiz do projeto, no mesmo nivel das pastas
    `quantum/` e `utils/`, e rode:

        python expected_values_all_article_cases_with_xy.py

Saida:
    results/expected_values_all_cases_xy/expected_values_all_cases_xy*/
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import qutip as qt
from tqdm import tqdm

from quantum.hamiltonian import g_t, h_closed, h_open
from quantum.operators import get_operators, get_collapse
from quantum.run import solve


# ============================================================
# CONFIGURACAO GERAL
# ============================================================
RUN_CLOSED_CASES = True   # False -> salva apenas const_aberto e var_aberto_*
N = 2                    # qubit base size
Nb = 45                  # field base size
alpha = np.sqrt(5)

ROOT = "results/expected_values_all_cases_xy"
BASE_NAME = "expected_values_all_cases_xy"


# ============================================================
# UTILIDADES
# ============================================================
def make_exp_folder(base_name=BASE_NAME, root=ROOT):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def to_jsonable(obj):
    """Converte numpy scalars/arrays para tipos serializaveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def as_real_array(values, name="observable"):
    """Converte expectativas do QuTiP para array real.

    Os operadores usados aqui sao Hermitianos, entao a parte imaginaria deve ser
    apenas residuo numerico pequeno.
    """
    arr = np.asarray(values, dtype=np.complex128)
    arr = np.real_if_close(arr, tol=1000)
    if np.iscomplexobj(arr):
        max_imag = float(np.max(np.abs(np.imag(arr)))) if arr.size else 0.0
        if max_imag > 1e-9:
            print(f"Aviso: {name} tem parte imaginaria maxima {max_imag:.3e}; salvando parte real.")
        arr = np.real(arr)
    return np.asarray(arr, dtype=float)


def make_time_grid(tmax):
    """Mesma estrutura temporal dos scripts do projeto."""
    tmax = float(tmax)
    return np.concatenate([
        np.linspace(0, 1, 100, endpoint=False),
        np.linspace(1, tmax, 200, endpoint=True),
    ])


def eval_g_single(tt, args):
    """Wrapper para a assinatura g_t(t, args)."""
    try:
        return float(g_t(float(tt), args))
    except TypeError:
        return float(g_t(args, float(tt)))


def coupling_curve(t, args, constant=False):
    if constant:
        return np.full_like(np.asarray(t, dtype=float), float(args.get("g0", 1.0)), dtype=float)
    return np.array([eval_g_single(tt, args) for tt in t], dtype=float)


def selected_points(values):
    labels = ["min", "av", "max"]
    return [
        {
            "label": label,
            "scan_index": int(i),
            "scan_value": float(values[i]),
            "display_value": float(values[i]),
        }
        for i, label in enumerate(labels)
    ]


# ============================================================
# ESTADOS INICIAIS DO ARTIGO
# ============================================================
def initial_state(state_kind, N=2, Nb=45, alpha=np.sqrt(5)):
    """Estados iniciais usados nos diferentes blocos do projeto."""
    if state_kind == "coherence":
        # Mesmo padrao de coerence.py e enegy.py
        return qt.tensor(
            qt.basis(N, 0) + qt.basis(N, 1),
            qt.coherent(Nb, alpha),
        ).unit()

    if state_kind == "magic":
        # Mesmo padrao de quantum_magic_cos_var_w.py
        theta = np.arccos(1 / np.sqrt(3))
        qubit = (
            np.cos(theta / 2) * qt.basis(N, 0)
            + np.exp(1j * np.pi / 4) * np.sin(theta / 2) * qt.basis(N, 1)
        )
        return qt.tensor(qubit, qt.coherent(Nb, alpha)).unit()

    if state_kind == "wigner":
        # Mesmo padrao do expected_values_all_time_cos_var_w.py:
        # qubit em |0> e campo em estado tipo gato.
        return qt.tensor(
            qt.basis(N, 0),
            qt.coherent(Nb, alpha) + qt.coherent(Nb, -alpha),
        ).unit()

    if state_kind == "entanglement":
        # Mesmo padrao de entanglment.py
        return (
            qt.tensor(qt.basis(N, 0), qt.coherent(Nb, -alpha))
            + qt.tensor(qt.basis(N, 1), qt.coherent(Nb, alpha))
        ).unit()

    raise ValueError(f"state_kind desconhecido: {state_kind}")


# ============================================================
# DEFINICAO DOS CASOS DO NOTEBOOK DO ARTIGO
# ============================================================
def cos_scan_values(tmax):
    """Tres omegas: omega_min, omega_av, omega_max.

    Para tmax=50, recupera o padrao dos scripts longos: wmax = 10*pi/50.
    Para tmax=15, recupera o padrao Wigner: wmax = 10*pi/15.
    """
    wmax = 10 * np.pi / float(tmax)
    return np.array([wmax / 4, wmax / 2, wmax], dtype=float)


ARTICLE_CASES = [
    # Coherence: Fig. 1, 2, 3
    {
        "case_id": "fig1_coherence_gauss_epsilon",
        "resource": "coherence",
        "state_kind": "coherence",
        "coupling": "gauss",
        "scan_name": "epsilon",
        "scan_values": np.array([6.0, 8.0, 10.0]),
        "fixed_args": {"T": 25.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "fig2_coherence_gauss_T",
        "resource": "coherence",
        "state_kind": "coherence",
        "coupling": "gauss",
        "scan_name": "T",
        "scan_values": np.array([15.0, 25.0, 35.0]),
        "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "fig3_coherence_cos_omega",
        "resource": "coherence",
        "state_kind": "coherence",
        "coupling": "cos",
        "scan_name": "w",
        "scan_values": cos_scan_values(50.0),
        "fixed_args": {"phi": 0.0},
        "tmax": 50.0,
    },

    # Magic: mesmos parametros das Figs. 1, 2, 3
    {
        "case_id": "magic_gauss_epsilon",
        "resource": "magic",
        "state_kind": "magic",
        "coupling": "gauss",
        "scan_name": "epsilon",
        "scan_values": np.array([6.0, 8.0, 10.0]),
        "fixed_args": {"T": 25.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "magic_gauss_T",
        "resource": "magic",
        "state_kind": "magic",
        "coupling": "gauss",
        "scan_name": "T",
        "scan_values": np.array([15.0, 25.0, 35.0]),
        "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "magic_cos_omega",
        "resource": "magic",
        "state_kind": "magic",
        "coupling": "cos",
        "scan_name": "w",
        "scan_values": cos_scan_values(50.0),
        "fixed_args": {"phi": 0.0},
        "tmax": 50.0,
    },

    # Wigner: Figs. 4, 5, 6. Escala temporal curta do notebook: tmax=15.
    {
        "case_id": "fig4_wigner_gauss_epsilon",
        "resource": "wigner",
        "state_kind": "wigner",
        "coupling": "gauss",
        "scan_name": "epsilon",
        "scan_values": np.array([1.8, 2.4, 3.0]),
        "fixed_args": {"T": 7.5, "sigma": -1.0},
        "tmax": 15.0,
    },
    {
        "case_id": "fig5_wigner_gauss_T",
        "resource": "wigner",
        "state_kind": "wigner",
        "coupling": "gauss",
        "scan_name": "T",
        "scan_values": np.array([4.5, 7.5, 10.5]),
        "fixed_args": {"epsilon": 2.4, "sigma": -1.0},
        "tmax": 15.0,
    },
    {
        "case_id": "fig6_wigner_cos_omega",
        "resource": "wigner",
        "state_kind": "wigner",
        "coupling": "cos",
        "scan_name": "w",
        "scan_values": cos_scan_values(15.0),
        "fixed_args": {"phi": 0.0},
        "tmax": 15.0,
    },

    # Entanglement: Figs. 7, 8, 9
    {
        "case_id": "fig7_entanglement_gauss_epsilon",
        "resource": "entanglement",
        "state_kind": "entanglement",
        "coupling": "gauss",
        "scan_name": "epsilon",
        "scan_values": np.array([6.0, 8.0, 10.0]),
        "fixed_args": {"T": 25.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "fig8_entanglement_gauss_T",
        "resource": "entanglement",
        "state_kind": "entanglement",
        "coupling": "gauss",
        "scan_name": "T",
        "scan_values": np.array([15.0, 25.0, 35.0]),
        "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
        "tmax": 50.0,
    },
    {
        "case_id": "fig9_entanglement_cos_omega",
        "resource": "entanglement",
        "state_kind": "entanglement",
        "coupling": "cos",
        "scan_name": "w",
        "scan_values": cos_scan_values(50.0),
        "fixed_args": {"phi": 0.0},
        "tmax": 50.0,
    },
]


# ============================================================
# ARGUMENTOS DO HAMILTONIANO
# ============================================================
def base_args_for_case(case, scan_value=None):
    args = {
        "g0": 1,
        "eta": 1,
        "kappa": 1e-1,
        "gamma": 0,
        "gamma_phi": 1e-2,
        "coupling": case["coupling"],
    }

    args.update(case.get("fixed_args", {}))

    if case["coupling"] == "cos":
        args.setdefault("phi", 0.0)
        args["w"] = float(scan_value if scan_value is not None else case["scan_values"][1])

    elif case["coupling"] == "gauss":
        args.setdefault("sigma", -1.0)
        # Para o caso constante o h_closed ignora epsilon/T, mas deixamos tudo
        # completo para salvar metadados e permitir g(t) se necessario.
        if case["scan_name"] == "epsilon":
            args["epsilon"] = float(scan_value if scan_value is not None else case["scan_values"][1])
            args.setdefault("T", float(case["fixed_args"].get("T", 25.0)))
        elif case["scan_name"] == "T":
            args["T"] = float(scan_value if scan_value is not None else case["scan_values"][1])
            args.setdefault("epsilon", float(case["fixed_args"].get("epsilon", 8.0)))
        else:
            raise ValueError(f"scan_name gauss desconhecido: {case['scan_name']}")

    else:
        raise ValueError(f"coupling desconhecido: {case['coupling']}")

    return args


# ============================================================
# SALVAMENTO DE UMA SOLUCAO
# ============================================================
def extract_expected_values(sol):
    """Ordem esperada: [sx, sy, sz, nb, nb**2]."""
    if sol.expect is None or len(sol.expect) < 5:
        raise ValueError(
            "sol.expect nao contem os 5 observaveis esperados. "
            "Use obs_list = [sx, sy, sz, nb, nb**2]."
        )

    ex = as_real_array(sol.expect[0], "expect_X")
    ey = as_real_array(sol.expect[1], "expect_Y")
    ez = as_real_array(sol.expect[2], "expect_Z")
    en = as_real_array(sol.expect[3], "expect_N")
    en2 = as_real_array(sol.expect[4], "expect_N2")
    var_n = en2 - en**2
    var_n[np.isclose(var_n, 0.0, atol=1e-12)] = 0.0

    return {
        "expect_X": ex,
        "expect_Y": ey,
        "expect_Z": ez,
        "expect_N": en,
        "expect_N2": en2,
        "var_N": var_n,
    }


def save_observables(case_dir, label, t, args_for_coupling, sol, constant_coupling=False):
    obs = extract_expected_values(sol)
    g_values = coupling_curve(t, args_for_coupling, constant=constant_coupling)

    # Arquivos separados, estilo expected_values_all_time_cos_var_w.py
    for key, arr in obs.items():
        np.save(os.path.join(case_dir, f"{label}_{key}.npy"), arr)
    np.save(os.path.join(case_dir, f"{label}_g_t.npy"), g_values)

    df = pd.DataFrame({
        "time_index": np.arange(len(t), dtype=int),
        "time": np.asarray(t, dtype=float),
        "g_t": g_values,
        "expect_X": obs["expect_X"],
        "expect_Y": obs["expect_Y"],
        "expect_Z": obs["expect_Z"],
        "expect_N": obs["expect_N"],
        "expect_N2": obs["expect_N2"],
        "var_N": obs["var_N"],
    })
    df.to_csv(os.path.join(case_dir, f"{label}_observables.csv"), index=False)

    # Um NPZ unico tambem facilita carregar tudo depois.
    np.savez(
        os.path.join(case_dir, f"{label}_observables.npz"),
        t=np.asarray(t, dtype=float),
        g_t=g_values,
        **obs,
    )

    return df


def save_matrix_outputs(case_dir, prefix, labels, dfs):
    """Salva matrizes do tipo var_aberto_expect_X.npy com shape [scan, tempo]."""
    if len(dfs) == 0:
        return
    for col in ["g_t", "expect_X", "expect_Y", "expect_Z", "expect_N", "expect_N2", "var_N"]:
        mat = np.vstack([df[col].to_numpy(dtype=float) for df in dfs])
        np.save(os.path.join(case_dir, f"{prefix}_{col}.npy"), mat)
    pd.DataFrame({"row_index": range(len(labels)), "label": labels}).to_csv(
        os.path.join(case_dir, f"{prefix}_matrix_rows.csv"), index=False
    )


def summarize_label(label, case_id, scan_name, scan_value, display_value, df):
    row = {
        "label": label,
        "case_id": case_id,
        "scan_name": scan_name,
        "scan_value": scan_value,
        "display_value": display_value,
    }
    for col in ["expect_X", "expect_Y", "expect_Z", "expect_N", "expect_N2", "var_N"]:
        arr = df[col].to_numpy(dtype=float)
        row[f"{col}_initial"] = float(arr[0])
        row[f"{col}_final"] = float(arr[-1])
        row[f"{col}_min"] = float(np.min(arr))
        row[f"{col}_max"] = float(np.max(arr))
        row[f"{col}_mean"] = float(np.mean(arr))
    return row


# ============================================================
# RODAR UM CASO COMPLETO
# ============================================================
def run_case(case, root_dir):
    case_id = case["case_id"]
    case_dir = os.path.join(root_dir, case_id)
    os.makedirs(case_dir, exist_ok=True)

    print(f"\n=== Rodando caso: {case_id} ===")

    t = make_time_grid(case["tmax"])
    np.save(os.path.join(case_dir, "t.npy"), t)

    scan_points = selected_points(case["scan_values"])
    scan_df = pd.DataFrame(scan_points)
    scan_df.to_csv(os.path.join(case_dir, "scan_points_used.csv"), index=False)
    np.save(os.path.join(case_dir, f"{case['scan_name']}_list_selected.npy"), case["scan_values"])

    with open(os.path.join(case_dir, "case_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(case), f, indent=2, ensure_ascii=False)

    state0 = initial_state(case["state_kind"], N=N, Nb=Nb, alpha=alpha)
    sz, sp, sm, b, nb, _ = get_operators(N, Nb)

    # Pauli X e Y no mesmo espaco tensorial dos operadores do projeto.
    sx = sp + sm
    sy = -1j * (sp - sm)

    obs_list = [sx, sy, sz, nb, nb**2]

    summary_rows = []

    # -------------------------
    # Constante: aberto/fechado
    # -------------------------
    args_const = base_args_for_case(case, scan_value=None)
    c_ops_const = get_collapse(args_const, sm, sz, b)
    H_const = h_closed(args_const, b, sp, sm)

    sol_const_aberto = solve(
        H_const,
        state0,
        t,
        c_ops_const,
        obs_list,
        args_const,
        store_state=False,
        open=True,
    )
    df_const_aberto = save_observables(
        case_dir,
        "const_aberto",
        t,
        args_const,
        sol_const_aberto,
        constant_coupling=True,
    )
    summary_rows.append(summarize_label("const_aberto", case_id, "const", np.nan, np.nan, df_const_aberto))

    if RUN_CLOSED_CASES:
        sol_const = solve(
            H_const,
            state0,
            t,
            None,
            obs_list,
            args_const,
            store_state=False,
            open=False,
        )
        df_const = save_observables(
            case_dir,
            "const",
            t,
            args_const,
            sol_const,
            constant_coupling=True,
        )
        summary_rows.append(summarize_label("const", case_id, "const", np.nan, np.nan, df_const))

    # -------------------------
    # Variavel: min / av / max
    # -------------------------
    var_aberto_dfs = []
    var_aberto_labels = []
    var_dfs = []
    var_labels = []

    for row in tqdm(scan_points, desc=case_id):
        scan_label = row["label"]
        scan_value = row["scan_value"]
        display_value = row["display_value"]

        args_var = base_args_for_case(case, scan_value=scan_value)
        c_ops_var = get_collapse(args_var, sm, sz, b)
        H_var = h_open(b, sp, sm)

        label_aberto = f"var_aberto_{scan_label}"
        sol_var_aberto = solve(
            H_var,
            state0,
            t,
            c_ops_var,
            obs_list,
            args_var,
            store_state=False,
            open=True,
        )
        df_var_aberto = save_observables(
            case_dir,
            label_aberto,
            t,
            args_var,
            sol_var_aberto,
            constant_coupling=False,
        )
        summary_rows.append(
            summarize_label(label_aberto, case_id, case["scan_name"], scan_value, display_value, df_var_aberto)
        )
        var_aberto_dfs.append(df_var_aberto)
        var_aberto_labels.append(label_aberto)

        if RUN_CLOSED_CASES:
            label_fechado = f"var_{scan_label}"
            sol_var = solve(
                H_var,
                state0,
                t,
                None,
                obs_list,
                args_var,
                store_state=False,
                open=False,
            )
            df_var = save_observables(
                case_dir,
                label_fechado,
                t,
                args_var,
                sol_var,
                constant_coupling=False,
            )
            summary_rows.append(
                summarize_label(label_fechado, case_id, case["scan_name"], scan_value, display_value, df_var)
            )
            var_dfs.append(df_var)
            var_labels.append(label_fechado)

    save_matrix_outputs(case_dir, "var_aberto", var_aberto_labels, var_aberto_dfs)
    if RUN_CLOSED_CASES:
        save_matrix_outputs(case_dir, "var", var_labels, var_dfs)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(case_dir, "summary_expected_values.csv"), index=False)

    return {
        "case_id": case_id,
        "case_dir": case_dir,
        "n_time": int(len(t)),
        "n_scan": int(len(scan_points)),
        "ran_closed_cases": bool(RUN_CLOSED_CASES),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    root_dir = make_exp_folder()
    print(f"Saida principal: {root_dir}")

    with open(os.path.join(root_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("=== EXPECTED VALUES ALL ARTICLE CASES WITH X/Y ===\n")
        f.write(f"created_at: {datetime.now().isoformat()}\n")
        f.write(f"N: {N}\n")
        f.write(f"Nb: {Nb}\n")
        f.write(f"alpha: {float(alpha)}\n")
        f.write(f"RUN_CLOSED_CASES: {RUN_CLOSED_CASES}\n")
        f.write("observables: X, Y, Z, N, N2, Var(N), g(t)\n")

    case_map = pd.DataFrame([
        {
            "case_id": case["case_id"],
            "resource": case["resource"],
            "state_kind": case["state_kind"],
            "coupling": case["coupling"],
            "scan_name": case["scan_name"],
            "scan_values": json.dumps(to_jsonable(case["scan_values"])),
            "fixed_args": json.dumps(to_jsonable(case["fixed_args"])),
            "tmax": case["tmax"],
        }
        for case in ARTICLE_CASES
    ])
    case_map.to_csv(os.path.join(root_dir, "case_map.csv"), index=False)

    global_summary = []
    for case in ARTICLE_CASES:
        global_summary.append(run_case(case, root_dir))

    pd.DataFrame(global_summary).to_csv(os.path.join(root_dir, "global_summary.csv"), index=False)
    print(f"\n✅ Tudo salvo em: {root_dir}")


if __name__ == "__main__":
    main()
