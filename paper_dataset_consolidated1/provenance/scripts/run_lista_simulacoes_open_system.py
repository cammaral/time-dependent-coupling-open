"""
run_lista_simulacoes_open_system.py

Arquivo unico para rodar a lista de simulacoes do PDF
"sistema_aberto_acoplamento_variavel-4.pdf".

Coloque este arquivo na raiz do projeto, no mesmo nivel da pasta quantum/.
Rode com:

    python run_lista_simulacoes_open_system.py

O arquivo segue o estilo dos scripts antigos:
  - usa quantum.hamiltonian, quantum.operators e quantum.run;
  - salva tudo em results/;
  - salva .npy, .csv e .json;
  - pula resultados ja existentes quando SKIP_EXISTING = True.

Observacao importante sobre a notacao:
  No texto, "only dephasing" usa gamma = 1e-3.
  No codigo antigo, o colapso de dephasing esta em gamma_phi, enquanto gamma
  seria decaimento atomico via sigma_minus. Por isso aqui eu uso:
      gamma = 0.0
      gamma_phi = 1e-3
  para os blocos com dephasing.
"""

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import qutip as qt
from tqdm import tqdm
from scipy import integrate

from quantum.hamiltonian import g_t, h_closed, h_open
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from quantum.non_classicality import coerence, entanglement


# ============================================================
# CONFIGURACAO GERAL
# ============================================================
ROOT = "results/lista_simulacoes_open_system"

N = 2
Nb = 45
ALPHA = np.sqrt(5)

# Grade da Wigner igual ao padrao antigo. Reduza NX/NP se ficar pesado.
X_MIN, X_MAX, NX = -7.5, 7.5, 200
P_MIN, P_MAX, NP = -7.5, 7.5, 200

# Blocos principais do PDF.
RUN_BLOCKS = [
    "only_dephasing",
    "only_cavity_damping",
    "specific_parameters",
]

RESOURCES = [
    "wigner",
    "coherence",
    "magic",
    "entanglement",
]

# Para testar rapido, coloque um numero pequeno, por exemplo 3.
# Para rodar exatamente a lista completa do PDF, deixe None.
SCAN_LIMIT = None

# Mantem parecido com os scripts antigos, mas com possibilidade de retomar.
SKIP_EXISTING = True
RUN_CONST_OPEN = True
RUN_CLOSED = False
STOP_ON_ERROR = False

# Se True, ao rodar de novo ele tenta continuar a ultima pasta open_system_sims*.
# Para forcar uma rodada nova, coloque False.
RESUME_LAST_RUN = True


# ============================================================
# FUNCOES PEQUENAS
# ============================================================
def make_exp_folder(base_name="open_system_sims", root="results"):
    os.makedirs(root, exist_ok=True)

    if RESUME_LAST_RUN:
        existing = []
        for name in os.listdir(root):
            if not name.startswith(base_name):
                continue
            suffix = name.replace(base_name, "", 1)
            if suffix.isdigit():
                existing.append((int(suffix), os.path.join(root, name)))
        if existing:
            existing.sort()
            return existing[-1][1]

    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def time_grid(tmax):
    # Mesma logica dos arquivos antigos: mais pontos no inicio.
    n_after = 200 if float(tmax) == 50.0 else 100
    return np.concatenate([
        np.linspace(0, 1, 100, endpoint=False),
        np.linspace(1, tmax, n_after, endpoint=True),
    ])


def safe_float_name(x):
    s = f"{float(x):.12g}"
    s = s.replace("-", "m").replace("+", "")
    s = s.replace(".", "p").replace("/", "_")
    return s


def eval_g_single(tt, args):
    return float(g_t(float(tt), args))


def coupling_curve(t, args, constant=False):
    if constant:
        return np.full_like(t, float(args.get("g0", 1.0)), dtype=float)
    return np.array([eval_g_single(tt, args) for tt in t], dtype=float)


def expect_real(op, state):
    val = qt.expect(op, state)
    val = np.real_if_close(val)
    return float(np.real(val))


def initial_state(kind, N, Nb, alpha):
    # Mantido no mesmo padrao dos scripts antigos.
    if kind == "coherence":
        return qt.tensor(
            qt.basis(N, 0) + qt.basis(N, 1),
            qt.coherent(Nb, alpha),
        ).unit()

    if kind == "magic":
        theta = np.arccos(1 / np.sqrt(3))
        qubit = (
            np.cos(theta / 2) * qt.basis(N, 0)
            + np.exp(1j * np.pi / 4) * np.sin(theta / 2) * qt.basis(N, 1)
        )
        return qt.tensor(qubit, qt.coherent(Nb, alpha)).unit()

    if kind == "wigner":
        return qt.tensor(
            qt.basis(N, 0),
            qt.coherent(Nb, alpha) + qt.coherent(Nb, -alpha),
        ).unit()

    if kind == "entanglement":
        return (
            qt.tensor(qt.basis(N, 0), qt.coherent(Nb, -alpha))
            + qt.tensor(qt.basis(N, 1), qt.coherent(Nb, alpha))
        ).unit()

    raise ValueError(f"Estado inicial desconhecido: {kind}")


def field_wigner_negativity(states, xvec, pvec):
    # Versao local para evitar problema de compatibilidade do scipy.integrate.simpson
    # com chamadas posicionais antigas.
    vals = []
    for state in states:
        w = qt.wigner(state.ptrace(1), xvec, pvec)
        waux = integrate.simpson(np.abs(w), x=xvec, axis=-1)
        integral = integrate.simpson(waux, x=pvec, axis=-1)
        vals.append(0.5 * (float(np.real(integral)) - 1.0))
    return np.array(vals, dtype=float)


def qubit_magic_sre2(states, subsystem=0, eps=1e-15):
    # Stabilizer Renyi entropy de ordem 2 para um qubit:
    # M2 = log((1 + |<X>|^2 + |<Y>|^2 + |<Z>|^2) /
    #          (1 + |<X>|^4 + |<Y>|^4 + |<Z>|^4))
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    out = []
    for state in states:
        rho = state.ptrace(subsystem)
        px = qt.expect(X, rho)
        py = qt.expect(Y, rho)
        pz = qt.expect(Z, rho)
        ax, ay, az = abs(px), abs(py), abs(pz)
        s2 = 1.0 + ax**2 + ay**2 + az**2
        s4 = 1.0 + ax**4 + ay**4 + az**4
        out.append(np.log(max(s2, eps) / max(s4, eps)))
    return np.real_if_close(np.array(out, dtype=float))


def compute_metric(resource, states, xvec=None, pvec=None):
    if resource == "coherence":
        return np.array(coerence(states), dtype=float)
    if resource == "magic":
        return np.array(qubit_magic_sre2(states), dtype=float)
    if resource == "wigner":
        return field_wigner_negativity(states, xvec, pvec)
    if resource == "entanglement":
        return np.array(entanglement(states), dtype=float)
    raise ValueError(f"Resource desconhecido: {resource}")


def save_observables(folder, label, states, t, args, sz, nb, constant=False):
    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    exp_N = []
    exp_N2 = []
    exp_Z_global = []
    exp_X_qubit = []
    exp_Y_qubit = []
    exp_Z_qubit = []

    for state in states:
        rho_q = state.ptrace(0)
        exp_N.append(expect_real(nb, state))
        exp_N2.append(expect_real(nb ** 2, state))
        exp_Z_global.append(expect_real(sz, state))
        exp_X_qubit.append(expect_real(X, rho_q))
        exp_Y_qubit.append(expect_real(Y, rho_q))
        exp_Z_qubit.append(expect_real(Z, rho_q))

    exp_N = np.array(exp_N, dtype=float)
    exp_N2 = np.array(exp_N2, dtype=float)
    var_N = exp_N2 - exp_N**2
    g_vals = coupling_curve(t, args, constant=constant)

    df = pd.DataFrame({
        "time": t,
        "g_t": g_vals,
        "expect_N": exp_N,
        "expect_N2": exp_N2,
        "var_N": var_N,
        "expect_Z_global": np.array(exp_Z_global, dtype=float),
        "expect_X_qubit": np.array(exp_X_qubit, dtype=float),
        "expect_Y_qubit": np.array(exp_Y_qubit, dtype=float),
        "expect_Z_qubit": np.array(exp_Z_qubit, dtype=float),
    })

    df.to_csv(os.path.join(folder, f"{label}_observables.csv"), index=False)
    np.save(os.path.join(folder, f"{label}_g_t.npy"), g_vals)
    return df


def summarize_metric(t, y):
    y = np.asarray(y, dtype=float)
    return {
        "metric_initial": float(y[0]),
        "metric_max": float(np.nanmax(y)),
        "metric_mean": float(np.nanmean(y)),
        "metric_final": float(y[-1]),
        "time_at_max": float(t[int(np.nanargmax(y))]),
    }


def metric_label(resource):
    if resource == "wigner":
        return "N_W(t)"
    if resource == "coherence":
        return "C_q(t)"
    if resource == "magic":
        return "M_2(t)"
    if resource == "entanglement":
        return "E_N(t)"
    return resource


def state_kind_from_resource(resource):
    return resource


def loss_args(block_name):
    # No codigo antigo:
    #   gamma     -> decaimento atomico por sigma_minus
    #   gamma_phi -> dephasing por sigma_z
    if block_name == "only_dephasing":
        return {"kappa": 0.0, "gamma": 0.0, "gamma_phi": 1e-2}
    if block_name == "only_cavity_damping":
        return {"kappa": 1e-1, "gamma": 0.0, "gamma_phi": 0.0}
    if block_name == "specific_parameters":
        return {"kappa": 1e-1, "gamma": 0.0, "gamma_phi": 1e-2}
    raise ValueError(f"Bloco desconhecido: {block_name}")


def base_args(block_name, coupling):
    args = {
        "g0": 1.0,
        "eta": 1.0,
        "coupling": coupling,
        **loss_args(block_name),
    }
    if coupling == "cos":
        args.update({"w": 0.0, "phi": 0.0})
    if coupling == "gauss":
        args.update({"sigma": -1.0, "epsilon": None, "T": None})
    return args


def add_case(cases, block, resource, coupling, scan_name, scan_display_name,
             scan_values, fixed_args, tmax, note=""):
    if block not in RUN_BLOCKS:
        return
    if resource not in RESOURCES:
        return

    scan_values = np.asarray(scan_values, dtype=float)
    if SCAN_LIMIT is not None:
        scan_values = scan_values[:int(SCAN_LIMIT)]

    case_id = f"{block}_{resource}_{coupling}_{scan_display_name}"
    cases.append({
        "case_id": case_id,
        "block": block,
        "resource": resource,
        "state_kind": state_kind_from_resource(resource),
        "coupling": coupling,
        "scan_name": scan_name,
        "scan_display_name": scan_display_name,
        "scan_values": scan_values,
        "fixed_args": dict(fixed_args),
        "tmax": float(tmax),
        "metric_label": metric_label(resource),
        "note": note,
    })


def build_cases():
    cases = []

    # ========================================================
    # 1) Only dephasing: kappa = 0, gamma_phi = 1e-3, g0 = 1
    # 2) Only cavity damping: kappa = 1e-2, gamma_phi = 0, g0 = 1
    # ========================================================
    for block in ["only_dephasing", "only_cavity_damping"]:
        # Wigner Negativity, t de 0 a 15
        add_case(
            cases, block, "wigner", "gauss",
            "epsilon", "zeta",
            np.linspace(1.8, 3.0, 250, endpoint=True),
            {"T": 7.5, "sigma": -1.0},
            15.0,
            "Wigner: gaussian varying width; plotting av index in Python = 125.",
        )
        add_case(
            cases, block, "wigner", "gauss",
            "T", "T",
            np.linspace(4.5, 10.5, 100, endpoint=True),
            {"epsilon": 2.4, "sigma": -1.0},
            15.0,
            "Wigner: gaussian varying peak time; plotting av index in Python = 50.",
        )
        add_case(
            cases, block, "wigner", "cos",
            "w", "omega",
            np.linspace(0.0, 2.0 * np.pi / 3.0, 100, endpoint=True),
            {"phi": 0.0},
            15.0,
            "Wigner: cosine varying frequency; plotting min index in Python = 25, av index = 50.",
        )

        # Coherence, Magic, Entanglement, t de 0 a 50
        for resource in ["coherence", "magic", "entanglement"]:
            add_case(
                cases, block, resource, "gauss",
                "epsilon", "zeta",
                np.linspace(6.0, 10.0, 100, endpoint=True),
                {"T": 25.0, "sigma": -1.0},
                50.0,
                f"{resource}: gaussian varying width; plotting av index in Python = 50.",
            )
            add_case(
                cases, block, resource, "gauss",
                "T", "T",
                np.linspace(15.0, 35.0, 100, endpoint=True),
                {"epsilon": 8.0, "sigma": -1.0},
                50.0,
                f"{resource}: gaussian varying peak time; plotting av index in Python = 50.",
            )
            add_case(
                cases, block, resource, "cos",
                "w", "omega",
                np.linspace(0.0, np.pi / 5.0, 200, endpoint=True),
                {"phi": 0.0},
                50.0,
                f"{resource}: cosine varying frequency; plotting min index in Python = 50, av index = 100.",
            )

    # ========================================================
    # 3) Specific parameters: kappa = 1e-2, gamma_phi = 1e-3, g0 = 1
    # ========================================================
    block = "specific_parameters"

    add_case(
        cases, block, "wigner", "gauss",
        "epsilon", "zeta_specific",
        np.array([2.4]),
        {"T": 7.5, "sigma": -1.0},
        15.0,
        "Specific Wigner: zeta_av = 2.4.",
    )
    add_case(
        cases, block, "wigner", "gauss",
        "T", "T_specific",
        np.array([7.5]),
        {"epsilon": 2.4, "sigma": -1.0},
        15.0,
        "Specific Wigner: T_av = 7.5.",
    )
    add_case(
        cases, block, "wigner", "cos",
        "w", "omega_specific",
        np.array([np.pi / 6.0, np.pi / 3.0]),
        {"phi": 0.0},
        15.0,
        "Specific Wigner: omega_min = pi/6 and omega_av = pi/3.",
    )

    for resource in ["coherence", "magic", "entanglement"]:
        add_case(
            cases, block, resource, "gauss",
            "epsilon", "zeta_specific",
            np.array([8.0]),
            {"T": 25.0, "sigma": -1.0},
            50.0,
            f"Specific {resource}: zeta_av = 8.",
        )
        add_case(
            cases, block, resource, "gauss",
            "T", "T_specific",
            np.array([25.0]),
            {"epsilon": 8.0, "sigma": -1.0},
            50.0,
            f"Specific {resource}: T_av = 25.",
        )
        add_case(
            cases, block, resource, "cos",
            "w", "omega_specific",
            np.array([np.pi / 20.0, np.pi / 10.0]),
            {"phi": 0.0},
            50.0,
            f"Specific {resource}: omega_min = pi/20 and omega_av = pi/10.",
        )

    return cases


def run_one_solution(case_dir, label, case, args, state0, t, c_ops, obs_list,
                     sz, sp, sm, b, nb, xvec, pvec, open_system=True, constant=False):
    metric_path = os.path.join(case_dir, f"{label}.npy")
    obs_path = os.path.join(case_dir, f"{label}_observables.csv")

    if SKIP_EXISTING and os.path.exists(metric_path) and os.path.exists(obs_path):
        return np.load(metric_path)

    H = h_closed(args, b, sp, sm) if constant else h_open(b, sp, sm)

    if open_system:
        sol = solve(H, state0, t, c_ops, obs_list, args)
    else:
        sol = solve(H, state0, t, None, obs_list, args, open=False)

    metric = compute_metric(case["resource"], sol.states, xvec=xvec, pvec=pvec)
    np.save(metric_path, metric)
    save_observables(case_dir, label, sol.states, t, args, sz, nb, constant=constant)
    return metric


def run_case(case, root_dir):
    case_dir = os.path.join(root_dir, case["case_id"])
    os.makedirs(case_dir, exist_ok=True)

    done_path = os.path.join(case_dir, "DONE.txt")
    if SKIP_EXISTING and os.path.exists(done_path):
        print(f"[skip] {case['case_id']}")
        return pd.read_csv(os.path.join(case_dir, "summary.csv"))

    t = time_grid(case["tmax"])
    xvec = np.linspace(X_MIN, X_MAX, NX)
    pvec = np.linspace(P_MIN, P_MAX, NP)

    args0 = base_args(case["block"], case["coupling"])
    args0.update(case["fixed_args"])

    sz, sp, sm, b, nb, I = get_operators(N, Nb)
    obs_list = [sz, nb, nb ** 2]
    c_ops = get_collapse(args0, sm, sz, b)
    state0 = initial_state(case["state_kind"], N, Nb, ALPHA)

    # Salva metadados antes de rodar, para facilitar auditoria se parar no meio.
    metadata = dict(case)
    metadata["scan_values"] = list(np.asarray(case["scan_values"], dtype=float))
    metadata["fixed_args"] = dict(case["fixed_args"])
    metadata["base_args"] = dict(args0)
    metadata["N"] = int(N)
    metadata["Nb"] = int(Nb)
    metadata["alpha"] = float(ALPHA)
    metadata["nx_wigner"] = int(NX)
    metadata["np_wigner"] = int(NP)
    metadata["created_or_updated_at"] = datetime.now().isoformat()
    metadata["run_const_open"] = bool(RUN_CONST_OPEN)
    metadata["run_closed"] = bool(RUN_CLOSED)

    with open(os.path.join(case_dir, "case_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    np.save(os.path.join(case_dir, "t.npy"), t)
    np.save(os.path.join(case_dir, "scan_values.npy"), np.asarray(case["scan_values"], dtype=float))

    summary_rows = []

    print(f"\n[{case['case_id']}] n_scan = {len(case['scan_values'])}")

    if RUN_CONST_OPEN:
        metric = run_one_solution(
            case_dir, "const_aberto", case, args0, state0, t, c_ops, obs_list,
            sz, sp, sm, b, nb, xvec, pvec,
            open_system=True, constant=True,
        )
        summary_rows.append({
            "case_id": case["case_id"],
            "label": "const_aberto",
            "block": case["block"],
            "resource": case["resource"],
            "coupling": case["coupling"],
            "scan_name": "const",
            "scan_index_python": -1,
            "scan_value": np.nan,
            **summarize_metric(t, metric),
        })

        if RUN_CLOSED:
            metric = run_one_solution(
                case_dir, "const", case, args0, state0, t, c_ops, obs_list,
                sz, sp, sm, b, nb, xvec, pvec,
                open_system=False, constant=True,
            )
            summary_rows.append({
                "case_id": case["case_id"],
                "label": "const",
                "block": case["block"],
                "resource": case["resource"],
                "coupling": case["coupling"],
                "scan_name": "const",
                "scan_index_python": -1,
                "scan_value": np.nan,
                **summarize_metric(t, metric),
            })

    var_metrics = []
    args_rows = []

    for i, value in enumerate(tqdm(case["scan_values"], desc=case["case_id"])):
        args = dict(args0)
        args[case["scan_name"]] = float(value)

        label = f"var_aberto_{i:04d}_{case['scan_display_name']}_{safe_float_name(value)}"
        metric = run_one_solution(
            case_dir, label, case, args, state0, t, c_ops, obs_list,
            sz, sp, sm, b, nb, xvec, pvec,
            open_system=True, constant=False,
        )
        var_metrics.append(metric)
        args_rows.append({
            **args,
            "label": label,
            "scan_index_python": int(i),
            "scan_index_text_1_based": int(i + 1),
            "scan_name": case["scan_name"],
            "scan_display_name": case["scan_display_name"],
            "scan_value": float(value),
        })
        summary_rows.append({
            "case_id": case["case_id"],
            "label": label,
            "block": case["block"],
            "resource": case["resource"],
            "coupling": case["coupling"],
            "scan_name": case["scan_name"],
            "scan_display_name": case["scan_display_name"],
            "scan_index_python": int(i),
            "scan_index_text_1_based": int(i + 1),
            "scan_value": float(value),
            **summarize_metric(t, metric),
        })

        if RUN_CLOSED:
            closed_label = label.replace("var_aberto", "var")
            closed_metric = run_one_solution(
                case_dir, closed_label, case, args, state0, t, c_ops, obs_list,
                sz, sp, sm, b, nb, xvec, pvec,
                open_system=False, constant=False,
            )
            summary_rows.append({
                "case_id": case["case_id"],
                "label": closed_label,
                "block": case["block"],
                "resource": case["resource"],
                "coupling": case["coupling"],
                "scan_name": case["scan_name"],
                "scan_display_name": case["scan_display_name"],
                "scan_index_python": int(i),
                "scan_index_text_1_based": int(i + 1),
                "scan_value": float(value),
                **summarize_metric(t, closed_metric),
            })

    var_metrics = np.array(var_metrics, dtype=float)
    np.save(os.path.join(case_dir, "var_aberto.npy"), var_metrics)
    pd.DataFrame(args_rows).to_csv(os.path.join(case_dir, "args_per_scan.csv"), index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(case_dir, "summary.csv"), index=False)

    with open(done_path, "w", encoding="utf-8") as f:
        f.write(f"done_at: {datetime.now().isoformat()}\n")

    return summary


def main():
    root_dir = make_exp_folder(base_name="open_system_sims", root=ROOT)
    cases = build_cases()

    run_info = {
        "created_at": datetime.now().isoformat(),
        "root_dir": root_dir,
        "run_blocks": RUN_BLOCKS,
        "resources": RESOURCES,
        "scan_limit": SCAN_LIMIT,
        "skip_existing": SKIP_EXISTING,
        "resume_last_run": RESUME_LAST_RUN,
        "run_const_open": RUN_CONST_OPEN,
        "run_closed": RUN_CLOSED,
        "N": int(N),
        "Nb": int(Nb),
        "alpha": float(ALPHA),
        "wigner_grid": {
            "x_min": X_MIN,
            "x_max": X_MAX,
            "nx": NX,
            "p_min": P_MIN,
            "p_max": P_MAX,
            "np": NP,
        },
        "note": (
            "gamma_phi is used for dephasing. gamma is kept zero unless you explicitly "
            "want atomic relaxation instead of pure dephasing."
        ),
    }
    with open(os.path.join(root_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    case_map = []
    all_summaries = []
    errors = []

    for case in cases:
        case_map.append({
            "case_id": case["case_id"],
            "block": case["block"],
            "resource": case["resource"],
            "state_kind": case["state_kind"],
            "coupling": case["coupling"],
            "scan_name": case["scan_name"],
            "scan_display_name": case["scan_display_name"],
            "n_scan": len(case["scan_values"]),
            "tmax": case["tmax"],
            "metric_label": case["metric_label"],
            "fixed_args": json.dumps(case["fixed_args"], ensure_ascii=False),
            "note": case["note"],
        })

        try:
            summary = run_case(case, root_dir)
            all_summaries.append(summary)
        except Exception as exc:
            err = {
                "case_id": case["case_id"],
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            pd.DataFrame(errors).to_csv(os.path.join(root_dir, "errors.csv"), index=False)
            print(f"\nERRO em {case['case_id']}: {exc}")
            if STOP_ON_ERROR:
                raise

    pd.DataFrame(case_map).to_csv(os.path.join(root_dir, "case_map.csv"), index=False)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(root_dir, "summary_all_cases.csv"), index=False
        )

    if errors:
        pd.DataFrame(errors).to_csv(os.path.join(root_dir, "errors.csv"), index=False)
        print(f"\nFinalizado com {len(errors)} erro(s). Veja errors.csv em: {root_dir}")
    else:
        print(f"\nTudo salvo em: {root_dir}")


if __name__ == "__main__":
    main()
