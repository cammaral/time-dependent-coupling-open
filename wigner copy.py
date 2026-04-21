import numpy as np
import qutip as qt
from tqdm import tqdm
from quantum.hamiltonian import g_t, h_closed, h_open
import pandas as pd
import matplotlib.pyplot as plt
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from quantum.non_classicality import wigner_negativity
from scipy import integrate
from fractions import Fraction
import os
import json
from datetime import datetime

def make_exp_folder(base_name="wigner_exp", root="results"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1

def get_snapshot_indices(t, t_min=0.0, t_max=10.0, n_snapshots=10):
    target_times = np.linspace(t_min, t_max, n_snapshots)
    indices = [int(np.argmin(np.abs(t - tt))) for tt in target_times]
    actual_times = [float(t[i]) for i in indices]
    return target_times, indices, actual_times


def save_states_and_wigner_snapshots(sol, t, xvec, pvec, save_root, label):
    os.makedirs(save_root, exist_ok=True)

    target_times, indices, actual_times = get_snapshot_indices(
        t, t_min=0.0, t_max=10.0, n_snapshots=10
    )

    meta_rows = []

    for k, (ttarget, idx, tactual) in enumerate(zip(target_times, indices, actual_times)):
        state = sol.states[idx]

        # salva o estado completo
        qt.qsave(state, os.path.join(save_root, f"{label}_state_{k:02d}"))

        # pega só o modo bosônico para calcular a Wigner
        rho_field = qt.ptrace(state, 1)

        # calcula e salva a Wigner
        W = qt.wigner(rho_field, xvec, pvec)
        np.save(os.path.join(save_root, f"{label}_wigner_{k:02d}.npy"), W)

        # salva também uma figura para inspeção
        plt.figure(figsize=(6, 5))
        plt.contourf(xvec, pvec, W, 100, cmap="RdBu_r")
        plt.colorbar(label="W")
        plt.xlabel("x")
        plt.ylabel("p")
        plt.title(f"{label} | target={ttarget:.3f} | actual={tactual:.3f}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_root, f"{label}_wigner_{k:02d}.png"), dpi=200)
        plt.close()

        meta_rows.append({
            "snapshot_id": k,
            "target_time": float(ttarget),
            "actual_time": float(tactual),
            "time_index": int(idx),
            "state_file": f"{label}_state_{k:02d}.qu",
            "wigner_file_npy": f"{label}_wigner_{k:02d}.npy",
            "wigner_file_png": f"{label}_wigner_{k:02d}.png",
        })

    pd.DataFrame(meta_rows).to_csv(
        os.path.join(save_root, f"{label}_snapshots.csv"),
        index=False
    )

# ==========================
# PARAMETERS
# ==========================
eps = 1e-10
limite = 1e-1
t = np.concatenate([
    np.linspace(0, 1, 100, endpoint=False),
    np.linspace(1, 15, 100, endpoint=True)
])

N = 2     # Qubit Base Size
Nb = 45   # Field Base Size



#Linear e exp
"""
args = {
    'g0': 1,
    'eta': 1,
    'w': 2*np.pi/50,
    'kappa': 1e-1,
    'gamma': 0,
    'gamma_phi': 1e-2,
    'coupling': 'exp_mod',
    'phi': 0,
    'zeta': 0.7,
    #'lambda': 0.1
}
"""


args = {
    'g0': 1,
    'eta': 1,
    'sigma': -1,
    'kappa': 1e-1,
    'gamma': 0,
    'gamma_phi': 1e-2,
    'coupling': 'gauss',
    'epsilon': 1.5,
    'T': None
}


extra = 'gauss1'

#======= LINEAR/ EXP / TRIG =======
#wmax = 0.05
#w_list = np.linspace(wmax/5, wmax, 100, endpoint=True)
#phi_max = np.pi/2
#phi_list = np.linspace(0, phi_max, 100)

xvec = np.linspace(-7.5, 7.5, 250)
pvec = np.linspace(-7.5, 7.5, 250)


#======= GAUSS ========
#epmax = 10
Tmax = 12.0
T_list = np.linspace(0, Tmax, 250, endpoint=True)
#ep_list = np.linspace(0.1, epmax, 200, endpoint=True)
# ==========================
# INITIAL STATE
# ==========================
alpha = np.sqrt(5)

phi0 = qt.tensor(qt.basis(N,0), qt.coherent(Nb,alpha) + qt.coherent(Nb,-alpha)).unit()
print("oi")
# ==========================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ==========================
save_dir = make_exp_folder(base_name=f"wigner_{args['coupling']}", root="results")

# snapshot inicial dos args (antes do loop mudar w)
args_init = dict(args)

# salvar TXT com parâmetros e resumos
with open(os.path.join(save_dir, "run_info.txt"), "w", encoding="utf-8") as f:
    f.write("=== COERENCE RUN INFO ===\n")
    f.write(f"created_at: {datetime.now().isoformat()}\n\n")

    f.write(f"eps: {eps}\n")
    f.write(f"limite: {limite}\n")
    f.write(f"N: {N}\n")
    f.write(f"Nb: {Nb}\n")
    f.write(f"alpha: {float(alpha)}\n")
    f.write(f"extra: {extra}\n")
    
    #f.write(f"wmax: {float(wmax)}\n") # lin/exp
    #f.write(f"len(w_list): {len(w_list)}\n\n")# lin/exp
    
    # Variacao de phi
    #f.write(f"phi_max: {float(phi_max)}\n") # lin/exp
    #f.write(f"len(phi_list): {len(phi_list)}\n\n")# lin/exp
    f.write(f"len(t): {len(t)}\n")

    #f.write(f'epmax: {float(epmax)}\n') #gauss
    #f.write(f'epmax: {float(Tmax)}\n') #gauss
    #f.write(f"len(eplist): {len(T_list)}\n\n") #gauss
    #f.write(f"len(eplist): {len(ep_list)}\n\n") #gauss
  

    f.write("args (initial):\n")
    f.write(json.dumps(args_init, indent=2, ensure_ascii=False))
    f.write("\n\n")

    f.write("t summary:\n")
    f.write(f"  t_min: {float(np.min(t))}\n")
    f.write(f"  t_max: {float(np.max(t))}\n")
    f.write(f"  first_5: {t[:5].tolist()}\n")
    f.write(f"  last_5: {t[-5:].tolist()}\n\n")

    f.write("w_list summary:\n")
    #f.write(f"  T_min: {float(np.min(T_list))}\n") #gauss
    #f.write(f"  T_max: {float(np.max(T_list))}\n") #gauss
    #f.write(f"  first_5: {T_list[:5].tolist()}\n") #gauss
    #f.write(f"  last_5: {T_list[-5:].tolist()}\n") #gauss
    
    #f.write(f"  T_min: {float(np.min(T_list))}\n") #gauss
    #f.write(f"  T_max: {float(np.max(T_list))}\n") #gauss
    #f.write(f"  first_5: {T_list[:5].tolist()}\n") #gauss
    #f.write(f"  last_5: {T_list[-5:].tolist()}\n") #gauss


    #f.write(f"  w_min: {float(np.min(w_list))}\n") # lin/exp
    #f.write(f"  w_max: {float(np.max(w_list))}\n") # lin/exp
    #f.write(f"  first_5: {w_list[:5].tolist()}\n") # lin/exp
    #f.write(f"  last_5: {w_list[-5:].tolist()}\n") # lin/exp
    #f.write(f"  phi_min: {float(np.min(phi_list))}\n") # lin/exp
    #f.write(f"  phi_max: {float(np.max(phi_list))}\n") # lin/exp
    #f.write(f"  first_5: {phi_list[:5].tolist()}\n") # lin/exp
    #f.write(f"  last_5: {phi_list[-5:].tolist()}\n") # lin/exp
# salvar arrays completos e args em formato carregável
np.save(os.path.join(save_dir, "t.npy"), t)
#np.save(os.path.join(save_dir, "w_list.npy"), w_list) # lin/exp
#np.save(os.path.join(save_dir, "phi_list.npy"), phi_list) # lin/exp
np.save(os.path.join(save_dir, "T_list.npy"), T_list) #gauss

with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
    json.dump(args_init, f, indent=2, ensure_ascii=False)

# ==========================
# OPERATORS
# ==========================
sz, sp, sm, b, nb, I = get_operators(N, Nb)
obs_list = [sz, nb, nb**2]

# ==========================
# DECAY AND DEPHASING
# ==========================
c_ops = get_collapse(args, sm, sz, b)

# ==========================
# CLOSED HAMILTONIAN (CONST)
# ==========================
H1 = h_closed(args, b, sp, sm)

state0 = phi0.copy()
sol_const_aberto = solve(H1, state0, t, c_ops, obs_list, args)
sol_const = solve(H1, state0, t, None, obs_list, args, open=False)

C_const_aberto = wigner_negativity(sol_const_aberto.states, xvec, pvec, one_mode=False)
C_const = wigner_negativity(sol_const.states, xvec, pvec, one_mode=False)
# salvar const / const_aberto
np.save(os.path.join(save_dir, "const_aberto.npy"), np.array(C_const_aberto))
np.save(os.path.join(save_dir, "const.npy"), np.array(C_const))
save_states_and_wigner_snapshots(
    sol_const_aberto,
    t,
    xvec,
    pvec,
    save_root=os.path.join(save_dir, "snapshots_const_aberto"),
    label="const_aberto"
)

save_states_and_wigner_snapshots(
    sol_const,
    t,
    xvec,
    pvec,
    save_root=os.path.join(save_dir, "snapshots_const"),
    label="const"
)
# ==========================
# OPEN HAMILTONIAN (VAR over w)
# ==========================
c_list_aberto = []
c_list = []
args_per_w = []

#for phi in tqdm(phi_list):
#    args['phi'] = float(phi)
#for w in tqdm(w_list): # lin/exp
#    args['w'] = float(w) # lin/exp
for ep in tqdm(T_list): #gauss
    args["T"] = float(ep) #gauss
    #print("Passei", w)
    H = h_open(b, sp, sm)

    sol_var_aberto = solve(H, state0, t, c_ops, obs_list, args)
    sol_var = solve(H, state0, t, None, obs_list, args)

    C_var_aberto = wigner_negativity(sol_var_aberto.states, xvec, pvec, one_mode=False)
    C_var = wigner_negativity(sol_var.states, xvec, pvec, one_mode=False)

    c_list.append(C_var)
    c_list_aberto.append(C_var_aberto)
    snap_dir_var_aberto = os.path.join(save_dir, f"snapshots_var_aberto_T_{ep:.4f}")
    snap_dir_var = os.path.join(save_dir, f"snapshots_var_T_{ep:.4f}")

    save_states_and_wigner_snapshots(
        sol_var_aberto,
        t,
        xvec,
        pvec,
        save_root=snap_dir_var_aberto,
        label=f"var_aberto_T_{ep:.4f}"
    )

    save_states_and_wigner_snapshots(
        sol_var,
        t,
        xvec,
        pvec,
        save_root=snap_dir_var,
        label=f"var_T_{ep:.4f}"
    )
    # guarda o args real usado em cada w
    args_per_w.append({**args})

# salvar var / var_aberto (listas completas)
np.save(os.path.join(save_dir, "var_aberto.npy"), np.array(c_list_aberto, dtype=object))
np.save(os.path.join(save_dir, "var.npy"), np.array(c_list, dtype=object))

# salvar histórico de args (inclui w em cada passo)
pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_T.csv"), index=False)
#pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_w.csv"), index=False)
#pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_phi.csv"), index=False)

print(f"✅ Tudo salvo em: {save_dir}")
