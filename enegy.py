import numpy as np
import qutip as qt
from tqdm import tqdm
from quantum.hamiltonian import g_t, h_closed, h_open
import pandas as pd
import matplotlib.pyplot as plt
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from scipy import integrate
from fractions import Fraction
import os
import json
from datetime import datetime

def make_exp_folder(base_name="energy_exp", root="results/energy"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1

# ==========================
# PARAMETERS
# ==========================
eps = 1e-10
limite = 1e-1
t = np.concatenate([
    np.linspace(0, 1, 100, endpoint=False),
    np.linspace(1, 50, 200, endpoint=True)
])

N = 2     # Qubit Base Size
Nb = 45   # Field Base Size

# Linear e exp
'''
args = {
    'g0': 1,
    'eta': 1,
    'w': 2*np.pi/50,
    'kappa': 1e-1,
    'gamma': 0,
    'gamma_phi': 1e-2,
    'coupling': 'cos',
    'phi': 0
}

'''
args = {
    'g0': 1,
    'eta': 1,
    'sigma': -1,
    'kappa': 1e-1,
    'gamma': 0,
    'gamma_phi': 1e-2,
    'coupling': 'gauss',
    'epsilon': 8,
    'T': None
}


extra = 'gauss1'

# ======= LINEAR / EXP / TRIG =======
#wmax = 10*np.pi/50
#w_list = np.linspace(0, wmax, 200)

# ======= GAUSS ========
#epmax = 1
Tmax = 35
T_list = np.linspace(15, Tmax, 100, endpoint=True)
# ep_list = np.linspace(1e-5, epmax, 100, endpoint=True)

# ==========================
# INITIAL STATE
# ==========================
alpha = np.sqrt(5)
phi0 = qt.tensor(qt.basis(N, 0) + qt.basis(N, 1), qt.coherent(Nb, alpha)).unit()
# phi0 = (qt.tensor(qt.basis(N,0), qt.coherent(Nb, -alpha)) + qt.tensor(qt.basis(N,1), qt.coherent(Nb, alpha))).unit()

# ==========================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ==========================
save_dir = make_exp_folder(base_name=f"energy_{args['coupling']}", root="results/energy")

# snapshot inicial dos args (antes do loop mudar w)
args_init = dict(args)

with open(os.path.join(save_dir, "run_info.txt"), "w", encoding="utf-8") as f:
    f.write("=== ENERGY RUN INFO ===\n")
    f.write(f"created_at: {datetime.now().isoformat()}\n\n")

    f.write(f"eps: {eps}\n")
    f.write(f"limite: {limite}\n")
    f.write(f"N: {N}\n")
    f.write(f"Nb: {Nb}\n")
    f.write(f"alpha: {float(alpha)}\n")
    f.write(f"extra: {extra}\n")

    #f.write(f"wmax: {float(wmax)}\n")
    #f.write(f"len(w_list): {len(w_list)}\n\n")
    #f.write(f"len(t): {len(t)}\n")

    #f.write(f'epmax: {float(epmax)}\n')
    f.write(f'Tmax: {float(Tmax)}\n')
    f.write(f"len(T_list): {len(T_list)}\n\n")
    # f.write(f"len(ep_list): {len(ep_list)}\n\n")

    f.write("args (initial):\n")
    f.write(json.dumps(args_init, indent=2, ensure_ascii=False))
    f.write("\n\n")

    f.write("t summary:\n")
    f.write(f"  t_min: {float(np.min(t))}\n")
    f.write(f"  t_max: {float(np.max(t))}\n")
    f.write(f"  first_5: {t[:5].tolist()}\n")
    f.write(f"  last_5: {t[-5:].tolist()}\n\n")

    f.write("w_list summary:\n")
    #f.write(f"  w_min: {float(np.min(w_list))}\n")
    #f.write(f"  w_max: {float(np.max(w_list))}\n")
    #f.write(f"  first_5: {w_list[:5].tolist()}\n")
    #f.write(f"  last_5: {w_list[-5:].tolist()}\n")

np.save(os.path.join(save_dir, "t.npy"), t)
np.save(os.path.join(save_dir, "T_list.npy"), T_list)

with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
    json.dump(args_init, f, indent=2, ensure_ascii=False)

# ==========================
# OPERATORS
# ==========================
sz, sp, sm, b, nb, I = get_operators(N, Nb)

# observáveis:
# 0 -> <sz> do íon
# 1 -> <n>  do campo
obs_list = [sz, nb]

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

# expectativas
sz_const_aberto = np.array(sol_const_aberto.expect[0], dtype=float)
nb_const_aberto = np.array(sol_const_aberto.expect[1], dtype=float)

sz_const = np.array(sol_const.expect[0], dtype=float)
nb_const = np.array(sol_const.expect[1], dtype=float)

# salvar const / const_aberto
np.save(os.path.join(save_dir, "const_aberto_sz.npy"), sz_const_aberto)
np.save(os.path.join(save_dir, "const_aberto_nb.npy"), nb_const_aberto)

np.save(os.path.join(save_dir, "const_sz.npy"), sz_const)
np.save(os.path.join(save_dir, "const_nb.npy"), nb_const)

# opcional: salvar também em csv
df_const = pd.DataFrame({
    "t": t,
    "sz_const": sz_const,
    "nb_const": nb_const,
    "sz_const_aberto": sz_const_aberto,
    "nb_const_aberto": nb_const_aberto,
})
df_const.to_csv(os.path.join(save_dir, "const_expectations.csv"), index=False)

# ==========================
# OPEN HAMILTONIAN (VAR over w)
# ==========================
sz_list_aberto = []
nb_list_aberto = []

sz_list = []
nb_list = []

args_per_w = []

# for phi in tqdm(phi_list):
#     args['phi'] = float(phi)

#for w in tqdm(w_list):
    #args['w'] = float(w)

for ep in tqdm(T_list):
    args["T"] = float(ep)

    H = h_open(b, sp, sm)

    sol_var_aberto = solve(H, state0, t, c_ops, obs_list, args)
    sol_var = solve(H, state0, t, None, obs_list, args)

    sz_var_aberto = np.array(sol_var_aberto.expect[0], dtype=float)
    nb_var_aberto = np.array(sol_var_aberto.expect[1], dtype=float)

    sz_var = np.array(sol_var.expect[0], dtype=float)
    nb_var = np.array(sol_var.expect[1], dtype=float)

    sz_list.append(sz_var)
    nb_list.append(nb_var)

    sz_list_aberto.append(sz_var_aberto)
    nb_list_aberto.append(nb_var_aberto)

    args_per_w.append({**args})

# transformar em arrays 2D: (len(w_list), len(t))
sz_list = np.array(sz_list, dtype=float)
nb_list = np.array(nb_list, dtype=float)

sz_list_aberto = np.array(sz_list_aberto, dtype=float)
nb_list_aberto = np.array(nb_list_aberto, dtype=float)

# salvar var / var_aberto
np.save(os.path.join(save_dir, "var_sz.npy"), sz_list)
np.save(os.path.join(save_dir, "var_nb.npy"), nb_list)

np.save(os.path.join(save_dir, "var_aberto_sz.npy"), sz_list_aberto)
np.save(os.path.join(save_dir, "var_aberto_nb.npy"), nb_list_aberto)

# salvar histórico de args
pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_T.csv"), index=False)

print(f"✅ Tudo salvo em: {save_dir}")