import numpy as np
import qutip as qt
from tqdm import tqdm
from quantum.hamiltonian import g_t, h_closed, h_open
import pandas as pd
import matplotlib.pyplot as plt
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from quantum.non_classicality import coerence
from scipy import integrate
from fractions import Fraction
import os
import json
from datetime import datetime

def make_exp_folder(base_name="coerence_exp", root="results"):
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


'''
#Linear e exp
args = {
    'g0': 1,
    'eta': 1,
    'w': 0,
    'kappa': 1e-1,
    'gamma': 0,
    'gamma_phi': 1e-2,
    'coupling': 'linear',
    'phi': 0.5#np.log(1e-3)
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


extra = 'gauss6'

#======= LINEAR/ EXP =======
#wmax = 1/100
#w_list = np.linspace(-1/100, wmax, 100)

#======= GAUSS ========
#epmax = 1
Tmax = 35
T_list = np.linspace(15, Tmax, 100, endpoint=True)
#ep_list = np.linspace(1e-5, epmax, 100, endpoint=True)
# ==========================
# INITIAL STATE
# ==========================
alpha = np.sqrt(5)
phi0 = qt.tensor(qt.basis(N, 0) + qt.basis(N, 1), qt.coherent(Nb, alpha)).unit()
# phi0 = (tensor(basis(N,0), coherent(Nb, -alpha)) + tensor(basis(N,1), coherent(Nb, alpha))).unit()

# ==========================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ==========================
save_dir = make_exp_folder(base_name=f"coerence_{args['coupling']}", root="results")

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
    f.write(f"len(t): {len(t)}\n")

    #f.write(f'epmax: {float(epmax)}\n') #gauss
    f.write(f'epmax: {float(Tmax)}\n') #gauss
    f.write(f"len(eplist): {len(T_list)}\n\n") #gauss
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
    f.write(f"  T_min: {float(np.min(T_list))}\n") #gauss
    f.write(f"  T_max: {float(np.max(T_list))}\n") #gauss
    f.write(f"  first_5: {T_list[:5].tolist()}\n") #gauss
    f.write(f"  last_5: {T_list[-5:].tolist()}\n") #gauss
    #f.write(f"  w_min: {float(np.min(w_list))}\n") # lin/exp
    #f.write(f"  w_max: {float(np.max(w_list))}\n") # lin/exp
    #f.write(f"  first_5: {w_list[:5].tolist()}\n") # lin/exp
    #f.write(f"  last_5: {w_list[-5:].tolist()}\n") # lin/exp

# salvar arrays completos e args em formato carregável
np.save(os.path.join(save_dir, "t.npy"), t)
#np.save(os.path.join(save_dir, "w_list.npy"), w_list) # lin/exp
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

C_const_aberto = coerence(sol_const_aberto.states)
C_const = coerence(sol_const.states)

# salvar const / const_aberto
np.save(os.path.join(save_dir, "const_aberto.npy"), np.array(C_const_aberto))
np.save(os.path.join(save_dir, "const.npy"), np.array(C_const))

# ==========================
# OPEN HAMILTONIAN (VAR over w)
# ==========================
c_list_aberto = []
c_list = []
args_per_w = []

#for w in tqdm(w_list): # lin/exp
#    args['w'] = float(w) # lin/exp
for ep in tqdm(T_list): #gauss
    args["T"] = float(ep) #gauss
    H = h_open(b, sp, sm)

    sol_var_aberto = solve(H, state0, t, c_ops, obs_list, args)
    sol_var = solve(H, state0, t, None, obs_list, args)

    C_var_aberto = coerence(sol_var_aberto.states)
    C_var = coerence(sol_var.states)

    c_list.append(C_var)
    c_list_aberto.append(C_var_aberto)

    # guarda o args real usado em cada w
    args_per_w.append({**args})

# salvar var / var_aberto (listas completas)
np.save(os.path.join(save_dir, "var_aberto.npy"), np.array(c_list_aberto, dtype=object))
np.save(os.path.join(save_dir, "var.npy"), np.array(c_list, dtype=object))

# salvar histórico de args (inclui w em cada passo)
#pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_w.csv"), index=False)
pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_T.csv"), index=False)

print(f"✅ Tudo salvo em: {save_dir}")
