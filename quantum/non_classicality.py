import qutip as qt
import numpy as np
from scipy import integrate
from utils.utils import diagonaliza

def coerence(states):
    C = np.zeros(len(states))
    for i in range(len(states)):
        estado = (states[i]).ptrace(0)
        estado = estado * estado.dag()
        estado_d = diagonaliza(estado)
        aux = qt.entropy_vn(estado_d, base=np.e) - qt.entropy_vn(estado, base=np.e)
        C[i]=aux
    return C

def entanglement(states):
    e_q_f = []
    for i,state in enumerate(states):
        e_q_f.append(qt.negativity(state, 0, method='tracenorm', logarithmic=True))
    return  e_q_f

def wigner_negativity(states, xvec, pvec, one_mode=False):
    ns = []
    if one_mode:
        w = qt.wigner(states, xvec, pvec)
        waux = integrate.simpson(abs(w), xvec)
        aux0 = integrate.simpson(waux, pvec)
        #if aux0 <0: aux0 = 0
        ns = 0.5*(aux0  - 1)
    else:
        for _, state in enumerate(states):
            w = qt.wigner(state.ptrace(1), xvec, pvec)
            waux = integrate.simpson(abs(w), xvec)
            aux0 = integrate.simpson(waux, pvec)
            #if aux0 <0: aux0 = 0
            aux = 0.5*(aux0  - 1)
            ns.append(aux)
    return ns

