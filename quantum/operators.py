import qutip as qt
import numpy as np

def get_operators(N=2, Nb=25):
    sz = qt.tensor(qt.sigmaz(), qt.qeye(Nb)) # Pauli-Z
    sp = qt.tensor(qt.sigmap(), qt.qeye(Nb)) # |1><0|
    sm = qt.tensor(qt.sigmam(), qt.qeye(Nb)) # |0><1|
    b = qt.tensor(qt.qeye(N), qt.destroy(Nb)) # Annihilation operator
    nb = b.dag() * b                 # Number operator
    I = qt.tensor(qt.qeye(N), qt.qeye(Nb))    # Identity operator
    return sz, sp, sm, b, nb, I

def get_collapse(args, sm, sz, b):
    return [np.sqrt(args['gamma']) * sm, np.sqrt(args['gamma_phi']) * sz, np.sqrt(args['kappa']) * b]
