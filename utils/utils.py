import numpy as np
import qutip as qt

def diagonaliza(state):
    matriz_diag = np.diag(np.diag(state.full()))
    return qt.Qobj(matriz_diag)