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
        C[i] = aux
    return C


def entanglement(states):
    e_q_f = []
    for i, state in enumerate(states):
        e_q_f.append(qt.negativity(state, 0, method="tracenorm", logarithmic=True))
    return e_q_f


def wigner_negativity(states, xvec, pvec, one_mode=False):
    ns = []
    if one_mode:
        w = qt.wigner(states, xvec, pvec)
        waux = integrate.simpson(abs(w), xvec)
        aux0 = integrate.simpson(waux, pvec)
        ns = 0.5 * (aux0 - 1)
    else:
        for _, state in enumerate(states):
            w = qt.wigner(state.ptrace(1), xvec, pvec)
            waux = integrate.simpson(abs(w), xvec)
            aux0 = integrate.simpson(waux, pvec)
            aux = 0.5 * (aux0 - 1)
            ns.append(aux)
    return ns


# ============================================================
# QUANTUM MAGIC: Stabilizer Renyi Entropy M2 for one qubit
# ============================================================
# For the reduced qubit state rho_q,
#   S2 = 1 + |<X>|^2 + |<Y>|^2 + |<Z>|^2
#   S4 = 1 + |<X>|^4 + |<Y>|^4 + |<Z>|^4
#   M2 = log(S2 / S4)
# This is the same convention used in the notebook sent with the project.


def stabilizer_renyi_entropy_qubit_from_expectations(px, py, pz, eps=1e-15):
    """
    Stabilizer Renyi entropy of order 2 for a single-qubit state.

    Parameters
    ----------
    px, py, pz : float or array-like
        Pauli expectation values <X>, <Y>, <Z> of the reduced qubit.
    eps : float
        Numerical floor to avoid log division problems.

    Returns
    -------
    M2 : ndarray or float
        Quantum magic/nonstabilizerness measured by SRE-2.
    S2 : ndarray or float
        Sum of squared Pauli expectations including identity.
    S4 : ndarray or float
        Sum of fourth powers of Pauli expectations including identity.
    purity : ndarray or float
        Qubit purity reconstructed from the Bloch vector:
        Tr(rho^2) = (1 + |r|^2)/2.
    magic_proxy_l1 : ndarray or float
        The old notebook diagnostic: 1 + |<X>| + |<Y>| + |<Z>|.
        This is not the SRE, but is useful for comparison.
    """
    px = np.asarray(px, dtype=np.complex128)
    py = np.asarray(py, dtype=np.complex128)
    pz = np.asarray(pz, dtype=np.complex128)

    ax = np.abs(px)
    ay = np.abs(py)
    az = np.abs(pz)

    S2 = 1.0 + ax**2 + ay**2 + az**2
    S4 = 1.0 + ax**4 + ay**4 + az**4

    ratio = np.maximum(S2, eps) / np.maximum(S4, eps)
    M2 = np.log(ratio)

    # Remove tiny negative values from numerical roundoff only.
    M2 = np.where((M2 < 0.0) & (M2 > -1e-12), 0.0, M2)

    purity = 0.5 * S2
    magic_proxy_l1 = 1.0 + ax + ay + az
    return np.real_if_close(M2), np.real_if_close(S2), np.real_if_close(S4), np.real_if_close(purity), np.real_if_close(magic_proxy_l1)


def qubit_pauli_expectations_from_states(states, subsystem=0):
    """
    Compute <X>, <Y>, <Z> for the reduced qubit of each state.

    The global state can be a ket or a density matrix. The default
    subsystem=0 matches the current Jaynes-Cummings code, where the first
    subsystem is the two-level atom/qubit and the second is the field.
    """
    px = []
    py = []
    pz = []

    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()

    for state in states:
        rho_q = state.ptrace(subsystem)
        px.append(qt.expect(X, rho_q))
        py.append(qt.expect(Y, rho_q))
        pz.append(qt.expect(Z, rho_q))

    return np.array(px), np.array(py), np.array(pz)


def quantum_magic(states, subsystem=0):
    """
    Compute SRE-2 quantum magic for a list of states.

    Returns only M2 to keep the interface similar to coerence(),
    entanglement(), and wigner_negativity(). If you want S2/S4/purity,
    use quantum_magic_details().
    """
    px, py, pz = qubit_pauli_expectations_from_states(states, subsystem=subsystem)
    M2, _, _, _, _ = stabilizer_renyi_entropy_qubit_from_expectations(px, py, pz)
    return M2


def quantum_magic_details(states, subsystem=0):
    """
    Compute SRE-2 quantum magic and diagnostic quantities.

    Returns
    -------
    dict with keys:
        M2, px, py, pz, S2, S4, purity, magic_proxy_l1
    """
    px, py, pz = qubit_pauli_expectations_from_states(states, subsystem=subsystem)
    M2, S2, S4, purity, magic_proxy_l1 = stabilizer_renyi_entropy_qubit_from_expectations(px, py, pz)
    return {
        "M2": M2,
        "px": px,
        "py": py,
        "pz": pz,
        "S2": S2,
        "S4": S4,
        "purity": purity,
        "magic_proxy_l1": magic_proxy_l1,
    }
