from tqdm import tqdm
import numpy as np
from sympy import *
from functools import lru_cache

x = symbols('x', real=True)

def raise_hyperbolic(n, psi, lam):
    if n < 0:
        return psi
    result = -I * diff(psi, x) + I * (lam - n) * tanh(x) * psi
    return raise_hyperbolic(n - 1, result, lam)

def get_aux_ground_state_hyperbolic(n, lam):
    norm = sqrt(gamma(lam - n + Rational(1, 2)) / (sqrt(pi) * gamma(lam - n)))
    return norm * sech(x) ** (lam - n)

@lru_cache(maxsize=128)
def get_eigenstate_hyperbolic(n, lam):
    psi_0 = get_aux_ground_state_hyperbolic(n, lam)
    En = n * (2 * lam - n)
    norm = sqrt(gamma(2 * lam - 2*n + 1) / (gamma(n+1) * gamma(2 * lam - n + 1)))
    return (En, norm * raise_hyperbolic(n - 1, psi_0, lam))


def change_basis_hyperbolic(domain, psi_0, lam_val):
    basis = []
    for n in tqdm(range(int(lam_val) + 1)):
        if n == lam_val:
            continue
        energy, psi_n = get_eigenstate_hyperbolic(n, lam_val)

        def psi_numeric(xval):
            return complex(psi_n.subs(x, xval).evalf())
        psi_n_fn = np.vectorize(psi_numeric, otypes=[complex])
        
        psi_n_vals = psi_n_fn(domain)
        coeff = np.trapz(np.conjugate(psi_n_vals) * psi_0, x=domain)
        basis.append((energy, coeff, psi_n_vals))
    return basis

if __name__ == "__main__":
    # integrate every hyperbolic eigenstate over a domain to see if its abs(psi)**2 == 1
    domain = np.linspace(-10, 10, 1000)
    lam = 8
    eigenstates = [get_eigenstate_hyperbolic(n, lam) for n in range(lam)]
    for en, psi in eigenstates:
        psi_fn = np.vectorize(lambda xval: complex(psi.subs(x, xval).evalf()), otypes=[complex])
        vals = psi_fn(domain)
        print(en, np.trapz(np.conjugate(vals) * vals, x=domain))
