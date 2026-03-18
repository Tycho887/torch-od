import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre, eval_legendre
from numba import njit
import itertools
import matplotlib.pyplot as plt

def get_legendre_deriv(n, x):
    if n == 0:
        return np.zeros_like(x)
    return n / (x**2 - 1) * (x * eval_legendre(n, x) - eval_legendre(n - 1, x))

def precompute_1d_integrals(max_deg, max_power):
    n_points = int(np.ceil((2 * max_deg + max_power + 1) / 2)) + 1 
    nodes, weights = roots_legendre(n_points)
    
    M = np.zeros((max_deg + 1, max_deg + 1, max_power + 1))
    D = np.zeros((max_deg + 1, max_deg + 1, max_power + 1))
    
    for a in range(max_deg + 1):
        Pa = eval_legendre(a, nodes)
        dPa = get_legendre_deriv(a, nodes)
        for b in range(max_deg + 1):
            Pb = eval_legendre(b, nodes)
            for p in range(max_power + 1):
                xp = nodes**p
                if (a + b + p) % 2 == 0:
                    M[a, b, p] = np.sum(weights * Pa * Pb * xp)
                if (a - 1 + b + p) % 2 == 0 and a > 0:
                    D[a, b, p] = np.sum(weights * dPa * Pb * xp)
    return M, D

@njit
def assemble_koopman_matrix(basis_degrees, terms, M, D):
    b, d = basis_degrees.shape
    K = np.zeros((b, b))
    
    # Precompute squared norms for orthonormal projection
    norm_sq = np.ones(b)
    for j in range(b):
        for dim in range(d):
            norm_sq[j] *= 2.0 / (2.0 * basis_degrees[j, dim] + 1.0)
            
    for i in range(b):
        for j in range(b):
            k_val = 0.0
            for t in range(terms.shape[0]):
                target_var = int(terms[t, 0])
                coeff = terms[t, 1]
                powers = terms[t, 2:]
                
                term_val = coeff
                is_zero = False
                
                for dim in range(d):
                    deg_i = basis_degrees[i, dim]
                    deg_j = basis_degrees[j, dim]
                    power = int(powers[dim])
                    
                    if dim == target_var:
                        if (deg_i - 1 + deg_j + power) % 2 != 0 or deg_i == 0:
                            is_zero = True
                            break
                        term_val *= D[deg_i, deg_j, power]
                    else:
                        if (deg_i + deg_j + power) % 2 != 0:
                            is_zero = True
                            break
                        term_val *= M[deg_i, deg_j, power]
                        
                    if term_val == 0:
                        is_zero = True
                        break
                        
                if not is_zero:
                    k_val += term_val
            
            K[i, j] = k_val / norm_sq[j]
    return K

# ==========================================
# 3. System Definition & Execution
# ==========================================
d = 8
J2 = 1.08262668e-3

terms = np.array([
    [0, -1.0,     0, 1, 0, 0, 0, 0, 0, 0],
    [0, -3*J2,    2, 0, 1, 1, 3, 0, 0, 0],
    [0, -9*J2,    1, 0, 1, 1, 4, 0, 0, 0],
    [0, -6*J2,    0, 0, 1, 1, 5, 0, 0, 0],
    [1, 1.0,      1, 0, 0, 0, 0, 0, 0, 0],
    [1, 4.5*J2,   2, 0, 2, 0, 3, 0, 0, 0],
    [1, -1.5*J2,  2, 0, 0, 0, 3, 0, 0, 0],
    [1, 9.0*J2,   1, 0, 2, 0, 4, 0, 0, 0],
    [1, -3.0*J2,  1, 0, 0, 0, 4, 0, 0, 0],
    [1, 4.5*J2,   0, 0, 2, 0, 5, 0, 0, 0],
    [1, -1.5*J2,  0, 0, 0, 0, 5, 0, 0, 0],
    [2, 1.0,      0, 0, 0, 1, 0, 0, 0, 0],
    [3, -1.0,     0, 0, 1, 0, 0, 0, 0, 0],
    [3, -3*J2,    1, 0, 1, 0, 3, 0, 0, 2],
    [3, -3*J2,    0, 0, 1, 0, 4, 0, 0, 2],
    [4, 3*J2,     1, 0, 1, 1, 4, 0, 0, 0],
    [4, 3*J2,     0, 0, 1, 1, 5, 0, 0, 0],
    [5, -3*J2,    1, 0, 2, 0, 0, 0, 1, 0],
    [5, -3*J2,    0, 0, 2, 0, 1, 0, 1, 0],
    [6, 12*J2,    1, 0, 1, 1, 3, 0, 1, 0],
    [6, 12*J2,    0, 0, 1, 1, 4, 0, 1, 0],
    [6, 6*J2,     1, 0, 1, 0, 0, 0, 2, 1],
    [6, 6*J2,     0, 0, 1, 0, 1, 0, 2, 1],
    [7, 3*J2,     1, 0, 1, 1, 3, 0, 0, 1],
    [7, 3*J2,     0, 0, 1, 1, 4, 0, 0, 1]
])

def exact_dynamics(t, x):
    """Computes exact ODE dynamics directly from the terms array."""
    dx = np.zeros(d)
    for t_idx in range(terms.shape[0]):
        target = int(terms[t_idx, 0])
        val = terms[t_idx, 1]
        for dim in range(d):
            val *= (x[dim] ** terms[t_idx, 2+dim])
        dx[target] += val
    return dx

# Make sure values are somewhat small/normalized so they stay in [-1, 1]
x0 = np.array([0.1, 0.05, 0.1, -0.05, 0.2, 0.1, 0.0, 0.1])
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 200)

sol_exact = solve_ivp(exact_dynamics, t_span, x0, t_eval=t_eval, method='DOP853', rtol=1e-13, atol=1e-13)

# ==========================================
# 4. Loop Orders to Compare Error
# ==========================================
for max_order in [3, 5, 7, 9]:
    basis = []
    for comb in itertools.product(range(max_order + 1), repeat=d):
        total_degree = sum(comb)
        if total_degree <= max_order and total_degree % 2 != 0:
            basis.append(comb)

    basis_degrees = np.array(basis)
    b = len(basis)
    print(f"\nEvaluating Koopman Operator (Order {max_order}, Basis dimension: {b})")

    max_deg = np.max(basis_degrees)
    max_power = int(np.max(terms[:, 2:]))
    M, D = precompute_1d_integrals(max_deg, max_power)

    K = assemble_koopman_matrix(basis_degrees, terms, M, D)

    def koopman_dynamics(t, L):
        return K @ L

    L0 = np.zeros(b)
    for i, degrees in enumerate(basis_degrees):
        val = 1.0
        for dim in range(d):
            val *= eval_legendre(degrees[dim], x0[dim])
        L0[i] = val

    sol_koopman = solve_ivp(koopman_dynamics, t_span, L0, t_eval=t_eval, method='RK45')

    # Index for Lambda (degree 1 in dim 0, degree 0 elsewhere)
    idx_Lambda = basis.index((1, 0, 0, 0, 0, 0, 0, 0))
    Lambda_koopman = sol_koopman.y[idx_Lambda, :]
    
    error = np.linalg.norm(sol_exact.y[0] - Lambda_koopman) / np.linalg.norm(sol_exact.y[0])
    print(f"L2 Error for Lambda: {error:.6f}")
    
    plt.plot(t_eval, Lambda_koopman, label=f'Koopman Order {max_order}')

plt.plot(t_eval, sol_exact.y[0], 'k--', label='Exact ODE')
plt.title('Koopman Approximation of J2 Perturbation (Lambda)')
plt.xlabel('Time')
plt.ylabel('Lambda')
plt.legend()
plt.show()