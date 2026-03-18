import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre, eval_legendre
from numba import njit
import itertools
import matplotlib.pyplot as plt

# ==========================================
# 1. 1D Quadrature & Basis Precomputation
# ==========================================

def get_legendre_deriv(n, x):
    """Evaluates the derivative of the n-th Legendre polynomial."""
    if n == 0:
        return np.zeros_like(x)
    # Recurrence relation for Legendre derivative
    return n / (x**2 - 1) * (x * eval_legendre(n, x) - eval_legendre(n - 1, x))

def precompute_1d_integrals(max_deg, max_power):
    """
    Precomputes 1D integrals using Gauss-Legendre quadrature.
    """
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
                
                # Standard mass matrix M: parity is a + b + p
                if (a + b + p) % 2 == 0:
                    M[a, b, p] = np.sum(weights * Pa * Pb * xp)
                
                # Derivative matrix D: parity is (a - 1) + b + p
                if (a - 1 + b + p) % 2 == 0 and a > 0:
                    D[a, b, p] = np.sum(weights * dPa * Pb * xp)
                
    return M, D

@njit
def assemble_koopman_matrix(basis_degrees, terms, M, D):
    """
    Assembles the Koopman matrix using precomputed 1D integrals
    and explicitly normalizes the orthogonal basis.
    """
    b, d = basis_degrees.shape
    K = np.zeros((b, b))
    
    for i in range(b):
        for j in range(b):
            k_val = 0.0
            
            # Compute squared norm of the receiving basis function L_j
            norm_sq_j = 1.0
            for dim in range(d):
                norm_sq_j *= 2.0 / (2.0 * basis_degrees[j, dim] + 1.0)
            
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
                        # Advected variable check (Derivative parity)
                        if (deg_i - 1 + deg_j + power) % 2 != 0 or deg_i == 0:
                            is_zero = True
                            break
                        term_val *= D[deg_i, deg_j, power]
                    else:
                        # Standard variable check
                        if (deg_i + deg_j + power) % 2 != 0:
                            is_zero = True
                            break
                        term_val *= M[deg_i, deg_j, power]
                        
                    if term_val == 0:
                        is_zero = True
                        break
                        
                if not is_zero:
                    k_val += term_val
            
            # Normalize the projection
            K[i, j] = k_val / norm_sq_j
            
    return K

# ==========================================
# 3. System Definition & Execution
# ==========================================

# Dynamics for Duffing Oscillator: dx1/dt = x2, dx2/dt = -x1 - eps*x1^3
eps = 0.5
d = 2
max_order = 9

# Define terms: [target_var, coefficient, power_x1, power_x2]
# Term 1: dx1/dt = 1.0 * x1^0 * x2^1
# Term 2: dx2/dt = -1.0 * x1^1 * x2^0
# Term 3: dx2/dt = -eps * x1^3 * x2^0
terms = np.array([
    [0, 1.0, 0, 1],
    [1, -1.0, 1, 0],
    [1, -eps, 3, 0]
])

# Generate Basis Functions (combinations where sum of degrees <= max_order)
basis = [comb for comb in itertools.product(range(max_order + 1), repeat=d) if sum(comb) <= max_order]
basis_degrees = np.array(basis)
b = len(basis)

# Precompute 1D Integrals
max_deg = np.max(basis_degrees)
max_power = int(np.max(terms[:, 2:]))
M, D = precompute_1d_integrals(max_deg, max_power)

# Assemble Matrix
K = assemble_koopman_matrix(basis_degrees, terms, M, D)

# ==========================================
# 4. Simulation and Comparison
# ==========================================

def exact_dynamics(t, x):
    return [x[1], -x[0] - eps * x[0]**3]

x0 = np.array([0.5, 0.0]) # Initial condition
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 200)

# Solve Exact ODE
sol_exact = solve_ivp(exact_dynamics, t_span, x0, t_eval=t_eval, method='RK45')

# Solve Koopman Linear System: dL/dt = K^T L  (Using transpose because of convention in eq 10)
def koopman_dynamics(t, L):
    return K @ L

# Lift initial conditions to extended space
L0 = np.zeros(b)
for i, degrees in enumerate(basis_degrees):
    val = 1.0
    for dim in range(d):
        val *= eval_legendre(degrees[dim], x0[dim])
    L0[i] = val

sol_koopman = solve_ivp(koopman_dynamics, t_span, L0, t_eval=t_eval, method='RK45')

# Extract x1 and x2 from the Koopman state (they correspond to basis degrees [1,0] and [0,1])
idx_x1 = basis.index((1, 0))
idx_x2 = basis.index((0, 1))

x1_koopman = sol_koopman.y[idx_x1, :]
x2_koopman = sol_koopman.y[idx_x2, :]

print(sol_exact.y[0])
print(x1_koopman)

# Output error metrics
error_x1 = np.linalg.norm(sol_exact.y[0] - x1_koopman) / np.linalg.norm(sol_exact.y[0])
error_x2 = np.linalg.norm(sol_exact.y[1] - x2_koopman) / np.linalg.norm(sol_exact.y[1])

print(f"L2 Error in x1: {error_x1:.4f}")
print(f"L2 Error in x2: {error_x2:.4f}")