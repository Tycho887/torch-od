import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre, eval_legendre
from scipy.linalg import expm
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

# Base terms for J2 General Formulation
terms = np.array([
    # [Target, Coeff, L, eta, s, gamma, kappa, beta, chi, rho]
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

# Define the Molniya Initial Conditions mapped to the modified 8D state
# (These represent a highly eccentric orbit, e=0.74, inc=63.4 deg)
# Note: For strict verification, these would be derived directly from Eq 27 and 30.
x0_physical = np.array([0.15, 0.0, 0.0, 0.45, 0.05, 0.0, 0.01, 0.45])

# Define maximum absolute expected bounds for the orbit to map to [-1, 1]
# Values must be chosen so that np.max(abs(x(theta))) < bounds
bounds = np.array([
    0.5,    # max Lambda
    1.0,    # max eta
    1.0,    # max s (sin latitude is inherently bound by 1)
    1.0,    # max gamma
    0.1,    # max kappa
    np.pi,  # max beta
    0.1,    # max chi
    1.0     # max rho
])

# Scale the initial conditions
x0_normalized = x0_physical / bounds

# Create the Normalized Terms Array
# If dx/dtheta = c * (y^p * z^q), then for scaled variables u = x/Sx, v = y/Sy, w = z/Sz:
# S_x * du/dtheta = c * (S_y*v)^p * (S_z*w)^q 
# du/dtheta = [c * (S_y^p * S_z^q) / S_x] * v^p * w^q
norm_terms = terms.copy()
for i in range(terms.shape[0]):
    target = int(terms[i, 0])
    coeff = terms[i, 1]
    powers = terms[i, 2:]
    
    scale_factor = np.prod(bounds ** powers) / bounds[target]
    norm_terms[i, 1] = coeff * scale_factor

def exact_dynamics_normalized(theta, x_norm):
    """Computes exact ODE dynamics using the scaled domain."""
    dx_norm = np.zeros(d)
    for t_idx in range(norm_terms.shape[0]):
        target = int(norm_terms[t_idx, 0])
        val = norm_terms[t_idx, 1]
        for dim in range(d):
            val *= (x_norm[dim] ** norm_terms[t_idx, 2+dim])
        dx_norm[target] += val
    return dx_norm

# Integrate over regularized time theta (e.g., 0 to 2*pi roughly corresponds to one orbit)
theta_span = (0, 2 * np.pi)
theta_eval = np.linspace(theta_span[0], theta_span[1], 200)

# Exact baseline integration (using stringent DOP853 parameters)
sol_exact_norm = solve_ivp(
    exact_dynamics_normalized, 
    theta_span, 
    x0_normalized, 
    t_eval=theta_eval, 
    method='DOP853', 
    rtol=1e-13, 
    atol=1e-13
)

# Extract physical truth for Lambda
Lambda_exact_physical = sol_exact_norm.y[0] * bounds[0]

# ==========================================
# 4. Koopman Analytical Propagation
# ==========================================
for max_order in [3, 5, 7]:
    basis = []
    for comb in itertools.product(range(max_order + 1), repeat=d):
        total_degree = sum(comb)
        if total_degree <= max_order and total_degree % 2 != 0:
            basis.append(comb)

    basis_degrees = np.array(basis)
    b = len(basis)
    print(f"\nEvaluating Koopman Operator (Order {max_order}, Basis dimension: {b})")

    max_deg = np.max(basis_degrees)
    max_power = int(np.max(norm_terms[:, 2:]))
    M, D = precompute_1d_integrals(max_deg, max_power)

    K = assemble_koopman_matrix(basis_degrees, norm_terms, M, D)

    # Lift normalized initial conditions to extended space
    L0 = np.zeros(b)
    for i, degrees in enumerate(basis_degrees):
        val = 1.0
        for dim in range(d):
            val *= eval_legendre(degrees[dim], x0_normalized[dim])
        L0[i] = val

    # Analytical propagation using Matrix Exponential: L(theta) = exp(K*theta) @ L0
    idx_Lambda = basis.index((1, 0, 0, 0, 0, 0, 0, 0))
    Lambda_koopman_physical = np.zeros_like(theta_eval)

    # Note: For large matrices, repeated expm can be slow. 
    # Eigendecomposition (K = V E V^-1) is much faster for scaling to 11th order.
    for step, theta in enumerate(theta_eval):
        L_theta = expm(K * theta) @ L0
        # Map back to physical space
        Lambda_koopman_physical[step] = L_theta[idx_Lambda] * bounds[0] 
    
    error = np.linalg.norm(Lambda_exact_physical - Lambda_koopman_physical) / np.linalg.norm(Lambda_exact_physical)
    print(f"L2 Error for Lambda: {error:.8f}")
    
    plt.plot(theta_eval, Lambda_koopman_physical, label=f'Koopman Order {max_order}')

plt.plot(theta_eval, Lambda_exact_physical, 'k--', label='Exact ODE')
plt.title('Koopman Approximation of J2 Perturbation (Lambda) - Molniya')
plt.xlabel('Regularized Time (Theta)')
plt.ylabel('Lambda (Physical)')
plt.legend()
plt.show()