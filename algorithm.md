# Batch Processor with Consider Covariance Analysis

## 1. System Definitions and Inputs

We define our full state space size as $N_{total} = 20$. We logically partition this into two sets:
*   **$n$ Estimated Parameters ($x$):** The parameters we are solving for (e.g., 6 Orbital Elements + 3 Drag + Frequency + Bias).
*   **$q$ Consider Parameters ($c$):** The parameters we are *not* solving for, but which have known uncertainties (e.g., station coordinates, atmospheric model biases).

**Inputs:**
*   $\mathbf{y}$: Residual vector ($10,000 \times 1$). The difference between Observed Doppler and Computed Doppler based on the current TLE.
*   $H_{total}$: The full Jacobian matrix ($10,000 \times 20$), computed via SIMD.
*   $R$: Measurement noise covariance ($10,000 \times 10,000$). Since Doppler noise is generally independent, this is a diagonal matrix where $R_{ii} = \sigma_{Doppler}^2$.
*   $\bar{P}_{cc}$: The *a priori* covariance of the consider parameters ($q \times q$).
*   $\bar{P}_{x}$: (Optional) The *a priori* covariance of the estimated parameters ($n \times n$), if using prior information.

---

## 2. Partitioning the Jacobian

Since $H_{total}$ is computed in one pass, we logically split it into two sub-matrices:

$$
H_{total} = \left[ H_x \;|\; H_c \right]
$$

*   $H_x$: The first $n$ columns (sensitivities of Doppler to Estimate parameters).
*   $H_c$: The last $q$ columns (sensitivities of Doppler to Consider parameters).

---

## 3. The Maximum Likelihood Step (Estimation)

To satisfy the Maximum Likelihood condition for Gaussian noise, we solve the **Weighted Least Squares** problem. We must accumulate the "Normal Equations" to solve for the state update $\hat{x}$.

### A. Compute Normal Matrices
Instead of handling the massive $10,000 \times 10,000$ matrix $R$ explicitly, we use vector operations. Since $R$ is diagonal with value $\sigma^2$:

1.  **Information Matrix ($M_{xx}$)** ($n \times n$):
    $$M_{xx} = H_x^T R^{-1} H_x + \bar{P}_x^{-1}$$
    *Implementation:* This is the weighted dot product of the columns of $H_x$.

2.  **Right-Hand Side ($N_x$)** ($n \times 1$):
    $$N_x = H_x^T R^{-1} \mathbf{y} + \bar{P}_x^{-1}\bar{x}$$
    *Implementation:* Weighted dot product of $H_x$ columns against the residual vector $\mathbf{y}$.

### B. Solve for the Update
Solve the linear system for the optimal correction $\hat{x}$:

$$\hat{x} = M_{xx}^{-1} N_x$$

Add this $\hat{x}$ to your current State estimate. *Note: You typically iterate this step (re-computing residuals and H) until $\hat{x}$ is small, satisfying the non-linear least squares problem.*

---

## 4. The Consider Covariance Step (Analysis)

Once the estimator has converged, we calculate the **Consider Covariance**. This tells us how the uncertainty in the "Consider" parameters ($c$) corrupts our estimate of $x$.

### A. Compute Cross-Coupling Matrix ($M_{xc}$)
We need the relationship between the Estimate and Consider parts of the Jacobian:

$$M_{xc} = H_x^T R^{-1} H_c$$
*(Dimension: $n \times q$)*

### B. Compute Sensitivity Matrix ($S_{xc}$)
This matrix defines how a change in a consider parameter would shift our estimated state:

$$S_{xc} = - M_{xx}^{-1} M_{xc}$$
*(Dimension: $n \times q$)*

### C. Compute Total Covariance ($P_{total}$)
The final covariance matrix for your estimated parameters is the sum of the noise-only covariance ($P_{noise}$) and the consider-induced covariance ($P_{consider}$):

1.  **Noise Component:** $P_{noise} = M_{xx}^{-1}$
2.  **Consider Component:** $P_{consider} = S_{xc} \bar{P}_{cc} S_{xc}^T$
3.  **Total:**
    $$P_{total} = P_{noise} + P_{consider}$$

---

## 5. Summary of Algorithm Flow

1.  **SIMD Compute:** Calculate $H_{total}$ ($10k \times 20$) and residuals $\mathbf{y}$ ($10k \times 1$).
2.  **Split:** View $H_{total}$ as $[H_x | H_c]$.
3.  **Accumulate (Estimate):** $M_{xx} = H_x^T W H_x$ and $N_x = H_x^T W \mathbf{y}$.
4.  **Accumulate (Consider):** $M_{xc} = H_x^T W H_c$.
5.  **Solve:** $\hat{x} = M_{xx}^{-1} N_x$.
6.  **Sensitivity:** $S_{xc} = -M_{xx}^{-1} M_{xc}$.
7.  **Final Covariance:** $P_{total} = M_{xx}^{-1} + S_{xc} \bar{P}_{cc} S_{xc}^T$.
