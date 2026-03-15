# Advanced Optimization: JIT, Gradients, and PyTorch Optimizers

This guide demonstrates how to accelerate the propagator with TorchScript, compute Jacobians, and use standard PyTorch optimizers for orbit determination.

## JIT Compilation

Compile the propagator and your forward model for maximum performance:

```python
compiled_propagate = torch.jit.script(sgp4_propagate)

@torch.jit.script
def compiled_forward_model(x: torch.Tensor, tsinces: torch.Tensor,
                           st_pos: torch.Tensor, st_vel: torch.Tensor,
                           center_freq: float, ...) -> torch.Tensor:
    # ... (full forward pass)
    return y_corrected
```

The compiled version eliminates Python overhead and is used in `example.py` and `benchmark.py`.

## Automatic Differentiation

Use `torch.func.jacfwd` to compute the Jacobian of your forward model w.r.t. the estimated parameters:

```python
from torch.func import jacfwd

def residual(x):
    return forward_model(x) - y_obs

J = jacfwd(residual)(x)   # Jacobian matrix
```

You can then form the normal equations or use the Jacobian in custom optimization loops.

## Using PyTorch Optimizers

Instead of the Gauss‑Newton solver, you can minimize a loss function (e.g., mean squared error) with any PyTorch optimizer.

### Example: LBFGS

```python
x = x_init.clone().requires_grad_()
optimizer = torch.optim.LBFGS([x], lr=1.0, max_iter=20)

def closure():
    optimizer.zero_grad()
    y_pred = forward_model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y_obs)
    loss.backward()
    return loss

for _ in range(5):
    optimizer.step(closure)
```

### Example: Adam

```python
x = x_init.clone().requires_grad_()
optimizer = torch.optim.Adam([x], lr=1e-2)
for step in range(200):
    optimizer.zero_grad()
    y_pred = forward_model(x)
    loss = torch.nn.functional.mse_loss(y_pred, y_obs)
    loss.backward()
    optimizer.step()
```

These approaches give you full flexibility to add regularisation, handle non‑Gaussian noise, or combine multiple measurement types.