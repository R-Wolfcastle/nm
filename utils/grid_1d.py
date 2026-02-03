



# 1st party
from typing import Tuple
# 3rd party
import jax
import jax.numpy as jnp

# ------------------------------------------------------------
# Small shift helpers (zero-padded, “safe”)
# ------------------------------------------------------------
def _shift_right(a: jnp.ndarray, k: int = 1) -> jnp.ndarray:
    """y[i] = a[i- k], with zeros where i-k < 0"""
    return jnp.pad(a, (k, 0))[0:a.size]

def _shift_left(a: jnp.ndarray, k: int = 1) -> jnp.ndarray:
    """y[i] = a[i+ k], with zeros where i+k >= n"""
    return jnp.pad(a, (0, k))[k:]


# ------------------------------------------------------------
# Interpolation: cell-centre (CC) → face-centre (FC)
# ------------------------------------------------------------
def interp_cc_with_ghosts_to_fc_function_1d(n: int):
    """
    Input: var with 1 ghost on each end, shape (n+2,)
    Output: face-centred var, shape (n+1,), using simple arithmetic mean.
    """
    def _interp(var_with_ghosts: jnp.ndarray) -> jnp.ndarray:
        # Faces sit between consecutive cell-centres (including ghosts)
        # => there are len(var)-1 faces
        return 0.5 * (var_with_ghosts[1:] + var_with_ghosts[:-1])
    return jax.jit(_interp)

def interp_cc_to_fc_function_1d(n: int):
    """
    Input: interior var, shape (n,)
    Output: face-centred var, shape (n+1,)
      - interior faces: average neighbours
      - boundary faces: copy boundary cell (same convention as 2‑D counterpart)
    """
    def _interp(var: jnp.ndarray) -> jnp.ndarray:
        fc = jnp.zeros((n+1,), dtype=var.dtype)
        fc = fc.at[1:-1].set(0.5 * (var[1:] + var[:-1]))
        fc = fc.at[0].set(var[0])
        fc = fc.at[-1].set(var[-1])
        return fc
    return jax.jit(_interp)


# ------------------------------------------------------------
# Gradients
# ------------------------------------------------------------
def cc_gradient_function_1d(dx: float):
    """
    Central gradient at CC locations using 1 ghost each side:
      Input: var with ghosts, shape (n+2,)
      Output: dvar_dx at interior CC (n,)
    """
    def _grad(var_with_ghosts: jnp.ndarray) -> jnp.ndarray:
        return (var_with_ghosts[2:] - var_with_ghosts[:-2]) * (0.5 / dx)
    return jax.jit(_grad)

def fc_gradient_function_1d(dx: float):
    """
    Central difference on faces (requires CC array with ghosts):
      Input: var with ghosts, shape (n+2,)
      Output: dvar_dx on faces, shape (n+1,)
    """
    def _grad_faces(var_with_ghosts: jnp.ndarray) -> jnp.ndarray:
        # derivative across each face is forward - backward over dx
        return (var_with_ghosts[1:] - var_with_ghosts[:-1]) / dx
    return jax.jit(_grad_faces)


# ------------------------------------------------------------
# Ghost cells (Dirichlet/odd; Neumann/even; Periodic)
# ------------------------------------------------------------
def add_reflection_ghost_cells_1d(n: int):
    """
    'Reflection' (odd) ghosting appropriate for a Dirichlet‑type variable:
      u_ghost_left  = -u[0]
      u_ghost_right = -u[-1]
    """
    def _apply(u_int: jnp.ndarray) -> jnp.ndarray:
        u = jnp.zeros((n+2,), dtype=u_int.dtype)
        u = u.at[1:-1].set(u_int)
        u = u.at[0].set(-u[1])
        u = u.at[-1].set(-u[-2])
        return u
    return jax.jit(_apply)

def add_continuation_ghost_cells_1d(n: int):
    """
    'Continuation' (even) ghosting appropriate for a Neumann‑type variable:
      h_ghost_left  =  h[0]
      h_ghost_right =  h[-1]
    """
    def _apply(h_int: jnp.ndarray) -> jnp.ndarray:
        h = jnp.zeros((n+2,), dtype=h_int.dtype)
        h = h.at[1:-1].set(h_int)
        h = h.at[0].set(h[1])
        h = h.at[-1].set(h[-2])
        return h
    return jax.jit(_apply)

def add_periodic_ghost_cells_1d(n: int):
    """
    Periodic ghosting:
      u_ghost_left  = u[-1]
      u_ghost_right = u[0]
    """
    def _apply(u_int: jnp.ndarray) -> jnp.ndarray:
        u = jnp.zeros((n+2,), dtype=u_int.dtype)
        u = u.at[1:-1].set(u_int)
        u = u.at[0].set(u[-2])
        u = u.at[-1].set(u[1])
        return u
    return jax.jit(_apply)

def apply_scalar_ghost_cells_to_tuple_1d(scalar_ghost_function):
    """Convenience: apply the same scalar ghosting to several fields."""
    def _apply(*args: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        return tuple(scalar_ghost_function(a) for a in args)
    return jax.jit(_apply)


# ------------------------------------------------------------
# Simple 1-D morphology (for calving-front adjacency)
# ------------------------------------------------------------
def binary_dilation_1d(mask: jnp.ndarray) -> jnp.ndarray:
    """3-cell dilation with logical padding (1-D)."""
    left = _shift_right(mask, 1)
    right = _shift_left(mask, 1)
    return mask | left | right

def binary_erosion_1d(mask: jnp.ndarray) -> jnp.ndarray:
    """3-cell erosion with logical padding (1-D)."""
    left = _shift_right(mask, 1)
    right = _shift_left(mask, 1)
    return mask & left & right


# ------------------------------------------------------------
# 1-D nearest-neighbour and linear CF extrapolators
# ------------------------------------------------------------
def nn_extrapolate_over_cf_function_1d(thk: jnp.ndarray):
    """
    Nearest-neighbour extrapolation into ocean cells adjacent to ice.
    Picks the neighbour (left or right) with largest |value|.
    """
    ice = thk > 0

    def _extrap(cc_field: jnp.ndarray) -> jnp.ndarray:
        left_has = _shift_right(ice, 1)
        right_has = _shift_left(ice, 1)
        left_val = _shift_right(cc_field, 1)
        right_val = _shift_left(cc_field, 1)

        # choose side with larger |value| (only among available sides)
        both = left_has & right_has
        choose_left = jnp.where(
            both, jnp.abs(left_val) >= jnp.abs(right_val),
            left_has & ~right_has
        )
        picked = jnp.where(choose_left, left_val, right_val)

        # only overwrite in ocean cells adjacent to ice
        cf_adjacent = (~ice) & (left_has | right_has)
        return jnp.where(cf_adjacent, picked, cc_field)

    return jax.jit(_extrap)

def linear_extrapolate_over_cf_function_1d(thk: jnp.ndarray):
    """
    Linear extrapolation into ocean cells adjacent to ice using the direction
    with the longer contiguous run of ice (up to 2 cells, like your 2-D code):
      u_extrap = 2*u1 - u2  if two cells exist, else u1
    """
    ice = thk > 0

    def _extrap(cc_field: jnp.ndarray) -> jnp.ndarray:
        # availability (1 or 2 cells) on each side
        L1 = _shift_right(ice, 1)
        L2 = L1 & _shift_right(ice, 2)
        R1 = _shift_left(ice, 1)
        R2 = R1 & _shift_left(ice, 2)

        # values
        uL1, uL2 = _shift_right(cc_field, 1), _shift_right(cc_field, 2)
        uR1, uR2 = _shift_left(cc_field, 1), _shift_left(cc_field, 2)

        scoreL = L1.astype(jnp.int32) + L2.astype(jnp.int32)
        scoreR = R1.astype(jnp.int32) + R2.astype(jnp.int32)
        choose_left = scoreL >= scoreR  # tie → left

        uL = jnp.where(L2, 2.0 * uL1 - uL2, uL1)
        uR = jnp.where(R2, 2.0 * uR1 - uR2, uR1)
        u_pick = jnp.where(choose_left, uL, uR)

        cf_adjacent = (~ice) & (L1 | R1)
        return jnp.where(cf_adjacent, u_pick, cc_field)

    return jax.jit(_extrap)


# ------------------------------------------------------------
# Divergence of a (1-D) tensor field: ∂x σ_xx
# ------------------------------------------------------------
def divergence_of_tensor_field_function_1d(dx: float, periodic: bool = False):
    """
    Input: tf[...,0,0] the σ_xx component as a 1‑D array of length n.
    Output: ∂x σ_xx, central differences with zero-gradient at boundaries by default,
            or periodic if periodic=True.
    """
    def _div(tf: jnp.ndarray) -> jnp.ndarray:
        sig_xx = tf[..., 0, 0]  # expect shape (n,)
        n = sig_xx.size
        grad = jnp.zeros_like(sig_xx)

        # interior
        grad = grad.at[1:-1].set((sig_xx[2:] - sig_xx[:-2]) * (0.5 / dx))

        if periodic and n >= 2:
            grad = grad.at[0].set((sig_xx[1] - sig_xx[-1]) * (0.5 / dx))
            grad = grad.at[-1].set((sig_xx[0] - sig_xx[-2]) * (0.5 / dx))
        else:
            # zero-gradient boundary (matches your 2‑D “assume ≈0 at edges” intent)
            grad = grad.at[0].set(0.0)
            grad = grad.at[-1].set(0.0)
        return grad
    return jax.jit(_div)


# ------------------------------------------------------------
# Sparsity for 1-D symmetric stencils
# ------------------------------------------------------------
def symmetric_stencil_sparsity_1d(n: int, radius: int, periodic: bool = False):
    """
    Return COO (row, col) indices for a symmetric 1‑D banded stencil
    with offsets [-radius, …, 0, …, +radius].

    Example: radius=1 → tridiagonal; radius=2 → pentadiagonal, etc.
    """
    offsets = jnp.arange(-radius, radius + 1)
    rows = jnp.repeat(jnp.arange(n), offsets.size)

    # For periodic: wrap with modulo; for non-periodic: drop out-of-bounds
    base_cols = jnp.tile(jnp.arange(n), (offsets.size, 1)).T  # (n, 2r+1)
    cols_all = (base_cols + offsets)  # (n, 2r+1)
    if periodic:
        cols_all = (cols_all + n) % n
        cols = cols_all.reshape(-1)
        return rows, cols
    else:
        mask = (cols_all >= 0) & (cols_all < n)
        rows_keep = jnp.repeat(jnp.arange(n), offsets.size)[mask.reshape(-1)]
        cols = cols_all.reshape(-1)[mask.reshape(-1)]
        return rows_keep, cols


def apply_symmetric_stencil_1d(a: jnp.ndarray, x: jnp.ndarray, periodic: bool = False) -> jnp.ndarray:
    """
    Vectorised matvec for a symmetric stencil.
      a: shape (2r+1,) coefficients with a[r] the centre,
      x: shape (n,)
    Boundary handling:
      - periodic=True: wrap; else, omit off-domain taps (Dirichlet-like).
    """
    r = (a.size - 1) // 2
    n = x.size

    acc = jnp.zeros_like(x)
    for k, coeff in enumerate(a):
        off = k - r
        if periodic:
            x_shift = jnp.roll(x, -off)
            acc = acc + coeff * x_shift
        else:
            if off < 0:
                # uses x[0:-off] for i >= -off
                contrib = jnp.zeros_like(x)
                contrib = contrib.at[-off:].set(x[:n+off])  # off negative
                acc = acc + coeff * contrib
            elif off > 0:
                contrib = jnp.zeros_like(x)
                contrib = contrib.at[:n-off].set(x[off:])
                acc = acc + coeff * contrib
            else:
                acc = acc + coeff * x
    return acc

