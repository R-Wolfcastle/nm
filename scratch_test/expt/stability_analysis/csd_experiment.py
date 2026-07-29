"""
Critical-slowing-down experiment, continuous-time version.

d(delta h)/dt = T @ delta h,   T = dH/dh - (dH/d[u,v]) (dG/d[u,v])^{-1} (dG/dh)

where G, H are BOTH built from compute_uvh_residuals_function_fully_nonlinear
_givenT_noextrap (residuals.py), the same coupled residual your
make_coupled_picnewton_solver_function uses:
  - G(u,v,h) = (x_mom_residual, y_mom_residual)   [its first two outputs]
  - H(u,v,h) = continuous dh/dt, obtained by dividing the third output
    (adv_residual) exactly through by delta_t and dx*dy -- delta_t
    appears linearly in that residual, so this is an EXACT algebraic
    reduction of the FOU advection scheme to its continuous-time rate,
    not an approximation (see chat for the derivation). PPM is not used
    here, per your request to work with FOU for this.

Stability threshold is now the ordinary continuous-time one: Re(lambda)=0
(NOT the mu=1 threshold from the earlier discrete-one-step-map version --
that version is superseded by this file).

As before: (dG/d[u,v])^{-1} is never formed explicitly -- only used via a
single PETSc (la_solver) solve per matvec, reusing one Jacobian assembled
at the base state. Leading eigenpair extraction uses the same
LinearOperator + ARPACK pattern as soa_test_refactor.py's
compute_evecs_ad/compute_evecs_sosa, swapped from eigsh to eigs (non-
symmetric) and 'LA' to 'LR' (largest real part).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigs, eigsh

nm_home = os.environ['NM_HOME']

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c

from grid import (interp_cc_with_ghosts_to_fc_function, add_ghost_cells_fcts,
                  gl_aware_driving_stress_function, beta_function,
                  fc_gradient_functions, cc_gradient_function,
                  linear_extrapolate_over_cf_function_cornersafe,
                  fc_viscosity_function_new_givenT,
                  fc_velocity_gradient_function_cf_safe)

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                  make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap

from residuals import compute_ssa_uv_residuals_function_wextrap,\
                      compute_ssa_uv_residuals_function_pnotC_givenT_noextrap

sys.path.insert(1, os.path.join(nm_home, 'utils'))
from sparsity_utils import basis_vectors_and_coords_2d_square_stencil, \
                            make_sparse_jacrev_fct_shared_basis
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp_given_csr


# ---------------------------------------------------------------------
# 1. Exact continuous-time FOU advection rate, reduced algebraically
#    from compute_uvh_residuals_function_fully_nonlinear_givenT_noextrap's
#    adv_residual (see derivation in module docstring / chat).
# ---------------------------------------------------------------------

def make_continuous_fou_advection_rate(ny, nx, dx, dy, ice_mask,
                                        add_uv_ghost_cells, add_s_ghost_cells,
                                        interp_cc_to_fc):

    def H(u, v, h, source, h_star_for_mask):
        u_full, v_full = add_uv_ghost_cells(u, v)
        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        #now-classic calving-front face-thickness fudge
        mask_ew, mask_ns = interp_cc_to_fc(add_s_ghost_cells(ice_mask))
        ff_ew = jnp.nan_to_num(1 / mask_ew, nan=0.0, posinf=0.0, neginf=0.0)
        ff_ns = jnp.nan_to_num(1 / mask_ns, nan=0.0, posinf=0.0, neginf=0.0)
        u_fc_ew, v_fc_ns = u_fc_ew * ff_ew, v_fc_ns * ff_ns

        h_full = add_s_ghost_cells(h)
        h_fc_fou_ew = jnp.where(u_fc_ew > 0, h_full[1:-1, :-1], h_full[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns > 0, h_full[1:, 1:-1], h_full[-1:, 1:-1])

        flux_div = (u_fc_ew[:, 1:] * h_fc_fou_ew[:, 1:] - u_fc_ew[:, :-1] * h_fc_fou_ew[:, :-1]) / dx + \
                   (v_fc_ns[:-1, :] * h_fc_fou_ns[:-1, :] - v_fc_ns[1:, :] * h_fc_fou_ns[1:, :]) / dy

        flux_div = jnp.where(h_star_for_mask > 1e-2, flux_div, 0.0)

        return source - flux_div

    return H


# ---------------------------------------------------------------------
# 2. G, and the (u,v)-block sparse Jacobian machinery, both built from
#    compute_uvh_residuals_function_fully_nonlinear_givenT_noextrap --
#    mirrors make_coupled_picnewton_solver_function's setup (residuals.py
#    lines ~848-886), radius=1 to match its actual stencil, but restricted
#    to the (u,v) block only (active_indices=(0,1)) since h is held fixed
#    for this linearisation, not part of what we're solving for here.
# ---------------------------------------------------------------------


def build_matvec_machinery(ny, nx, dy, dx, b,
                           ice_mask, mucoef_0,
                           C_0, sliding):

    temperature_field = jnp.zeros((ny, nx)) + 250.15

    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx)
    hgrads_fct                                 = gl_aware_driving_stress_function(dy, dx)

    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    beta_fct = beta_function(b, sliding, None)

    get_uv_residuals_nonlinear = compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field,
                                                       hgrads_fct)

    H = make_continuous_fou_advection_rate(ny, nx, dx, dy, ice_mask,
                                           add_uv_ghost_cells, add_scalar_ghost_cells,
                                           interp_cc_to_fc)

    def G(u_1d, v_1d, q, p, h_1d):   # NOTE: (u,v,q,p,h) order, matching the real call sites
        return get_uv_residuals_nonlinear(u_1d, v_1d, q, p, h_1d)

    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(
        ny, nx, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny * nx), len(basis_vectors))
    mask = (i_coordinate_sets >= 0)

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
        basis_vectors, i_coordinate_sets, j_coordinate_sets, mask, 2,
        active_indices=(0, 1))

    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]

    coords = jnp.stack([
        jnp.concatenate([i_coordinate_sets, i_coordinate_sets,
                          i_coordinate_sets + (ny * nx), i_coordinate_sets + (ny * nx)]),
        jnp.concatenate([j_coordinate_sets, j_coordinate_sets + (ny * nx),
                          j_coordinate_sets, j_coordinate_sets + (ny * nx)])
    ])

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
        coords, (ny * nx * 2, ny * nx * 2), indirect=False,
        ksp_type="gmres", preconditioner="hypre", monitor_ksp=False)

    return dict(G=G, H=H, sparse_jacrev=sparse_jacrev, la_solver=la_solver, mask=mask)

def build_grounded_mask(h_star, b, min_thickness=50.0):
    """Restrict perturbations to grounded ice with a comfortable safety
    margin above flotation-thickness/edge artifacts -- excludes the
    floating buffer and any thin numerical remnants near the domain edge."""
    grounded = (h_star + b) > (h_star * (1 - c.RHO_I / c.RHO_W))
    thick_enough = h_star > min_thickness
    return np.asarray((grounded & thick_enough).reshape(-1))

def restrict_matvec(full_matvec, mask):
    idx = np.where(mask)[0]
    n_sub = len(idx)
    def sub_matvec(x_sub):
        x_full = np.zeros(mask.shape[0])
        x_full[idx] = x_sub
        y_full = np.array(full_matvec(x_full))
        return y_full[idx]
    return sub_matvec, n_sub, idx

def embed(x_sub, idx, n_full):
    x_full = np.zeros(n_full)
    x_full[idx] = x_sub
    return x_full

#def build_matvec_machinery_wextrap(ny, nx, dy, dx, b,
#                           ice_mask, mucoef_0,
#                           C_0, sliding):
#
#    temperature_field = jnp.zeros((ny, nx)) + 250.15
#
#
#    interp_cc_to_fc = interp_cc_with_ghosts_to_fc_function(ny, nx)
#    ew_gradient, ns_gradient = fc_gradient_functions(dy, dx)
#    cc_gradient = cc_gradient_function(dy, dx)
#    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx)
#    extrapolate_over_cf = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
#    hgrads_fct = gl_aware_driving_stress_function(dy, dx)
#
#    viscosity_fct = fc_viscosity_function_new_givenT(ny, nx, dy, dx,
#                                                       extrapolate_over_cf,
#                                                       add_uv_ghost_cells,
#                                                       add_scalar_ghost_cells,
#                                                       interp_cc_to_fc,
#                                                       ew_gradient, ns_gradient,
#                                                       ice_mask, mucoef_0,
#                                                       temperature_field)
#
#    beta_fct = beta_function(b, sliding, None)   # matches your working default
#
#    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_wextrap(
#        ny, nx, dy, dx, b,
#        beta_fct, ice_mask,
#        interp_cc_to_fc,
#        ew_gradient, ns_gradient,
#        cc_gradient,
#        add_uv_ghost_cells,
#        add_scalar_ghost_cells,
#        extrapolate_over_cf,
#        mucoef_0, C_0,
#        temperature_field,
#        hgrads_fct)
#
#    H = make_continuous_fou_advection_rate(ny, nx, dx, dy, ice_mask,
#                                            add_uv_ghost_cells, add_scalar_ghost_cells,
#                                            interp_cc_to_fc)
#
#    def G(u_1d, v_1d, q, p, h_1d):   # NOTE: (u,v,q,p,h) order, matching the real call sites
#        return get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d)
#
#
#    #interp_cc_to_fc = interp_cc_with_ghosts_to_fc_function(ny, nx)
#    #add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx)
#    #hgrads_fct = gl_aware_driving_stress_function(dy, dx)
#
#    #fc_velocity_gradient = fc_velocity_gradient_function_cf_safe(
#    #    dy, dx, ny, nx, ice_mask, add_uv_ghost_cells, add_scalar_ghost_cells)
#
#    #beta_fct = beta_function(b, sliding)
#
#    #get_uvh_residuals = compute_uvh_residuals_function_fully_nonlinear_givenT_noextrap(
#    #    ny, nx, dy, dx, b,
#    #    beta_fct, ice_mask,
#    #    interp_cc_to_fc,
#    #    fc_velocity_gradient,
#    #    add_uv_ghost_cells,
#    #    add_scalar_ghost_cells,
#    #    hgrads_fct,
#    #    mucoef_0, C_0,
#    #    temperature_field)
#
#    #H = make_continuous_fou_advection_rate(ny, nx, dx, dy, ice_mask,
#    #                                        add_uv_ghost_cells, add_scalar_ghost_cells,
#    #                                        interp_cc_to_fc)
#
#    #def G(u_1d, v_1d, h_1d, q, p):
#    #    # h_t/source/delta_t are dummy values here: the momentum residuals
#    #    # (indices 0,1) don't depend on them at all -- only adv_residual
#    #    # (index 2, unused here since H is computed separately/exactly).
#    #    x_mom, y_mom, _ = get_uvh_residuals(u_1d, v_1d, h_1d, q, p,
#    #                                         h_1d, jnp.zeros((ny, nx)), 1.0)
#    #    return x_mom, y_mom
#
#    # --- (u,v)-block sparse Jacobian / PETSc solver, radius=1 to match
#    #     fc_velocity_gradient_function_cf_safe's actual stencil width ---
#
#    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(
#        ny, nx, 1)
#
#    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
#    j_coordinate_sets = jnp.tile(jnp.arange(ny * nx), len(basis_vectors))
#    mask = (i_coordinate_sets >= 0)
#
#    sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
#        basis_vectors, i_coordinate_sets, j_coordinate_sets, mask, 2,
#        active_indices=(0, 1))
#
#    i_coordinate_sets = i_coordinate_sets[mask]
#    j_coordinate_sets = j_coordinate_sets[mask]
#
#    coords = jnp.stack([
#        jnp.concatenate([i_coordinate_sets, i_coordinate_sets,
#                          i_coordinate_sets + (ny * nx), i_coordinate_sets + (ny * nx)]),
#        jnp.concatenate([j_coordinate_sets, j_coordinate_sets + (ny * nx),
#                          j_coordinate_sets, j_coordinate_sets + (ny * nx)])
#    ])
#
#    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
#        coords, (ny * nx * 2, ny * nx * 2), indirect=False,
#        ksp_type="gmres", preconditioner="hypre", monitor_ksp=False)
#
#    return dict(G=G, H=H, sparse_jacrev=sparse_jacrev, la_solver=la_solver, mask=mask)


# ---------------------------------------------------------------------
# 3. Matrix-free T @ delta_h, continuous-time, at a fixed base state
# ---------------------------------------------------------------------

def make_tangent_propagator_matvec(ny, nx, machinery, u_star, v_star, h_star,
                                    q, p, accumulation_val):
    G = machinery['G']
    H = machinery['H']
    sparse_jacrev = machinery['sparse_jacrev']
    la_solver = machinery['la_solver']
    mask = machinery['mask']

    u_1d = u_star.reshape(-1)
    v_1d = v_star.reshape(-1)
    h_1d = h_star.reshape(-1)
    n = ny * nx

    # (dG/d[u,v]) at the base state -- assembled once, reused every matvec
    #dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(G, (u_1d, v_1d, h_1d, q, p))
    dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(G, (u_1d, v_1d, q, p, h_1d))
    
    nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],
                                     dJv_du[mask], dJv_dv[mask]])

    source_field = jnp.where(h_star > 0, accumulation_val, 0.0)

    def G_of_h(h_1d_probe):
        x_mom, y_mom = G(u_1d, v_1d, q, p, h_1d_probe)
        return jnp.concatenate([x_mom, y_mom])
    
    #def G_of_h(h_1d_probe):
    #    x_mom, y_mom = G(u_1d, v_1d, h_1d_probe, q, p)
    #    return jnp.concatenate([x_mom, y_mom])

    def H_of_h(h_probe_2d):
        return H(u_star, v_star, h_probe_2d, source_field, h_star)

    def H_of_uv(u_probe_2d, v_probe_2d):
        return H(u_probe_2d, v_probe_2d, h_star, source_field, h_star)

    def Tv(delta_h_flat):
        delta_h = jnp.asarray(delta_h_flat).reshape((ny, nx))

        _, dGdh_v = jax.jvp(G_of_h, (h_1d,), (delta_h.reshape(-1),))

        x = la_solver(nz_jac_values, dGdh_v)
        # implicit-function-theorem sign -- CHECK THIS FIRST if results
        # look wrong: delta_[u,v] = -(dG/d[u,v])^{-1} (dG/dh) delta_h
        delta_u = -x[:n].reshape((ny, nx))
        delta_v = -x[n:].reshape((ny, nx))

        _, dHdh_v = jax.jvp(H_of_h, (h_star,), (delta_h,))
        _, dHduv_v = jax.jvp(H_of_uv, (u_star, v_star), (delta_u, delta_v))

        return (dHdh_v + dHduv_v).reshape(-1)

    return Tv

def make_tangent_propagator_transpose_matvec(ny, nx, machinery, u_star, v_star, h_star,
                                              q, p, accumulation_val):
    G = machinery['G']
    H = machinery['H']
    sparse_jacrev = machinery['sparse_jacrev']
    la_solver = machinery['la_solver']
    mask = machinery['mask']

    u_1d = u_star.reshape(-1)
    v_1d = v_star.reshape(-1)
    h_1d = h_star.reshape(-1)
    n = ny * nx

    dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(G, (u_1d, v_1d, q, p, h_1d))
    nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask], dJv_du[mask], dJv_dv[mask]])

    source_field = jnp.where(h_star > 0, accumulation_val, 0.0)

    def G_of_h(h_1d_probe):
        x_mom, y_mom = G(u_1d, v_1d, q, p, h_1d_probe)
        return jnp.concatenate([x_mom, y_mom])

    def H_of_h(h_probe_2d):
        return H(u_star, v_star, h_probe_2d, source_field, h_star)

    def H_of_uv(u_probe_2d, v_probe_2d):
        return H(u_probe_2d, v_probe_2d, h_star, source_field, h_star)

    def TTa(alpha_flat):
        alpha = jnp.asarray(alpha_flat).reshape((ny, nx))

        _, vjp_H_uv = jax.vjp(H_of_uv, u_star, v_star)
        beta_u, beta_v = vjp_H_uv(alpha)                       # (dH/d[u,v])^T @ v
        beta = jnp.concatenate([beta_u.reshape(-1),
                                beta_v.reshape(-1)])

        gamma = la_solver(nz_jac_values, beta, transpose=True)   # (dG/d[u,v])^{-T} @ w1

        _, vjp_G_h = jax.vjp(G_of_h, h_1d)
        (w1,) = vjp_G_h(gamma)                             # (dG/dh)^T @ w2

        _, vjp_H_h = jax.vjp(H_of_h, h_star)
        (w2,) = vjp_H_h(alpha)                               # (dH/dh)^T @ v

        return (w2.reshape(-1) - w1)

    return TTa


def leading_eigenpair_matrix_free(Tv_fct, n, k=1):
    """Same LinearOperator + ARPACK pattern as soa_test_refactor.py's
    compute_evecs_ad/compute_evecs_sosa (eigsh -> eigs since T is not
    symmetric; 'LA' -> 'LR' since we want largest real part, the
    continuous-time stability threshold being Re(lambda)=0)."""
    T_op = LinearOperator(
        shape=(n, n), dtype=np.float64,
        matvec=lambda x: np.array(Tv_fct(x))
    )
    vals, vecs = eigs(T_op, k=k, which='LR', tol=1e-8, maxiter=2000)
    order = np.argsort(-np.real(vals))
    lam = vals[order][0]
    vec = np.real(vecs[:, order][:, 0])
    vec = vec / (np.linalg.norm(vec) + 1e-30)
    return lam, vec


def numerical_abscissa_matrix_free(Tv_fct, TTa_fct, n, k=1):
    def Sv(x):
        return 0.5 * (np.array(Tv_fct(x)) + np.array(TTa_fct(x)))
    S_op = LinearOperator(shape=(n, n), dtype=np.float64, matvec=Sv)
    vals, vecs = eigsh(S_op, k=k, which='LA', tol=1e-8, maxiter=2000)
    idx = np.argmax(vals)
    return float(vals[idx]), vecs[:, idx]


# ---------------------------------------------------------------------
# 4. Driver: decreasing-accumulation sweep, quasi-static vs transient
#    (unchanged in spirit from before -- just Re(lambda)=0 as the
#    threshold now, not mu=1, and using the momentum_solver/
#    advection_stepper pair from make_picnewton_velocity_solver_function
#    _full_cvjp for the actual forward integration, same as your working
#    driver -- only the diagnostic eigenvalue machinery is continuous-time)
# ---------------------------------------------------------------------

#def relax_to_steady_state(momentum_solver, advection_stepper, u, v, thk,
#                           q, p, acc_val, dt, n_outer_steps, dhdt_tol,
#                           resolution, cfl_scale=0.9):
#    
#    history = {"t": [], "dt": [], "x_gl": [], "thk_gl": [], "max_speed": [],
#               "max_dhdt": [], "accumulation": []}
#
#    for outer_i in range(n_outer_steps):
#        u, v = momentum_solver(q, p, u, v, thk)
#
#        source = jnp.where(thk > 0, acc_val, 0.0)
#
#        dt = cfl_scale*(resolution/jnp.max(jnp.sqrt(u**2+v**2)))
#
#        thk_new = advection_stepper(u.reshape(-1), v.reshape(-1), thk.reshape(-1),
#                                     source=source, delta_t=dt)
#        
#        thk_new = jnp.maximum(thk_new, 0.0)
#        dhdt = float(jnp.max(jnp.abs((thk_new - thk) / dt)))
#        thk = thk_new
#
#        if dhdt < dhdt_tol:
#            break
#        
#        if not outer_i%10:
#            print(f"Iteration: {outer_i}, max dhdt={dhdt:.4e} m/a")
#            :wa
#
#
#    return u, v, thk, dhdt, outer_i + 1

def relax_to_steady_state(momentum_solver, advection_stepper, u, v, thk,
                           q, p, acc_val, x, b, n_outer_steps, dhdt_tol,
                           cfl_factor=0.45, max_delta_t=5.0, t0=0.0,
                           output_dir=None, resolution=None, tag=""):

    t = t0
    history = {"t": [], "dt": [], "x_gl": [], "thk_gl": [], "max_speed": [],
               "max_dhdt": [], "accumulation": []}

    dhdt = float('inf')
    for outer_i in range(n_outer_steps):
        u, v = momentum_solver(q, p, u, v, thk)

        max_speed = float(jnp.max(jnp.sqrt(u ** 2 + v ** 2)))
        delta_t = cfl_factor * (float(x[1] - x[0]) / max(max_speed, 1e-6))
        delta_t = min(delta_t, max_delta_t)

        accumulation = jnp.where(thk > 0, acc_val, 0.0)

        thk_old = thk
        thk = advection_stepper(u.reshape(-1), v.reshape(-1), thk.reshape(-1),
                                 source=accumulation, delta_t=delta_t)
        thk = jnp.maximum(thk, 0.0)

        # NOTE: ice_mask below is diagnostic only -- the real ice_mask used
        # by momentum_solver/advection_stepper was fixed when they were
        # constructed (calving front is fixed for the life of the solver
        # object), so this doesn't feed back into anything.
        ice_mask_now = (thk > 0).astype(int)

        t += delta_t
        dhdt = float(jnp.max(jnp.abs((thk - thk_old) / delta_t)))

        gl_idx = int(grounding_line_index(thk, b))
        x_gl = float(x[gl_idx]) if gl_idx >= 0 else float("nan")
        thk_gl = float(thk[0, gl_idx]) if gl_idx >= 0 else float("nan")

        history["t"].append(t)
        history["dt"].append(delta_t)
        history["x_gl"].append(x_gl)
        history["thk_gl"].append(thk_gl)
        history["max_speed"].append(max_speed)
        history["max_dhdt"].append(dhdt)
        history["accumulation"].append(acc_val)

        if dhdt < dhdt_tol:
            print(f"REACHED DHDT TOLERANCE THRESHOLD: {dhdt}")
            break

        if not outer_i%10:
            print(f"********** xgl: {x_gl},  dhdt: {dhdt}")

    #if output_dir is not None:
    #    plot_transect_result(x, b, thk, history, output_dir=output_dir,
    #                          resolution=resolution, tag=tag)

    return u, v, thk, ice_mask_now, dhdt, outer_i + 1, t, history


def grounding_line_index(thk, b):
    s_gnd = thk + b
    s_flt = thk * (1.0 - c.RHO_I / c.RHO_W)
    grounded = (s_gnd > s_flt) & (thk > 0)
    grounded = grounded[0]
    idx = jnp.where(grounded, jnp.arange(grounded.shape[0]), -1)
    return jnp.max(idx)

def plot_transect_result(x, b, thk, history, output_dir=None, resolution=None, tag="",
                          sa_evec=None, na_evec=None):
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)

    s_gnd = thk[0] + b[0]
    s_flt = thk[0] * (1.0 - c.RHO_I / c.RHO_W)
    surface = jnp.maximum(s_gnd, s_flt)

    base = surface-thk[0]

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=False)

    axes[0].plot(x / 1000, b[0], color="saddlebrown", label="bed")
    axes[0].plot(x / 1000, jnp.where(thk[0]>0, base, jnp.nan), color="teal", label="base")
    axes[0].plot(x / 1000, jnp.where(thk[0]>0, surface, jnp.nan), color="steelblue", label="surface")
    axes[0].fill_between(x / 1000, base, surface, where=(thk[0] > 0),
                          color="lightblue", alpha=0.5)
    axes[0].set_ylabel("elevation (m)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Transect profile, {tag} ({resolution} m)")

    if sa_evec is not None:
        axes[1].plot(x / 1000, -np.abs(sa_evec), label='SA')
    if na_evec is not None:
        axes[1].plot(x / 1000, -np.abs(na_evec), label='NA')
    axes[1].set_ylabel("eigenvector dh (m)")
    axes[1].legend()

    axes[2].plot(history["t"], np.array(history["x_gl"]) / 1000)
    axes[2].set_ylabel("x_gl (km)")
    axes[2].set_xlabel("time (yr)")

    axes[3].semilogy(history["t"], history["max_dhdt"])
    axes[3].set_ylabel("max |dh/dt| (m/yr)")
    axes[3].set_xlabel("time (yr)")

    axes[0].set_ylim(-1000, 5000)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f"transect_{tag}.png"), dpi=150)
    plt.close()


def run_accumulation_sweep(domain, acc_values, dt=0.1,
                           n_pic_iterations=7,
                           n_newt_iterations=3, sliding="basic_weertman",
                           branch="quasistatic", 
                           output_dir=None,
                           max_outer_steps=1000,
                           dhdt_tol=1e-3, k_eigs=1,
                           u0=None, v0=None, h0=None):
    (lx, ly, nr, nc, x, y, delta_x, delta_y,
     thk, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain

    thk = jnp.load(f"{nm_home}/bits_of_data/schoof_instability_stuff/6/thk_1000m_step9999.npy")

    temp_field = jnp.zeros_like(C_0) + 250.15

    ny, nx = nr, nc
    p = jnp.zeros_like(q)

    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
        ny, nx, delta_y, delta_x, b, ice_mask,
        n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
        adv_method="FOU",
        sliding=sliding, temperature_field=temp_field)

    u = u0 if u0 is not None else jnp.where(thk > 1e-10, 1.0, 0.0)
    v = v0 if v0 is not None else jnp.zeros_like(u)
    h = h0 if h0 is not None else thk

    history = dict(acc=[], x_gl=[], dhdt=[], eval=[],
                   sa_evec=[], na_evec=[], omega=[], gap=[])
    n = ny * nx

    t_cum = 0.0
    for i, acc_val in enumerate(acc_values):

        if branch == "quasistatic":
            u, v, h, ice_mask_now, dhdt, n_used, t_cum, step_history = relax_to_steady_state(
                momentum_solver, advection_stepper, u, v, h, q, p, acc_val,
                x, b, max_outer_steps, dhdt_tol, t0=t_cum,
                output_dir=output_dir, resolution=delta_x, tag=f"acc{i:03d}_{acc_val:.4f}")
        else:  # "transient": one step only, never wait for convergence
            u, v = momentum_solver(q, p, u, v, h)
            source = jnp.where(h > 0, acc_val, 0.0)
            h_old = h
            h = advection_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1),
                                   source=source, delta_t=dt)
            h = jnp.maximum(h, 0.0)
            dhdt = float(jnp.max(jnp.abs((h - h_old) / dt)))

        machinery = build_matvec_machinery(ny, nx, delta_y, delta_x, b, ice_mask,
                                            mucoef_0, C_0, sliding)


        mask = build_grounded_mask(h, b)



        print("making tangent propagator")
        Tv = make_tangent_propagator_matvec(ny, nx, machinery, u, v, h, q, p, acc_val)
        #making subsetting to grounded area version
        Tv_sub, n_sub, idx = restrict_matvec(Tv, mask)
        print("computing leading eigenpair of T")
        lam, sa_evec_sub = leading_eigenpair_matrix_free(Tv_sub, n_sub, k=k_eigs)
        #lam, sa_evec = leading_eigenpair_matrix_free(Tv, n, k=k_eigs)

        print("making adjoint of tangent propagator")
        TTa = make_tangent_propagator_transpose_matvec(ny, nx, machinery, u, v, h, q, p, acc_val)
        TTa_sub, _, _      = restrict_matvec(TTa, mask)
        print("computing numerical abscissa of T")
        #omega, na_evec = numerical_abscissa_matrix_free(Tv, TTa, n)
        omega, na_evec_sub = numerical_abscissa_matrix_free(Tv_sub, TTa_sub, n_sub)

        #make evecs bigger again so can plot on full x-axis
        sa_evec = embed(sa_evec_sub, idx, n)
        na_evec = embed(na_evec_sub, idx, n)

        if output_dir is not None and branch == "quasistatic":
            plot_transect_result(x, b, h, step_history, output_dir=output_dir,
                                  resolution=delta_x, tag=f"acc{i:03d}_{acc_val:.4f}",
                                  sa_evec=sa_evec, na_evec=na_evec)

        grounded_now = jnp.where((h + b) > (h * (1 - c.RHO_I / c.RHO_W)), 1, 0)
        gl_idx = int(jnp.max(jnp.where(grounded_now[0] > 0, jnp.arange(nx), -1)))
        x_gl = float(x[gl_idx]) if gl_idx >= 0 else float('nan')

        history['acc'].append(acc_val)
        history['x_gl'].append(x_gl)
        history['dhdt'].append(dhdt)
        history['eval'].append(lam)
        history['sa_evec'].append(sa_evec)
        history['na_evec'].append(na_evec)
        history['omega'].append(omega)
        history['gap'].append(omega - np.real(lam))
        print(f"  ... alpha={np.real(lam):.4e}  omega={omega:.4e}  gap={omega-np.real(lam):.4e}")

        print(f"[{branch}] acc={acc_val:.4f} m/yr  x_gl={x_gl/1000:.2f} km  "
              f"dhdt={dhdt:.4e}  "
              f"alpha={np.real(lam):.4e}  omega={omega:.4e}  gap={omega-np.real(lam):.4e}")
        

    return history


if __name__ == "__main__":

    output_dir = os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/13/")
    os.makedirs(output_dir, exist_ok=True)

    # Import your (now-fixed) Schoof transect domain here, e.g.:
    from standard_domains import schoof2007_transect_domain#, schoof_scaled
    domain = schoof2007_transect_domain(resolution=1000)

    # Decreasing-accumulation sweep, matching Schoof's a=0.3 m/yr default
    # as the high-accumulation starting point.
    acc_values = np.linspace(0.30, 0.05, 15)

    max_itns_to_ss = 2000

    resA = run_accumulation_sweep(domain, acc_values, branch="quasistatic",
                                  max_outer_steps=max_itns_to_ss, output_dir=output_dir)
    # resB = run_accumulation_sweep(domain, acc_values, branch="transient")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(resA['acc'], np.real(resA['eval']), 'o-', label='quasistatic')
    #axes[0].plot(resB['acc'], np.real(resB['eval']), 's-', label='transient')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('accumulation (m/yr)'); axes[0].set_ylabel('leading eigenvalue')
    axes[0].legend()
    axes[0].invert_xaxis()
    
    axes[1].plot(resA['acc'], np.real(resA['omega']), 'o-', label='omega')
    #axes[1].plot(resA['acc'], np.real(resA['gap']), 'o-', label='gap')
    axes[1].set_xlabel('accumulation (m/yr)');
    axes[1].set_ylabel('numerical abscissa (a^-1)')
    axes[1].legend()
    axes[1].invert_xaxis()
    

    axes[2].plot(resA['acc'], np.array(resA['x_gl'])/1000, 'o-', label='quasistatic')
    #axes[1].plot(resB['acc'], resB['x_gl'], 's-', label='transient')
    axes[2].set_xlabel('accumulation (m/yr)'); axes[1].set_ylabel('x_gl (km)')
    axes[2].legend()
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csd_accumulation_sweep.png"), dpi=150)
    pass
