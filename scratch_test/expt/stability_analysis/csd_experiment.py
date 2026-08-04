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
from plotting_stuff import plotgeom, plotboth

from sparsity_utils import basis_vectors_and_coords_2d_square_stencil, \
                            make_sparse_jacrev_fct_shared_basis

from standard_domains import schoof2007_transect_domain, schoof_scaled

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                 make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                                         make_coupled_picnewton_solver_function
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp_given_csr

from residuals import compute_ssa_uv_residuals_function_wextrap,\
                      compute_ssa_uv_residuals_function_pnotC_givenT_noextrap

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


def build_matvec_machinery(ny, nx, dy, dx, b,
                           ice_mask, mucoef_0,
                           C_0, sliding):

    temperature_field = jnp.zeros((ny, nx)) + 263.15

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


def solve_steady_state_direct(impl_solver, q, p, u, v, h, acc_val,
                              delta_t_start=20.0, delta_t_growth_factor=4.0,
                              delta_t_max=1e6, n_outer=15, dhdt_tol=1e-9):

    delta_t = delta_t_start
    for outer_i in range(n_outer):
        u, v, h_new = impl_solver(q, p, u, v, h, delta_t, accm=acc_val)
        dhdt = float(jnp.max(jnp.abs((h_new - h) / delta_t)))
        h = h_new
        print(f"outer {outer_i}: delta_t={delta_t:.1f}  effective max|dh/dt|={dhdt:.4e}")
        if dhdt < dhdt_tol:
            break
        delta_t = min(delta_t * delta_t_growth_factor, delta_t_max)
    return u, v, h, dhdt, outer_i + 1

def save_history(history, output_dir, filename="history.npz"):
    """Persist history to disk every time it's called (not just at the
    end of a sweep), so a long run's progress survives an interruption.
    Handles ragged/complex-valued entries (eval is complex, sa_evec/
    na_evec are arrays) by falling back to dtype=object where needed."""
    if output_dir is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    save_dict = {}
    for k, v in history.items():
        try:
            save_dict[k] = np.array(v)
        except Exception:
            save_dict[k] = np.array(v, dtype=object)
    np.savez(os.path.join(output_dir, filename), **save_dict)

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

    



def plot_ss(x, thk, b, speed, sa_evec, na_evec,
            savepath, title=None,
            axis_limits=None, show_plots=False):

    s_gnd = b + thk
    s_flt = thk*(1-c.RHO_I/c.RHO_W)
    surface = jnp.maximum(s_gnd, s_flt)

    base = surface-thk

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    ax1 = axes[0]
    ax2 = ax1.twinx()
    ax3 = axes[1]

    ax1.plot(x / 1000, b, color="saddlebrown", label="bed")
    ax1.plot(x / 1000, jnp.where(thk>0, base, jnp.nan), color="teal", label="base")
    ax1.plot(x / 1000, jnp.where(thk>0, surface, jnp.nan), color="steelblue", label="surface")
    ax1.fill_between(x / 1000, base, surface,
                     where=(thk > 0),
                     color="lightblue", alpha=0.5)
    ax1.set_ylabel("elevation (m)")
    #ax1.legend(fontsize=8)
    #ax1.set_title(f"Transect profile, {tag} ({resolution} m)")

    ax2.plot(x / 1000, jnp.where(speed>1e-10, speed, jnp.nan), color='k', 
                                 marker=".", markersize=0.1, label="speed")
    ax2.set_ylabel("Speed (m a^-1)")

    if sa_evec is not None:
        ax3.plot(x / 1000, -np.where(np.abs(sa_evec)>1e-10, np.abs(sa_evec), np.nan), label='SA')
    if na_evec is not None:
        ax3.plot(x / 1000, -np.where(np.abs(na_evec)>1e-10, np.abs(na_evec), np.nan), label='NA')
    ax3.set_ylabel("eigenvector dh (m)")
    ax3.set_xlim(ax1.get_xlim())
    ax3.legend()
   

    if axis_limits is not None:
        if axis_limits[0] is not None:
            ax1.set_ylim(axis_limits[0])
        if axis_limits[1] is not None:
            ax2.set_ylim(axis_limits[1])

    if title is not None:
        plt.title(title)
  
    plt.savefig(savepath, dpi=200)



def steady_state_accumulation_sweep(domain, acc_values, dt=0.1,
                                    n_pic_iterations=40,
                                    n_newt_iterations=5,
                                    sliding="basic_weertman",
                                    output_dir=None,
                                    max_outer_steps=1000,
                                    dhdt_tol=1e-3, k_eigs=1,
                                    u0=None, v0=None, h0=None):
    (lx, ly, nr, nc, x, y, delta_x, delta_y,
     thk, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain

    thk = jnp.load(f"{nm_home}/bits_of_data/schoof_instability_stuff/6/thk_1000m_step9999.npy")

    temp_field = jnp.zeros_like(C_0) + 263.15

    ny, nx = nr, nc
    p = jnp.zeros_like(q)

    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
        ny, nx, delta_y, delta_x, b, ice_mask,
        n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
        adv_method="FOU",
        sliding=sliding, temperature_field=temp_field)


    impl_solver = make_coupled_picnewton_solver_function(
                    ny, nx, delta_y, delta_x, b, ice_mask,
                    n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
                    sliding=sliding, temperature_field=temp_field
                                                        )

    u = u0 if u0 is not None else jnp.where(thk > 1e-10, 1.0, 0.0)
    v = v0 if v0 is not None else jnp.zeros_like(u)
    h = h0 if h0 is not None else thk

    history = dict(acc=[], x_gl=[], dhdt=[], eval=[],
                   sa_evec=[], na_evec=[], omega=[], gap=[])
    n = ny * nx

    t_cum = 0.0
    for i, acc_val in enumerate(acc_values):

        #u, v, h, ice_mask_now, dhdt, n_used, t_cum, step_history = relax_to_steady_state(
        #    momentum_solver, advection_stepper, u, v, h, q, p, acc_val,
        #    x, b, max_outer_steps, dhdt_tol, t0=t_cum,
        #    output_dir=output_dir, resolution=delta_x, tag=f"acc{i:03d}_{acc_val:.4f}")
        u, v, h, dhdt, n_used = solve_steady_state_direct(impl_solver, q, p,
                                       u, v, h, acc_val, delta_t_growth_factor=4,
                                       delta_t_max=1e6, n_outer=15, dhdt_tol=1e-9)


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

        plot_ss(x, h[0,:], b[0,:], u[0,:],
                sa_evec, na_evec,
                os.path.join(output_dir, f"acc{i:03d}_{acc_val:.4f}.png"),
                title=f"Steady state for accumulation: {acc_val:.4f} m a^-1",
                axis_limits=[[-1500, 5500],[0, 5000]])

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

        print(f" acc={acc_val:.4f} m/yr  x_gl={x_gl/1000:.2f} km  "
              f"dhdt={dhdt:.4e}  "
              f"alpha={np.real(lam):.4e}  omega={omega:.4e}  gap={omega-np.real(lam):.4e}")
    
        save_history(history, output_dir, "steady_state_history.npz")

    return history

def unsteady_accumulation_sweep(domain, acc_values, dt=0.1,
                                n_pic_iterations=40,
                                n_newt_iterations=5,
                                sliding="basic_weertman",
                                output_dir=None,
                                max_outer_steps=1000,
                                dhdt_tol=1e-3, k_eigs=1,
                                u0=None, v0=None, h0=None):
    (lx, ly, nr, nc, x, y, delta_x, delta_y,
     thk, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain

    thk = jnp.load(f"{nm_home}/bits_of_data/schoof_instability_stuff/6/thk_1000m_step9999.npy")

    temp_field = jnp.zeros_like(C_0) + 263.15

    ny, nx = nr, nc
    p = jnp.zeros_like(q)

    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
        ny, nx, delta_y, delta_x, b, ice_mask,
        n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
        adv_method="FOU",
        sliding=sliding, temperature_field=temp_field)


    impl_solver = make_coupled_picnewton_solver_function(
                    ny, nx, delta_y, delta_x, b, ice_mask,
                    n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
                    sliding=sliding, temperature_field=temp_field
                                                        )

    u = u0 if u0 is not None else jnp.where(thk > 1e-10, 1.0, 0.0)
    v = v0 if v0 is not None else jnp.zeros_like(u)
    h = h0 if h0 is not None else thk

    history = dict(acc=[], x_gl=[], dhdt=[], eval=[],
                   sa_evec=[], na_evec=[], omega=[], gap=[])
    n = ny * nx

    t_cum = 0.0
    for i, acc_val in enumerate(acc_values):
        

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

        plot_ss(x, h[0,:], b[0,:], u[0,:],
                sa_evec, na_evec,
                os.path.join(output_dir, f"acc{i:03d}_{acc_val:.4f}.png"),
                title=f"Steady state for accumulation: {acc_val:.4f} m a^-1",
                axis_limits=[[-1500, 5500],[0, 1500]])

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


def steady_experiment():

    output_dir = os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/17/")
    os.makedirs(output_dir, exist_ok=True)

    # Import your (now-fixed) Schoof transect domain here, e.g.:
    domain = schoof2007_transect_domain(resolution=1000)
    #domain = schoof_scaled(resolution=250)

    # Decreasing-accumulation sweep, matching Schoof's a=0.3 m/yr default
    # as the high-accumulation starting point.
    #t = np.linspace(1, -1, 20)
    #acc_values = 1 + 0.5 * t**3
    acc_values = np.linspace(0.5, 0.05, 40)

    max_itns_to_ss = 2000

    resA = steady_state_accumulation_sweep(domain, acc_values,
                                           max_outer_steps=max_itns_to_ss,
                                           output_dir=output_dir)

    #resB = run_accumulation_sweep(domain, acc_values, branch="transient")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(resA['acc'], np.real(resA['eval']), 'o-')#, label='quasistatic')
    #axes[0].plot(resB['acc'], np.real(resB['eval']), 's-', label='transient')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('accumulation (m/yr)')
    axes[0].set_ylabel('spectral abscissa (a^-1)')
    #axes[0].legend()
    axes[0].invert_xaxis()
    
    axes[1].plot(resA['acc'], np.real(resA['omega']), 'o-')
    #axes[1].plot(resA['acc'], np.real(resA['gap']), 'o-', label='gap')
    axes[1].set_xlabel('accumulation (m/yr)');
    axes[1].set_ylabel('numerical abscissa (a^-1)')
    #axes[1].legend()
    axes[1].invert_xaxis()
    

    axes[2].plot(resA['acc'], np.array(resA['x_gl'])/1000, 'o-')#, label='quasistatic')
    #axes[1].plot(resB['acc'], resB['x_gl'], 's-', label='transient')
    axes[2].set_xlabel('accumulation (m/yr)')
    axes[2].set_ylabel('x_gl (km)')
    #axes[2].legend()
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csd_accumulation_sweep.png"), dpi=150)


def time_forced_sweep(domain, acc_start=0.5, acc_end=0.0, t_total=50_000.0,
                      sample_interval=1000.0,
                      start_in_steady_state=False,
                      n_pic_iterations=25, n_newt_iterations=4,
                      sliding="basic_weertman",
                      output_dir=None,
                      integrator="implicit", delta_t_implicit=10.0,
                      cfl_factor=0.45, max_delta_t=5.0,
                      k_eigs=1,
                      noise_amplitude=0.25, 
                      noise_correlation_time=50.0,
                      noise_seed=0, acc_floor=-0.5,
                      u0=None, v0=None, h0=None,
                      compute_na=True):
    """
    Ramp accumulation LINEARLY IN TIME from acc_start to acc_end over
    t_total years, PLUS a stochastic component modelled as an
    Ornstein-Uhlenbeck (mean-reverting) process added on top of the
    deterministic ramp:
        noise(t+dt) = noise(t)*exp(-dt/tau) + sigma*sqrt(1-exp(-2dt/tau))*N(0,1)
    exact for any step size dt (not a small-dt approximation), reducing
    to fresh white noise each step as tau -> 0. noise_amplitude=sigma is
    the STATIONARY std dev of the noise (m/yr); noise_correlation_time
    =tau is its decorrelation timescale (yr). noise_amplitude=0 (default)
    recovers the original purely-deterministic ramp.

    The realised (noisy) accumulation is clipped at acc_floor (default
    0) before being used as the source term -- accumulation going
    negative isn't meaningful in this simple forcing model.

    Sample (spectral abscissa, numerical abscissa, geometry) every
    sample_interval years of elapsed time -- since delta_t won't
    generally land exactly on multiples of sample_interval, we trigger
    a sample whenever t crosses the next scheduled sample time.

    integrator="implicit" (default): fully-implicit coupled (u,v,h) step
        via make_coupled_picnewton_solver_function, using a FIXED,
        moderate delta_t_implicit (years) -- deliberately NOT ramped up
        the way solve_steady_state_direct does, since growing delta_t
        toward a pseudo-steady-state solve would defeat the purpose of
        this experiment (tracing genuinely transient behavior, not a
        sequence of instantaneous steady states). Pick delta_t_implicit
        small enough that several steps occur between each
        sample_interval, so the trajectory between samples is actually
        resolved, not just teleported.

    integrator="explicit": the original CFL-limited explicit
        momentum_solver + advection_stepper pair, kept for comparison.
    """
    (lx, ly, nr, nc, x, y, delta_x, delta_y,
     thk, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain

    #thk = jnp.load(f"{nm_home}/bits_of_data/schoof_instability_stuff/6/thk_1000m_step9999.npy")

    temp_field = jnp.zeros_like(C_0) + 263.15

    ny, nx = nr, nc
    p = jnp.zeros_like(q)

    if integrator == "explicit":
        momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
            ny, nx, delta_y, delta_x, b, ice_mask,
            n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
            adv_method="FOU",
            sliding=sliding, temperature_field=temp_field)
    elif integrator == "implicit":
        impl_solver = make_coupled_picnewton_solver_function(
            ny, nx, delta_y, delta_x, b, ice_mask,
            n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
            sliding=sliding, temperature_field=temp_field)
        
        if start_in_steady_state:
            print("Solving initial (NOT VERY!!!!!!) steady state!")
            u, v, h, dhdt, n_used = solve_steady_state_direct(impl_solver, q, p,
                                           jnp.zeros_like(q), jnp.zeros_like(q),
                                           thk, acc_start, delta_t_growth_factor=2,
                                           delta_t_max=1e6, n_outer=15, dhdt_tol=0.5)
            #print("Solving initial steady state!")
            #u, v, h, dhdt, n_used = solve_steady_state_direct(impl_solver, q, p,
            #                               jnp.zeros_like(q), jnp.zeros_like(q),
            #                               thk, acc_start, delta_t_growth_factor=4,
            #                               delta_t_max=1e6, n_outer=15, dhdt_tol=1e-10)

    else:
        raise ValueError(f"unknown integrator {integrator!r}")

    if not start_in_steady_state:
        u = u0 if u0 is not None else jnp.where(thk > 1e-10, 1.0, 0.0)
        v = v0 if v0 is not None else jnp.zeros_like(u)
        h = h0 if h0 is not None else thk

    def acc_ramp(t):
        frac = min(max(t / t_total, 0.0), 1.0)
        return acc_start + (acc_end - acc_start) * frac

    rng = np.random.default_rng(noise_seed)
    acc_noise = 0.0  # current OU-process value, mean-reverts to 0

    def ou_advance(noise, delta_t, tau, sigma):
        if sigma <= 0:
            return 0.0
        tau = max(tau, 1e-6)  # guard div-by-zero; tiny tau ~= fresh draw each step anyway
        decay = np.exp(-delta_t / tau)
        return noise * decay + sigma * np.sqrt(max(1.0 - decay ** 2, 0.0)) * rng.standard_normal()

    history = dict(t=[], acc=[], acc_ramp=[], x_gl=[], dhdt=[], eval=[],
                    sa_evec=[], na_evec=[], omega=[], gap=[], thk=[], u=[], v=[])
    n = ny * nx

    t = 0.0
    next_sample = 0.0  # sample immediately at t=0 too, before any stepping
    step_i = 0

    while (t < t_total) and (np.min(h[h>0])>100):

        if integrator == "implicit":
            delta_t = min(delta_t_implicit, t_total - t)
        else:
            u, v = momentum_solver(q, p, u, v, h)
            max_speed = float(jnp.max(jnp.sqrt(u ** 2 + v ** 2)))
            delta_t = cfl_factor * (float(delta_x) / max(max_speed, 1e-6))
            delta_t = min(delta_t, max_delta_t)
            delta_t = min(delta_t, t_total - t)  # don't overshoot the ramp's end

        acc_noise = ou_advance(acc_noise, delta_t, noise_correlation_time, noise_amplitude)
        acc_val = max(acc_ramp(t) + acc_noise, acc_floor)

        if integrator == "implicit":
            u, v, h_new = impl_solver(q, p, u, v, h, delta_t, accm=acc_val)
            h_old = h
            h = jnp.maximum(h_new, 0.0)
        else:
            source = jnp.where(h > 0, acc_val, 0.0)
            h_old = h
            h = advection_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1),
                                   source=source, delta_t=delta_t)
            h = jnp.maximum(h, 0.0)

        dhdt = float(jnp.max(jnp.abs((h - h_old) / delta_t)))

        t += delta_t
        step_i += 1

        if t >= next_sample or t >= t_total:
            print(f"=== sampling at t={t:.1f} yr (acc={acc_val:.4f} m/yr) ===")

            machinery = build_matvec_machinery(ny, nx, delta_y, delta_x, b, ice_mask,
                                                mucoef_0, C_0, sliding)
            mask = build_grounded_mask(h, b)

            Tv = make_tangent_propagator_matvec(ny, nx, machinery, u, v, h, q, p, acc_val)
            Tv_sub, n_sub, idx = restrict_matvec(Tv, mask)
            lam, sa_evec_sub = leading_eigenpair_matrix_free(Tv_sub, n_sub, k=k_eigs)

            if compute_na:
                TTa = make_tangent_propagator_transpose_matvec(ny, nx, machinery, u, v,
                                                               h, q, p, acc_val)
                TTa_sub, _, _ = restrict_matvec(TTa, mask)
                omega, na_evec_sub = numerical_abscissa_matrix_free(Tv_sub, TTa_sub, n_sub)
            else:
                omega, na_evec_sub = None, None

            sa_evec = embed(sa_evec_sub, idx, n)
            na_evec = embed(na_evec_sub, idx, n)

            if output_dir is not None:
                plot_ss(x, h[0, :], b[0, :], u[0, :], sa_evec, na_evec,
                        os.path.join(output_dir, f"t{int(round(t)):06d}yr.png"),
                        title=f"t={t:.0f} yr, accumulation={acc_val:.4f} m a^-1",
                        axis_limits=[[-1000, 3000], [0, 1500]])

            grounded_now = jnp.where((h + b) > (h * (1 - c.RHO_I / c.RHO_W)), 1, 0)
            gl_idx = int(jnp.max(jnp.where(grounded_now[0] > 0, jnp.arange(nx), -1)))
            x_gl = float(x[gl_idx]) if gl_idx >= 0 else float('nan')

            history['t'].append(t)
            history['acc'].append(acc_val)
            history['acc_ramp'].append(acc_ramp(t))
            history['x_gl'].append(x_gl)
            history['dhdt'].append(dhdt)
            history['eval'].append(lam)
            history['sa_evec'].append(sa_evec)
            history['na_evec'].append(na_evec)
            history['omega'].append(omega)
            history['gap'].append(omega - np.real(lam))
            history['thk'].append(np.array(h))
            history['u'].append(np.array(u))
            history['v'].append(np.array(v))


            print(f"  t={t:.0f} yr  acc={acc_val:.4f} m/yr  x_gl={x_gl/1000:.2f} km  "
                  f"dhdt={dhdt:.4e}  alpha={np.real(lam):.4e}  omega={omega:.4e}  "
                  f"gap={omega-np.real(lam):.4e}")

            save_history(history, output_dir, "time_forced_history.npz")

            while next_sample <= t:
                next_sample += sample_interval

        elif not step_i % 200:
            print(f"step {step_i}: t={t:.1f} yr  acc={acc_val:.4f} m/yr  dhdt={dhdt:.4e}")

    return history



def load_snapshot_at_year(dir_, year, filename="time_forced_history.npz"):
    """Load a saved time_forced_sweep history and pull out the full state
    (thk, u, v, eigenvectors, accumulation) at the sample closest to the
    requested year. Returns a plain dict, plus the ACTUAL sample time
    used (may differ slightly from `year` since sampling only happens at
    CFL/implicit step boundaries, not exactly on round numbers)."""
    path = os.path.join(dir_, filename)
    data = np.load(path, allow_pickle=True)

    t_arr = np.asarray(data['t'], dtype=float)
    idx = int(np.argmin(np.abs(t_arr - year)))
    t_actual = float(t_arr[idx])

    snapshot = dict(
        t=t_actual,
        thk=np.asarray(data['thk'][idx]),
        u=np.asarray(data['u'][idx]),
        v=np.asarray(data['v'][idx]),
        na_evec=np.asarray(data['na_evec'][idx]),
        sa_evec=np.asarray(data['sa_evec'][idx]),
        acc=float(np.asarray(data['acc'])[idx]),
        eval=complex(np.asarray(data['eval'])[idx]),
        omega=float(np.asarray(data['omega'])[idx]),
    )
    print(f"loaded snapshot: requested year={year}, actual sample t={t_actual:.1f} yr, "
          f"acc={snapshot['acc']:.4f} m/yr")
    return snapshot


def add_na_perturbation(dir_, year, perturbation_amplitude=10.0,
                         domain=None, t_total=2000.0, sample_interval=100.0,
                         output_dir=None):
    """
    Load the state at `year` from a previous time_forced_sweep run in
    `dir_`, perturb thickness by perturbation_amplitude * na_evec (the
    numerical-abscissa-maximising direction), and run it forward under
    CONSTANT accumulation (frozen at whatever the actual forcing was at
    that moment -- no more ramp, no more noise) to see whether the
    perturbation actually grows in the real nonlinear dynamics, not just
    in the linearised T.
    """
    if domain is None:
        domain = schoof_scaled(resolution=250, buffer_km=4, C_0_val=10_000, z_scale=1)

    if output_dir is None:
        output_dir = os.path.join(dir_, f"na_perturbation_y{int(round(year))}")

    snap = load_snapshot_at_year(dir_, year)

    accumulation = snap['acc']  # held constant for the whole forward run

    na_evec_2d = snap['na_evec'].reshape(snap['thk'].shape)
    h_perturbed = jnp.asarray(snap['thk']) + perturbation_amplitude * jnp.asarray(na_evec_2d)
    h_perturbed = jnp.maximum(h_perturbed, 0.0)
    h_control = jnp.asarray(snap['thk'])

    print(f"perturbation applied: amplitude={perturbation_amplitude} m, "
          f"max|na_evec|={float(jnp.max(jnp.abs(na_evec_2d))):.4e}, "
          f"max|h_perturbed - h_control|={float(jnp.max(jnp.abs(h_perturbed-h_control))):.4f} m")

    return run_forward_with_perturbation(
        domain, accumulation, h_perturbed, h_control=h_control,
        u0=jnp.asarray(snap['u']), v0=jnp.asarray(snap['v']),
        t_total=t_total, sample_interval=sample_interval,
        output_dir=output_dir, tag=f"na_y{int(round(snap['t']))}")


def add_sa_perturbation(dir_, year, perturbation_amplitude=10.0,
                         domain=None, t_total=2000.0, sample_interval=100.0,
                         output_dir=None):
    """Same as add_na_perturbation but perturbs along sa_evec (the
    leading eigenvector of T itself) instead of na_evec -- a natural
    control experiment: if na_evec genuinely identifies a MORE dangerous
    direction than sa_evec, a matched-amplitude sa perturbation should
    grow less (or not at all) compared to the na one."""
    if domain is None:
        domain = schoof_scaled(resolution=250, buffer_km=4, C_0_val=10_000, z_scale=1)

    if output_dir is None:
        output_dir = os.path.join(dir_, f"sa_perturbation_y{int(round(year))}")

    snap = load_snapshot_at_year(dir_, year)

    accumulation = snap['acc']

    sa_evec_2d = snap['sa_evec'].reshape(snap['thk'].shape)
    h_perturbed = jnp.asarray(snap['thk']) + perturbation_amplitude * jnp.asarray(sa_evec_2d)
    h_perturbed = jnp.maximum(h_perturbed, 0.0)
    h_control = jnp.asarray(snap['thk'])

    return run_forward_with_perturbation(
        domain, accumulation, h_perturbed, h_control=h_control,
        u0=jnp.asarray(snap['u']), v0=jnp.asarray(snap['v']),
        t_total=t_total, sample_interval=sample_interval,
        output_dir=output_dir, tag=f"sa_y{int(round(snap['t']))}")


def run_forward_with_perturbation(domain, accumulation, h_perturbed, h_control=None,
                                   u0=None, v0=None,
                                   n_pic_iterations=40, n_newt_iterations=5,
                                   sliding="basic_weertman",
                                   t_total=2000.0, sample_interval=100.0,
                                   cfl_factor=0.45, max_delta_t=2.0,
                                   output_dir=None, tag=""):
    """
    Step the perturbed state (and, if given, a matched unperturbed
    control) forward under CONSTANT accumulation using the explicit
    momentum_solver + advection_stepper pair -- deliberately NOT the
    large-step implicit integrator here, since this experiment is
    specifically about resolving short-timescale transient growth, which
    big implicit steps would risk numerically damping away.

    The key diagnostic is history['divergence']: ||h_perturbed(t) -
    h_control(t)||, tracked over time. If it grows (even temporarily)
    before eventually decaying, that's direct empirical evidence of real
    transient amplification in the full nonlinear model -- not just a
    property of the linearised T.
    """
    (lx, ly, nr, nc, x, y, delta_x, delta_y,
     thk0, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain

    ny, nx = nr, nc
    p = jnp.zeros_like(q)
    temp_field = jnp.zeros_like(C_0) + 263.15

    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
        ny, nx, delta_y, delta_x, b, ice_mask,
        n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
        adv_method="FOU",
        sliding=sliding, temperature_field=temp_field)

    u_p = u0 if u0 is not None else jnp.where(h_perturbed > 1e-10, 1.0, 0.0)
    v_p = v0 if v0 is not None else jnp.zeros_like(u_p)
    h_p = h_perturbed

    run_control = h_control is not None
    if run_control:
        u_c, v_c = u_p, v_p
        h_c = h_control

    history = dict(t=[], x_gl_perturbed=[], dhdt_perturbed=[], divergence=[])
    if run_control:
        history['x_gl_control'] = []

    def gl_x(h):
        grounded_now = jnp.where((h + b) > (h * (1 - c.RHO_I / c.RHO_W)), 1, 0)
        gl_idx = int(jnp.max(jnp.where(grounded_now[0] > 0, jnp.arange(nx), -1)))
        return float(x[gl_idx]) if gl_idx >= 0 else float('nan')

    t = 0.0
    next_sample = 0.0
    step_i = 0
    source = None  # set below once thickness masks are known each step

    while t < t_total:

        u_p, v_p = momentum_solver(q, p, u_p, v_p, h_p)
        max_speed = float(jnp.max(jnp.sqrt(u_p ** 2 + v_p ** 2)))
        delta_t = cfl_factor * (float(delta_x) / max(max_speed, 1e-6))
        delta_t = min(delta_t, max_delta_t)
        delta_t = min(delta_t, t_total - t)

        source_p = jnp.where(h_p > 0, accumulation, 0.0)
        h_p_old = h_p
        h_p = advection_stepper(u_p.reshape(-1), v_p.reshape(-1), h_p.reshape(-1),
                                 source=source_p, delta_t=delta_t)
        h_p = jnp.maximum(h_p, 0.0)
        dhdt_p = float(jnp.max(jnp.abs((h_p - h_p_old) / delta_t)))

        if run_control:
            u_c, v_c = momentum_solver(q, p, u_c, v_c, h_c)
            source_c = jnp.where(h_c > 0, accumulation, 0.0)
            h_c = advection_stepper(u_c.reshape(-1), v_c.reshape(-1), h_c.reshape(-1),
                                     source=source_c, delta_t=delta_t)
            h_c = jnp.maximum(h_c, 0.0)

        t += delta_t
        step_i += 1

        if t >= next_sample or t >= t_total:
            divergence = float(jnp.linalg.norm(h_p - h_c)) if run_control else float('nan')

            history['t'].append(t)
            history['x_gl_perturbed'].append(gl_x(h_p))
            history['dhdt_perturbed'].append(dhdt_p)
            history['divergence'].append(divergence)
            if run_control:
                history['x_gl_control'].append(gl_x(h_c))

            print(f"  t={t:.1f} yr  x_gl_pert={history['x_gl_perturbed'][-1]/1000:.3f} km  "
                  f"divergence=||h_pert-h_control||={divergence:.4e}")

            if output_dir is not None:
                save_history(history, output_dir, f"perturbation_history_{tag}.npz")

            while next_sample <= t:
                next_sample += sample_interval

    if output_dir is not None:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        axes[0].plot(history['t'], np.array(history['x_gl_perturbed']) / 1000, 'o-', label='perturbed')
        if run_control:
            axes[0].plot(history['t'], np.array(history['x_gl_control']) / 1000, 's-', label='control')
        axes[0].set_ylabel('x_gl (km)'); axes[0].legend()

        axes[1].semilogy(history['t'], history['divergence'], 'o-')
        axes[1].set_ylabel('||h_pert - h_control||')

        axes[2].semilogy(history['t'], history['dhdt_perturbed'], 'o-')
        axes[2].set_ylabel('max |dh/dt| (perturbed)')
        axes[2].set_xlabel('time (yr)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"perturbation_growth_{tag}.png"), dpi=150)
        plt.close()

    return history

#def time_forced_sweep(domain,
#                      start_in_steady_state=False,
#                      acc_start=0.5, acc_end=0.0, t_total=50_000.0,
#                      sample_interval=1000.0,
#                      n_pic_iterations=10, n_newt_iterations=4,
#                      sliding="basic_weertman",
#                      output_dir=None,
#                      integrator="implicit", delta_t_implicit=25.0,
#                      cfl_factor=0.9, max_delta_t=5.0,
#                      k_eigs=1,
#                      u0=None, v0=None, h0=None):
#    """
#    Ramp accumulation LINEARLY IN TIME from acc_start to acc_end over
#    t_total years. Sample (spectral abscissa, numerical abscissa,
#    geometry) every sample_interval years of elapsed time -- since
#    delta_t won't generally land exactly on multiples of sample_interval,
#    we trigger a sample whenever t crosses the next scheduled sample time.
#
#    integrator="implicit" (default): fully-implicit coupled (u,v,h) step
#        via make_coupled_picnewton_solver_function, using a FIXED,
#        moderate delta_t_implicit (years) -- deliberately NOT ramped up
#        the way solve_steady_state_direct does, since growing delta_t
#        toward a pseudo-steady-state solve would defeat the purpose of
#        this experiment (tracing genuinely transient behavior, not a
#        sequence of instantaneous steady states). Pick delta_t_implicit
#        small enough that several steps occur between each
#        sample_interval, so the trajectory between samples is actually
#        resolved, not just teleported.
#
#    integrator="explicit": the original CFL-limited explicit
#        momentum_solver + advection_stepper pair, kept for comparison.
#    """
#    (lx, ly, nr, nc, x, y, delta_x, delta_y,
#     thk, b, C_0, mucoef_0, q, ice_mask, surface, grounded) = domain
#
#    #thk = jnp.load(f"{nm_home}/bits_of_data/schoof_instability_stuff/6/thk_1000m_step9999.npy")
#
#    temp_field = jnp.zeros_like(C_0) + 263.15
#
#    ny, nx = nr, nc
#    p = jnp.zeros_like(q)
#
#    if integrator == "explicit":
#        momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
#            ny, nx, delta_y, delta_x, b, ice_mask,
#            n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
#            adv_method="FOU",
#            sliding=sliding, temperature_field=temp_field)
#    elif integrator == "implicit":
#        impl_solver = make_coupled_picnewton_solver_function(
#            ny, nx, delta_y, delta_x, b, ice_mask,
#            n_pic_iterations, n_newt_iterations, mucoef_0, C_0,
#            sliding=sliding, temperature_field=temp_field)
#
#        if start_in_steady_state:
#            print("Solving initial steady state!")
#            u, v, h, dhdt, n_used = solve_steady_state_direct(impl_solver, q, p,
#                                           jnp.zeros_like(q), jnp.zeros_like(q),
#                                           thk, acc_start, delta_t_growth_factor=4,
#                                           delta_t_max=1e6, n_outer=15, dhdt_tol=1e-10)
#
#
#    else:
#        raise ValueError(f"unknown integrator {integrator!r}")
#
#    if not start_in_steady_state:
#        u = u0 if u0 is not None else jnp.where(thk > 1e-10, 1.0, 0.0)
#        v = v0 if v0 is not None else jnp.zeros_like(u)
#        h = h0 if h0 is not None else thk
#
#    def acc_of_t(t):
#        frac = min(max(t / t_total, 0.0), 1.0)
#        return acc_start + (acc_end - acc_start) * frac
#
#    history = dict(t=[], acc=[], x_gl=[], dhdt=[], eval=[],
#                    sa_evec=[], na_evec=[], omega=[], gap=[])
#    n = ny * nx
#
#    t = 0.0
#    next_sample = 0.0  # sample immediately at t=0 too, before any stepping
#    step_i = 0
#
#    while t < t_total:
#        acc_val = acc_of_t(t)
#
#        if integrator == "implicit":
#            delta_t = min(delta_t_implicit, t_total - t)
#            u, v, h_new = impl_solver(q, p, u, v, h, delta_t, accm=acc_val)
#            h_old = h
#            h = jnp.maximum(h_new, 0.0)
#        else:
#            u, v = momentum_solver(q, p, u, v, h)
#            max_speed = float(jnp.max(jnp.sqrt(u ** 2 + v ** 2)))
#            delta_t = cfl_factor * (float(delta_x) / max(max_speed, 1e-6))
#            delta_t = min(delta_t, max_delta_t)
#            delta_t = min(delta_t, t_total - t)  # don't overshoot the ramp's end
#
#            source = jnp.where(h > 0, acc_val, 0.0)
#            h_old = h
#            h = advection_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1),
#                                   source=source, delta_t=delta_t)
#            h = jnp.maximum(h, 0.0)
#
#        dhdt = float(jnp.max(jnp.abs((h - h_old) / delta_t)))
#
#        t += delta_t
#        step_i += 1
#
#        if t >= next_sample or t >= t_total:
#            print(f"=== sampling at t={t:.1f} yr (acc={acc_val:.4f} m/yr) ===")
#
#            machinery = build_matvec_machinery(ny, nx, delta_y, delta_x, b, ice_mask,
#                                                mucoef_0, C_0, sliding)
#            mask = build_grounded_mask(h, b)
#
#            Tv = make_tangent_propagator_matvec(ny, nx, machinery, u, v, h, q, p, acc_val)
#            Tv_sub, n_sub, idx = restrict_matvec(Tv, mask)
#            lam, sa_evec_sub = leading_eigenpair_matrix_free(Tv_sub, n_sub, k=k_eigs)
#
#            TTa = make_tangent_propagator_transpose_matvec(ny, nx, machinery, u, v, h, q, p, acc_val)
#            TTa_sub, _, _ = restrict_matvec(TTa, mask)
#            omega, na_evec_sub = numerical_abscissa_matrix_free(Tv_sub, TTa_sub, n_sub)
#
#            sa_evec = embed(sa_evec_sub, idx, n)
#            na_evec = embed(na_evec_sub, idx, n)
#
#            if output_dir is not None:
#                plot_ss(x, h[0, :], b[0, :], u[0, :], sa_evec, na_evec,
#                        os.path.join(output_dir, f"t{int(round(t)):06d}yr.png"),
#                        title=f"t={t:.0f} yr, accumulation={acc_val:.4f} m a^-1",
#                        axis_limits=[[-1500, 5500], [0, 1500]])
#
#            grounded_now = jnp.where((h + b) > (h * (1 - c.RHO_I / c.RHO_W)), 1, 0)
#            gl_idx = int(jnp.max(jnp.where(grounded_now[0] > 0, jnp.arange(nx), -1)))
#            x_gl = float(x[gl_idx]) if gl_idx >= 0 else float('nan')
#
#            history['t'].append(t)
#            history['acc'].append(acc_val)
#            history['x_gl'].append(x_gl)
#            history['dhdt'].append(dhdt)
#            history['eval'].append(lam)
#            history['sa_evec'].append(sa_evec)
#            history['na_evec'].append(na_evec)
#            history['omega'].append(omega)
#            history['gap'].append(omega - np.real(lam))
#
#            print(f"  t={t:.0f} yr  acc={acc_val:.4f} m/yr  x_gl={x_gl/1000:.2f} km  "
#                  f"dhdt={dhdt:.4e}  alpha={np.real(lam):.4e}  omega={omega:.4e}  "
#                  f"gap={omega-np.real(lam):.4e}")
#
#            save_history(history, output_dir, "time_forced_history.npz")
#
#            while next_sample <= t:
#                next_sample += sample_interval
#
#        elif not step_i % 200:
#            print(f"step {step_i}: t={t:.1f} yr  acc={acc_val:.4f} m/yr  dhdt={dhdt:.4e}")
#
#    return history

def unsteady_experiment():
    output_dir = os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/17/")
    os.makedirs(output_dir, exist_ok=True)

    # Import your (now-fixed) Schoof transect domain here, e.g.:
    domain = schoof2007_transect_domain(resolution=1000)
    #domain = schoof_scaled(resolution=250)

    # Decreasing-accumulation sweep, matching Schoof's a=0.3 m/yr default
    # as the high-accumulation starting point.
    #t = np.linspace(1, -1, 20)
    #acc_values = 1 + 0.5 * t**3
    acc_values = np.linspace(0.5, 0.05, 40)

    max_itns_to_ss = 2000

    resA = steady_state_accumulation_sweep(domain, acc_values, branch="quasistatic",
                                           max_outer_steps=max_itns_to_ss,
                                           output_dir=output_dir)

    #resB = run_accumulation_sweep(domain, acc_values, branch="transient")

    fig, axes = plt.subplots(3, 1, figsize=(8, 8))
    axes[0].plot(resA['acc'], np.real(resA['eval']), 'o-')#, label='quasistatic')
    #axes[0].plot(resB['acc'], np.real(resB['eval']), 's-', label='transient')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_xlabel('accumulation (m/yr)')
    axes[0].set_ylabel('spectral abscissa (a^-1)')
    #axes[0].legend()
    axes[0].invert_xaxis()
    
    axes[1].plot(resA['acc'], np.real(resA['omega']), 'o-')
    #axes[1].plot(resA['acc'], np.real(resA['gap']), 'o-', label='gap')
    axes[1].set_xlabel('accumulation (m/yr)');
    axes[1].set_ylabel('numerical abscissa (a^-1)')
    #axes[1].legend()
    axes[1].invert_xaxis()
    

    axes[2].plot(resA['acc'], np.array(resA['x_gl'])/1000, 'o-')#, label='quasistatic')
    #axes[1].plot(resB['acc'], resB['x_gl'], 's-', label='transient')
    axes[2].set_xlabel('accumulation (m/yr)')
    axes[2].set_ylabel('x_gl (km)')
    #axes[2].legend()
    axes[2].invert_xaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csd_accumulation_sweep.png"), dpi=150)

def unsteady_time_forced_experiment():
    output_dir = os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/31/")
    os.makedirs(output_dir, exist_ok=True)

    #domain = schoof2007_transect_domain(resolution=1000,
    #                                    C_0_val=4000)
    domain = schoof_scaled(resolution=250,
                           buffer_km=4,
                           C_0_val=14_000,
                           z_scale=0.5)

    resA = time_forced_sweep(domain, acc_start=0.4, acc_end=-0.4,
                             t_total=2_000.0, sample_interval=50.0,
                             noise_amplitude=0.4,
                             noise_correlation_time=50.0,
                             delta_t_implicit=5.0,
                             acc_floor=-1,
                             output_dir=output_dir,
                             start_in_steady_state=True)

    fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(resA['t'], np.real(resA['eval']), 'o-')
    axes[0].axhline(0, color='k', lw=0.5)
    axes[0].set_ylabel('spectral abscissa (a^-1)')

    axes[1].plot(resA['t'], np.real(resA['omega']), 'o-')
    axes[1].set_ylabel('numerical abscissa (a^-1)')

    axes[2].plot(resA['t'], np.array(resA['x_gl']) / 1000, 'o-')
    axes[2].set_ylabel('x_gl (km)')

    axes[3].plot(resA['t'], resA['acc'], '.', ms=3, alpha=0.5, label='noisy')
    axes[3].plot(resA['t'], resA['acc_ramp'], 'k-', lw=1, label='ramp')
    axes[3].set_ylabel('accumulation (m/yr)')
    axes[3].set_xlabel('time (yr)')
    axes[3].legend()

    axes[4].plot(resA['t'], np.array(resA['dhdt']), 'o-')
    axes[4].set_ylabel('dh/dt (m a^-1)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csd_time_forced_sweep.png"), dpi=150)


if __name__ == "__main__":
    #steady_experiment()
    unsteady_time_forced_experiment()
    #add_na_perturbation(os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/27/"), 
    #                    8000, 
    #                    output_dir=os.path.join(nm_home, "bits_of_data", "schoof_instability_stuff/27/")
