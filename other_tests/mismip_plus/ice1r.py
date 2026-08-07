#1st party
import os
import sys


#3rd party
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import *
from standard_domains import mismip_domain_symm
from plotting_stuff import show_vel_field, show_vel_field_2

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt,\
                              make_time_marcher,\
                              make_coupled_quasi_newton_solver_function,\
                              make_coupled_picnewton_solver_function,\
                              implicit_forward_solver,\
                              solve_steady_state_direct,\
                              make_diva3d_solver

from standard_domains import mismip_domain, mismip_domain_symm


resolution = 1000
n_levels = 32
n_pic_iterations = 10
n_newt_iterations = 7
max_n_diva_iterations = 40

(
    lx, ly, nr, nc,
    x, y, delta_x,
    delta_y, _, b,
    C_0, mucoef_0, q,
    ice_mask, surface,
    grounded
) = mismip_domain_symm(resolution=resolution, half=True)
p = jnp.zeros_like(q)


xx, yy = jnp.meshgrid(x, y)

temp_field = jnp.zeros_like(q)+265

nm_data_home = f"{nm_home}/bits_of_data/mismip_plus_experiments/full/"

A_ACC     = 0.3        # m a^-1, constant surface accumulation, applied everywhere there's ice
OMEGA     = 0.2        # a^-1,   Ice1 melt-rate factor (Eq. 17)
Z0        = 100.0      # m,      Ice1 melt cutoff depth (Eq. 17)
HC0       = 75.0       # m,      Ice1 reference cavity thickness (Eq. 17)
ICE2_MELT = 100.0      # m a^-1, Ice2 sub-shelf melt rate
ICE2_X0   = 480_000.0  # m,      Ice2 melt only applied where x > this


def floating_ice(h, b):
    return jnp.where((h + b) > (h * (1 - c.RHO_I / c.RHO_W)), 0, 1)


def make_ice0_accumulation(a_acc=A_ACC):
    """Ice0 / the "no melting" re-advance phases (Ice1ra, Ice2ra): constant
    surface accumulation only, m_i = 0."""
    def accumulation_function(h, b, ice_mask):
        return a_acc * ice_mask
    return accumulation_function


make_no_melt_accumulation = make_ice0_accumulation


def make_ice1_accumulation(a_acc=A_ACC, omega=OMEGA, z0=Z0, Hc0=HC0):
    
    def accumulation_function(h, b, ice_mask):
        floating = floating_ice(h, b)
        base = -h * c.RHO_I / c.RHO_W
        Hc = base - b
        melt_rate = omega * jnp.tanh(Hc / Hc0) * jnp.maximum((-base - z0), 0) * floating * ice_mask
        return a_acc * ice_mask - melt_rate
    
    return accumulation_function


def make_ice2_accumulation(a_acc=A_ACC, melt_rate_val=ICE2_MELT, x_threshold=ICE2_X0, xx=xx):
    """Ice2(r/rr): constant surface accumulation + a fixed melt rate applied
    only where x > x_threshold, standing in for a sequence of large
    calving events removing ice far from the grounding line."""
    
    def accumulation_function(h, b, ice_mask):
        floating = floating_ice(h, b)
        melt = jnp.where((xx > x_threshold) & (floating > 0) & (ice_mask > 0),
                          melt_rate_val, 0.0)
        return a_acc * ice_mask - melt
    
    return accumulation_function

# ============================================================================
# Grounding-line extraction from a saved thickness field
# ============================================================================
def extract_grounding_line(thk, b, x, y):
    """Extract grounding-line points from a 2-D thickness field, by linear
    interpolation of the flotation criterion f = h - h_f between grid
    columns straddling each row's grounded/floating transition(s),
    following the point-data convention of Sect. 2.3 of Asay-Davis et al.
    (2016) (one or more xGL/yGL points per row that actually has a
    grounding line; rows with no transition contribute none).

    thk, b : (ny, nx) arrays (works directly on a loaded thickness .npy
              together with the module-level `b`)
    x, y   : 1-D coordinate vectors of length nx, ny

    Returns (xGL, yGL) as 1-D numpy arrays.
    """
    thk = np.asarray(thk)
    b = np.asarray(b)
    x = np.asarray(x)
    y = np.asarray(y)

    h_f = np.maximum(0.0, -(c.RHO_W / c.RHO_I) * b)
    f = thk - h_f          # > 0 grounded, < 0 floating
    ice = thk > 0

    xGL, yGL = [], []
    for j in range(thk.shape[0]):
        row_f, row_ice = f[j, :], ice[j, :]
        for i in range(len(x) - 1):
            if not (row_ice[i] and row_ice[i + 1]):
                continue
            if row_f[i] == 0.0:
                xGL.append(x[i]); yGL.append(y[j])
                continue
            if (row_f[i] > 0) != (row_f[i + 1] > 0):
                frac = row_f[i] / (row_f[i] - row_f[i + 1])
                xGL.append(x[i] + frac * (x[i + 1] - x[i]))
                yGL.append(y[j])

    return np.array(xGL), np.array(yGL)


def extract_grounding_line_from_file(thickness_npy_path, b=b, x=x, y=y):
    thk = jnp.load(thickness_npy_path)
    return extract_grounding_line(thk, b, x, y)


def grounded_area(thk, b, delta_x, delta_y, mirror_y=True):
    thk = np.asarray(thk)
    b = np.asarray(b)
    h_f = np.maximum(0.0, -(c.RHO_W / c.RHO_I) * b)
    is_grounded = (thk > 0) & (thk > h_f)
    weights = np.ones(thk.shape[0])
    if mirror_y:
        weights[1:] = 2.0
    return float(np.sum(is_grounded * weights[:, None]) * delta_x * delta_y)

# ============================================================================
# Plotting with an independently-scaled x/y aspect ratio
# ============================================================================
def show_field_scaled(field, x=x, y=y, ax=None, cmap="viridis", vmin=None, vmax=None,
                       cbar_label=None, title=None, y_exaggeration=6.0,
                       xlabel="x (km)", ylabel="y (km)", figsize=(10, 4)):
    
    x = np.asarray(x)
    y = np.asarray(y)
    field = np.asarray(field)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    extent = [x[0] / 1e3, x[-1] / 1e3, y[0] / 1e3, y[-1] / 1e3]
    im = ax.imshow(field, origin="lower", extent=extent, cmap=cmap,
                    vmin=vmin, vmax=vmax, aspect=y_exaggeration)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if cbar_label:
        cbar.set_label(cbar_label)
    if own_fig:
        plt.tight_layout()

    return ax, im


def show_field_with_gl_scaled(field, thk, b=b, x=x, y=y, ax=None, gl_color="k", **kwargs):
    """show_field_scaled(...) with the grounding line (computed from thk, b)
    overlaid as a line."""
    ax, im = show_field_scaled(field, x=x, y=y, ax=ax, **kwargs)
    xGL, yGL = extract_grounding_line(thk, b, x, y)
    if len(xGL):
        order = np.argsort(yGL)
        ax.plot(np.asarray(xGL)[order] / 1e3, np.asarray(yGL)[order] / 1e3,
                gl_color + "-", lw=1.5)
    return ax, im


def plot_grounded_area_timeseries(series, ax=None, title="Grounded area vs time"):
    """series: dict[label] -> (times (a), areas (m^2)), mirroring Fig. 4 of
    Asay-Davis et al. (2016). Areas are shown in 10^3 km^2, matching the
    paper's y-axis."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    for label, (t, a) in series.items():
        ax.plot(np.asarray(t), np.asarray(a) / 1e9, label=label)
    ax.set_xlabel("Time (a)")
    ax.set_ylabel(r"Grounded area ($10^3$ km$^2$)")
    ax.legend()
    if title:
        ax.set_title(title)
    if own_fig:
        plt.tight_layout()
    return ax




#def explicit_SSA_ice1ra_experiment():
#    thk = jnp.load(f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy")
#   
#    expl_dir_ = f"{nm_home}/bits_of_data/mismip_plus_experiments/all_experimenta/ice1/expl_ssa/"
#    os.makedirs(expl_dir_, exist_ok=True)
#
#    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(
#                                                  nr, nc,
#                                                  delta_y,
#                                                  delta_x,
#                                                  b,
#                                                  ice_mask,
#                                                  n_pic_iterations,
#                                                  n_newt_iterations,
#                                                  mucoef_0,
#                                                  C_0,
#                                                  sliding="basic_weertman",
#                                                  temperature_field=temp_field,
#                                                )
#    
#    time_marcher = make_time_marcher(momentum_solver, advection_stepper, 
#                                     delta_x, b,
#                                     max_n_timesteps=1000,
#                                     accumulation_function=accumulation_function_1, 
#                                     dir_=expl_dir_,
#                                     max_t=100)
#    
#    u_va, v_va, thk_final, dhdt_final = time_marcher(q, p, thk)
#
#
#explicit_SSA_ice1r_experiment()



# ============================================================================
# Experiment runners (SSA/Picard-Newton and DIVA)
# ============================================================================

def ssa_momentum_and_advection():
    return make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt(
        nr, nc, delta_y, delta_x, b,
        n_pic_iterations, n_newt_iterations,
        mucoef_0, C_0, sliding="basic_weertman", temperature_field=temp_field,
        adv_method="PPM")
 
 
def diva_momentum_and_advection():
    return make_diva3d_solver(
        nr, nc, delta_y, delta_x, n_levels, b, ice_mask, n_diva_iterations,
        mucoef_0, C_0, sliding="basic_weertman", temperature_field=temp_field)
 
 
solvers = {"ssa": ssa_momentum_and_advection, "diva": diva_momentum_and_advection}


def run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                  accumulation_function, out_dir,
                                  max_n_timesteps=1000, max_t=100, t_start=0.0):
    os.makedirs(out_dir, exist_ok=True)
    time_marcher = make_time_marcher(momentum_solver, advection_stepper,
                                     delta_x, b,
                                     max_n_timesteps=max_n_timesteps,
                                     accumulation_function=accumulation_function,
                                     dir_=out_dir, max_t=max_t, t_start=t_start)
    return time_marcher(q, p, thk_init)

 
def solver_out_root(solver):
    return os.path.join(nm_data_home, solver)


def run_ice0(thk_init, solver="ssa", max_t=100, max_n_timesteps=1000):
    """100-year control run, m_i = 0."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice0")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice0_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t)


def run_ice1r(thk_init, solver="ssa", max_t=100, max_n_timesteps=1000):
    """100-year run with Eq. (17) melt-induced retreat."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice1r")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice1_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t)


def run_ice1ra(thk_init, t_start=100, solver="ssa", max_t=200, max_n_timesteps=2000):
    """Continues from the end of Ice1r with no melting (advance phase)."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice1ra")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_no_melt_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t, t_start=t_start)


def run_ice1rr(thk_init, t_start=100, solver="ssa", max_t=1000, max_n_timesteps=9000):
    """Optional: continues Ice1r with melting on to t=1000a."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice1rr")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice1_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t, t_start=t_start)


def run_ice2r(thk_init, solver="ssa", max_t=100, max_n_timesteps=1000):
    """100-year "calving-event" run: fixed 100 m/a melt where x>480km."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice2r")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice2_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t)


def run_ice2ra(thk_init, t_start=100, solver="ssa", max_t=200, max_n_timesteps=2000):
    """Continues from the end of Ice2r with no melting (advance phase)."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice2ra")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_no_melt_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t, t_start=t_start)


def run_ice2rr(thk_init, t_start=100, solver="ssa", max_t=1000, max_n_timesteps=9000):
    """Optional: continues Ice2r with melting on to t=1000a."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice2rr")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice2_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t, t_start=t_start)


#def explicit_SSA_ice1r_experiment():
#    thk = jnp.load(f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy")
#    expl_dir_ = f"{nm_home}/bits_of_data/mismip_plus_experiments/ice1r/expl_ssa/1/"
#    os.makedirs(expl_dir_, exist_ok=True)
#
#    momentum_solver, advection_stepper = _ssa_momentum_and_advection()
#
#    time_marcher = make_time_marcher(momentum_solver, advection_stepper,
#                                     delta_x, b,
#                                     max_n_timesteps=1000,
#                                     accumulation_function=make_ice1_accumulation(),
#                                     dir_=expl_dir_,
#                                     max_t=100)
#
#    return time_marcher(q, p, thk)


if __name__ == "__main__":
    thk_init = jnp.load(
        f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy"
    )
    #run_ice1r(thk_init, solver="ssa")

    #starting_thickness = jnp.load(
    #        solver_out_root("ssa")+"/ice1r/thickness_WmSlidingC1e4_1km_res_HalfDomain_100.0years.npy"
    #                             )
    #run_ice1ra(starting_thickness, solver="ssa")
    #run_ice1rr(starting_thickness, solver="ssa")

    run_ice2r(thk_init, solver="ssa")
    #run_ice2ra(solver="ssa")
    ## run_ice2rr(solver="ssa")

    # --- DIVA suite (same experiments, DIVA velocity solve) -------------
    # run_ice1r(thk_init, solver="diva")
    # run_ice1ra(solver="diva")
    # run_ice2r(thk_init, solver="diva")
    # run_ice2ra(solver="diva")

    # --- example post-processing ----------------------------------------
    # xGL, yGL = extract_grounding_line_from_file(
    #     _latest_thickness_file(os.path.join(solver_out_root("ssa"), "ice1r")))
    # show_field_with_gl_scaled(thk_init, thk_init, title="Ice1r initial state",
    #                            cbar_label="thickness (m)")
    # plt.show()
