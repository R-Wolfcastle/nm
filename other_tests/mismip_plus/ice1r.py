#1st party
import os
import sys
import re
import glob

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
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
n_pic_iterations = 50
n_newt_iterations = 50
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

temp_field = jnp.zeros_like(q)+265.43

nm_data_home = f"{nm_home}/bits_of_data/mismip_plus_experiments/full_attepmt_schoof/"

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
                       cbar_label=None, title=None, y_exaggeration=4.0,
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
#                                                  sliding="schoof",
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
# Loading a saved thickness series + summary plots
# ============================================================================
_THICKNESS_RE = re.compile(
    r"thickness_WmSlidingC1e4_1km_res_HalfDomain_([0-9]+\.[0-9]+)years\.npy$")

def load_thickness_series(dir_):
    """Scan dir_ for thickness_WmSlidingC1e4_1km_res_HalfDomain_<t>years.npy
    files and return (times, thicknesses): times as a sorted 1-D numpy
    array (years), thicknesses as a list of (ny,nx) numpy arrays in the
    same order."""
    files = glob.glob(os.path.join(dir_, "thickness_WmSlidingC1e4_1km_res_HalfDomain_*years.npy"))
    entries = []
    for f in files:
        m = _THICKNESS_RE.search(os.path.basename(f))
        if m:
            entries.append((float(m.group(1)), f))
    if not entries:
        raise FileNotFoundError(f"No matching thickness files found in {dir_}")
    entries.sort(key=lambda e: e[0])
    times = np.array([t for t, _ in entries])
    thicknesses = [np.load(f) for _, f in entries]
    return times, thicknesses


def grounded_area_series(dir_):
    """(times, areas in m^2) for every saved thickness snapshot in dir_."""
    times, thks = load_thickness_series(dir_)
    areas = np.array([grounded_area(thk, b, delta_x, delta_y) for thk in thks])
    return times, areas


def combined_grounded_area_series(root_dir, experiment, solver="ssa"):
    """Concatenates the r and ra phases (e.g. ice1r + ice1ra) of `experiment`
    ('ice1' or 'ice2') into one continuous (times, areas) timeline. The ra
    phase's saved times already continue from the r phase's final time (via
    make_time_marcher's t_start), so this is just a concatenate + sort."""
    r_dir = os.path.join(root_dir, f"{experiment}r")
    ra_dir = os.path.join(root_dir, f"{experiment}ra")

    t_r, a_r = grounded_area_series(r_dir)
    t_ra, a_ra = grounded_area_series(ra_dir)

    t_all = np.concatenate([t_r, t_ra])
    a_all = np.concatenate([a_r, a_ra])
    order = np.argsort(t_all)
    return t_all[order], a_all[order]


def plot_grounded_area_ice1_ice2(root_dir=None, solver="ssa", out_path=None):
    """Grounded area vs time for ice1(r+ra) and ice2(r+ra) on one figure,
    matching the style of Fig. 4 in Asay-Davis et al. (2016)."""
    root_dir = root_dir or solver_out_root(solver)

    series = {
        "Ice1 (r+ra)": combined_grounded_area_series(root_dir, "ice1", solver=solver),
        "Ice2 (r+ra)": combined_grounded_area_series(root_dir, "ice2", solver=solver),
    }

    ax = plot_grounded_area_timeseries(series, title="Grounded area: Ice1 vs Ice2")

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return ax

def plot_experiment_summary_panels(root_dir, experiment, target_times=(0, 100, 200),
                                    solver="ssa", speed_log=True, out_path=None):
    """For `experiment` ('ice1' or 'ice2'), find the saved thickness
    snapshot closest to each of target_times (searching the combined r+ra
    series), solve for velocity at each, and save a figure with thickness
    (top row) and speed (bottom row) -- each with the grounding line
    overlaid -- one column per requested time."""
    root_dir = root_dir or _solver_out_root(solver)
    r_dir = os.path.join(root_dir, f"{experiment}r")
    ra_dir = os.path.join(root_dir, f"{experiment}ra")

    t_r, thk_r = load_thickness_series(r_dir)
    t_ra, thk_ra = load_thickness_series(ra_dir)
    times = np.concatenate([t_r, t_ra])
    thicknesses = thk_r + thk_ra
    order = np.argsort(times)
    times = times[order]
    thicknesses = [thicknesses[i] for i in order]

    momentum_solver, _ = _SOLVER_BUILDERS[solver]()

    n = len(target_times)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 7))
    if n == 1:
        axes = axes.reshape(2, 1)

    speed_kwargs = dict(cmap="RdYlBu_r")

    for col, target_t in enumerate(target_times):
        actual_t, thk = _closest_snapshot(times, thicknesses, target_t)
        u_va, v_va = diagnostic_velocity(thk, momentum_solver)
        speed = np.sqrt(u_va**2 + v_va**2)

        ax_thk, im_thk = show_field_with_gl_scaled(
            thk, thk, ax=axes[0, col], cmap="Blues", vmin=0,
            title=f"t = {actual_t:.1f} a  --  thickness",
            cbar_label="thickness (m)")

        if speed_log:
            norm = LogNorm(vmin=max(np.min(speed[speed > 0]), 1e-2), vmax=np.max(speed) + 1e-6)
            ax_spd, im_spd = show_field_with_gl_scaled(
                speed, thk, ax=axes[1, col], vmin=None, vmax=None,
                title=f"t = {actual_t:.1f} a  --  speed", cbar_label="speed (m/a)",
                **speed_kwargs)
            im_spd.set_norm(norm)
        else:
            show_field_with_gl_scaled(
                speed, thk, ax=axes[1, col],
                title=f"t = {actual_t:.1f} a  --  speed", cbar_label="speed (m/a)",
                **speed_kwargs)

    fig.suptitle(f"{experiment} ({solver})", y=1.02)
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig, axes


# ============================================================================
# Experiment runners (SSA/Picard-Newton and DIVA)
# ============================================================================

def ssa_momentum_and_advection():
    return make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt(
        nr, nc, delta_y, delta_x, b,
        n_pic_iterations, n_newt_iterations,
        mucoef_0, C_0, sliding="schoof", temperature_field=temp_field,
        adv_method="PPM")
 
 
def diva_momentum_and_advection():
    return make_diva3d_solver(
        nr, nc, delta_y, delta_x, n_levels, b, ice_mask, max_n_diva_iterations,
        mucoef_0, C_0, sliding="schoof", temperature_field=temp_field)
 
 
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


def run_ice0(thk_init, t_start=789.7538, solver="ssa", max_t=5000, max_n_timesteps=10_000):
    """100-year control run, m_i = 0."""
    momentum_solver, advection_stepper = solvers[solver]()
    out_dir = os.path.join(solver_out_root(solver), "ice0")
    return run_time_marched_experiment(momentum_solver, advection_stepper, thk_init,
                                       make_ice0_accumulation(), out_dir,
                                       max_n_timesteps=max_n_timesteps, max_t=max_t,
                                       t_start=t_start)


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

    #################### SSA ########################

    ######Ice1############

    #thk_init = jnp.load(
    #    #f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy"
    #    f"{nm_home}/bits_of_data/mismip_plus_experiments/full_attepmt_schoof/ssa/ice0/thickness_WmSlidingC1e4_1km_res_HalfDomain_789.7538years.npy"
    #)
    #
    #run_ice0(thk_init, solver="ssa")
    #raise
    
    thk_init = jnp.load(
        #f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy"
        f"{nm_home}/bits_of_data/mismip_plus_experiments/full_attepmt_schoof/ssa/ice0/thickness_WmSlidingC1e4_1km_res_HalfDomain_1360.1363years.npy"
    )
    
    run_ice1r(thk_init, solver="ssa")

    starting_thickness = jnp.load(
            solver_out_root("ssa")+"/ice1r/thickness_WmSlidingC1e4_1km_res_HalfDomain_100.0000years.npy"
                                 )
    run_ice1ra(starting_thickness, solver="ssa")
    run_ice1rr(starting_thickness, solver="ssa")


    ######Ice2############


    #thk_init = jnp.load(
    #    f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy"
    #)
    #run_ice2r(thk_init, solver="ssa")

    #starting_thickness = jnp.load(
    #        solver_out_root("ssa")+"/ice2r/thickness_WmSlidingC1e4_1km_res_HalfDomain_100.0000years.npy"
    #        #solver_out_root("ssa")+"/ice2ra/thickness_WmSlidingC1e4_1km_res_HalfDomain_160.7852years.npy"
    #                             )
    #run_ice2ra(starting_thickness, solver="ssa")
    ####run_ice2rr(starting_thickness, solver="ssa")

   
    raise







    ################### DIVA ########################

    #####Ice1############
    
    thk_init = jnp.load(
        f"{nm_home}/bits_of_data/damage/mismip/diva/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_DIVA_765.7years.npy"
    )
    run_ice1r(thk_init, solver="diva")

    #starting_thickness = jnp.load(
    #        solver_out_root("diva")+"/ice1r/thickness_WmSlidingC1e4_1km_res_HalfDomain_100.0000years.npy"
    #                             )
    #run_ice1ra(starting_thickness, solver="diva")
    ####run_ice1rr(starting_thickness, solver="diva")


    ######Ice2############


    #run_ice2r(thk_init, solver="diva")
    #
    #starting_thickness = jnp.load(
    #        solver_out_root("diva")+"/ice2r/thickness_WmSlidingC1e4_1km_res_HalfDomain_100.0000years.npy"
    #        #solver_out_root("ssa")+"/ice2ra/thickness_WmSlidingC1e4_1km_res_HalfDomain_160.7852years.npy"
    #                             )
    #run_ice2ra(starting_thickness, solver="diva")
    ####run_ice2rr(starting_thickness, solver="diva")


    #plot_grounded_area_ice1_ice2(solver_out_root("ssa"), solver="ssa",
    #    out_path=f"{nm_data_home}/plots/ssa_grounded_area_ice1_vs_ice2.png")
    #plot_grounded_area_ice1_ice2(solver_out_root("diva"), solver="diva",
    #    out_path=f"{nm_data_home}/plots/diva_grounded_area_ice1_vs_ice2.png")
    
    #plot_experiment_summary_panels(solver_out_root("ssa"), "ice1",
    #    target_times=(0, 100, 200), solver="ssa",
    #    out_path=f"{nm_data_home}/plots/ssa/ice1_summary.png")
    #
    #plot_experiment_summary_panels(solver_out_root("ssa"), "ice2",
    #    target_times=(0, 100, 200), solver="ssa",
    #    out_path=f"{nm_data_home}/plots/ssa/ice2_summary.png")
