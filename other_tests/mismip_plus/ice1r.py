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

(
    lx, ly, nr, nc,
    x, y, delta_x,
    delta_y, _, b,
    C_0, mucoef_0, q,
    ice_mask, surface,
    grounded
) = mismip_domain_symm(resolution=resolution, half=True)
p = jnp.zeros_like(q)


temp_field = jnp.zeros_like(q)+265


def explicit_SSA_ice1r_experiment():
    thk = jnp.load(f"{nm_home}/bits_of_data/damage/mismip/expl/2/thickness_WmSlidingC1e4_1km_res_HalfDomain_8998.4years.npy")
   
    expl_dir_ = f"{nm_home}/bits_of_data/mismip_plus_experiments/ice1r/expl_ssa/1/"
    os.makedirs(expl_dir_, exist_ok=True)

    def accumulation_function_1(h, b, ice_mask):
        omega = 0.2
        z0    = 100
        Hc0   = 75
        
        floating = jnp.where((h+b)>(h*(1-c.RHO_I/c.RHO_W)), 0, 1)
        
        base = -h*c.RHO_I/c.RHO_W

        Hc = base - b

        melt_rate = - omega * jnp.tanh(Hc/Hc0) * jnp.maximum((-base - z0), 0) * floating * ice_mask

        return melt_rate

    momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(nr, nc,
                                                  delta_y,
                                                  delta_x,
                                                  b,
                                                  ice_mask,
                                                  n_pic_iterations,
                                                  n_newt_iterations,
                                                  mucoef_0,
                                                  C_0,
                                                  sliding="basic_weertman",
                                                  temperature_field=temp_field,
                                                )
    
    time_marcher = make_time_marcher(momentum_solver, advection_stepper, 
                                     delta_x, b,
                                     max_n_timesteps=1000,
                                     accumulation_function=accumulation_function_1, 
                                     dir_=expl_dir_,
                                     max_t=100)
    
    u_va, v_va, thk_final, dhdt_final = time_marcher(q, p, thk)


explicit_SSA_ice1r_experiment()
