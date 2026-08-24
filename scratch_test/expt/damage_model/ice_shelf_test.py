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
from plotting_stuff import show_vel_field, show_vel_field_2, make_plot_mismip_field_function
from grid import add_ghost_cells_fcts, cc_gradient_function, interp_cc_with_ghosts_to_fc_function

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt,\
                              make_time_marcher,\
                              make_coupled_quasi_newton_solver_function,\
                              make_coupled_picnewton_solver_function,\
                              implicit_forward_solver,\
                              solve_steady_state_direct,\
                              make_diva3d_solver,\
                              make_advection_stepper

from standard_domains import mismip_domain, mismip_domain_symm, ice_shelf

sys.path.insert(1, os.path.join(nm_home, 'physics'))
from damage import nye_vertical_damage_function, nye_vertical_damage_surfandbase_function



resolution = 1000
n_levels = 100
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


thk = jnp.load(
    f"{nm_home}/bits_of_data/mismip_plus_experiments/full_attepmt_schoof/ssa/ice0/thickness_WmSlidingC1e4_1km_res_HalfDomain_789.7538years.npy"
)
    



#(
#    lx, ly, nr, nc,
#    x, y, delta_x,
#    delta_y, thk, b,
#    C_0, mucoef_0,
#    q, p,
#    ice_mask, surface,
#    grounded
#) = ice_shelf(resolution=resolution)

#plt.imshow(ice_mask)
#plt.colorbar()
#plt.show()
#plt.close()
#plt.imshow(grounded)
#plt.colorbar()
#plt.show()
#plt.close()
#plt.imshow(C_0)
#plt.colorbar()
#plt.show()
#plt.close()
#plt.imshow(surface)
#plt.colorbar()
#plt.show()
#plt.close()


plot_mismip_field = make_plot_mismip_field_function(b, x, y, y_exaggeration=2)

def ssa_momentum_and_advection():
    return make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt(
        nr, nc, delta_y, delta_x, b,
        n_pic_iterations, n_newt_iterations,
        mucoef_0, C_0, sliding="schoof", temperature_field=None,
        adv_method="PPM")

vel_solver, adv_stepper = ssa_momentum_and_advection()

add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
cc_gradient                                = cc_gradient_function(delta_y, delta_x)
interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(nr, nc)

damage_adv_stepper = make_advection_stepper(nc, nr, delta_x, delta_y,
                                            interp_cc_to_fc,
                                            add_uv_ghost_cells,
                                            add_scalar_ghost_cells,
                                            method="PPM",
                                            conservative=False)

damage_maker = nye_vertical_damage_surfandbase_function(nr, nc, delta_y, delta_x,
                                            add_uv_ghost_cells,
                                            add_scalar_ghost_cells,
                                            cc_gradient,
                                            mucoef_0,
                                            temp_cc=None)
#damage_maker = nye_vertical_damage_function(nr, nc, delta_y, delta_x,
#                                            add_uv_ghost_cells,
#                                            add_scalar_ghost_cells,
#                                            cc_gradient,
#                                            mucoef_0,
#                                            temp_cc=None)
                                            

damage = jnp.zeros_like(thk)+0.01
#damage = damage.at[30:, -70:-65].set(0.75)

u, v = jnp.zeros_like(thk), jnp.zeros_like(thk)

h = thk

ice_mask = jnp.where(h>0, 1, 0)

outdir = f"{nm_home}/bits_of_data/damage_figures/mismip/5/"
os.makedirs(outdir, exist_ok=True)

t_cum = 0
for i in range(200):

    #plot_mismip_field(damage, h, vmin=0, vmax=1, cmap="gnuplot2_r",
    #                  cbar_label="Damage", filepath=f"{outdir}/dam{i}.png",
    #                  title=f"Damage {t_cum:.2f} years"
    #                  )
    plot_mismip_field(damage, h, vmin=0, vmax=0.9, cmap="RdBu_r",
                      cbar_label="Damage", filepath=f"{outdir}/dam{i}.png",
                      title=f"Damage {t_cum:.2f} years"
                      )

    #plt.imshow(damage, vmin=0, vmax=1, cmap="gnuplot2_r")
    #plt.colorbar()
    #plt.title(f"Damage {t_cum:.2f} years")
    #plt.savefig(f"{outdir}/dam{i}.png")
    #plt.close()    
    
    q = jnp.log((1-damage)/mucoef_0)

    #if i==0:
    u, v = vel_solver(q, p, u, v, h)
   
    plot_mismip_field(jnp.sqrt(u**2 + v**2), h, cmap="RdYlBu_r", vmin=0, vmax=4000,
                      cbar_label="Speed (m a^-1)", filepath=f"{outdir}/speed{i}.png",
                      title=f"Speed {t_cum:.2f} years"
                      )

    #plt.imshow(jnp.sqrt(u**2 + v**2 ), cmap="RdYlBu_r", vmin=0, vmax=5000)
    #plt.colorbar()
    #plt.title(f"Speed {t_cum:.2f} years")
    #plt.savefig(f"{outdir}/speed{i}.png")
    #plt.close()    

    z_coords = define_z_coordinates(b, h, n_levels)
    
    delta_t = 0.95*(delta_x/jnp.max(jnp.sqrt(u**2 + v**2)))
    t_cum += delta_t

    damage_adv = damage_adv_stepper(u.reshape(-1), v.reshape(-1), damage.reshape(-1), source=0, delta_t=delta_t)
    damage = damage_maker(q, u, v, z_coords)
    
    #damage = damage_adv
    damage = jnp.minimum(jnp.maximum(damage, damage_adv)*ice_mask, 0.9)

    h = adv_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1), source=0, delta_t=delta_t)
    ice_mask = jnp.where(h>0, 1, 0)
