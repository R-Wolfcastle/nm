#1st party
import os
import sys

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from osgeo import gdal

#local apps
nm_home = os.environ['NM_HOME']

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import *
from plotting_stuff import make_plot_mismip_field_function
from grid import add_ghost_cells_fcts, cc_gradient_function, interp_cc_with_ghosts_to_fc_function

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt

sys.path.insert(1, os.path.join(nm_home, 'physics'))
from damage import make_isotropic_creep_damage_stepper


#in_dir  = f"{nm_home}/bits_of_data/COOKING_TEA_BREAK/annual_ip_data_wpp/500m_res/"
#out_dir = f"{nm_home}/bits_of_data/COOKING_TEA_BREAK/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0008_50its_measuresCprior/500m_res/"

in_dir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_data_wpp/500m_res/"
out_dir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0008_50its_measuresCprior/500m_res/"

res = 500

def define_cook_problem(year):
    with xr.open_dataset(f"{in_dir}/{year}.nc") as nc_file:
        phi_0, q_ig, C_0, p_ig,\
        topg, thk, speed_obs, uc  = (np.flipud(nc_file[var_].values) for var_ in
                                     ["phi_0", "q_ig", "c_one_0", "p_ig", "topg", "thk", "uo", "uc"])

    nr, nc = topg.shape
    ice_mask = np.where(thk>0.01, 1, 0)

    q_out_ds = gdal.Open(f"{out_dir}/q_out_{year}.tiff", gdal.GA_ReadOnly)
    q_out    = q_out_ds.ReadAsArray()
    q_out_ds = None
    p_out_ds = gdal.Open(f"{out_dir}/p_out_{year}.tiff", gdal.GA_ReadOnly)
    p_out    = p_out_ds.ReadAsArray()
    p_out_ds = None

    phi_out = phi_0*jnp.exp(q_out)
    C_out = C_0*jnp.exp(p_out)

    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)

    return nr*res, nc*res, nr, nc,\
           thk, topg, C_out, phi_out,\
           q_out, jnp.zeros_like(q_out),\
           ice_mask, grounded


lx, ly, nr, nc,\
thk, b, C, mucoef_0,\
q, p, ice_mask, grounded = define_cook_problem("2025")

#plt.imshow(1-mucoef_0*jnp.exp(q))
#plt.show()

mucoef_0 = jnp.where(grounded==1, mucoef_0, 1)

delta_y, delta_x = res, res
x = jnp.arange(nc)*res
y = jnp.arange(nr)*res

n_levels = 100
n_pic_iterations = 50
n_newt_iterations = 50

plot_field = make_plot_mismip_field_function(b, x, y, reflect=False, y_exaggeration=1)

temperature_field = 253.15

vel_solver, adv_stepper = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap_dt(
    nr, nc, delta_y, delta_x, b,
    n_pic_iterations, n_newt_iterations,
    mucoef_0, C, sliding="linear",
    adv_method="PPM",
    temperature_field=temperature_field)

add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
cc_gradient                                = cc_gradient_function(delta_y, delta_x)
interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(nr, nc)

damage_stepper = make_isotropic_creep_damage_stepper(nc, nr, delta_x, delta_y,
                                                     interp_cc_to_fc,
                                                     add_uv_ghost_cells,
                                                     add_scalar_ghost_cells,
                                                     cc_gradient,
                                                     mucoef_0,
                                                     method="PPM")

outdir = f"{nm_home}/bits_of_data/damage_figures/cook_creep/8/"
os.makedirs(outdir, exist_ok=True)


def initialize_damage_from_va(va_damage, D_background=0.01, D_max=c.dmg.D_max):
    f = jnp.clip((va_damage - D_background)/(D_max - D_background), 0.0, 1.0)
    n_ruptured = jnp.round(0.5*f*n_levels).astype(jnp.int32)
    layer_idx = jnp.arange(n_levels)
    ruptured = (layer_idx[None,None,:] < n_ruptured[...,None]) | \
               (layer_idx[None,None,:] >= n_levels - n_ruptured[...,None])
    return jnp.where(ruptured, D_max, D_background)


u, v = jnp.zeros_like(thk), jnp.zeros_like(thk)
h = thk

z_coords = define_z_coordinates(b, h, n_levels)


va_damage = 1-mucoef_0*jnp.exp(q)
damage = initialize_damage_from_va(va_damage)

#damage = jnp.zeros((nr, nc, n_levels)) + 0.01
#va_damage = vertically_average(damage, z_coords)

t_cum = 0
for i in range(1000):
    
    grounded = jnp.where((b+h)>(h*(1-c.RHO_I/c.RHO_W)), 1, 0)

    plot_field(va_damage, h, vmin=0, vmax=0.9, cmap="RdBu_r",
              cbar_label="Damage", filepath=f"{outdir}/dam{i}.png",
              title=f"Damage {t_cum:.2f} years", reflect_gl_y=True)
    plt.close()

    q = jnp.log((1-va_damage)/mucoef_0)
    u, v = vel_solver(q, p, u, v, h)

    plot_field(jnp.sqrt(u**2 + v**2), h, cmap="RdYlBu_r", vmin=0, vmax=2000,
              cbar_label="Speed (m a^-1)", filepath=f"{outdir}/speed{i}.png",
              title=f"Speed {t_cum:.2f} years", reflect_gl_y=True)
    plt.close()

    delta_t = 0.95*(delta_x/jnp.max(jnp.sqrt(u**2 + v**2)))

    damage_new, va_damage_new, delta_t_used = damage_stepper(q, u, v, damage, z_coords, delta_t)
    t_cum += delta_t_used

    #NOTE: ONLY UPDATE FLOATING ICE
    damage = damage*grounded[..., None] + damage_new*(1-grounded[..., None])
    va_damage = va_damage*grounded + va_damage_new*(1-grounded)

    #h = adv_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1), source=0, delta_t=delta_t)

    z_coords_new = define_z_coordinates(b, h, n_levels)
    damage = interp_field_onto_new_zs(damage, z_coords, z_coords_new)
    z_coords = z_coords_new
