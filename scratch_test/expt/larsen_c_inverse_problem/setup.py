
#1st party
from pathlib import Path
import sys
import time
from functools import partial

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils")
from sparsity_utils import (scipy_coo_to_csr,
                           basis_vectors_and_coords_2d_square_stencil,
                           make_sparse_jacrev_fct_new, make_sparse_jacrev_fct_shared_basis)

import constants_years as c
#import constants as c
from plotting_stuff import (show_vel_field, make_gif, show_damage_field,
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette)
from grid import *

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers")
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp
from nonlinear_solvers import (forward_adjoint_and_second_order_adjoint_solvers,
                                make_newton_velocity_solver_function_custom_vjp,
                                make_picard_velocity_solver_function_custom_vjp,
                                make_picnewton_velocity_solver_function_cvjp)

#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp, custom_jvp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy
from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
import xarray as xr

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)


def soft_grow(binary, sigma=1.0, mode='reflect'):
    bin_float = binary.astype(np.float32)
    blurred   = gaussian_filter(bin_float, sigma=sigma, mode=mode)
    
    # Keep original ones exactly 1, while letting blur extend into zeros
    out = np.maximum(blurred, bin_float)
    
    # If you want literal 1.0 on original foreground to avoid any FP quirks:
    out[binary.astype(bool)] = 1.0
    return out


def setup_ls_domain():

    data_nc_fp = "/Users/eartsu/new_model/testing/nm/bits_of_data/LC_EnvBm.nc"
    
    ds = xr.open_dataset(data_nc_fp)
    
    x = ds["x"]
    y = ds["y"]
    
    nc_res = x[1]-x[0]
    
    #res_inc_factor = 32
    res_inc_factor = 16
    
    new_res = nc_res*res_inc_factor
    
    new_x = ds["x"].values[::res_inc_factor]
    new_y = ds["y"].values[::res_inc_factor]

    #lx = new_x[-1]-new_x[0]
    #ly = new_y[-1]-new_y[0]

    
    thk  = ds["thk"].interp(x=new_x, y=new_y, method="nearest")
    topg = ds["topg"].interp(x=new_x, y=new_y, method="nearest")
    
    uo = ds["uo"].interp(x=new_x, y=new_y, method="nearest")
    uc = ds["uc"].interp(x=new_x, y=new_y, method="nearest")
    
    thk  = jnp.array(thk[80:-50,25:-25])[::-1, :]
    topg = jnp.array(topg[80:-50,25:-25])[::-1, :]
    uo = jnp.array(uo[80:-50,25:-25])[::-1, :]
    uc = jnp.array(uc[80:-50,25:-25])[::-1, :]
    
    
    #do some erosion away from CF and GL!
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    
    
    #thk  = jnp.array(thk[40:-25,12:-12])[::-1, :]
    #topg = jnp.array(topg[40:-25,12:-12])[::-1, :]
    #uo = jnp.array(uo[40:-25,12:-12])[::-1, :]
    #uc = jnp.array(uc[40:-25,12:-12])[::-1, :]
    
    
    #plt.imshow(uo)
    #plt.show()
    #plt.imshow(uc, vmin=0, vmax=1)
    #plt.show()
    
    
    nr, nc = thk.shape
    
    
    b = topg.copy()
    
    
    s_gnd = thk + b
    s_flt = thk * (1-c.RHO_I/c.RHO_W)
    
    grounded = jnp.where(s_gnd>=s_flt, 1, 0)
    
    
    C = jnp.zeros_like(grounded)+1e4
    C = C*soft_grow(grounded)
    #C = jnp.where(soft_grow(grounded==1, 1e4, 0)
    C = jnp.where(thk==0, 1, C)
    
    #plt.figure(figsize=(6,6))
    ##plt.imshow(jnp.log10(C), vmin=0, vmax=20, cmap="RdBu_r")
    #plt.imshow(jnp.log10(C), cmap="RdBu_r")
    ##plt.imshow(thk, vmin=0, vmax=600, cmap="magma")
    ##plt.imshow(thk, cmap="magma")
    #plt.show()
    
    
    delta_x = new_x[1]-new_x[0]
    delta_y = new_y[1]-new_y[0]
    

    mucoef_0 = jnp.ones_like(thk)*3
    q = jnp.zeros_like(thk)

    ice_mask = jnp.where(thk>0, 1, 0)

    #plt.imshow(ice_mask)
    #plt.show()
    
    
    return nr, nc, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, uo, uc

nr, nc, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, uo, uc = setup_ls_domain()


#plt.imshow(uo)
#plt.colorbar()
#plt.show()
#raise

n_pic_iterations = 30
n_newt_iterations = 15
u_init = jnp.zeros_like(thk)
v_init = jnp.zeros_like(thk)


def misfit(u_mod, v_mod, q, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    return jnp.sum(mask.reshape(-1) * (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2)

solver = make_picnewton_velocity_solver_function_cvjp(nr, nc,
                                                      delta_y,
                                                      delta_x,
                                                      b, ice_mask,
                                                      n_pic_iterations,
                                                      n_newt_iterations,
                                                      mucoef_0)
#u_out, v_out = solver(q, C, u_init, v_init, thk)
#
#show_vel_field(u_out, v_out, vmin=0, vmax=3000)
#raise


def lbfgsb_function(misfit_functional, misfit_fctl_args=None, iterations=50):
    def reduced_functional(q):
        u_out, v_out = solver(q.reshape(nr, nc), C, u_init, v_init, thk)
        return misfit_functional(u_out, v_out, q, *misfit_fctl_args)

    get_grad = jax.grad(reduced_functional)

    def lbfgsb(initial_guess):
        print("starting opt")
        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(reduced_functional, 
                           initial_guess, 
                           jac = lambda x: get_grad(x), 
                           method="L-BFGS-B", 
                           bounds=[(-3, 2)] * initial_guess.size, 
                           options={"maxiter": iterations} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb

lbfgsb = lbfgsb_function(misfit, (uo, uc), iterations=1)
q_out = lbfgsb(jnp.zeros_like(thk).reshape(-1))
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/potential_q_5.npy", q_out)

#q_out = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/potential_q_5.npy")

plt.imshow(q_out.reshape((nr, nc)))
plt.colorbar()
plt.show()

u_out, v_out = solver(q_out.reshape((nr,nc)), C, u_init, v_init, thk)
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/vel_q_5.npy", jnp.stack([u_out, v_out]))

#vel = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/vel_q_5.npy")

u_out = vel[0]
v_out = vel[1]

show_vel_field(u_out, v_out, vmin=0, vmax=1000, cmap="RdYlBu_r")
#
#plt.figure(figsize=(8,6))
#plt.imshow(uo, vmin=0, vmax=1000, cmap="RdYlBu_r")
#plt.colorbar()
#plt.show()


phi = jnp.where(C==0, jnp.exp(q_out.reshape(nr,nc)), 0)

plt.figure(figsize=(8,6))
plt.imshow(phi, vmin=0, vmax=2, cmap="RdBu_r")
#plt.imshow(phi, cmap="RdBu_r")
plt.colorbar()
plt.show()







