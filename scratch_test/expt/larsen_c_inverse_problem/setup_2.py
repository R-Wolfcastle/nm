
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
                                make_picnewton_velocity_solver_function_cvjp,
                                make_picnewton_velocity_solver_function_full_cvjp)

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


def soft_grow(binary, sigma=1, mode='reflect'):
    bin_float = binary.astype(np.float32)
    blurred   = gaussian_filter(bin_float, sigma=sigma, mode=mode)
    
    # Keep original ones exactly 1, while letting blur extend into zeros
    out = np.maximum(blurred, bin_float)
    
    # If you want literal 1.0 on original foreground to avoid any FP quirks:
    out[binary.astype(bool)] = 1.0
    return out

from skimage.measure import label
from skimage.segmentation import clear_border

def label_all_sk(img, connectivity=2):
    return label(img.astype(bool), connectivity=connectivity)

def label_interior_sk(img, connectivity=2):
    lab = label(img.astype(bool), connectivity=connectivity)
    lab = clear_border(lab)
    if lab.max():
        u = np.unique(lab); u = u[u>0]
        remap = np.zeros(u.max()+1, int); remap[u] = np.arange(1, u.size+1)
        lab = remap[lab]
    return lab

def interior_islands(img, connectivity=2):
    lab = label(img.astype(bool), connectivity=connectivity)
    lab = clear_border(lab)
    return (lab > 0).astype(np.uint8)


def vertically_averaged_temperature():
    data_nc_fp = "/Users/eartsu/new_model/testing/nm/bits_of_data/LC_IntEnergy.nc"

    ds = xr.open_dataset(data_nc_fp)

    x = ds["x"]
    y = ds["y"]

    nc_res = x[1]-x[0]

    res_inc_factor = 16

    new_res = nc_res*res_inc_factor

    new_x = ds["x"].values[::res_inc_factor]
    new_y = ds["y"].values[::res_inc_factor]

    temp = ds["internalEnergy0012"].interp(x=new_x, y=new_y, method="nearest")

    plt.imshow(temp)
    plt.colorbar()
    plt.show()


#vertically_averaged_temperature()
def setup_ls_domain_v_low_res():

    data_nc_fp = "/Users/eartsu/new_model/testing/nm/bits_of_data/LC_EnvBm.nc"
    
    ds = xr.open_dataset(data_nc_fp)
    
    x = ds["x"]
    y = ds["y"]
    
    nc_res = x[1]-x[0]
    
    res_inc_factor = 32
    
    new_res = nc_res*res_inc_factor
    
    new_x = ds["x"].values[::res_inc_factor]
    new_y = ds["y"].values[::res_inc_factor]

    #lx = new_x[-1]-new_x[0]
    #ly = new_y[-1]-new_y[0]

    
    thk  = ds["thk"].interp(x=new_x, y=new_y, method="nearest")
    topg = ds["topg"].interp(x=new_x, y=new_y, method="nearest")

    #plt.imshow(jnp.array(thk)[::-1, :], vmin=0, vmax=750)
    #plt.colorbar()
    #plt.show()

    uo = ds["uo"].interp(x=new_x, y=new_y, method="nearest")
    uc = ds["uc"].interp(x=new_x, y=new_y, method="nearest")


    #C = ds["btrc"].interp(x=new_x, y=new_y, method="nearest")

    
    thk  = jnp.array(thk[40:-20,10:-10])[::-1, :]
    topg = jnp.array(topg[40:-20,10:-10])[::-1, :]
    uo = jnp.array(uo[40:-20,10:-10])[::-1, :]
    uc = jnp.array(uc[40:-20,10:-10])[::-1, :]
    #C = jnp.array(C[80:-50,25:-25])[::-1, :]
    
    
    nr, nc = thk.shape
    
    
    b = topg.copy()
    
    
    s_gnd = thk + b
    s_flt = thk * (1-c.RHO_I/c.RHO_W)

    surface = jnp.maximum(s_gnd, s_flt)
    
    grounded = jnp.where(s_gnd>=s_flt, 1, 0)

    #pinning_points = interior_islands(grounded)
    bawden = (label_interior_sk(grounded.astype(bool))==1).astype(int)
   
    C = jnp.where(((grounded==1) & (bawden==0)), 1e8, 0)
    C = jnp.where(bawden==1, 1e2, C)
    #C = jnp.zeros_like(grounded)+1e8
    #C = C*soft_grow(grounded)
    #C = jnp.where(soft_grow(grounded==1, 1e4, 0)
    C = jnp.where(thk==0, 1, C)
    
    #plt.figure(figsize=(6,6))
    ##plt.imshow(jnp.log10(C), vmin=0, vmax=20, cmap="RdBu_r")
    #plt.imshow(jnp.log10(C), cmap="RdBu_r")
    ##plt.imshow(C)
    #plt.colorbar()
    ##plt.imshow(thk, vmin=0, vmax=600, cmap="magma")
    ##plt.imshow(thk, cmap="magma")
    #plt.show()
    #raise
   
    #uc = jnp.where(grounded==1, 0, uc)

    #do some erosion away from CF and GL!
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    
    
    delta_x = new_x[1]-new_x[0]
    delta_y = new_y[1]-new_y[0]
    

    mucoef_0 = jnp.ones_like(thk)#*3
    q = jnp.zeros_like(thk)

    ice_mask = jnp.where(thk>0, 1, 0)

    #plt.imshow(ice_mask)
    #plt.show()
    
    
    return nr, nc, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, uo, uc, surface

def setup_ls_domain():

    data_nc_fp = "/Users/eartsu/new_model/testing/nm/bits_of_data/LC_EnvBm.nc"
    
    ds = xr.open_dataset(data_nc_fp)
    
    x = ds["x"]
    y = ds["y"]
    
    nc_res = x[1]-x[0]
    
    res_inc_factor = 16
    
    new_res = nc_res*res_inc_factor
    
    new_x = ds["x"].values[::res_inc_factor]
    new_y = ds["y"].values[::res_inc_factor]

    #lx = new_x[-1]-new_x[0]
    #ly = new_y[-1]-new_y[0]

    
    thk  = ds["thk"].interp(x=new_x, y=new_y, method="nearest")
    topg = ds["topg"].interp(x=new_x, y=new_y, method="nearest")

    #plt.imshow(jnp.array(thk)[::-1, :], vmin=0, vmax=750)
    #plt.colorbar()
    #plt.show()

    uo = ds["uo"].interp(x=new_x, y=new_y, method="nearest")
    uc = ds["uc"].interp(x=new_x, y=new_y, method="nearest")


    #C = ds["btrc"].interp(x=new_x, y=new_y, method="nearest")

    
    thk  = jnp.array(thk[80:-49,25:-25])[::-1, :]
    topg = jnp.array(topg[80:-49,25:-25])[::-1, :]
    uo = jnp.array(uo[80:-49,25:-25])[::-1, :]
    uc = jnp.array(uc[80:-49,25:-25])[::-1, :]
    #C = jnp.array(C[80:-50,25:-25])[::-1, :]
    
    
    nr, nc = thk.shape
    
    
    b = topg.copy()
    
    
    s_gnd = thk + b
    s_flt = thk * (1-c.RHO_I/c.RHO_W)

    surface = jnp.maximum(s_gnd, s_flt)
    
    grounded = jnp.where(s_gnd>=s_flt, 1, 0)

    #pinning_points = interior_islands(grounded)
    bawden = (label_interior_sk(grounded.astype(bool))==1).astype(int)
   
    C = jnp.where(((grounded==1) & (bawden==0)), 1e8, 0)
    C = jnp.where(bawden==1, 1e2, C)
    #C = jnp.zeros_like(grounded)+1e8
    #C = C*soft_grow(grounded)
    #C = jnp.where(soft_grow(grounded==1, 1e4, 0)
    C = jnp.where(thk==0, 1, C)
    
    #plt.figure(figsize=(6,6))
    ##plt.imshow(jnp.log10(C), vmin=0, vmax=20, cmap="RdBu_r")
    #plt.imshow(jnp.log10(C), cmap="RdBu_r")
    ##plt.imshow(C)
    #plt.colorbar()
    ##plt.imshow(thk, vmin=0, vmax=600, cmap="magma")
    ##plt.imshow(thk, cmap="magma")
    #plt.show()
    #raise
   
    #uc = jnp.where(grounded==1, 0, uc)

    #do some erosion away from CF and GL!
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    
    
    delta_x = new_x[1]-new_x[0]
    delta_y = new_y[1]-new_y[0]
    

    mucoef_0 = jnp.ones_like(thk)#*3
    q = jnp.zeros_like(thk)

    ice_mask = jnp.where(thk>0, 1, 0)

    #plt.imshow(ice_mask)
    #plt.show()
    
    
    return nr, nc, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, uo, uc, surface


#nr, nc, delta_x, delta_y, thk, b, C_0, mucoef_0, q, ice_mask, uo, uc, surface = setup_ls_domain()
nr, nc, delta_x, delta_y, thk, b, C_0, mucoef_0, q, ice_mask, uo, uc, surface = setup_ls_domain_v_low_res()

cf_cells = (thk>0) & ~binary_erosion(thk>0)

add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
gradient_function = cc_gradient_function(delta_y, delta_x)


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_pic_iterations = 25
n_newt_iterations = 25



def misfit(u_mod, v_mod, q, p, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    return jnp.sum(mask.reshape(-1) * (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2)


def regularised_misfit(u_mod, v_mod, q, p, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    
    misfit_term = jnp.sum(mask.reshape(-1) * \
                          (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2
                          #)/(nr*nc*100)
                         )/(nr*nc*1000)
    #NOTE NOTE NOTE NOTE NOTE THE 1000 not 100 as it was in the not-so-low-res version!



    phi = mucoef_0*jnp.exp(q.reshape((nr, nc)))
    dphi_dx, dphi_dy = gradient_function(phi)

    phi_regn_term = 1e4 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2)
                              )
    
    C = C_0*jnp.exp(p.reshape((nr, nc)))
    dC_dx, dC_dy = gradient_function(C)

    C_regn_term = 1e8 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2)
                              )

    jax.debug.print("misfit_term: {x}", x=misfit_term)
    jax.debug.print("phi_regn_term: {x}", x=phi_regn_term)
    jax.debug.print("C_regn_term: {x}", x=C_regn_term)
    

    #return misfit_term, regn_term, misfit_term + regn_term
    return misfit_term + phi_regn_term + C_regn_term
    #return phi_regn_term + C_regn_term



def lbfgsb_function(misfit_functional, misfit_fctl_args=(), iterations=50):
    def reduced_functional(qp):
        q = qp[:(nr*nc)]
        p = qp[(nr*nc):]
        u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
        return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)

    #get_grad_basic = jax.grad(reduced_functional)
    #def get_grad(x):
    #    grad = get_grad_basic(x)
    #    return grad*(1-cf_cells.astype(int).reshape(-1))
    get_grad = jax.grad(reduced_functional)

    def lbfgsb(initial_guess):
        print("starting opt")
        ##initial gradient descent:
        #gr = get_grad(initial_guess)
        #plt.imshow(gr[:(nr*nc)].reshape((nr,nc)))
        #plt.colorbar()
        #plt.show()
        #plt.imshow(gr[(nr*nc):].reshape((nr,nc)))
        #plt.colorbar()
        #plt.show()
        #raise


        ##initial gradient descent:
        #initial_guess = initial_guess - 1e-7*get_grad(initial_guess)
        #plt.imshow(new_initial_guess[:(nr*nc)].reshape((nr,nc)))
        #plt.colorbar()
        #plt.show()
        #plt.imshow(new_initial_guess[(nr*nc):].reshape((nr,nc)))
        #plt.colorbar()
        #plt.show()
        #raise
        ##NOTE: REMOVE THE ABOVE!!!

        result = scinimize(reduced_functional,
                           initial_guess,
                           jac = get_grad,
                           method="L-BFGS-B",
                           bounds=[(-0.2, 0.2)] * initial_guess.size,
                           options={"maxiter": 2} #Note: disp is depricated
                          )

        new_initial_guess = result.x

        #plt.imshow(result.x[:(nr*nc)].reshape((nr,nc)))
        #plt.colorbar()
        #plt.show()

        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(reduced_functional, 
                           new_initial_guess, 
                           jac = get_grad, 
                           method="L-BFGS-B", 
                           bounds=[(-1.6, 1)] * initial_guess.size, 
                           options={"maxiter": iterations} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb


def initial_guess_for_C(speed_obs):
    surf_extended = add_scalar_ghost_cells(surface)
    dsdx, dsdy = gradient_function(surf_extended)

    rhs_squared = (c.RHO_I * c.g * thk)**2 * (dsdx**2 + dsdy**2)

    lhs_squared = speed_obs ** (2/3)

    cig = jnp.minimum(jnp.sqrt(rhs_squared/(lhs_squared + 1e-12)), 1e12)

    cig = jnp.where(thk>0, jnp.where(C==0, 0, cig), 1)

    cig = cig.at[:1,:].set(1e12)
    cig = cig.at[-1:,:].set(1e12)
    cig = cig.at[:,:1].set(1e12)

    return cig

#plt.imshow(uo)
#plt.imshow(uc, cmap="Grays_r", alpha=0.5)
#plt.show()
#raise


solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                         delta_y, delta_x,
                                                         b, ice_mask,
                                                         n_pic_iterations,
                                                         n_newt_iterations,
                                                         mucoef_0, C_0,
                                                           #sliding="basic_weertman")
                                                         sliding="linear")

q_initial_guess = jnp.zeros_like(thk).reshape(-1)
p_initial_guess = jnp.zeros_like(thk).reshape(-1)

qp_initial_guess = jnp.zeros((2*nr*nc,))


#def reg_test(q_flat):
#    q = q_flat.reshape(nr, nc)
#    phi = mucoef_0 * jnp.exp(q)
#    dphi_dx, dphi_dy = gradient_function(phi)  # suspect
#    return jnp.sum(dphi_dx**2 + dphi_dy**2)
#
#g = jax.grad(reg_test)(jnp.ones(nr*nc))
#print("||grad|| =", jnp.linalg.norm(g))
#raise

#uin, vin = solver(q_initial_guess.reshape(nr, nc),
#                    p_initial_guess.reshape(nr, nc),
#                    u_init, v_init, thk)
#
#show_vel_field(uin, vin, cmap="RdYlBu_r", vmin=0, vmax=1000)
##raise


lbfgsb_iterator = lbfgsb_function(regularised_misfit, (uo, uc), iterations=20)





qp_out = lbfgsb_iterator(qp_initial_guess)
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/qp_out_large_10its_1.npy", qp_out)
#qp_out = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/qp_out_large_10its_1.npy")

q_out = qp_out[:(nr*nc)].reshape((nr,nc))
p_out = qp_out[(nr*nc):].reshape((nr,nc))

u_mod_end, v_mod_end = solver(q_out, p_out, u_init, v_init, thk)
show_vel_field(u_mod_end, v_mod_end, cmap="RdYlBu_r", vmin=0, vmax=1000)


plt.imshow(q_out)
plt.colorbar()
plt.show()

phi_out = mucoef_0*jnp.exp(q_out)
plt.imshow(phi_out, vmin=0, vmax=3, cmap="cubehelix")
#plt.imshow(phi_out, cmap="cubehelix")
plt.colorbar()
plt.show()

C_out = C_0*jnp.exp(p_out)
plt.imshow(C_out, vmin=0, vmax=2000, cmap="magma")
plt.colorbar()
plt.show()

#raise


plt.figure(figsize=(8,6))
plt.imshow(jnp.sqrt(u_mod_end**2 + v_mod_end**2) - uo, vmin=-250, vmax=250, cmap="RdBu_r")
plt.colorbar()
plt.show()











