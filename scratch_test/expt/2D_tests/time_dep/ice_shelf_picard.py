
#1st party
import sys
import time

##local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers/")
from nonlinear_solvers import make_newton_coupled_solver_function,\
        make_newton_velocity_solver_function_custom_vjp_dynamic_thk,\
        make_picard_velocity_solver_function_custom_vjp

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,\
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette


#3rd party
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

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)



def ice_shelf():
    lx = 160_000
    ly = 200_000
    
    resolution = 2000 #m
    
    nr = int(ly/resolution)
    nc = int(lx/resolution)
    
    lx = nr*resolution
    ly = nc*resolution
    
    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)
    
    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    thk_profile = 500# - 300*x/lx
    thk = jnp.zeros((nr, nc))+thk_profile
    thk = thk.at[:, -2:].set(0)
    
    b = jnp.zeros_like(thk)-600
    
    mucoef = jnp.ones_like(thk)
    
    C = jnp.zeros_like(thk)
    C = C.at[:4, :].set(1e8)
    C = C.at[:, :4].set(1e8)
    C = C.at[-4:,:].set(1e8)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q



#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = tiny_ice_shelf()
lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = ice_shelf()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 20


ice_mask = jnp.where(thk>0, 1, 0)


vel_solver = make_picard_velocity_solver_function_custom_vjp(nr, nc,
                                                         delta_y,
                                                         delta_x,
                                                         b, ice_mask,
                                                         n_iterations, mucoef_0)

ui = u_init.copy()
vi = v_init.copy()

ui, vi = vel_solver(q, C, ui, vi, thk)

show_vel_field(ui, vi)
raise


timestep = 10 #year



def expl_ts():
    vel_solver = make_newton_velocity_solver_function_custom_vjp_dynamic_thk(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             C, b,
                                                             n_iterations, mucoef_0)
    
    thickness_update = make_advect_scalar_field_fou_function(nc, nr, 
                                                             delta_x, delta_y,
                                                             vel_bcs="rflc")
    
    
    
    ui = u_init.copy()
    vi = v_init.copy()
    hi = thk.copy()
    for i in range(10):
        ui, vi = vel_solver(q, ui, vi, hi)
        #show_vel_field(u_out, v_out)
        hi = thickness_update(ui, vi, hi, 0, timestep, hi)
    #plt.plot(hi[10,:])
    #plt.show()
    plt.imshow(hi, vmin=0, vmax=500)
    plt.show()


def impl_ts():
    uvh_solver = make_newton_coupled_solver_function(nr, nc,
                                                     delta_y,
                                                     delta_x,
                                                     C, b,
                                                     n_iterations, mucoef_0)
    
    vel_solver = make_newton_velocity_solver_function_custom_vjp_dynamic_thk(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             C, b,
                                                             n_iterations, mucoef_0)
   
    ui = u_init.copy()
    vi = v_init.copy()

    ui, vi = vel_solver(q, ui, vi, thk)
    #show_vel_field(ui,vi)

    hi = thk.copy()

    for i in range(10):
        ui, vi, hi = uvh_solver(q, ui, vi, hi, 0, timestep)

        show_vel_field(ui,vi)

        plt.imshow(hi, vmin=0, vmax=500)
        plt.show()

#expl_ts()
impl_ts()

































def tiny_ice_shelf():
    lx = 150_0
    ly = 150_0
    resolution = 50_0 #m
    
    nr = int(ly/resolution)
    nc = int(lx/resolution)
    
    lx = nr*resolution
    ly = nc*resolution
    
    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)
    
    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    thk_profile = 500# - 300*x/lx
    thk = jnp.zeros((nr, nc))+thk_profile
    thk = thk.at[:, -1:].set(0)
    
    b = jnp.zeros_like(thk)-600
    
    mucoef = jnp.ones_like(thk)
    
    C = jnp.zeros_like(thk)
    C = C.at[:1, :].set(1e8)
    C = C.at[:, :1].set(1e8)
    C = C.at[-1:,:].set(1e8)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q

