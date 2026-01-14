#1st party
import sys
import time

##local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers/")
from nonlinear_solvers import make_newton_coupled_solver_function,\
        make_newton_velocity_solver_function_custom_vjp_dynamic_thk

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
    
    resolution = 1000 #m
    
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



lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = ice_shelf()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10


vel_solver = make_newton_velocity_solver_function_custom_vjp_dynamic_thk(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             C, b,
                                                             n_iterations,
                                                             mucoef_0)

ui, vi = vel_solver(q, u_init, v_init, thk)

plt.plot(ui[:,155])
plt.show()


show_vel_field(ui, vi)
