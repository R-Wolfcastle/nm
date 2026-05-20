
#1st party
import sys
import time
import os


#3rd party
import numpy as np
import jax
#NOTE:
#jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import xarray as xr
import scipy
from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


##local apps
nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_pic_velocity_solver_function_acrobatic,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                              make_picnewton_velocity_solver_function_no_cf_extrap_expl_advection,\
                              make_pic_velocity_solver_function_gpusafe,\
                              make_pic_velocity_solver_function_expl_advection_gpusafe
sys.path.insert(1, os.path.join(nm_home, 'utils'))
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,\
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette
from grid import binary_erosion, cc_gradient_function, add_ghost_cells_fcts
import constants_years as c



np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)





def mucoef_rifted(x,y,resolution):

    L = len(x)*resolution
    W = len(y)*resolution
    
    # crack parallel to front
    L3 = 32.0e+3 #crack to front
    L2 = 4.0e+3  #crack width
    L1 = L - L3 - L2  #left hand boundary to crack
    
    #crack parallel to shear margin
    W1 = 32.0e+3 # y = 0 to crack
    W2 = 4.0e+3  # crack width
    W3 = W - W1 + W2

    mucoef = jnp.ones((len(y), len(x)))

    #rift parallel to front
    mucoef = jnp.where(((x>L1) & (x < L1 + L2) & (y > W1) & (y < W - W1)), 0.25, mucoef)
    #(higher) damage parallel to shear margin @ y = W1
    mucoef = jnp.where(((y > W1) & (y < W1 + W2)), 0.25, mucoef)
    #(lower) damage parallel to shear margin @y = W - W1
    mucoef = jnp.where(((y > W - W1 - W2) & (y < W - W1)), 0.5, mucoef)

    mucoef = jnp.flipud(mucoef)
            
    return mucoef


def stickiness(x, y, resolution):
    BETA_MAX = 2.0e+3
    BETA_MID = 1.0e+3
    
    L = len(x)*resolution
    W = len(y)*resolution
    
    # crack parallel to front
    L3 = 32.0e+3 #crack to front
    L2 = 4.0e+3  #crack width
    L1 = L - L3 - L2  #left hand boundary to crack
    
    #crack parallel to shear margin
    W1 = 32.0e+3 # y = 0 to crack
    W2 = 4.0e+3  # crack width
    W3 = W - W1 + W2

    beta = jnp.zeros_like(x)+BETA_MAX
    #beta = jnp.where(((y > W1) & (y < W - W1)), 0.01 * BETA_MID * (1.0 + jnp.cos(16.0 * jnp.pi * x/L)), beta)
    beta = jnp.where(((y > W1) & (y < W - W1)), BETA_MID * (1.0 + jnp.cos(16.0 * jnp.pi * x/L)), beta)

    return beta


def wonky_stream(resolution=500):
    lx = 128_000
    ly = 128_000

    nr = int(ly/resolution)
    nc = int(lx/resolution)

    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)
   
    xx, yy = jnp.meshgrid(x,y)

    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]


    thk = jnp.zeros((nr,nc)) + 512 - 256*x/lx
    thk = thk.at[:,-2:].set(0)
    b = jnp.zeros((nr, nc)) - 256 - 256*x/lx

   
    C = stickiness(xx, yy, resolution)
    
    grounded = jnp.where((b+thk)>thk*(1-0.917/1.027), 1, 0)
    C = jnp.where((grounded>0) | (thk==0), C, 0)
    
    C = C.at[:1,:].set(1e12)
    C = C.at[-1:,:].set(1e12)
    C = C.at[:,:1].set(1e12)
    

    surface = jnp.maximum(thk+b, thk * (1-c.RHO_I/c.RHO_W))

    b = jnp.where(C>1e11, 0.01+surface-thk, b)
    
    mucoef_0 = mucoef_rifted(xx, yy, resolution)

    q = jnp.zeros_like(C)

    ice_mask = jnp.where(thk>0, 1, 0)


    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded



print(jax.default_backend())
print(jax.devices())

for res in [4000, 2000, 1000, 500, 250, 125, 62.5]:

    lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = wonky_stream(res)
    
    print(f"DOFS: {jnp.log2(nr*nc)}")
    
    u_init = jnp.zeros_like(b) + 100
    v_init = jnp.zeros_like(b)
    
    n_iterations = 100
    
    n_timesteps = 5

    prognostic_solver = make_picnewton_velocity_solver_function_no_cf_extrap_expl_advection(nr, nc,
                                                         delta_y, delta_x,
                                                         b, ice_mask,
                                                         10, 5, n_timesteps,
                                                         mucoef_0, C,
                                                         sliding="basic_weertman")
    
    #prognostic_solver = make_pic_velocity_solver_function_expl_advection_gpusafe(nr, nc, delta_y, delta_x,
    #                                                   b, ice_mask, n_iterations,
    #                                                   mucoef_0, C, n_timesteps, sliding="basic_weertman")
    
    
    
    ## Warm-up
    #u, v, h = prognostic_solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
    #u.block_until_ready()
    
    # Timed run
    t0 = time.time()
    u, v, h = prognostic_solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
    u.block_until_ready()
    print("Total runtime:", time.time() - t0)
    
    ##raise
    #show_vel_field(u, v, cmap="RdYlBu_r", vmin=0, vmax=3000)
    #
    #plt.imshow(h)
    #plt.colorbar()
    #plt.show()
    
