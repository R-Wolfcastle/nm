
#1st party
import sys
import time

##local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers/")
from nonlinear_solvers import make_newton_coupled_solver_function,\
        make_newton_velocity_solver_function_custom_vjp_dynamic_thk,\
        make_newton_velocity_solver_function_custom_vjp,\
        make_picard_velocity_solver_function_custom_vjp

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,\
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette
from grid import binary_erosion
import constants_years as c

#3rd party
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
import scipy
from scipy.optimize import minimize as scinimize
import matplotlib.pyplot as plt


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)




def make_misfit_function_speed_basic(u_obs, u_c, reg_param, solver):

    def misfit_function(mucoef_internal):

        u_mod, v_mod = solver(mucoef_internal.reshape(u_obs.shape))

        u_mod = u_mod*c.S_PER_YEAR
        v_mod = v_mod*c.S_PER_YEAR

        misfit = jnp.sum(uc * (u_obs - jnp.sqrt(u_mod**2 + v_mod**2 + 1e-12))**2)/(u_mod.size)

        regularisation = reg_param * jnp.sum((1-mucoef_internal)**2)/(u_mod.size)

        return misfit + regularisation

    return misfit_function



def lbfgsb_function(misfit_function, iterations=50):
    def lbfgsb(initial_guess):

        get_grad = jax.grad(misfit_function)

        print("starting opt")
        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(misfit_function, 
                           initial_guess, 
                           jac = lambda x: get_grad(x), 
                           method="L-BFGS-B", 
                           bounds=[(0.1, 2)] * initial_guess.size, 
                           options={"maxiter": iterations} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb


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


def wonky_stream():
    lx = 128_000
    ly = 128_000

    resolution = 1000

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


    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask



    

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask = wonky_stream()

#plt.imshow(b)
#plt.colorbar()
#plt.show()
#raise


#plt.imshow(mucoef_0)
#plt.show()
#
#plt.imshow(C, vmin=0, vmax=2000)
#plt.show()
#
#raise

u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 50



#solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
#                                                         delta_y, delta_x,
#                                                         thk, b, C,
#                                                         n_iterations,
#                                                         mucoef_0)
#
#u_out, v_out = solver(q, u_init, v_init)
#show_vel_field(u_out, v_out, cmap="RdYlBu_r")

solver = make_picard_velocity_solver_function_custom_vjp(nr, nc,
                                                         delta_y, delta_x,
                                                         b, ice_mask,
                                                         n_iterations,
                                                         mucoef_0, 
                                                         sliding="basic_weertman")

u_out, v_out = solver(q, C, u_init, v_init, thk)

show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=2500)
raise



misfit_fct = make_misfit_function_speed_basic(uo, uc, 1e2, solver)
lbfgs_iterator = lbfgsb_function(misfit_fct, iterations=10)






