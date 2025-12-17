
#1st party
import sys
import time

##local apps
sys.path.insert(1, "/Users/eetss/new_model_code/src/nm/solvers/")
from nonlinear_solvers import make_newton_coupled_solver_function,\
        make_newton_velocity_solver_function_custom_vjp_dynamic_thk,\
        make_newton_velocity_solver_function_custom_vjp

sys.path.insert(1, "/Users/eetss/new_model_code/src/nm/utils/")
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



def wonky_stream():

    L = 128.0e+3
    W = 128.0e+3
    H0 = 512.0 # thickness at left boundary
    HL = 256.0 # thickness at CF
    B0 = -HL # bed elevation at left boundary
    BL = -H0 # bed elevation at CF
    
    # crack parallel to front
    L3 = 32.0e+3 #crack to front
    L2 = 4.0e+3  #crack width
    L1 = L - L3 - L2  #left hand boundary to crack
    
    #crack parallel to shear margin
    W1 = 32.0e+3 # y = 0 to crack
    W2 = 4.0e+3  # crack width
    W3 = W - W1 + W2


def topography(x,y):
    return B0*(x - L)/(-L) + BL*x/L

def thickness_basic(x,y):
    #basic wedge shape
    h = H0*(x-L)/(-L) + HL*x/L
    #strip if ocean
    if (x > L - 2.0e+3):
        h = 0.0
    return h
   
   
def mucoef_rifted(x,y,*etc):

    mucoef = 1.0

    #rift parallel to front
    if (x > L1) and (x < L1 + L2) and (y > W1) and (y < W - W1):
        mucoef = 0.25
            
    #(higher) damage parallel to shear margin @ y = W1
    if (y > W1) and (y < W1 + W2):
        mucoef = min(mucoef,0.25)
    #(lower) damage parallel to shear margin @y = W - W1
    if (y > W - W1 - W2) and (y < W - W1):
        mucoef = min(mucoef, 0.5)

    return mucoef

def larsen_c():

    data_nc_fp = "../bits_of_data/larsen_c/LC_EnvBm.nc"
    
    ds = xr.open_dataset(data_nc_fp)
    
    x = ds["x"]
    y = ds["y"]
    
    nc_res = x[1]-x[0]
    
    res_inc_factor = 16
    
    new_res = nc_res*res_inc_factor
    
    new_x = ds["x"].values[::res_inc_factor]
    new_y = ds["y"].values[::res_inc_factor]
    
    thk  = ds["thk"].interp(x=new_x, y=new_y, method="nearest")
    topg = ds["topg"].interp(x=new_x, y=new_y, method="nearest")
    
    uo = ds["uo"].interp(x=new_x, y=new_y, method="nearest")
    uc = ds["uc"].interp(x=new_x, y=new_y, method="nearest")
    
    thk  = jnp.array(thk[80:-50,25:-25])[::-1, :]
    topg = jnp.array(topg[80:-50,25:-25])[::-1, :]
    uo = jnp.array(uo[80:-50,25:-25])[::-1, :]
    uc = jnp.array(uc[80:-50,25:-25])[::-1, :]


    plt.imshow(uo)
    plt.show()

    plt.imshow(uc)
    plt.show()


    
    
    #do some erosion away from CF and GL!
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    uc = binary_erosion(uc)
    
    
    nr, nc = thk.shape
    
    
    b = topg.copy()
    
    
    s_gnd = thk + b
    s_flt = thk * (1-c.RHO_I/c.RHO_W)
    
    grounded = jnp.where(s_gnd>=s_flt, 1, 0)
    
    C = jnp.zeros_like(grounded)
    C = jnp.where(grounded==1, 1e16, 0)
    C = jnp.where(thk==0, 1, C)
    
    plt.imshow(C)
    plt.show()

    raise

    
    delta_x = new_x[1]-new_x[0]
    delta_y = new_y[1]-new_y[0]

    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q
    
    

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = larsen_c()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10


misfit_fct = make_misfit_function_speed_basic(uo, uc, 1e2, solver)
lbfgs_iterator = lbfgsb_function(misfit_fct, iterations=10)






