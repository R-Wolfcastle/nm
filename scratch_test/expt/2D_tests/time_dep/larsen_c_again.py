

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
import scipy
from scipy.optimize import minimize as scinimize




np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)



def larsen_c():

    data_nc_fp = "../../../bits_of_data/larsen_c/LC_EnvBm.nc"
    
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
    
    
    delta_x = new_x[1]-new_x[0]
    delta_y = new_y[1]-new_y[0]

    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q
    
    

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = larsen_c()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10



misfit_fct = make_misfit_function_speed_basic(uo, uc, 1e2, solver)
lbfgs_iterator = lbfgsb_function(misfit_fct, iterations=10)



