
#1st party
import sys
import time

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils")
import constants_years as c
from plotting_stuff import show_vel_field
from grid import binary_erosion

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers")
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg

from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
import xarray as xr


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)




def xr_load_crop_and_resample(path, tl, br, res):
    ds = xr.open_dataset(path)
    x0, y1 = tl
    x1, y0 = br


    #Shift everything by half a pixel because xarray is stoopid
    dx = float(ds.x[1] - ds.x[0])
    dy = float(ds.y[0] - ds.y[1])

    ds = ds.assign_coords(
        x = ds.x + dx/2,
        y = ds.y + dy/2
    )


    ds = ds.sel(x=slice(x0, x1), y=slice(y0, y1))

    new_x = np.arange(float(ds.x.min()), float(ds.x.max())+res, res)
    new_y = np.arange(float(ds.y.max()), float(ds.y.min())-res, -res)

    #better than coarsen etc I think.
    return ds.interp(x=new_x, y=new_y, method="linear")


def setup_domain(resolution=2000):

    data_nc_fp = "/Users/eartsu/Documents/misc_data/bedmachine_v3_ase_MAR-1980-2021_smooth_post_relax_50y_500m.nc"
    
    
    data_nc = xr_load_crop_and_resample(data_nc_fp, 
                                        (146_000, 852_000),
                                        (370_000, 510_000),
                                        resolution)

    phi, C, topg, thk = (data_nc[var_].values for var_ in ["mucoef",
                                                           "c_third",
                                                           #"c_one",
                                                           "topg", 
                                                           "thk"])

    C[:2, :] = 1e6
    C[-2:,:] = 1e6
    C[:, :2] = 1e6
    C[:,-2:] = 1e6

    #plt.imshow(thk>0)
    #plt.show()

    #thk_eroded = binary_erosion(thk)
    
    ice_mask = np.where(thk>0.01, 1, 0)

    return phi, C, topg, thk, ice_mask

res = 2000

phi, C, topg, thk, ice_mask = setup_domain(res)
nr, nc = phi.shape


print(jnp.min(C))
print(jnp.min(thk))


u_init = jnp.zeros((nr,nc))
v_init = u_init.copy()

n_pic_iterations = 60
n_newt_iterations = 0

solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                         res, res,
                                                         topg, ice_mask,
                                                         n_pic_iterations,
                                                         n_newt_iterations,
                                                         phi, C,
                                                         sliding="basic_weertman")
                                                         #sliding="linear")

u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0)




















