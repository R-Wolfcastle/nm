
#1st party
import sys
import time
import os
import gc
import re
    

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

#local apps
nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from plotting_stuff import show_vel_field
from grid import binary_erosion, binary_dilation,\
        cc_gradient_function, add_ghost_cells_fcts,\
        cc_resistive_and_deviatoric_stress_tensors,\
        linear_extrapolate_over_cf_function_cornersafe,\
        face_free_cells
from standard_domains import tiny_ice_shelf, wonky_stream, wonky_stream_rotated

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_pic_velocity_solver_function_densetest,\
                              make_pic_velocity_solver_function_gpusafe,\
                              make_pic_velocity_solver_function_expl_advection_gpusafe,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                              make_picnewton_vel_expl_dam_solver_function_noextrap#,\
                              #make_picnewton_vel_expl_dam_solver_function

from linear_solvers import create_petsc_operator_solver

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg

from scipy.sparse.linalg import cg, LinearOperator, eigsh, minres, gmres
from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian2DKernel, convolve
import xarray as xr
import rioxarray
from osgeo import gdal, osr
from scipy import ndimage as ndi


def little_ice_shelf():
    
    Lx = 100_000
    Ly = 60_000
    
    resolution = 1000
    
    x = jnp.arange(0, Lx, resolution)
    y = jnp.arange(0, Ly, resolution)
    
    nx = int(Lx/resolution)
    ny = int(Ly/resolution)
    
    b = jnp.zeros((ny,nx))-1000

    thk_profile = 1000*(1-(500/1000)*x/Lx)
    thk = jnp.zeros((ny, nx))+thk_profile[None,:]
    thk = thk.at[:, -1].set(0)

    b = b.at[:5, :].set(-thk[:5, :]*c.RHO_I/c.RHO_W + 0.1)
    b = b.at[-5:,:].set(-thk[-5:,:]*c.RHO_I/c.RHO_W + 0.1)
    b = b.at[:, :5].set(-thk[:, :5]*c.RHO_I/c.RHO_W + 0.1)

    #A = 6.338e-25
    ##B = 0.5 * (A**(-1/c.GLEN_N))
    #B = A**(-1/c.GLEN_N)
    #m = 3
    
    C = jnp.zeros_like(thk)+3.16e6
    C = C.at[15:-15, 1:].set(0)
    C = C.at[:, -20:].set(0)


    mucoef_0 = jnp.ones_like(C)
    q = jnp.zeros_like(mucoef_0)
    ice_mask = jnp.where(thk>0, 1, 0)
    
    surface = jnp.maximum(b+thk, thk*(1-c.RHO_I/c.RHO_W))

    grounded = jnp.where(b+thk > thk*(1-c.RHO_I/c.RHO_W), 1, 0)


    return ny*resolution, nx*resolution, ny, nx,\
           x, y, resolution, resolution, thk, b, C, mucoef_0,\
           q, ice_mask, surface, grounded



#bedmachine_fp = "/Users/eartsu/Documents/BedMachine/Antarctica/v4/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"

def open_shapefile_as_mask(in_shp, aoi, res):
    options = "-burn 1.0 -tr {} {} -init 0.0 -a_nodata 0.0 -te {} {} {} {} -tap".format(res, res, aoi[0], aoi[3], aoi[2], aoi[1])
    datafile = gdal.Rasterize('/vsimem/dataOddGeoFcts.tif', gdal.OpenEx(in_shp), options=options)
    data_ = datafile.ReadAsArray()
    datafile=None
    gdal.Unlink('/vsimem/dataOddGeoFcts.tif')
    return data_

def define_cook_problem(year):
    print(year)
    
    print("Extracting things from ncdf")
    with xr.open_dataset(
            f"{in_dir}/{year}.nc"
                        ) as nc_file:
        phi_0, q_ig, C_0, p_ig,\
        topg, thk, speed_obs, uc  = (np.flipud(nc_file[var_].values) for var_ in ["phi_0",
                                                                                  "q_ig",
                                                                                  "c_one_0",
                                                                                  "p_ig",
                                                                                  "topg",
                                                                                  "thk",
                                                                                  "uo",
                                                                                  "uc"
                                                                                 ])

    nr, nc = topg.shape
    
    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()

    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape


    print("defining forward solver")
    n_pic_iterations = 12
    n_newt_iterations = 10
    
    
    q_out_ds = gdal.Open(f"{out_dir}/q_out_{year}.tiff", gdal.GA_ReadOnly)
    q_out    = q_out_ds.ReadAsArray()
    q_out_ds = None
    p_out_ds = gdal.Open(f"{out_dir}/p_out_{year}.tiff", gdal.GA_ReadOnly)
    p_out    = p_out_ds.ReadAsArray()
    p_out_ds = None


    phi_out = phi_0*jnp.exp(q_out)
    #plt.imshow(phi_out, vmin=0, vmax=2, cmap="RdBu")
    #plt.title(str(year))
    #plt.colorbar()
    #plt.show()
    

    C_out = C_0*jnp.exp(p_out)
    #plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    #plt.colorbar()
    #plt.show()

    grounded = jnp.where((thk+topg)>(thk*c.RHO_I/c.RHO_W))

    return nr*res, nc*res, nr, nc,\
           thk, topg, C_out, phi_out,\
           q_out, jnp.zeros_like(q_out),\
           ice_mask, grounded



#lx, ly, nr,\
#nc, x, y,\
#delta_x, delta_y,\
#thk, b, C, mucoef_0,\
#q, ice_mask,\
#surface, grounded = \
#            little_ice_shelf()
#            #wonky_stream(resolution=2000)
#            #wonky_stream_rotated(resolution=2000)
#
#
#
#print(f"DOFS: {jnp.log2(nr*nc)}")
#
#u_init = jnp.zeros_like(b) + 100
#v_init = jnp.zeros_like(b)
#D_init = v_init.copy()
#D_init = D_init.at[10:35, -40:-38].set(0.9)







#in_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data_wpp/500m_res"
#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0008_50its_measuresCprior/500m_res/"


in_dir = "/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_data_wpp/500m_res/"
out_dir = "/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0008_50its_measuresCprior/500m_res/"



res = 500

tlxy = (1_020_000, -2_035_000)
brxy = (1_154_000, -2_148_000)



lx, ly, nr, nc,\
thk, b, C, mucoef_0,\
q, p, ice_mask, grounded = define_cook_problem("2024")



print(f"DOFS: {jnp.log2(nr*nc)}")

u_init = jnp.zeros_like(b)
v_init = jnp.zeros_like(b)

#D_init = 1 -  mucoef_0*jnp.exp(q)
#D_init = jnp.maximum(D_init, 0.01)
##D_init = jnp.zeros_like(q)

#mucoef_0 = mucoef_0*jnp.exp(q)
#q = jnp.zeros_like(q)

D_init = jnp.zeros_like(b)
mucoef_0 = jnp.where(grounded==1, mucoef_0, 1)




##There's some shit we have to deal with, in preventing negative damage
#mucoef =  mucoef_0*jnp.exp(q)
#
#mucoef_stiff = jnp.where(mucoef>1, mucoef, 1)


delta_y, delta_x = res, res



#solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
#                                                         delta_x, delta_y,
#                                                         b, ice_mask,
#                                                         10, 6,
#                                                         mucoef_0, C,
#                                                         sliding="linear")
#
#u, v = solver(q, p,  u_init, v_init, thk)
#
#raise


n_timesteps = 400

prognostic_solver = make_picnewton_vel_expl_dam_solver_function_noextrap(nr, nc,
                                                     delta_y, delta_x,
                                                     b, ice_mask,
                                                     2, 5, n_timesteps,
                                                     mucoef_0, C,
                                                     sliding="linear")

os.system(f"mkdir -p {nm_home}/solvers/nonlinear_solvers.py {nm_home}/bits_of_data/ss_damage_cook/11/")
os.system(f"cp {nm_home}/solvers/nonlinear_solvers.py {nm_home}/bits_of_data/ss_damage_cook/11/")

u, v, D = prognostic_solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk, D_init)


jnp.save(f"{nm_home}/bits_of_data/ss_damage_cook/11/D.npy", D)


from pathlib import Path
from PIL import Image


def make_speed_gif():
    dir_ = f"{nm_home}/bits_of_data/ss_damage_cook/11/"

    img_dir = Path(dir_)

    #pngs = sorted(img_dir.glob("*.png"))
    
    pngs = sorted(
        (
            f for f in img_dir.glob("*.png")
            if re.match(r"^\d+", f.stem)
        ),
        key=lambda f: int(re.match(r"^\d+", f.stem).group())
    )



    frames = [Image.open(f) for f in pngs]

    out_file = f"{dir_}/damgif.gif"

    frames[0].save(
        out_file,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )

    print(f"Saved GIF to {out_file}")

make_speed_gif()








