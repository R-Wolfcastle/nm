
#1st party
import sys
import time

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils")
import constants_years as c
from plotting_stuff import show_vel_field
from grid import binary_erosion, binary_dilation,\
        cc_gradient_function, add_ghost_cells_fcts,\
        cc_resistive_and_deviatoric_stress_tensors,\
        linear_extrapolate_over_cf_function_cornersafe

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers")
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_pic_velocity_solver_function_densetest

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg

from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
import xarray as xr
import rioxarray
from osgeo import gdal, osr
from scipy import ndimage as ndi



np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)





def clean_mask_scipy(mask, connectivity=1):
    """
    mask: bool or 0/1 array, foreground=True
    connectivity: 1 for 4-connectivity, 2 for 8-connectivity
    returns: mask with holes filled and islands removed
    """

    mask = mask.astype(bool)

    # 1) Fill holes in the foreground
    filled = ndi.binary_fill_holes(mask)

    # 2) Keep only the largest connected component in the foreground
    structure = ndi.generate_binary_structure(2, connectivity)  # 4 or 8 connectivity
    labeled, n = ndi.label(filled, structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    # Find label of the largest component
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background count ignored
    keep_label = counts.argmax()
    cleaned = (labeled == keep_label)

    return cleaned



def smooth_gaussian(a, sigma=1.5):
    """
    a      : 2D array
    sigma  : smoothing length in pixels
    """
    return ndi.gaussian_filter(a, sigma=sigma)


def principal_and_max_shear(stress):
    """
    stress: array with shape (nx, ny, 2, 2), where
        stress[0,0] = sxx, stress[0,1] = sxy,
        stress[1,0] = syx, stress[1,1] = syy

    Returns:
        sigma1, sigma2, tau_max of shape (nx, ny)
    """

    vals, vecs = jnp.linalg.eigh(stress)

    sigma2 = vals[:,:,0]
    sigma1 = vals[:,:,1]
    shear = 0.5 * (sigma1 - sigma2)

    return sigma1, sigma2, shear



def align_tensor_with_flow(u, v, tensor):
   
    theta = jnp.arctan2(v, u)

    c = np.cos(theta)
    s = np.sin(theta)

    # Rotation matrix at each gridpoint:
    # [ [c,  s],
    #   [-s, c] ]

    R = jnp.zeros((c.shape[0], c.shape[1], 2, 2))
    R = R.at[..., 0, 0].set(c)
    R = R.at[..., 0, 1].set(s)
    R = R.at[..., 1, 0].set(-s)
    R = R.at[..., 1, 1].set(c)

    Rt = jnp.swapaxes(R, -1, -2)

    tensor_aligned = R @ tensor @ Rt

    return tensor_aligned

def align_tensor_with_flow_new(u, v, tensor):
    """
    Rotate a 2x2 tensor field into the flow-aligned frame where e1 points along (u,v).
    tensor: (nr, nc, 2, 2); u,v: (nr, nc)
    Returns tensor_aligned with same shape.
    """
    # Robust angle (handles signs correctly)
    theta = jnp.arctan2(v, u)

    c = jnp.cos(theta)
    s = jnp.sin(theta)

    # Build rotation Q(+theta) = [[c, -s], [s, c]] at each grid point
    Q = jnp.stack(
        [jnp.stack([c, -s], axis=-1),
         jnp.stack([s,  c], axis=-1)],
        axis=-2
    )  # (nr, nc, 2, 2)

    # Transpose on the last two axes
    Qt = jnp.swapaxes(Q, -1, -2)

    # Rotate: T' = Q T Q^T
    tensor_aligned = Q @ tensor @ Qt
    return tensor_aligned


def plot_orientations(u, v):
    ny, nx = u.shape
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    plt.figure(figsize=(6,6))
    plt.quiver(X, Y, u, v, angles='xy', scale=20)
    plt.gca().invert_yaxis()   # optional (image-style coordinates)
    plt.axis('equal')
    plt.show()




def level_set_gl_normal(thickness, bed, add_scalar_ghost_cells,
                        gradient_function):
    sg = bed+thickness
    sf = thickness*(1-c.RHO_I/c.RHO_W)

    sg = add_scalar_ghost_cells(sg)
    sf = add_scalar_ghost_cells(sf)

    varphi = smooth_gaussian(sg-sf, sigma=3)

    dp_dx, dp_dy = gradient_function(-varphi)
    dp_dx_n = dp_dx/jnp.sqrt(dp_dx**2 + dp_dy**2)
    dp_dy_n = dp_dy/jnp.sqrt(dp_dx**2 + dp_dy**2)
    
    return dp_dx_n, dp_dy_n


def grounding_line_flux(gl_normal_x, gl_normal_y, u, v, h):
    return jnp.sum(h*(u*gl_normal_x+v*gl_normal_y))
    

def gl_flux_functional(gl_normal_x, gl_normal_y, h, solver):

    def gl_flux(q, p):
        u, v = solver(q, p, jnp.zeros_like(h), jnp.zeros_like(h), h)
        return grounding_line_flux(gl_normal_x, gl_normal_y, u, v, h)
        
    return gl_flux



def geocode_array_1(array, bounding_crs_coords, resolution, filename, compression=None):
    array = np.array(array)
    drv = gdal.GetDriverByName("GTiff")
    if compression:
      ds = drv.Create(filename, array.shape[1], array.shape[0],
                      1, gdal.GDT_Float32,
                      options=['COMPRESS={}'.format(compression)])
    else:
      ds = drv.Create(filename, array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform([bounding_crs_coords[0], resolution, 0,
                        bounding_crs_coords[1], 0, -resolution])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(3031)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()


def open_shapefile_as_mask(in_shp, aoi, res):
    options = "-burn 1.0 -tr {} {} -init 0.0 -a_nodata 0.0 -te {} {} {} {} -tap".format(res, res, aoi[0], aoi[3], aoi[2], aoi[1])
    data_ = gdal.Rasterize('/vsimem/dataOddGeoFcts.tif', gdal.OpenEx(in_shp), options=options).ReadAsArray()
    gdal.Unlink('/vsimem/dataOddGeoFcts.tif')
    return data_


def analyse_year(year, tlxy, brxy, res):
    print(f"ANALYSING YEAR {year}")

    print("Extracting things from ncdf")
    #with xr.open_dataset(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ip_data/{year}.nc") as nc_file:
    #    mucoef_0, q_ig, C_0, p_ig,\
    #    topg, thk, speed_obs, uc, temp  = (np.flipud(nc_file[var_].values) for var_ in ["mucoef_0",
    #                                                                   "q_ig",
    #                                                                   "c_one_0",
    #                                                                   "p_ig",
    #                                                                   "topg",
    #                                                                   "thk",
    #                                                                   "uo",
    #                                                                   "uc",
    #                                                                   "temp"
    #                                                                   ])
    
    vars_to_load = ["mucoef_0", "q_ig", "c_one_0", "p_ig",
                    "topg", "thk", "uo", "uc", "temp"]
    
    with xr.open_dataset(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ip_data/{year}.nc") as nc_file:
        loaded = [np.flipud(nc_file[v].values).copy(order="C") for v in vars_to_load]
    
    (mucoef_0, q_ig, C_0, p_ig,
     topg, thk, speed_obs, uc, temp) = loaded

    uc = jnp.where(jnp.isfinite(speed_obs), uc, 0)
    speed_obs = jnp.where(jnp.isfinite(speed_obs), speed_obs, 0)

    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    gradient_function                          = cc_gradient_function(res, res)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)


    randd_stress_function = cc_resistive_and_deviatoric_stress_tensors(nr, nc, res, res,
                                                       extrapolate_over_cf,
                                                       add_uv_ghost_cells, add_scalar_ghost_cells,
                                                       gradient_function, mucoef_0, temp)


    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    grounded = clean_mask_scipy(grounded).astype(int)
    grounded = add_scalar_ghost_cells(grounded).astype(int)
    gl_main  = (grounded & ~binary_erosion(grounded))[1:-1,1:-1]
    grounded = grounded[1:-1, 1:-1]

    plt.imshow(grounded)
    plt.show()
    raise


    gradient_function = cc_gradient_function(res, res)
    gl_normal_x, gl_normal_y = level_set_gl_normal(thk, topg, add_scalar_ghost_cells,
                                                   gradient_function)
    gl_normal_x = gl_normal_x*gl_main
    gl_normal_y = gl_normal_y*gl_main


    border_cells = jnp.zeros_like(thk)
    border_cells = border_cells.at[:5,:].set(1)
    border_cells = border_cells.at[-5:,:].set(1)
    border_cells = border_cells.at[:,:5].set(1)
    border_cells = border_cells.at[:,-5:].set(1)
    #plt.imshow(border_cells)
    #plt.show()
    border_cells_reduced_flat = border_cells[1:-1,1:-1].astype(int).reshape(-1)
    border_cells_flat = border_cells.astype(int).reshape(-1)
    border_cells_double_flat = jnp.concatenate((border_cells_flat, border_cells_flat))

    cf_cells = (thk>0) & ~binary_erosion(thk>0)



    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()
    
    n_pic_iterations = 9
    n_newt_iterations = 5
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             mucoef_0, C_0,
                                                             sliding="linear",
                                                             temperature_field=temp)


    gl_flux_function = gl_flux_functional(gl_normal_x, gl_normal_y, thk, solver)
    gl_flux_gradient_fct = jax.grad(gl_flux_function, argnums=0)


    qp_out = jnp.load(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/qp_out_{year}.npy")
    
    q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    p_out = qp_out[(nr*nc):].reshape((nr,nc))

    geocode_array_1(q_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/q_out_{year}.tiff")
    geocode_array_1(p_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/p_out_{year}.tiff")


    #plt.imshow(p_out, vmin=-4, vmax=4, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(q_out, vmin=-2, vmax=2, cmap="RdBu")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(q_out-q_ig.reshape((nr,nc)), vmin=-1, vmax=1, cmap="RdBu")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(mucoef_0, vmin=0, vmax=1, cmap="cubehelix")
    #plt.colorbar()
    #plt.show()
    
    phi_out = mucoef_0*jnp.exp(q_out)*ice_mask
    #plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
    #plt.colorbar()
    #plt.show()
    geocode_array_1(phi_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/phi_out_{year}.tiff")
    
    C_out = C_0*jnp.exp(p_out)*ice_mask
    #plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    #plt.colorbar()
    #plt.show()
    geocode_array_1(C_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/C_out_{year}.tiff")

   

    u_mod_end = jnp.load(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/u_mod_end_{year}.npy")
    v_mod_end = jnp.load(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/v_mod_end_{year}.npy")
    
    
    #show_vel_field(u_mod_end, v_mod_end, cmap="RdYlBu_r", vmin=0, vmax=4500)

    speed_out = jnp.sqrt(u_mod_end**2 + v_mod_end**2 + 1e-10)
    geocode_array_1(speed_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/speed_out_{year}.tiff")
   
    misfit = uc*(jnp.sqrt(u_mod_end**2 + v_mod_end**2 + 1e-10) - speed_obs)*ice_mask
    #plt.figure(figsize=(8,6))
    #plt.imshow(misfit, vmin=-500, vmax=500, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    geocode_array_1(phi_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/phi_out_{year}.tiff")
    
    
    dFdq = gl_flux_gradient_fct(q_out, p_out)*ice_mask*(1-grounded)
    geocode_array_1(dFdq, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/dFdq_{year}.tiff")
    
    #plt.imshow(dFdq)
    #plt.colorbar()
    #plt.show()
    
    
    #grounding_line_flux(gl_normal_x, gl_normal_y, u_mod_end, v_mod_end, thk)


    
    
    print("computing RST")
    rst, dst = randd_stress_function(q_out, u_mod_end, v_mod_end, temp)
    rst = rst*ice_mask[:,:,None,None]
    dst = dst*ice_mask[:,:,None,None]
    
    tau_1, tau_2, tau_shear = principal_and_max_shear(dst)
    
    s1, s2, shear = principal_and_max_shear(rst)
    
    sigma_1 = 2*tau_1 + tau_2
    sigma_2 = 2*tau_2 + tau_1
    
    
    
    #vm = jnp.sqrt(3)*jnp.sqrt(tau_1**2 + tau_2**2 + tau_1*tau_2 + 1e-10)
    #
    #plt.imshow(vm, vmin=0, vmax=5e5, cmap="magma")
    #plt.colorbar()
    #plt.show()
    
    #plt.imshow(s1, vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(s2, vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(shear, vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    
    geocode_array_1(s1, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/s1_{year}.tiff")
    
    
    aligned_rst = align_tensor_with_flow(u_mod_end, v_mod_end, rst)
    
    #plt.imshow(aligned_rst[:,:,0,0], vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(aligned_rst[:,:,1,0], vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(aligned_rst[:,:,1,1], vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    
    geocode_array_1(aligned_rst[:,:,0,0], (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ips_out/sxx_{year}.tiff")


    extraction_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/pig_through_time/data/misc/stress_extraction_region/mask.shp",
                                           (*tlxy, *brxy),
                                           res)

    print("MEAN ALONGFLOW EXTENSIONAL STRESS")
    print(np.nanmean(aligned_rst[:,:,0,0]*np.where(extraction_mask>0, 1, np.nan)))

tlxy = (-1_654_000, -190_000)
brxy = (-1_550_000, -346_000)

res = 1000
years = ["2006_07","2007_08","2008_09","2009_10","2010_11","2011_12","2012_13","2013_14","2014_15","2015_16","2016_17","2017_18","2018_19", "2019_20"]
years = ["2019_20"]
for year in years:
    analyse_year(year, tlxy, brxy, 1000)


