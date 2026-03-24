
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
from osgeo import gdal


msrs_annual_dir = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/annual/"
new_cfs_dir = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/pig_through_time/data/misc/calving_fronts/for_measures_annual/"

bedmachine_fp = "/Users/eartsu/Documents/misc_data/bedmachine_v3_ase_MAR-1980-2021_smooth_post_relax_50y_500m_coordshift.nc"
temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW_upsidedownlayers.nc"

data_locations_etc = {
   "2006_07":
   {"vel":msrs_annual_dir+"2006_2007/Antarctica_ice_velocity_2006_2007_1km_v01.nc",
    "mask":None},
   "2007_08":
   {"vel":msrs_annual_dir+"2007_2008/Antarctica_ice_velocity_2007_2008_1km_v01.nc",
    "mask":None},
   "2008_09":
   {"vel":msrs_annual_dir+"2008_2009/Antarctica_ice_velocity_2008_2009_1km_v01.nc",
    "mask":None},
   "2009_10":
   {"vel":msrs_annual_dir+"2009_2010/Antarctica_ice_velocity_2009_2010_1km_v01.nc",
    "mask":None},
   "2010_11":
   {"vel":msrs_annual_dir+"2010_2011/Antarctica_ice_velocity_2010_2011_1km_v01.nc",
    "mask":None},
   "2011_12":
   {"vel":msrs_annual_dir+"2011_2012/Antarctica_ice_velocity_2011_2012_1km_v01.nc",
    "mask":None},
   "2012_13":
   {"vel":msrs_annual_dir+"2012_2013/Antarctica_ice_velocity_2012_2013_1km_v01.nc",
    "mask":None},
   "2013_14":
   {"vel":msrs_annual_dir+"2013_2014/Antarctica_ice_velocity_2013_2014_1km_v01.nc",
    "mask":None},
   "2014_15":
   {"vel":msrs_annual_dir+"2014_2015/Antarctica_ice_velocity_2014_2015_1km_v01.nc",
    "mask":new_cfs_dir+"20141102.shp"},
   "2015_16":
   {"vel":msrs_annual_dir+"2015_2016/Antarctica_ice_velocity_2015_2016_1km_v01.nc",
    "mask":new_cfs_dir+"20150910.shp"},
   "2016_17":
   {"vel":msrs_annual_dir+"2016_2017/Antarctica_ice_velocity_2016_2017_1km_v01.nc",
    "mask":new_cfs_dir+"20160730.shp"},
   "2017_18":
   {"vel":msrs_annual_dir+"2017_2018/Antarctica_ice_velocity_2017_2018_1km_v01.1.nc",
    "mask":new_cfs_dir+"20171030.shp"},
   "2018_19":
   {"vel":msrs_annual_dir+"2018_2019/Antarctica_ice_velocity_2018_2019_1km_v01.1.nc",
    "mask":new_cfs_dir+"20190129.shp"},
   "2019_20":
   {"vel":msrs_annual_dir+"2019_2020/Antarctica_ice_velocity_2019_2020_1km_v01.1.nc",
    "mask":new_cfs_dir+"20200229.shp"},
        }


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





def get_main_gl(thk, topg):
    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)

    grounded = clean_mask_scipy(grounded).astype(int)

    grounded = add_scalar_ghost_cells(grounded).astype(int)

    gl = grounded & ~binary_erosion(grounded)

    return gl[1:-1,1:-1], grounded[1:-1,1:-1]



def shift_coords_and_save(in_path, out_path, dx, dy):
    """
    Load a NetCDF file with xarray, shift x and y coordinates,
    and save to a new NetCDF file.
    """
    ds = xr.open_dataset(in_path)

    # Shift coordinates
    ds = ds.assign_coords(
        x = ds.x + dx,
        y = ds.y + dy
    )

    # Save to new file
    ds.to_netcdf(out_path)

    return out_path


def open_shapefile_as_mask(in_shp, aoi, res):
    options = "-burn 1.0 -tr {} {} -init 0.0 -a_nodata 0.0 -te {} {} {} {} -tap".format(res, res, aoi[0], aoi[3], aoi[2], aoi[1])
    data_ = gdal.Rasterize('/vsimem/dataOddGeoFcts.tif', gdal.OpenEx(in_shp), options=options).ReadAsArray()
    gdal.Unlink('/vsimem/dataOddGeoFcts.tif')
    return data_



def load_geotiff_resampled(fp, target_grid, method="bilinear"):
    da = rioxarray.open_rasterio(fp).squeeze()
    return da.rio.reproject_match(target_grid, resampling=method)


def setup_comparison_data(resolution, tlx, tly):
    x0, y1 = tlxy
    x1, y0 = brxy

    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(
            x=("x", xs),
            y=("y", ys)
        )
    )
    target_grid = target_grid.rio.write_crs("EPSG:3031")

    vel_fp = r"/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"

    vel_data = load_geotiff_resampled(vel_fp, target_grid).values

    vel_data = jnp.maximum(0, vel_data)

    #plt.imshow(jnp.array(vel_data), vmin=0, vmax=4400, cmap="RdYlBu_r")
    #plt.colorbar()
    #plt.show()

    uc = jnp.where(jnp.isfinite(vel_data) & (vel_data>0), 1, 0)

    return jnp.array(vel_data), uc





def create_nc_files(resolution, tlxy, brxy):
    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"

    
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )
    target_grid_raster = target_grid.rio.write_crs("EPSG:3031")

    bed_nc = xr.open_dataset(bedmachine_fp)
    bed_nc = bed_nc.interp_like(target_grid, method="linear")
   
    temp_nc = xr.open_dataset(temp_fp)
    temp_nc = temp_nc.interp_like(target_grid, method="linear")


    #BM STUFF
    bm_phi, bm_C, topg, thk = (bed_nc[var_].values for var_ in ["mucoef",
                                                         "c_one",
                                                         "topg", 
                                                         "thk"])
    #SINK PINNING POINTS
    sink_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/pig_through_time/data/misc/sink_bedrock_mask.shp",
                                       (*tlxy, *brxy),
                                       resolution)
    topg = topg - 1000*sink_mask
    

    bm_C[:2, :] = 1e10
    bm_C[-2:,:] = 1e10
    bm_C[:, :2] = 1e10
    bm_C[:,-2:] = 1e10


    #TEMPERATURE
    temp = temp_nc["temperature"].values

    nr, nc = temp.shape

    qp_from_some_past_problem = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/qp_out_1km_OSreg_40its20ls_uniformIG_8e6_2e0.npy")

    q_ig = qp_from_some_past_problem[:(nr*nc)].reshape((nr, nc))

    mucoef_0 = jnp.ones((nr,nc))
    C_0 = bm_C*jnp.exp(qp_from_some_past_problem[(nr*nc):].reshape((nr, nc)))
    p_ig = jnp.zeros_like(C_0)


    for year in data_locations_etc.keys():
        vel_obs = xr.open_dataset(data_locations_etc[year]["vel"])\
                             .interp_like(target_grid, method="linear")
        uo, vo = vel_obs["VX"].values, vel_obs["VY"].values

        speed_obs = jnp.sqrt(1e-10 + uo**2 + vo**2)

        if data_locations_etc[year]["mask"] is not None:
            thk_mask = open_shapefile_as_mask(data_locations_etc[year]["mask"],
                                          (*tlxy, *brxy),
                                          resolution)
        else:
            thk_mask = jnp.zeros_like(thk)

        uo = uo*(1-thk_mask)
        vo = vo*(1-thk_mask)
        thickness = thk*(1-thk_mask)

        uc = jnp.where((thickness>0) &\
                       jnp.isfinite(speed_obs) &\
                       (speed_obs>0) &\
                       (speed_obs<5_500), 1, 0)


        xr.Dataset({"temp":     (("y", "x"), np.asarray(jnp.flipud(temp))       ),
                    "mucoef_0": (("y", "x"), np.asarray(jnp.flipud(mucoef_0))   ),
                    "q_ig":     (("y", "x"), np.asarray(jnp.flipud(q_ig))       ),
                    "c_one_0":  (("y", "x"), np.asarray(jnp.flipud(C_0))        ),
                    "p_ig":     (("y", "x"), np.asarray(jnp.flipud(p_ig))       ),
                    "topg":     (("y", "x"), np.asarray(jnp.flipud(topg))       ),
                    "thk":      (("y", "x"), np.asarray(jnp.flipud(thickness))  ),
                    "uo":       (("y", "x"), np.asarray(jnp.flipud(speed_obs))  ),
                    "uc":       (("y", "x"), np.asarray(jnp.flipud(uc))         )
                    },
                   coords={"x": bed_nc.x, "y": bed_nc.y}
        ).to_netcdf(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ip_data/{year}.nc")
    

    




def left_top_centred_gradient_function(field, res):
    dx = jnp.zeros_like(field)
    dy = jnp.zeros_like(field)

    dx = dx.at[:, :-1].set(field[:, 1:]-field[:,:-1])/res
    dy = dy.at[:-1, :].set(field[:-1,:]-field[1: ,:])/res

    return dx, dy


def regularised_misfit(u_mod, v_mod, q, p, speed_obs,\
                       mask, nr, nc, mucoef_0, C_0, res, border_cells_flat):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    
    misfit_term = jnp.sum(mask.reshape(-1) * \
                          (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2
                         )/(nr*nc)

    jax.debug.print("{x}, {y}, {z}", x=jnp.sum(mask), y=jnp.sum(speed_mod), z=jnp.sum(speed_obs))

    #Assume that things are, on average, wrong by 100ma^-1. So, divide by 10_000:
    misfit_term = misfit_term/10_000


    phi = mucoef_0*jnp.exp(q.reshape((nr, nc)))
    dphi_dx, dphi_dy = left_top_centred_gradient_function(phi, res)


    #The coefficients are at least an order of magnitude smaller than Steph's choices of:
    #alpha_phi = 1e11 (for me, that would be 1e11/10_000 ~ 1e7)
    #alpha_C   = 1e3  (for me, that would be 1e3 /10_000 ~ 1e-1)

    #maybe 1e4 a good shout?
    phi_regn_term = 8e6 * jnp.sum( mask.reshape(-1) *\
                                (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                (1-border_cells_flat)
                              )/(nr*nc)
    


    C = C_0*jnp.exp(p.reshape((nr, nc)))
    dC_dx, dC_dy = left_top_centred_gradient_function(C, res)

    C_regn_term = 2e0 * jnp.sum( mask.reshape(-1) *\
                                (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                (1-border_cells_flat)
                              )/(nr*nc)


    jax.debug.print("misfit_term: {x}", x=misfit_term)
    jax.debug.print("phi_regn_term: {x}", x=phi_regn_term)
    jax.debug.print("C_regn_term: {x}", x=C_regn_term)

    #return misfit_term, regn_term, misfit_term + regn_term
    return misfit_term + phi_regn_term + C_regn_term





def run_ip_for_year(year, res, lbfgs_iterations):

    print(f"IP FOR YEAR {year}")

    print("Extracting things from ncdf")
    nc_file = xr.open_dataset(f"/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/annual_ip_data/{year}.nc")

    mucoef_0, q_ig, C_0, p_ig,\
    topg, thk, speed_obs, uc, temp  = (np.flipud(nc_file[var_].values) for var_ in ["mucoef_0",
                                                                   "q_ig",
                                                                   "c_one_0",
                                                                   "p_ig",
                                                                   "topg",
                                                                   "thk",
                                                                   "uo",
                                                                   "uc",
                                                                   "temp"
                                                                   ])

    uc = jnp.where(jnp.isfinite(speed_obs), uc, 0)
    speed_obs = jnp.where(jnp.isfinite(speed_obs), speed_obs, 0)

    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    gradient_function                          = cc_gradient_function(res, res)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)


    #def get_main_gl(thk, topg):
    #    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    #
    #    grounded = clean_mask_scipy(grounded).astype(int)
    #
    #    grounded = add_scalar_ghost_cells(grounded).astype(int)
    #
    #    gl = grounded & ~binary_erosion(grounded)
    #
    #    return gl[1:-1,1:-1], grounded[1:-1,1:-1]
    #
    # 
    #gl_main, grounded = get_main_gl(thk, topg)
    
    
    
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
   
    #plt.imshow(ice_mask)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(cf_cells)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(border_cells)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(q_ig)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(topg)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(thk)
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(mucoef_0)
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(C_0)
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(temp)
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(speed_obs)
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(uc)
    #plt.colorbar()
    #plt.show()


    print("defining solver")
    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()
    
    n_pic_iterations = 12
    n_newt_iterations = 8
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             mucoef_0, C_0,
                                                             sliding="linear",
                                                             temperature_field=temp)


    print("defining optimiser functions")
    def reduced_functional(qp):
        q = qp[:(nr*nc)]
        p = qp[(nr*nc):]
        u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
        return regularised_misfit(u_out, v_out, q, p, speed_obs, ice_mask, nr,\
                                  nc, mucoef_0, C_0, res, border_cells_flat)

    get_grad_basic = jax.grad(reduced_functional)
    def get_grad(x):
        grad = get_grad_basic(x)
        return grad*(1-border_cells_double_flat)
    #get_grad = jax.grad(reduced_functional)

    def lbfgsb(initial_guess):
        print("starting opt")

        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(reduced_functional, 
                           initial_guess, 
                           jac = get_grad, 
                           method="L-BFGS-B", 
                           bounds= [(-2, 0.5)] * int(initial_guess.size/2) + \
                                   [(-4, 4)] * int(initial_guess.size/2), 
                           #bounds= [(-2, 0.1)] * int(initial_guess.size/2) + \
                           #        [(-1, 1)] * int(initial_guess.size/2), 
                           #options={"maxiter": iterations, "maxls": 10} #Note: disp is depricated
                           options={"maxiter": lbfgs_iterations}
                          )

        return result.x


    
    qp_initial_guess = jnp.concatenate((q_ig.reshape(-1), p_ig.reshape(-1)))
    
    print("solving optimisation problem")
    qp_out = lbfgsb(qp_initial_guess)




    q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    p_out = qp_out[(nr*nc):].reshape((nr,nc))
    
    plt.imshow(p_out, vmin=-4, vmax=4, cmap="RdBu_r")
    plt.colorbar()
    plt.show()
    
    plt.imshow(q_out, vmin=-2, vmax=2, cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    plt.imshow(q_out-q_ig.reshape((nr,nc)), vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    plt.imshow(mucoef_0, vmin=0, vmax=1, cmap="cubehelix")
    plt.colorbar()
    plt.show()
    
    phi_out = mucoef_0*jnp.exp(q_out)
    plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
    plt.colorbar()
    plt.show()
    
    C_out = C_0*jnp.exp(p_out)
    plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    plt.colorbar()
    plt.show()







def setup_domain(resolution, tlxy, brxy, nc_fp):
    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"

    
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )

    nc = xr.open_dataset(bedmachine_fp)
    nc = nc.interp_like(target_grid, method="linear")


    mucoef_0, q_ig, C_0, p_ig,\
    topg, thk, speed_obs, uc, temp  = (bed_r[var_].values for var_ in ["mucoef_0",
                                                                   "q_ig",
                                                                   "c_one_0",
                                                                   "p_ig",
                                                                   "topg",
                                                                   "thk",
                                                                   "uo",
                                                                   "uc",
                                                                   "temp"
                                                                   ])

    C_0[:2, :] = 1e10
    C_0[-2:,:] = 1e10
    C_0[:, :2] = 1e10
    C_0[:,-2:] = 1e10

    ice_mask = np.where(thk>0.01, 1, 0)


    
tlxy = (-1_654_000, -190_000)
brxy = (-1_550_000, -346_000)

res = 1000


#create_nc_files(res, tlxy, brxy)

#run_ip_for_year("2006_07", res, 1)
run_ip_for_year("2017_18", res, 4)












