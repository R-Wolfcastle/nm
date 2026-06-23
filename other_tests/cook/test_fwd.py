
#1st party
import sys
import time
import os

#local apps
nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from plotting_stuff import show_vel_field
from grid import binary_erosion, binary_dilation,\
        cc_gradient_function, add_ghost_cells_fcts,\
        cc_resistive_and_deviatoric_stress_tensors,\
        linear_extrapolate_over_cf_function_cornersafe
from standard_domains import tiny_ice_shelf

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_pic_velocity_solver_function_densetest,\
                              make_pic_velocity_solver_function_gpusafe,\
                              make_pic_velocity_solver_function_expl_advection_gpusafe,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap
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
from osgeo import gdal
from scipy import ndimage as ndi



np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)


def smooth_gaussian(array, sigma=1.5):
    return ndi.gaussian_filter(array, sigma=sigma)

def smooth_gaussian_nan(array, sigma=1.5):
    kernel = Gaussian2DKernel(x_stddev=sigma)
    return convolve(array, kernel, preserve_nan=True, boundary="extend")

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


def open_vector_features_as_masks(in_vec, layer_name, field_name, aoi, res):
    masks = []
    
    src = gdal.OpenEx(in_vec) 
    layer = src.GetLayerByName(layer_name)


    for i in range(1, layer.GetFeatureCount()+1):
        feature = layer.GetFeature(i)

        fid = feature.GetFID()
        print(f"FID: {fid}, Name: {feature.GetField(field_name)}")

        options = (
            f"-burn 1.0 -tr {res} {res} -init 0.0 "
            f"-a_nodata 0.0 "
            f"-te {aoi[0]} {aoi[3]} {aoi[2]} {aoi[1]} "
            f"-tap -l {layer_name} "
            f"-where \"FID = {fid}\""
        )

        data_ = gdal.Rasterize(
            '/vsimem/temp.tif',
            src,
            options=options
        ).ReadAsArray()

        masks.append(data_)

        gdal.Unlink('/vsimem/temp.tif')
    return masks


#def open_mask(path, tl, br, resolution):


#shift_coords_and_save("/Users/eartsu/Documents/misc_data/bedmachine_v3_ase_MAR-1980-2021_smooth_post_relax_50y_500m.nc",
#                      "/Users/eartsu/Documents/misc_data/bedmachine_v3_ase_MAR-1980-2021_smooth_post_relax_50y_500m_coordshift.nc",
#                      dx=-1_839_750,
#                      dy=-879_750)
#
#raise




def make_temperature_from_ctlw():
    fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_bedmachine_CTplusLw_8km_24.nc"
    out_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW.nc"

    ds = xr.open_dataset(fp)
    var_names = ["internalEnergy00{:02d}".format(i) for i in range(24)]

    z_coordinates = np.array([0.0, 0.0712603, 0.14191442, 0.21137662, 0.27910062, 0.3445959,
    0.40744022, 0.46728806, 0.52387456, 0.5770154, 0.62660312, 0.67260068,
    0.71503302, 0.75397751, 0.78955403, 0.82191523, 0.85123747, 0.87771266,
    0.90154127, 0.92292652, 0.94206966, 0.95916636, 0.97440401, 0.98795991, 1.])

    
    layer_thicknesses = z_coordinates[1:] - z_coordinates[:-1]

    
    energy_stack = np.array([ds[v].values for v in var_names])

    #plt.imshow(np.flipud(energy_stack[0,:,:]))
    #plt.colorbar()
    #plt.show()
    #plt.imshow(np.flipud(energy_stack[10,:,:]))
    #plt.colorbar()
    #plt.show()
    #plt.imshow(np.flipud(energy_stack[20,:,:]))
    #plt.colorbar()
    #plt.show()

    #raise

    energy_avg = np.sum(
            energy_stack*layer_thicknesses[:, None, None],
                        axis=0
                                      ) 

    cp_ice = 2009  # J / (kg K)

    temperature = energy_avg / cp_ice #(no LW assumed!)

    #save verticall-averaged temp as new netcdf
    xr.Dataset({"temperature": (("y", "x"), temperature)},
               coords={"x": ds.x, "y": ds.y}
    ).to_netcdf(out_fp)

    #return temperature

#make_temperature_from_ctlw()
#raise



def load_geotiff_resampled(fp, target_grid, method="bilinear"):
    da = rioxarray.open_rasterio(fp).squeeze()
    return da.rio.reproject_match(target_grid, resampling=method)


def open_mega_annual_data(resolution, tlxy, brxy):

    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"


    vel_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/velocity_data/cook_iv_annual_means.nc"

    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )

    vel_nc  = xr.open_dataset(vel_fp)
    #vels = vel_nc["speed_myr"]
    vel_r  = vel_nc.interp_like(target_grid, method="linear")
    vels = vel_r["speed_myr"]


    return vels


def open_measures_ip_data(resolution, tlx, tly):
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

    plt.imshow(jnp.array(vel_data), vmin=0, vmax=800, cmap="RdYlBu_r")
    plt.colorbar()
    plt.show()

    uc = jnp.where(jnp.isfinite(vel_data) & (vel_data>0), 1, 0)

    plt.imshow(uc)
    plt.show()

    raise

    return jnp.array(vel_data), uc

def open_measures_iv(resolution, tlx, tly):
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

    return vel_data



year_date_dict = {
    "2016": "2016/08/08 00:00:00",
    "2017": "2017/06/28 00:00:00",
    "2018": "2018/06/29 00:00:00",
    "2019": "2019/07/18 00:00:00",
    "2020": "2020/06/30 00:00:00",
    "2021": "2021/06/19 00:00:00",
    "2022": "2022/06/14 00:00:00",
    "2023": "2023/06/09 00:00:00",
    "2024": "2024/06/15 00:00:00",
    "2025": "2025/07/16 00:00:00",
    "2026": "2026/05/19 00:00:00"
        }


def extrapolate_thickness(thickness, pixels=10, mask=None):
    """
    Extrapolate non-zero values outward by ~n_iter pixels.

    Parameters:
        thickness (2D np.array): thickness field (zeros = no data)
        pixels (int): number of pixels to extrapolate outward
        mask: mask over thickness to extrapolate out from

    Returns:
        np.array: extrapolated thickness field
    """

    out = thickness.copy().astype(float)
    if mask==None:
        mask = out > 0

    for _ in range(pixels):
        # grow mask by 1 pixel
        new_mask = binary_dilation(mask) & (~mask)

        if not np.any(new_mask):
            break

        # for each new pixel, average neighbouring known pixels
        inds = np.argwhere(new_mask)

        for i, j in inds:
            neighbours = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < out.shape[0] and
                        0 <= nj < out.shape[1] and
                        mask[ni, nj]):
                        neighbours.append(out[ni, nj])

            if neighbours:
                out[i, j] = np.mean(neighbours)

        mask = mask | new_mask

    return out



def setup_annual_data(resolution, tlxy, brxy, sink_pinning_points=True):

    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"


    bedmachine_fp = "/Users/eartsu/Documents/BedMachine/Antarctica/v4/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
    #temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW_upsidedownlayers.nc"
    #temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW.nc"
    
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )
    target_grid = target_grid.rio.write_crs("EPSG:3031")

    # load raw
    bed_nc  = xr.open_dataset(bedmachine_fp)
    #temp_nc = xr.open_dataset(temp_fp)

    # reproject / reindex both onto same grid
    bed_r  = bed_nc.interp_like(target_grid, method="linear")
    #temp_r = temp_nc.interp_like(target_grid, method="linear")

    
    #NOTE: no point getting surface out, as it disagrees with hydrostatic surface.
    topg, thick = (bed_r[var_].values for var_ in ["bed",
                                              "thickness"])
    #TODO: If you _do_ use the hydrostatic assumption, then there are all sorts of
    #weird floating scraggly bits near the grounding line, so this is a major fudge.
    thick = thick*1.01

    cf_masks = open_vector_features_as_masks(
                "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/cook_fronts_polys/front_shapes.gpkg",
                "front_shapes",
                "date",
                (*tlxy, *brxy),
                resolution
            )

    iv = open_mega_annual_data(resolution, tlxy, brxy)

    
    basin_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/basin.shp"
    basin_mask = open_shapefile_as_mask(basin_mask_fp, (*tlxy, *brxy), res).astype(bool)

    ice_shelf_uc_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/ice_shelf_uc_mask.shp"
    ice_shelf_uc_mask = open_shapefile_as_mask(ice_shelf_uc_mask_fp, 
                                               (*tlxy, *brxy),
                                               res).astype(bool)
    

    ice_free_and_ocean = jnp.where((topg+thick+0.1)>((thick+0.1)*(1-c.RHO_I/c.RHO_W)), 0, 1) *\
                         jnp.where(thick==0, 1, 0)

    
    extrapolated_thick = extrapolate_thickness(thick,
                                             pixels=int(20*(1000/resolution)), 
                                             mask=(1-ice_free_and_ocean))



    C_MAX = 2e4


    #plt.imshow(extrapolated_thk)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(extrapolated_thk-thk)
    #plt.colorbar()
    #plt.show()

    grounded_original = jnp.where((topg+thick)>(thick*(1-c.RHO_I/c.RHO_W)), 1, 0)

    years = [str(yr) for yr in range(2016, 2027)]

    i = 0
    for year in years:
        cf_mask = cf_masks[i]

        thk = jnp.maximum(thick, extrapolated_thick*cf_mask)
        thk = smooth_gaussian_nan(jnp.where(thk>0, thk, jnp.nan), sigma=1)

        #plt.imshow(thickness-thk)
        #plt.show()

        thk = jnp.where(jnp.isfinite(thk), thk, 0)
        thk = jnp.where(grounded_original, thick, thk*1.01)

   
        srf = jnp.maximum(topg+thk, thk*(1-c.RHO_I/c.RHO_W))
        grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)


        grounded_clean = clean_mask_scipy(grounded)
        grounded_clean_eroded = binary_erosion(binary_erosion(
                                    jnp.pad(grounded_clean, ((2,2),(2,2)), constant_values=1)
                                                             ))[2:-2, 2:-2]


        thk = jnp.where((~grounded_clean_eroded) & (~basin_mask), 0, thk)

        ice_mask = np.where(thk>0.01, 1, 0)


        uo = iv.sel(year=int(year)).values

        uc = jnp.where((jnp.isfinite(uo)) & (ice_shelf_uc_mask==1) & (ice_mask==1), 1, 0)
        uc = binary_erosion(binary_erosion(uc))
        uc = uc.at[:5,  :].set(0)
        uc = uc.at[-5:, :].set(0)
        uc = uc.at[:,  :5].set(0)
        uc = uc.at[:, -5:].set(0)

        uo = jnp.where(jnp.isfinite(uo), uo, 0)

        #plt.imshow(uo, vmin=0, vmax=1000, cmap="RdYlBu_r")
        #plt.colorbar()
        #plt.show()

        #plt.imshow(uc, vmin=0, vmax=1, cmap="RdYlBu_r")
        #plt.colorbar()
        #plt.show()


        i+=1

    raise



def c_init_from_measures_and_driving_stress(thk, topg)

    srf = jnp.maximum(topg+thk, thk*(1-c.RHO_I/c.RHO_W))

    
    grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    grounded_clean = clean_mask_scipy(grounded)
    grounded_clean_eroded = binary_erosion(binary_erosion(
                                jnp.pad(grounded_clean, ((2,2),(2,2)), constant_values=1)
                                                         ))[2:-2, 2:-2]
    grounding_zone = (grounded_clean_eroded==0) & (grounded_clean==1)

    ice_mask = np.where(thk>0.01, 1, 0)

    dsdx = (srf[:, 2:] - srf[:, :-2])/(2*resolution)
    dsdy = (srf[2:, :] - srf[:-2, :])/(2*resolution)

    dsdx = jnp.pad(dsdx, ((0,0), (1,1)), mode='constant', constant_values=0)
    dsdy = jnp.pad(dsdy, ((1,1), (0,0)), mode='constant', constant_values=0)

    vel_fp = r"/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"
    vel_data = load_geotiff_resampled(vel_fp, target_grid).values
    vel_data = jnp.maximum(0, vel_data)

    C_MAX = 2e4
   
    #C to balance driving
    C = c.RHO_I * c.g * thk * (dsdx**2 + dsdy**2 + 1e-10)**0.5 / (vel_data + 1e-10)
    
    #NOTE: This is where we decide what to do at the grounding line...
    #C = jnp.where(grounding_zone==1, jnp.minimum(C, 100), C)
    #C = jnp.where(grounded_clean==1, C, 0)
    #NOTE: NOT CHANGING PRIOR AT GL AT ALL
    C = jnp.where(grounded_clean==1, C, 0)

    C = C.at[:3,  :].set(C_MAX)
    C = C.at[-3:, :].set(C_MAX)
    C = C.at[:, -3:].set(C_MAX)
    C = C.at[:,  :3].set(C_MAX)
    
    #Set C to C_MAX outside basin
    C = jnp.where(basin_mask, C, C_MAX)
    C = jnp.where(ice_mask==0, 1, C)
    C = jnp.minimum(C_MAX, C)

    return C


def setup_domain(resolution, tlxy, brxy, sink_pinning_points=True):

    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"



    bedmachine_fp = "/Users/eartsu/Documents/BedMachine/Antarctica/v4/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
    #temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW_upsidedownlayers.nc"
    #temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW.nc"
    
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )
    target_grid = target_grid.rio.write_crs("EPSG:3031")

    # load raw
    bed_nc  = xr.open_dataset(bedmachine_fp)
    #temp_nc = xr.open_dataset(temp_fp)

    # reproject / reindex both onto same grid
    bed_r  = bed_nc.interp_like(target_grid, method="linear")
    #temp_r = temp_nc.interp_like(target_grid, method="linear")

    
    #NOTE: no point getting surface out, as it disagrees with hydrostatic surface.
    topg, thk = (bed_r[var_].values for var_ in ["bed",
                                              "thickness"])

    #TODO: If you _do_ use the hydrostatic assumption, then there are all sorts of
    #weird floating scraggly bits near the grounding line, so this is a major fudge.
    thk = thk*1.01


    srf = jnp.maximum(topg+thk, thk*(1-c.RHO_I/c.RHO_W))

    
    grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    grounded_clean = clean_mask_scipy(grounded)
    grounded_clean_eroded = binary_erosion(binary_erosion(
                                jnp.pad(grounded_clean, ((2,2),(2,2)), constant_values=1)
                                                         ))[2:-2, 2:-2]
    grounding_zone = (grounded_clean_eroded==0) & (grounded_clean==1)

    #plt.imshow(grounded_clean_eroded)
    #plt.show()
    #raise

    basin_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/basin.shp"
    basin_mask = open_shapefile_as_mask(basin_mask_fp, (*tlxy, *brxy), res).astype(bool)
    #Remo800floating ice outside the basin
    thk = jnp.where((~grounded_clean_eroded) & (~basin_mask), 0, thk)

    ice_mask = np.where(thk>0.01, 1, 0)

    dsdx = (srf[:, 2:] - srf[:, :-2])/(2*resolution)
    dsdy = (srf[2:, :] - srf[:-2, :])/(2*resolution)

    dsdx = jnp.pad(dsdx, ((0,0), (1,1)), mode='constant', constant_values=0)
    dsdy = jnp.pad(dsdy, ((1,1), (0,0)), mode='constant', constant_values=0)
   

    vel_fp = r"/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"
    vel_data = load_geotiff_resampled(vel_fp, target_grid).values
    vel_data = jnp.maximum(0, vel_data)

    C_MAX = 2e4
   
    #C to balance driving
    C = c.RHO_I * c.g * thk * (dsdx**2 + dsdy**2 + 1e-10)**0.5 / (vel_data + 1e-10)
    #Set C to zero on floating ice, 1 where there's no ice, and cap it at C_MAX 
    #C = jnp.minimum(C_MAX, 
    #                jnp.where(ice_mask==1, jnp.where(grounded_clean_eroded==1, C, 0), 1)
    #    )

    #plt.imshow(grounding_zone)
    #plt.show()

    #NOTE: This is where we decide what to do at the grounding line...
    #C = jnp.where(grounding_zone==1, jnp.minimum(C, 100), C)
    #C = jnp.where(grounded_clean==1, C, 0)
    #NOTE: NOT CHANGING PRIOR AT GL AT ALL
    C = jnp.where(grounded_clean==1, C, 0)

    C = C.at[:3,  :].set(C_MAX)
    C = C.at[-3:, :].set(C_MAX)
    C = C.at[:, -3:].set(C_MAX)
    C = C.at[:,  :3].set(C_MAX)
    
    #Set C to C_MAX outside basin
    C = jnp.where(basin_mask, C, C_MAX)
    C = jnp.where(ice_mask==0, 1, C)
    C = jnp.minimum(C_MAX, C)

    #plt.imshow(jnp.log10(C))
    ##plt.imshow(ice_mask, cmap="Grays", alpha=0.25)
    #plt.colorbar()
    #plt.show()
    #raise

    phi = jnp.ones_like(thk)




    uo = open_measures_iv(resolution, tlxy, brxy)

    uc = jnp.where((jnp.isfinite(uo)) & (basin_mask==1) & (ice_mask==1), 1, 0)
    uc = binary_erosion(binary_erosion(uc))
    uc = uc.at[:5,  :].set(0)
    uc = uc.at[-5:, :].set(0)
    uc = uc.at[:,  :5].set(0)
    uc = uc.at[:, -5:].set(0)


    plt.imshow(uo, vmin=0, vmax=1000, cmap="RdYlBu_r")
    plt.colorbar()
    plt.show()


#    temp_nc = xr_load_crop_and_resample(temp_fp,
#                                        tlxy,
#                                        brxy,
#                                        resolution)
#
#    temp = temp_r["temperature"].values


    if sink_pinning_points:
        sink_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/pig_through_time/data/misc/sink_bedrock_mask.shp",
                                           (*tlxy, *brxy),
                                           res)
        topg = topg - 1000*sink_mask


    temp = None
    
    return phi, C, topg, thk, ice_mask, temp, uo, uc, C_MAX

#gradients = np.array([8.451827167442678e-05, 0.0001420950469844517, 0.00021662014326661133,\
#                      0.00018216718582513477, 0.00020918385491679802, 0.0008558366456195366,\
#                      0.00022765815574714144, 0.0019849956275157276, 0.000901744602211164,\
#                      0.004946184474424708, 0.005171408843955474, 0.009807579297774016, \
#                      0.021759756518195623, 0.07886554292046398, 0.04192539689823826 ])[::-1]
#
#plt.scatter(range(len(gradients)), np.log10(gradients))
#plt.show()
#raise



res = 1000

#tlxy = (964_000,   -1_950_000)
#brxy = (1_180_000, -2_136_000)
tlxy = (1_020_000,   -2_020_000)
brxy = (1_154_000, -2_148_000)

#phi, C, topg, thk, ice_mask, temp, uo, uc, C_MAX = setup_domain(res, tlxy, brxy)
phi, C, topg, thk, ice_mask, temp, uo, uc, C_MAX = setup_annual_data(res, tlxy, brxy)
nr, nc = phi.shape
print(nr, nc)
raise
#vels = open_mega_annual_data(res, tlxy, brxy)

def run_fwd():

    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()
    
    n_pic_iterations = 12
    n_newt_iterations = 8
   
    #solver = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(nr, nc,
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi, C,
                                                             sliding="linear",
                                                             temperature_field=temp)
    
    u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
    
    show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=800)
    

#run_fwd()



#grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
#nearly_grounded = binary_dilation(grounded) & ~grounded
#topg = jnp.where(nearly_grounded>0, -thk*c.RHO_I/c.RHO_W + 0.1, topg)
#C = jnp.where(nearly_grounded>0, 100000, C)

#mucoef_0 = phi.copy()


######INVERSE PROBLEM GUBBINS:


def border_cells():
    border_cells = jnp.zeros((nr, nc))
    border_cells = border_cells.at[:5,:].set(1)
    border_cells = border_cells.at[-5:,:].set(1)
    border_cells = border_cells.at[:,:5].set(1)
    border_cells = border_cells.at[:,-5:].set(1)
    #plt.imshow(border_cells)
    #plt.show()
    border_cells_reduced_flat = border_cells[1:-1,1:-1].astype(int).reshape(-1)
    border_cells_flat = border_cells.astype(int).reshape(-1)

    return border_cells_flat


def left_top_centred_gradient_function(field):
    dx = jnp.zeros_like(field)
    dy = jnp.zeros_like(field)

    dx = dx.at[:, :-1].set(field[:, 1:]-field[:,:-1])/res
    dy = dy.at[:-1, :].set(field[:-1,:]-field[1: ,:])/res

    return dx, dy



def solve_ip_second_order():

    phi_0 = phi.copy()
    C_0 = C.copy()
   
    border_cells_flat = border_cells()
    border_cells_double_flat = jnp.concatenate((border_cells_flat, border_cells_flat))
    
    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()


    def regularised_misfit(u_mod, v_mod, q, p, speed_obs, mask):
        speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
        
        misfit_term = jnp.sum(mask.reshape(-1) * \
                              (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2 *\
                                    (1-border_cells_flat)
                             )/(nr*nc)
    
        #Assume that things are, on average, wrong by 100ma^-1. So, divide by 10_000:
        misfit_term = misfit_term/10_000
    
    
        phi = phi_0*jnp.exp(q.reshape((nr, nc)))
        dphi_dx, dphi_dy = left_top_centred_gradient_function(phi)
    
    
        #The coefficients are at least an order of magnitude smaller than Steph's choices of:
        #alpha_phi = 1e11 (for me, that would be 1e11/10_000 ~ 1e7)
        #alpha_C   = 1e3  (for me, that would be 1e3 /10_000 ~ 1e-1)
    
        #maybe 1e4 a good shout? #5e6 is ok!
        phi_regn_term = 1e6 * jnp.sum( mask.reshape(-1) *\
                                    (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
        
    
    
        C = C_0*jnp.exp(p.reshape((nr, nc)))
        dC_dx, dC_dy = left_top_centred_gradient_function(C)
    
        C_regn_term = 8e-3 * jnp.sum( mask.reshape(-1) *\
                                    (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
    

        #phi_box_constraint = 1e0 * jnp.sum(
        #    jax.nn.softplus(5*(phi - 4))**2 +
        #    jax.nn.softplus(10*(0.1 - phi))**2
        #) / (nr*nc)
        phi_box_constraint = 5e-1 * jnp.sum(
            (phi - phi_0)**2 +
            ((C - C_0)/1000)**2
        ) / (nr*nc) #+\
        #1e-2 * jnp.sum(
        #    jax.nn.softplus(5*(phi - 4))**2 +
        #    jax.nn.softplus(10*(0.1 - phi))**2
        #) / (nr*nc)


        #phi_box_constraint = jnp.sum( jnp.log(1+jnp.exp(5*(phi-4))) + jnp.log(1+jnp.exp(10*(0.1-phi))) )/(nr*nc)

        #box_terms = ( 1e-4 * jnp.sum((phi-phi_0)**2) + 1e-10 * jnp.sum((C-C_0)**2) ) /(nr*nc)


        jax.debug.print("misfit_term: {x}", x=misfit_term)
        jax.debug.print("phi_regn_term: {x}", x=phi_regn_term)
        jax.debug.print("C_regn_term: {x}", x=C_regn_term)
        jax.debug.print("phi_box_constraint: {x}", x=phi_box_constraint)
        jax.debug.print("Total cost: {x}", x=misfit_term + phi_regn_term + C_regn_term + phi_box_constraint)
   

        #return misfit_term, regn_term, misfit_term + regn_term
        return misfit_term + phi_regn_term + C_regn_term + phi_box_constraint
    
   
    def newton_function(misfit_functional, solver,  misfit_fctl_args=(), iterations=50):

        def reduced_functional(qp):
            q = qp[:(nr*nc)]
            p = qp[(nr*nc):]
            u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
            return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)
    
        #get_grad_basic = jax.grad(reduced_functional)
        #def get_grad(x):
        #    grad = get_grad_basic(x)
        #    return grad#*(1-border_cells_double_flat)

        #_, vjp_grad = jax.vjp(get_grad, jnp.zeros((nr*nc*2,)))
        #def hessian_vector_product_function(qp):
        #    _, hvp_function = jax.vjp(get_grad, qp)
        #    return hvp_function


        def newton(initial_guess):

            qp = initial_guess
            
            damping = 0*1e-3

            cost = jnp.inf

            
            g_old = jnp.inf

            qp_old = qp.copy()

            itns_since_damping_reduced = 0

            for itn in range(iterations):


                get_grad = jax.grad(reduced_functional)
                g = get_grad(qp)
                
                _, vjp_grad = jax.vjp(get_grad, qp)

                g_np = np.array(g)
            
                gnorm = np.linalg.norm(g_np)
                print(f"iter {itn}, ||g|| = {gnorm}")
                
                if itn == 0:
                    first_gnorm = gnorm.copy()
                

                #seems pretty good having damping be at 1e-3 for the first 20 iterations, 
                #then drop to 1e-4, then just stay there 
                #damping = 1e-3 * (1 - 9*itn)/190 #Decreases by a factor of 10 every 20 iterations
                #damping = 1e-3 if (itn<20) else 1e-4
                #damping = 0
                
                #if ((gnorm/first_gnorm < 0.1) or (itns_since_damping_reduced>0 and itns_since_damping_reduced%10==0)) and (damping>1e-5):
                #    print(f"REDUCING DAMPING TO {damping*0.1}")
                #    damping *= 0.1
                #    first_gnorm = gnorm.copy()
                #    itns_since_damping_reduced = 0
                
                #else:
                #    damping = 1e-2
                #NOTE: I want to add something that goes back a step and increases damping if
                #the gradient has increased
            
                def matvec(v_np):
                    v = jnp.array(v_np, dtype=qp.dtype)
            
                    (Hv,) = vjp_grad(v)
            
                    return np.array(Hv + damping * v)
            

                ########## SCIPY VERSION #################
                #A = LinearOperator((qp.size, qp.size), matvec=matvec)

                ##w, v = eigsh(A, 1)
                ##plt.imshow(v[:(nr*nc), 0].reshape((nr, nc)))
                ##plt.colorbar()
                ##plt.show()
                ##plt.imshow(v[(nr*nc):, 0].reshape((nr, nc)))
                ##plt.colorbar()
                ##plt.show()


                #
                ##v1 = np.random.randn(qp.size)
                ##v2 = np.random.randn(qp.size)
                ##
                ##Av1 = matvec(v1)
                ##Av2 = matvec(v2)
                #
                ##print(f"SYMMETRY CHECK: {np.dot(v1, Av2) - np.dot(v2, Av1)}")



                ##dqp_np, info = cg(A, -g_np, maxiter=20)
                ##dqp_np, info = gmres(A, -g_np, maxiter=20)
                #dqp_np, info = minres(A, -g_np, maxiter=60)
                #print(info)
                ########## END SCIPY VERSION #############



                ########## PETSc VERSION #################
                petsc_solver = create_petsc_operator_solver(matvec,
                                                            size=qp.size,
                                                            ksp_type="gmres",
                                                            preconditioner=None,
                                                            ksp_max_iter=40,
                                                            monitor_ksp=True)
                dqp_np = petsc_solver(-g_np)
                ########## END PETSc VERSION #############



                g_old = g.copy()

                dqp = jnp.array(dqp_np)
                

                #plt.imshow(dqp[:(nr*nc)].reshape((nr, nc)))
                #plt.colorbar()
                #plt.show()

                qp = qp + dqp

                #qp = jnp.concatenate( (
                #                       jnp.minimum( qp[:(nr*nc)], jnp.log(5/phi_0.reshape(-1)) ),
                #                       jnp.minimum( qp[(nr*nc):], jnp.log(C_MAX/(C_0.reshape(-1)+1e-5)) )
                #                      )
                #                    )
                qp = jnp.concatenate( (
                                       jnp.minimum( qp[:(nr*nc)], 1 ),
                                       jnp.minimum( qp[(nr*nc):], 3.5 )
                                      )
                                    )
                
                #plt.imshow( phi_0*jnp.exp( qp[:(nr*nc)].reshape((nr, nc)) ) , vmin=0, vmax=4, cmap="RdYlBu")
                #plt.colorbar()
                #plt.show()
                #
                #plt.imshow( jnp.log10(C_0*jnp.exp( qp[(nr*nc):].reshape((nr, nc)) )) , vmin=0, vmax=jnp.log10(C_MAX), cmap="Spectral")
                #plt.colorbar()
                #plt.show()


                


                #alpha = 1.0
                #for ls_iteration in range(1):
                #    print(f"LINE_SEARCH_ITERATION: {ls_iteration}")

                #    qp_trial = qp + alpha * dqp
                #    #cost_new = reduced_functional(qp_trial)
                #
                #    #if cost_new < 1e100:#cost:
                #    #    cost = cost_new.copy()
                #    #    print("LINE SEARCH TERMINATED")
                #    #    break
                #    #alpha *= 0.5
                #
                #qp = qp_trial
                
                ##damping update
                #if alpha == 1.0:
                #    damping *= 0.75
                #else:
                #    damping *= 5.0
                #damping = np.clip(damping, 1e-6, 1)


                itns_since_damping_reduced += 1

            #damping = 0
            #A = LinearOperator((qp.size, qp.size), matvec=matvec)


            #v1 = np.random.randn(qp.size)
            #v2 = np.random.randn(qp.size)
            #
            #Av1 = matvec(v1)
            #Av2 = matvec(v2)
            #
            #print(np.dot(v1, Av2) - np.dot(v2, Av1))



            #uncertainty_np, info = cg(A, g_np, maxiter=60)

            #plt.imshow(uncertainty_np[:(nr*nc)].reshape((nr, nc)))
            #plt.colorbar()
            #plt.show()
            #
            #plt.imshow(uncertainty_np[(nr*nc):].reshape((nr, nc)))
            #plt.colorbar()
            #plt.show()

            #print(f"PARAMETRIC UNCERTAINTY: {np.dot(g_np, uncertainty_np)}")

            return qp


#            for itn in range(iterations):
#
#                gradient = np.array(get_grad(qp))
#                
#                grad_norm = np.linalg.norm(gradient)
#                print(f"iter {itn}, ||g|| = {grad_norm}")
#
#                def hvp_function(pert_numpy):
#                    pert_jax = jnp.array(pert_numpy, dtype=qp_new.dtype)
#                    hvp = jax.vjp(get_grad, (qp,), (pert_jax,))[1]
#                    return np.array(hvp)
#
#
#                A = LinearOperator((nr*nc*2, nr*nc*2), matvec=hvp_function)
#
#                dqp, _ = cg(hvp_function, -gradient, maxiter=200)
#
#                qp = qp + dqp
#
#            return jnp.array(qp)

        return newton


    def lbfgsb_function(misfit_functional, solver, misfit_fctl_args=(), iterations=50):
        def reduced_functional(qp):
            q = qp[:(nr*nc)]
            p = qp[(nr*nc):]
            u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
            return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)
    
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
                               #options={"maxiter": iterations, "maxls": 10} #Note: disp is depricated
                               options={"maxiter": iterations}
                              )
    
            return result.x
        return lbfgsb

    n_pic_iterations = 12
    n_newt_iterations = 8
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi, C,
                                                             sliding="linear",
                                                             temperature_field=temp)
    
    
    q_initial_guess = jnp.zeros_like(thk).reshape(-1)
    p_initial_guess = jnp.zeros_like(thk).reshape(-1)
   
#    qp_initial_guess = jnp.load(
#    ##"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_1km_6its_8e6_5em1.npy"
#    #"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_1km_newt_1e4_5em1.npy"
#            f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_{res}m_newt_10its_1e4_5em1_softBox_lambdaZERO.npy"
#            )
    qp_initial_guess = jnp.zeros((2*nr*nc,))
    
    #lbfgsb_iterator = lbfgsb_function(regularised_misfit, solver, (uo, uc), iterations=40)
    #qp_out = lbfgsb_iterator(qp_initial_guess)
   
    ip_iterations = 15
    newton_iterator = newton_function(regularised_misfit, solver, (uo, uc), iterations=ip_iterations)
    qp_out = newton_iterator(qp_initial_guess)

    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_1km_6its_8e6_5em1.npy", qp_out)
    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_1km_newt_1e4_5em1.npy", qp_out)
    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_1km_newt_1e4_5em1i_round3.npy", qp_out)
    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_500m_newt_10its_1e4_5em1_softBox_lambdaZERO.npy", qp_out)
    #jnp.save(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_{res}m_newt_10its_1e4_5em1_softBox_lambdaZERO.npy", qp_out)
    #jnp.save(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_{res}m_newt_12to20its_1e4_5em1_softBox_lambda1em4.npy", qp_out)
    jnp.save(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/newton_tests/qp_out_{res}m_newt_{ip_iterations}its_1e6_8em3_softBoxQuadradic_lambdaZERO.npy", qp_out)
    
    q_out = qp_out[:(nr*nc)].reshape((nr, nc))
    p_out = qp_out[(nr*nc):].reshape((nr, nc))
    
    plt.imshow(uo, vmin=0, vmax=1000, cmap="RdYlBu_r")
    plt.colorbar()
    plt.show()
    
    plt.imshow(q_out)
    plt.colorbar()
    plt.show()

    plt.imshow(p_out)
    plt.colorbar()
    plt.show()

    plt.imshow(phi_0*jnp.exp(q_out), vmin=0, vmax=2, cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    C_out = C_0*jnp.exp(p_out)
    plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    plt.colorbar()
    plt.show()

    u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)

    show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=1000)


    plt.imshow(((u_out**2 + v_out**2)**(0.5)-uo) * uc, vmin=-100, vmax=100, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

    plt.imshow(((u_out**2 + v_out**2)**(0.5)-uo) * uc/(uo+1e-10), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

solve_ip_second_order()




