#1st party
import sys
import time
import os
import gc

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
from osgeo import gdal, osr
from scipy import ndimage as ndi


bedmachine_fp = "/Users/eartsu/Documents/BedMachine/Antarctica/v4/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"

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

def array_to_geotiff(array, original_tif, new_tif_dir, compression=None):
    driver = gdal.GetDriverByName('GTiff')
    ny = array.shape[0]
    nx = array.shape[1]
    if compression:
        new_data = driver.Create(new_tif_dir, nx, ny, 1, gdal.GDT_Float32, options=['COMPRESS={}'.format(compression)])
    else:
        new_data = driver.Create(new_tif_dir, nx, ny, 1, gdal.GDT_Float32)
    geo_transform = original_tif.GetGeoTransform()  #get GeoTranform from existing dataset
    projection = original_tif.GetProjection() #similarly get from orignal tifD
    new_data.SetGeoTransform(geo_transform)
    new_data.SetProjection(projection)

    new_data.GetRasterBand(1).WriteArray(array)

    new_data.FlushCache() #write to disk
    return new_data

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
    datafile = gdal.Rasterize('/vsimem/dataOddGeoFcts.tif', gdal.OpenEx(in_shp), options=options)
    data_ = datafile.ReadAsArray()
    datafile=None
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

def get_main_gl(thk, topg):
    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)

    grounded = clean_mask_scipy(grounded).astype(int)

    grounded = add_scalar_ghost_cells(grounded).astype(int)

    gl = grounded & ~binary_erosion(grounded)

    return gl[1:-1,1:-1], grounded[1:-1,1:-1]


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



def fill_gaps(field, mask, sigma=1, iterations=1):
    
    arr = np.array(field, dtype=float)
    mask = np.array(mask).astype(bool)

    arr[mask] = np.nan
    kernel = Gaussian2DKernel(x_stddev=sigma)

    filled = arr.copy()
    for _ in range(iterations):

        smooth = convolve(
            filled,
            kernel,
            boundary="extend",
            nan_treatment="interpolate",
            normalize_kernel=True,
            preserve_nan=False,
        )

        filled[mask] = smooth[mask]

    return jnp.asarray(filled)


def c_init_from_speed_and_driving_stress(speed_data, thk, topg, resolution, basin_mask):

    srf = jnp.maximum(topg+thk, thk*(1-c.RHO_I/c.RHO_W))

    
    grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    grounded_clean = clean_mask_scipy(grounded)
    #grounded_clean_eroded = binary_erosion(binary_erosion(
    #                            jnp.pad(grounded_clean, ((2,2),(2,2)), constant_values=1)
    #                                                     ))[2:-2, 2:-2]
    #grounding_zone = (grounded_clean_eroded==0) & (grounded_clean==1)


    ice_mask = np.where(thk>0.01, 1, 0)

    dsdx = (srf[:, 2:] - srf[:, :-2])/(2*resolution)
    dsdy = (srf[2:, :] - srf[:-2, :])/(2*resolution)

    dsdx = jnp.pad(dsdx, ((0,0), (1,1)), mode='constant', constant_values=0)
    dsdy = jnp.pad(dsdy, ((1,1), (0,0)), mode='constant', constant_values=0)

    C_MAX = 2e4
   
    #C to balance driving
    C = c.RHO_I * c.g * thk * (dsdx**2 + dsdy**2 + 1e-10)**0.5 / (speed_data + 1e-10)
    interp_mask = jnp.where((speed_data==0) | ~jnp.isfinite(speed_data), 1, 0)
    C = fill_gaps(C, interp_mask)
    if jnp.count_nonzero(~jnp.isfinite(C))>0:
        interp_mask = ~jnp.isfinite(C)
        C = fill_gaps(C, ~jnp.isfinite(C))

    #NOTE: This is where we decide what to do at the grounding line...
    #C = jnp.where(grounding_zone==1, jnp.minimum(C, 100), C)
    #C = jnp.where(grounded_clean==1, C, 0)
    #NOTE: NOT CHANGING PRIOR AT GL AT ALL
    C = jnp.where(grounded_clean==1, C, 0)
    #C = jnp.where(grounded==1, C, 0)

    C = C.at[:3,  :].set(C_MAX)
    C = C.at[-3:, :].set(C_MAX)
    C = C.at[:, -3:].set(C_MAX)
    C = C.at[:,  :3].set(C_MAX)
    
    #Set C to C_MAX outside basin
    C = jnp.where(basin_mask, C, C_MAX)
    C = jnp.where(ice_mask==0, 1, C)
    C = jnp.minimum(C_MAX, C)

    return C

def create_measures_nc_file(resolution, tlxy, brxy):
    
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
    

    gl_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/GroundingLine_Antarctica_v2.shp",
                                     (*tlxy, *brxy), res)

    pp_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/pp/pp.shp",
                                     (*tlxy, *brxy), res)

    
    basin_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/basin.shp"
    basin_mask = open_shapefile_as_mask(basin_mask_fp, (*tlxy, *brxy), res).astype(bool)

    
    ice_free_and_ocean = jnp.where((topg+thick+0.1)>((thick+0.1)*(1-c.RHO_I/c.RHO_W)), 0, 1) *\
                         jnp.where(thick==0, 1, 0)


    C_MAX = 2e4
    speed_fp = r"/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"
    uo = load_geotiff_resampled(speed_fp, target_grid).values
    uo = jnp.maximum(0, uo)

    grounded_original = jnp.where((topg+thick)>(thick*(1-c.RHO_I/c.RHO_W)), 1, 0)



    thk = smooth_gaussian_nan(jnp.where(thick>0, thick, jnp.nan), sigma=1.5*(1000/res))
    
    
    thk = jnp.where(jnp.isfinite(thk), thk, 0)
    


    thk = jnp.where(jnp.isfinite(thk), thk, 0)
    thk = jnp.where(grounded_original, thick, thk*1.01)



    #Ungrounding everything seaward of the measures mask
    topg = jnp.where(gl_mask==0, topg-1000, topg)

    grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)


    thk = jnp.where((~grounded) & (~basin_mask), 0, thk)

    #smooth the edges a bit!
    thk = jnp.where(binary_dilation(binary_erosion(thk>0.01))==1, thk, 0)

    #Get rid of free or dangling cells!
    thk = jnp.where(face_free_cells(thk), 0, thk)

    #Final removal of icebergs and islands
    thk = clean_mask_scipy(thk>0.01)*thk


    topg = np.where(pp_mask==1,
                   -thk*c.RHO_I/c.RHO_W + 0.1,
                   topg)


    grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)

    ice_mask = np.where(thk>0.01, 1, 0)

    uc = jnp.where((jnp.isfinite(uo)) &\
                   #(ice_shelf_uc_mask==1) &\
                   (basin_mask==1) &\
                   (ice_mask==1) &\
                   (uo>0) &\
                   (uo<1_800), 1, 0)

    uc = binary_erosion(uc)
    uc = uc.at[:5,  :].set(0)
    uc = uc.at[-5:, :].set(0)
    uc = uc.at[:,  :5].set(0)
    uc = uc.at[:, -5:].set(0)


    uo = jnp.where(jnp.isfinite(uo), uo, 0)
    
    phi_0 = jnp.ones_like(thk)
    q_ig = jnp.zeros_like(phi_0)
    C_0 = c_init_from_speed_and_driving_stress(uo, thk, topg,
                                               resolution, basin_mask)
    C_0 = np.where(pp_mask==1, 1000, C_0)
    p_ig = jnp.zeros_like(phi_0)


    ###plt.imshow(phi_0)
    ###plt.colorbar()
    ###plt.show()
    ###plt.imshow(q_ig)
    ###plt.colorbar()
    ###plt.show()
    #plt.imshow(np.log10(C_0))
    #plt.colorbar()
    #plt.show()
    #plt.imshow(grounded)
    #plt.show()
    ##plt.imshow(p_ig)
    ##plt.colorbar()
    ##plt.show()
    #plt.imshow(topg)
    #plt.colorbar()
    #plt.show()
    #plt.imshow(thk)
    #plt.colorbar()
    #plt.show()
    ##plt.imshow(uo)
    ##plt.colorbar()
    ##plt.show()
    ##plt.imshow(uc)
    ##plt.colorbar()
    ##plt.show()


    xr.Dataset({"phi_0":    (("y", "x"), np.asarray(jnp.flipud(phi_0))      ),
                "q_ig":     (("y", "x"), np.asarray(jnp.flipud(q_ig))       ),
                "c_one_0":  (("y", "x"), np.asarray(jnp.flipud(C_0) )       ),
                "p_ig":     (("y", "x"), np.asarray(jnp.flipud(p_ig))       ),
                "topg":     (("y", "x"), np.asarray(jnp.flipud(topg))       ),
                "thk":      (("y", "x"), np.asarray(jnp.flipud(thk))        ),
                "uo":       (("y", "x"), np.asarray(jnp.flipud(uo))         ),
                "uc":       (("y", "x"), np.asarray(jnp.flipud(uc))         )
                },
               coords={"x": xs, "y": ys}
    ).to_netcdf(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data_wpp/{res}m_res/measures.nc")

    return None

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
    #plt.imshow(thick>0)
    #plt.show()
    
    #weird floating scraggly bits near the grounding line, so this is a major fudge.
    thick = thick*1.01


    #Just in case there are any interpolation smudges
    #misc_mask = binary_erosion(binary_erosion(
    #                           jnp.pad((thick>0.1), ((2,2),(2,2)), constant_values=1)
    #                                         )
    #                          )
    #thick = np.array(thick*misc_mask[2:-2, 2:-2])

    cf_masks = open_vector_features_as_masks(
                "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/cook_fronts_polys/front_shapes.gpkg",
                "front_shapes",
                "date",
                (*tlxy, *brxy),
                resolution
            )

    iv = open_mega_annual_data(resolution, tlxy, brxy)


    gl_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Antarctic Boundaries/GroundingLine_Antarctica_v2.shp",
                                     (*tlxy, *brxy), res)

    pp_mask = open_shapefile_as_mask("/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/pp/pp.shp",
                                     (*tlxy, *brxy), res)
    
    basin_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/basin.shp"
    basin_mask = open_shapefile_as_mask(basin_mask_fp, (*tlxy, *brxy), res).astype(bool)

    ice_shelf_uc_mask_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/rough_basin/ice_shelf_uc_mask.shp"
    ice_shelf_uc_mask = open_shapefile_as_mask(ice_shelf_uc_mask_fp, 
                                               (*tlxy, *brxy),
                                               res).astype(bool)
    
    grounded_original = jnp.where((topg+thick)>(thick*(1-c.RHO_I/c.RHO_W)), 1, 0)
    
    ice_free_and_ocean = binary_dilation(jnp.where(grounded_original==1, 0, 1) *\
                                         jnp.where(thick==0, 1, 0))

    extrapolated_thick = extrapolate_thickness(thick,
                                             pixels=int(40*(1000/resolution)), 
                                             mask=(1-ice_free_and_ocean))
    
    #plt.imshow(extrapolated_thick)
    #plt.show()
    ##raise



    C_MAX = 2e4
    speed_for_c_esimation_fp = r"/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/Quantarctica3/Glaciology/MEaSUREs Ice Flow Velocity/MEaSUREs_IceFlowSpeed_450m.tif"
    speed_for_c_esimation = load_geotiff_resampled(speed_for_c_esimation_fp, target_grid).values
    speed_for_c_esimation = jnp.maximum(0, speed_for_c_esimation)


    years = [str(yr) for yr in range(2016, 2027)]
    #years = ["2025"]

    for i, year in enumerate(years):
        cf_mask = cf_masks[i]

        thk = jnp.maximum(thick, extrapolated_thick*cf_mask)
       
        thk = smooth_gaussian_nan(jnp.where(thk>0, thk, jnp.nan), sigma=1.5*1000/res)
        

        #plt.imshow(thickness-thk)
        #plt.show()

        thk = jnp.where(jnp.isfinite(thk), thk, 0)
        thk = jnp.where(grounded_original, thick, thk*1.01)
        

        
        #Ungrounding everything seaward of the measures mask
        topg = jnp.where(gl_mask==0, topg-1000, topg)
        
        grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)



        #grounded_clean = clean_mask_scipy(grounded)
        #grounded_clean_eroded = binary_erosion(binary_erosion(
        #                            jnp.pad(grounded_clean, ((2,2),(2,2)), constant_values=1)
        #                                                     ))[2:-2, 2:-2]

        #continue

        #thk = jnp.where((~grounded_clean_eroded) & (~basin_mask), 0, thk)
        thk = jnp.where((~grounded) & (~basin_mask), 0, thk)


        #mask1 = (thk>0.01)
        ##mask2 = clean_mask_scipy(mask1)

        #plt.imshow(mask1)
        ##plt.imshow(mask2, alpha=0.25)
        #plt.show()
        
        #smooth the edges a bit!
        thk = jnp.where(binary_dilation(binary_erosion(thk>0.01))==1, thk, 0)

        #Get rid of free or dangling cells!
        thk = jnp.where(face_free_cells(thk), 0, thk)

        #Final removal of icebergs and islands
        thk = clean_mask_scipy(thk>0.01)*thk
    

        #Add in the pinning point
        topg = np.where(pp_mask==1,
                       -thk*c.RHO_I/c.RHO_W + 0.1,
                       topg)

        
        grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    
        #plt.imshow(grounded)
        #plt.imshow(gl_mask, cmap="Grays_r", alpha=0.25)
        #plt.show()

        ice_mask = np.where(thk>0.01, 1, 0)


        uo = iv.sel(year=int(year)).values

        uc = jnp.where((jnp.isfinite(uo)) &\
                       #(ice_shelf_uc_mask==1) &\
                       (basin_mask==1) &\
                       (ice_mask==1) &\
                       (uo>0) &\
                       (uo<1_800), 1, 0)

        uc = binary_erosion(uc)
        uc = uc.at[:5,  :].set(0)
        uc = uc.at[-5:, :].set(0)
        uc = uc.at[:,  :5].set(0)
        uc = uc.at[:, -5:].set(0)


        uo = jnp.where(jnp.isfinite(uo), uo, 0)
    

        phi_0 = jnp.ones_like(thk)
        q_ig = jnp.zeros_like(phi_0)
        C_0 = c_init_from_speed_and_driving_stress(speed_for_c_esimation, thk, topg,
                                                   resolution, basin_mask)
    
        C_0 = np.where(pp_mask==1, 1000, C_0)
        
        p_ig = jnp.zeros_like(phi_0)



        ###plt.imshow(phi_0)
        ###plt.colorbar()
        ###plt.title(year)
        ###plt.show()
        ####plt.imshow(q_ig)
        ####plt.colorbar()
        ####plt.title(year)
        ####plt.show()
        #plt.imshow(np.log10(C_0))
        #plt.colorbar()
        #plt.title(year)
        #plt.show()
        ####plt.imshow(p_ig)
        ####plt.colorbar()
        ####plt.title(year)
        ####plt.show()
        #plt.imshow(topg)
        #plt.colorbar()
        #plt.title(year)
        #plt.show()
        #plt.imshow(thk)
        #plt.colorbar()
        #plt.title(year)
        #plt.show()
        #plt.imshow(thk>0.1)
        #plt.colorbar()
        #plt.title(year)
        #plt.show()
        ##plt.imshow(uo)
        ##plt.colorbar()
        ##plt.title(year)
        ##plt.show()
        ##plt.imshow(uc)
        ##plt.colorbar()
        ##plt.title(year)
        ##plt.show()




        xr.Dataset({"phi_0":    (("y", "x"), np.asarray(jnp.flipud(phi_0))      ),
                    "q_ig":     (("y", "x"), np.asarray(jnp.flipud(q_ig))       ),
                    "c_one_0":  (("y", "x"), np.asarray(jnp.flipud(C_0) )       ),
                    "p_ig":     (("y", "x"), np.asarray(jnp.flipud(p_ig))       ),
                    "topg":     (("y", "x"), np.asarray(jnp.flipud(topg))       ),
                    "thk":      (("y", "x"), np.asarray(jnp.flipud(thk))        ),
                    "uo":       (("y", "x"), np.asarray(jnp.flipud(uo))         ),
                    "uc":       (("y", "x"), np.asarray(jnp.flipud(uc))         )
                    },
                   coords={"x": xs, "y": ys}
        ).to_netcdf(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data_wpp/{res}m_res/{year}.nc")

    

    return None


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




def left_top_centred_gradient_function(field, res):
    dx = jnp.zeros_like(field)
    dy = jnp.zeros_like(field)

    dx = dx.at[:, :-1].set(field[:, 1:]-field[:,:-1])/res
    dy = dy.at[:-1, :].set(field[:-1,:]-field[1: ,:])/res

    return dx, dy


def border_cells(nr, nc):
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


def run_ip_for_year(year, res, newton_iterations):

    print(f"IP FOR YEAR {year}")

    print("Extracting things from ncdf")
    nc_file = xr.open_dataset(f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data/{year}.nc")

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

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    gradient_function                          = cc_gradient_function(res, res)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)


    
    border_cells_flat = border_cells(nr, nc)
    border_cells_double_flat = jnp.concatenate((border_cells_flat, border_cells_flat))
    
    cf_cells = (thk>0) & ~binary_erosion(thk>0)
   

    #plt.imshow(C_0)
    #plt.colorbar()
    #plt.show()

    #plt.imshow(thk)
    #plt.colorbar()
    #plt.show()

    plt.imshow(speed_obs, vmin=0, vmax=1000, cmap="RdYlBu_r")
    plt.colorbar()
    plt.imshow(uc, cmap="Grays_r", alpha=0.1)
    plt.show()

    #raise


    print("defining misfit")
    
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
        phi_regn_term = 5e6 * jnp.sum( mask.reshape(-1) *\
                                    (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
        
    
    
        C = C_0*jnp.exp(p.reshape((nr, nc)))
        dC_dx, dC_dy = left_top_centred_gradient_function(C)
    
        C_regn_term = 1e-2 * jnp.sum( mask.reshape(-1) *\
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
    

    print("defining newton optimiser")
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
            
            damping = 1e-3

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
                                                            ksp_max_iter=60,
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
        return newton

    print("defining forward solver")
    n_pic_iterations = 12
    n_newt_iterations = 10
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi_0, C_0,
                                                             sliding="linear")
    
    
    qp_initial_guess = jnp.concatenate((q_ig.reshape(-1), p_ig.reshape(-1)))
    
    ip_iterations = 35
    newton_iterator = newton_function(regularised_misfit, solver, (speed_obs, uc), iterations=ip_iterations)
    
    print("solving optimisation problem")
    #qp_out = newton_iterator(qp_initial_guess)


    #q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    #p_out = qp_out[(nr*nc):].reshape((nr,nc))
   
    #geocode_array_1(q_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/q_out_{ip_iterations}its_5e6_1em2_5em1_5em7_lambda1em3_{year}.tiff")
    #geocode_array_1(p_out, (tlxy[0], tlxy[1]), res, f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/p_out_{ip_iterations}its_5e6_1em2_5em1_5em7_lambda1em3_{year}.tiff")

    #q_out = gdal.Open("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/5000000.0_0.01_0.5_5e-07_lambda0.002_40its/q_out_2026.tiff", gdal.GA_ReadOnly).ReadAsArray()
    #p_out = gdal.Open("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/5000000.0_0.01_0.5_5e-07_lambda0.002_40its/p_out_2026.tiff", gdal.GA_ReadOnly).ReadAsArray()
    q_out = gdal.Open("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/10000000.0_0.1_1.0_1e-06_lambda0.002_40its/q_out_2026.tiff", gdal.GA_ReadOnly).ReadAsArray()
    p_out = gdal.Open("/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/10000000.0_0.1_1.0_1e-06_lambda0.002_40its/p_out_2026.tiff", gdal.GA_ReadOnly).ReadAsArray()

    plt.imshow(p_out, vmin=-4, vmax=4, cmap="RdBu_r")
    plt.colorbar()
    plt.show()
    
    plt.imshow(q_out, vmin=-2, vmax=2, cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    plt.imshow(q_out-q_ig.reshape((nr,nc)), vmin=-1, vmax=1, cmap="RdBu")
    plt.colorbar()
    plt.show()
    
    plt.imshow(phi_0, vmin=0, vmax=1, cmap="cubehelix")
    plt.colorbar()
    plt.show()
    
    phi_out = phi_0*jnp.exp(q_out)
    plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
    plt.colorbar()
    plt.show()
    
    C_out = C_0*jnp.exp(p_out)
    plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    plt.colorbar()
    plt.show()


    u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)

    show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=1000)


    plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc, vmin=-100, vmax=100, cmap="RdBu_r")
    plt.colorbar()
    plt.show()

    plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc/(speed_obs+1e-10), vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar()
    plt.show()



res = 250

tlxy = (1_020_000,  -2_035_000)
brxy = (1_154_000, -2_148_000)

#create_measures_nc_file(res, tlxy, brxy)
#create_nc_files(res, tlxy, brxy)
##create_monthly_speed_file(res, tlxy, brxy)
#raise



#run_ip_for_year("2026", res, 4)
#raise



def create_rst_functional(p_out, C_0, phi_0,
                          thickness, momentum_solver,
                          randd_stress_function, mask):

    def rst_fctl(q_flat):

        q_out = q_flat.reshape((nr, nc))
        
        u_out, v_out = solver(q_out, p_out,
                              jnp.zeros_like(thickness),
                              jnp.zeros_like(thickness),
                              thickness)
    
        rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
        
        s1, s2, shear = principal_and_max_shear(rst)
        
        mean_s1 = jnp.nanmean(jnp.where(mask==1,
                                        s1,
                                        jnp.nan)
                                      )
    
    
        return mean_s1
    return rst_fctl


def mean_stress_analysis_and_uq(year, mother_dir, damping):
    print(year)
    
    print("Extracting things from ncdf")
    with xr.open_dataset(
            f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data/{res}m_res/{year}.nc"
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

    measures_ctrl_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/measures_prior/"
    q_measures = gdal.Open(measures_ctrl_dir+"q_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()
    p_measures = gdal.Open(measures_ctrl_dir+"p_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()

    phi_0 = phi_0*jnp.exp(q_measures)
    C_0   = C_0 * jnp.exp(p_measures)

    thk = clean_mask_scipy(thk>0.01)*thk

    rst_bb_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/resistive_stress_bb/box.shp"
    rst_bb = open_shapefile_as_mask(rst_bb_fp, 
                                    (*tlxy, *brxy),
                                    res)


    nr, nc = topg.shape
    
    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    gradient_function                          = cc_gradient_function(res, res)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)

    
    randd_stress_function = cc_resistive_and_deviatoric_stress_tensors(nr, nc, res, res,
                                                       extrapolate_over_cf,
                                                       add_uv_ghost_cells, add_scalar_ghost_cells,
                                                       gradient_function, phi_0)

    print("defining forward solver")
    n_pic_iterations = 12
    n_newt_iterations = 10
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi_0, C_0,
                                                             sliding="linear")
    


    rst_functional = create_rst_functional(p_out, C_0, phi_0,
                                           thk, solver,
                                           randd_stress_function, rst_bb)


    q_flat = q_out.flatten()

    get_grad = jax.value_and_grad(rst_functional)

    rst, g = get_grad(q_flat)

    _, vjp_grad = jax.vjp(get_grad, q_flat)

    g_np = np.array(g)

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
                                                ksp_max_iter=60,
                                                monitor_ksp=True)
    dqp_np = petsc_solver(-g_np)
    ########## END PETSc VERSION #############



    g_old = g.copy()

    dqp = jnp.array(dqp_np)


    #plt.imshow(dqp[:(nr*nc)].reshape((nr, nc)))
    #plt.colorbar()
    #plt.show()

    qp = qp + dqp




def analyse_year(year, mother_dir):
    print(year)
    
    print("Extracting things from ncdf")
    with xr.open_dataset(
            f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data/{res}m_res/{year}.nc"
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

    measures_ctrl_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/measures_prior/"
    q_measures = gdal.Open(measures_ctrl_dir+"q_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()
    p_measures = gdal.Open(measures_ctrl_dir+"p_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()

    phi_0 = phi_0*jnp.exp(q_measures)
    C_0   = C_0 * jnp.exp(p_measures)

    thk = clean_mask_scipy(thk>0.01)*thk

    rst_bb_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/resistive_stress_bb/box.shp"
    rst_bb = open_shapefile_as_mask(rst_bb_fp, 
                                    (*tlxy, *brxy),
                                    res)


    #plt.imshow(phi_0)
    #plt.colorbar()
    #plt.title(year)
    #plt.show()
    #plt.imshow(q_ig)
    #plt.colorbar()
    #plt.title(year)
    #plt.show()
    ##plt.imshow(C_0)
    ##plt.colorbar()
    ##plt.title(year)
    ##plt.show()
    ##plt.imshow(p_ig)
    ##plt.colorbar()
    ##plt.title(year)
    ##plt.show()
    ##plt.imshow(topg)
    ##plt.colorbar()
    ##plt.title(year)
    ##plt.show()
    #plt.imshow(thk)
    #plt.colorbar()
    #plt.title(year)
    #plt.show()
    #plt.imshow(speed_obs)
    #plt.colorbar()
    #plt.title(year)
    #plt.show()
    #plt.imshow(uc)
    #plt.colorbar()
    #plt.title(year)
    #plt.show()

    #return None


    nr, nc = topg.shape
    
    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()

    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    gradient_function                          = cc_gradient_function(res, res)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)

    
    randd_stress_function = cc_resistive_and_deviatoric_stress_tensors(nr, nc, res, res,
                                                       extrapolate_over_cf,
                                                       add_uv_ghost_cells, add_scalar_ghost_cells,
                                                       gradient_function, phi_0)

    
    border_cells_flat = border_cells(nr, nc)
    border_cells_double_flat = jnp.concatenate((border_cells_flat, border_cells_flat))
    
    cf_cells = (thk>0) & ~binary_erosion(thk>0)
   

    print("defining forward solver")
    n_pic_iterations = 12
    n_newt_iterations = 10
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi_0, C_0,
                                                             sliding="linear")
    
    
    print(f"{mother_dir}/q_out_{year}.tiff")
    q_out_ds = gdal.Open(f"{mother_dir}/q_out_{year}.tiff", gdal.GA_ReadOnly)
    q_out    = q_out_ds.ReadAsArray()
    q_out_ds = None
    p_out_ds = gdal.Open(f"{mother_dir}/p_out_{year}.tiff", gdal.GA_ReadOnly)
    p_out    = p_out_ds.ReadAsArray()
    p_out_ds = None

    #plt.imshow(p_out, vmin=-4, vmax=4, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    #
    #plt.imshow(q_out, vmin=-2, vmax=2, cmap="RdBu")
    #plt.colorbar()
    #plt.show()
    
    #plt.imshow(q_out-q_ig.reshape((nr,nc)), vmin=-1, vmax=1, cmap="RdBu")
    #plt.colorbar()
    #plt.show()
    
    #plt.imshow(phi_0, vmin=0, vmax=1, cmap="cubehelix")
    #plt.colorbar()
    #plt.show()
    
    phi_out = phi_0*jnp.exp(q_out)
    plt.imshow(phi_out, vmin=0, vmax=2, cmap="RdBu")
    plt.title(str(year))
    plt.colorbar()
    plt.show()
    

    C_out = C_0*jnp.exp(p_out)
    plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    plt.colorbar()
    plt.show()


    raise

    geocode_array_1(phi_out, (tlxy[0], tlxy[1]), res, f"{mother_dir}/Phi_{year}.tiff")
    geocode_array_1(C_out, (tlxy[0], tlxy[1]), res, f"{mother_dir}/C_{year}.tiff")


    #continue


    u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)
    
    geocode_array_1(u_out, (tlxy[0], tlxy[1]), res, f"{mother_dir}/u_out_{year}.tiff")
    geocode_array_1(v_out, (tlxy[0], tlxy[1]), res, f"{mother_dir}/v_out_{year}.tiff")

    print(f"{mother_dir}/u_out_{year}.tiff")

    u_out_ds = gdal.Open(f"{mother_dir}/u_out_{year}.tiff", gdal.GA_ReadOnly)
    u_out = u_out_ds.ReadAsArray(); u_out_ds = None
    v_out_ds = gdal.Open(f"{mother_dir}/v_out_{year}.tiff", gdal.GA_ReadOnly)
    v_out = v_out_ds.ReadAsArray(); v_out_ds = None

    #plt.imshow(speed_obs, vmin=0, vmax=1000, cmap="RdYlBu_r")
    #plt.colorbar()
    #plt.imshow(uc, cmap="Grays_r", alpha=0.1)
    #plt.show()


    #show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=1000)


    #plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc, vmin=-100, vmax=100, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()

    #plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc/(speed_obs+1e-10), vmin=-1, vmax=1, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()



    #print("computing RST")
    rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
    rst = rst*ice_mask[:,:,None,None]
    dst = dst*ice_mask[:,:,None,None]
    
    tau_1, tau_2, tau_shear = principal_and_max_shear(dst)
    
    s1, s2, shear = principal_and_max_shear(rst)
    
    sigma_1 = 2*tau_1 + tau_2
    sigma_2 = 2*tau_2 + tau_1
    
    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    
    #plt.imshow(aligned_rst[:,:,0,0], vmin=-5e5, vmax=5e5, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()
    
    geocode_array_1(s1, (tlxy[0], tlxy[1]), res, f"{mother_dir}/s1_{year}.tiff")
    geocode_array_1(s2, (tlxy[0], tlxy[1]), res, f"{mother_dir}/s2_{year}.tiff")
    geocode_array_1(shear, (tlxy[0], tlxy[1]), res, f"{mother_dir}/shear_{year}.tiff")


    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    
    geocode_array_1(aligned_rst[:,:,0,0], (tlxy[0], tlxy[1]), res, f"{mother_dir}/sxx_{year}.tiff")

    mean_aligned_rst = jnp.nanmean(jnp.where(rst_bb==1,
                                             aligned_rst[:,:,0,0],
                                             jnp.nan)
                                  )

    print(f"YEAR: {year}, MEAN_RS: {mean_aligned_rst}")

    plt.close('all')

    ##jax.clear_caches()
    ##gc.collect()

    
    grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    grounded = clean_mask_scipy(grounded).astype(int)
    grounded = add_scalar_ghost_cells(grounded).astype(int)
    gl_main  = (grounded & ~binary_erosion(grounded))[1:-1,1:-1]
    grounded = grounded[1:-1, 1:-1]
    
    gradient_function = cc_gradient_function(res, res)
    gl_normal_x, gl_normal_y = level_set_gl_normal(thk, topg, add_scalar_ghost_cells,
                                                   gradient_function)
    gl_normal_x = gl_normal_x*gl_main
    gl_normal_y = gl_normal_y*gl_main

    gl_flux_function = gl_flux_functional(gl_normal_x, gl_normal_y, thk, solver)
    gl_flux_gradient_fct = jax.grad(gl_flux_function, argnums=0)

    dFdq = gl_flux_gradient_fct(q_out, p_out)*ice_mask*(1-grounded)

    #plt.imshow(dFdq)
    #plt.colorbar()
    #plt.show()
    geocode_array_1(dFdq, (tlxy[0], tlxy[1]), res, f"{mother_dir}/dFdq_{year}.tiff")




#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out/5000000.0_0.2_1.0_1e-06_lambda0.002_50its"
#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/1000000.0_0.2_1.0_1e-06_lambda0.002_40its/500m_res/"

#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/1000000.0_0.2_0.1_0.0001_lambda0.002_30its_measuresprior/500m_res/"
#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/30000.0_0.2_0.002_0.0001_lambda0.001_40its_measuresprior_wuncertainty/500m_res/"

#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/30000.0_0.2_0.002_0.0001_lambda0.001_40its_measuresCprior/500m_res/"
#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/30000.0_0.2_0.002_0.0001_lambda0.001_40its_measuresCprior/250m_res/"
#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ips_out_new/30000.0_0.2_0.002_0.0001_lambda0.0002_40its_measuresCprior/250m_res/"

#out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0008_50its_measuresCprior/500m_res/"
out_dir = "/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_out_wpp/30000.0_0.2_0.002_0.0001_lambda0.0002_50its_measuresCprior/250m_res/250m_res/"

####years = [str(y) for y in list(np.arange(2016, 2027))]
####years = ["2018", "2025"]
####for year in years:
####    analyse_year(year, out_dir)
#year = str(sys.argv[1])
##print(year)
###year = "2026"
#analyse_year(year, out_dir)



def make_stress_category_plot(year, mother_dir):
    s1_fp = mother_dir + f"s1_{year}.tiff"
    s2_fp = mother_dir + f"s2_{year}.tiff"

    s1_ds = gdal.Open(s1_fp, gdal.GA_ReadOnly)
    s2_ds = gdal.Open(s2_fp, gdal.GA_ReadOnly)

    s1, s2 = s1_ds.ReadAsArray(), s2_ds.ReadAsArray()

    cat = np.where((s1<0) & (s2<0), 1, 0)
    cat = np.where((s1<0) & (s2>0), 2, cat)
    cat = np.where((s1>0) & (s2<0), 3, cat)
    cat = np.where((s1>0) & (s2>0), 4, cat)

    array_to_geotiff(cat, s1_ds, mother_dir+f"s_category_{year}.tiff")

    s1_ds, s2_ds = None, None


def make_s_difference(year, reference_year, mother_dir, s_index=1):
    s1_ref_fp = mother_dir + f"s{s_index}_{reference_year}.tiff"
    s1_fp = mother_dir + f"s{s_index}_{year}.tiff"

    s1_ref_ds = gdal.Open(s1_ref_fp, gdal.GA_ReadOnly)
    s1_ds = gdal.Open(s1_fp, gdal.GA_ReadOnly)

    s1_ref, s1 = s1_ref_ds.ReadAsArray(), s1_ds.ReadAsArray()

    diff = s1 - s1_ref

    array_to_geotiff(diff, s1_ds, mother_dir+f"s{s_index}_diff_{year}_minus_{reference_year}.tiff")

    s1_ref_ds, s1_ds = None, None

def make_speed(year, mother_dir):

    ufp = mother_dir + f"u_out_{year}.tiff"
    vfp = mother_dir + f"v_out_{year}.tiff"

    u_ds = gdal.Open(ufp, gdal.GA_ReadOnly)
    v_ds = gdal.Open(vfp, gdal.GA_ReadOnly)

    u, v = u_ds.ReadAsArray(), v_ds.ReadAsArray()

    speed = np.sqrt(u**2+v**2)


    array_to_geotiff(speed, u_ds, mother_dir+f"speed_out_local_{year}.tiff")

    u_ds, v_ds = None, None


def extract_resistive_stress(years, mother_dir):
    bbfp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/resistive_stress_bb/box2.shp"
    bb = open_shapefile_as_mask(bbfp, (*tlxy, *brxy), res) 

    rss = []
    for i, year in enumerate(years):
        s1_fp = mother_dir + f"s1_{year}.tiff"
        s1_ds = gdal.Open(s1_fp, gdal.GA_ReadOnly)
        s1 = s1_ds.ReadAsArray()
        s1_ds = None

        rs = np.nanmean(np.where(bb==1, s1, np.nan))

        rss.append(rs)

    return np.array(rss)


years = [str(y) for y in list(np.arange(2016, 2027))]
for year in years:
    make_s_difference(year, 2016, out_dir, 1)
    ##make_stress_category_plot(year, out_dir)
    #make_speed(year, out_dir)
raise



rs = extract_resistive_stress(years, out_dir)/1000
us = np.sqrt([7294419909.476391, 6942386598.619005, 6769678238.747236, 7927953690.9748745, 7414583989.042897, 7277245715.351883, 7243184175.066392, 7522675428.67612, 7375284626.778178, 7783237593.993211, 10831510222.555117])/10000
dates = np.arange(2016, 2027)

plt.figure(figsize=(8, 4))

# Line (optional but nice for trend)
plt.plot(dates, rs, linestyle='--', color='black', linewidth=0.4)

plt.fill_between(dates, rs-us, rs+us, color='black', alpha=0.1)

# Scatter with '+' markers
plt.scatter(dates, rs, marker='+', color='black', s=30)

plt.xlabel("Year")
plt.ylabel("Along-flow resistive stress (kPa)")

# Clean styling
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
#plt.xlim(2015.5, 2026.5)


plt.tight_layout()
plt.show()


raise












#rs = np.array([97039.03545612257, 132646.31613723727, 138682.16732310294, 140784.78063693293, 136853.91356746785, 136210.27747721903, 137767.4219814884, 134739.38131396973, 130192.18558037902, 130923.8931986658, 135751.07467300972])[::-1]/1000
rs = np.array([156227.97732940855, 154311.99236583678, 153340.15077655963, 158229.94702829217, 157123.0060316737, 155486.80947724177, 156055.29467703885, 159143.22726466184, 155824.10169532595, 143300.76538809133, 140918.662519562])
us = np.sqrt([7294419909.476391, 6942386598.619005, 6769678238.747236, 7927953690.9748745, 7414583989.042897, 7277245715.351883, 7243184175.066392, 7522675428.67612, 7375284626.778178, 7783237593.993211, 10831510222.555117])
dates = np.arange(2016, 2027)



plt.figure(figsize=(8, 4))

# Line (optional but nice for trend)
plt.plot(dates, rs, linestyle='--', color='black', linewidth=0.4)

plt.fill_between(dates, rs-us, rs+us, color='black', alpha=0.1)

# Scatter with '+' markers
plt.scatter(dates, rs, marker='+', color='black', s=30)

plt.xlabel("Year")
plt.ylabel("Along-flow resistive stress (kPa)")

# Clean styling
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
plt.xlim(2015.5, 2026.5)


plt.tight_layout()
plt.show()

raise


def perturbation_experiment(qp_dir, geom_dir, tiff_outdir, year):

    berg_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/cook_study/geometry_data/predicted_berg.geojson"
    berg = open_shapefile_as_mask(berg_fp, (*tlxy, *brxy), res)

    
    print("Extracting things from ncdf")
    with xr.open_dataset(f"{geom_dir}/{year}.nc") as nc_file:
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

    thk_perturbed = thk * (1-berg)
    #Final removal of icebergs and islands
    thk_perturbed= clean_mask_scipy(thk_perturbed>0.01)*thk_perturbed


    nr, nc = topg.shape
    
    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()

    print("Defining grid operations")

    ice_mask = np.where(thk>0.01, 1, 0)

    nr, nc = ice_mask.shape

    n_pic_iterations = 12
    n_newt_iterations = 10
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi_0, C_0,
                                                             sliding="linear")

    
    C_0 = jnp.where((thk_perturbed>0.01), C_0, 1)
    solver_pert = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, (thk_perturbed>0.01),
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi_0, C_0,
                                                             sliding="linear")
    
    q_out_ds = gdal.Open(f"{qp_dir}/q_out_{year}.tiff", gdal.GA_ReadOnly)
    q_out    = q_out_ds.ReadAsArray()
    q_out_ds = None
    p_out_ds = gdal.Open(f"{qp_dir}/p_out_{year}.tiff", gdal.GA_ReadOnly)
    p_out    = p_out_ds.ReadAsArray()
    p_out_ds = None
    
    #u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)
    #geocode_array_1(u_out, (tlxy[0], tlxy[1]), res, f"{tiff_outdir}/u_out_{year}.tiff")
    #geocode_array_1(v_out, (tlxy[0], tlxy[1]), res, f"{tiff_outdir}/v_out_{year}.tiff")

    print(f"{tiff_outdir}/u_out_{year}.tiff")

    u_out_ds = gdal.Open(f"{tiff_outdir}/u_out_{year}.tiff", gdal.GA_ReadOnly)
    u_out = u_out_ds.ReadAsArray(); u_out_ds = None
    v_out_ds = gdal.Open(f"{tiff_outdir}/v_out_{year}.tiff", gdal.GA_ReadOnly)
    v_out = v_out_ds.ReadAsArray(); v_out_ds = None

    #show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=1800)

    u_per, v_per = solver_pert(q_out, p_out, u_init, v_init, thk_perturbed)
    geocode_array_1(u_per, (tlxy[0], tlxy[1]), res, f"{tiff_outdir}/u_pert_{year}.tiff")
    geocode_array_1(v_per, (tlxy[0], tlxy[1]), res, f"{tiff_outdir}/v_pert_{year}.tiff")
    
    u_per_ds = gdal.Open(f"{tiff_outdir}/u_pert_{year}.tiff", gdal.GA_ReadOnly)
    u_per = u_per_ds.ReadAsArray(); u_per_ds = None
    v_per_ds = gdal.Open(f"{tiff_outdir}/v_pert_{year}.tiff", gdal.GA_ReadOnly)
    v_per = v_per_ds.ReadAsArray(); v_per_ds = None
    
    #show_vel_field(u_per, v_per, cmap="RdYlBu_r", vmin=0, vmax=1800)

    speed_out = np.sqrt(u_out**2 + v_out**2 + 1e-10)
    speed_out_perturbed = np.sqrt(u_per**2 + v_per**2 + 1e-10)

    #geocode_array_1((speed_out_perturbed-speed_out)*(thk_perturbed>0.01),\
    #                (tlxy[0], tlxy[1]), res,\
    #                f"{tiff_outdir}/speed_diff_{year}.tiff")

    plt.imshow((speed_out_perturbed-speed_out)*(thk_perturbed>0.01),
               vmin=-300, vmax=300, cmap="RdBu_r")
    plt.colorbar()
    plt.show()



#qp_dir = f"{out_dir}"
#geom_dir = f"/Users/eartsu/new_model/testing/nm/bits_of_data/COOKING_TEA_BREAK/annual_ip_data/{res}m_res"
#tiff_outdir = qp_dir
#year = "2024"
#perturbation_experiment(qp_dir, geom_dir, tiff_outdir, year)



gs = np.log10(np.array([0.1525600329177573, 0.07151608208916231, 0.06180718819212839, 0.026994424132206044, 0.08428949450579289, 0.01375419606464239, 0.005881335577326926, 0.004344938169024285, 0.0034078018988263985, 0.0028312506482688934, 0.0024342994308766634, 0.002146276903204865, 0.0019397934999022125, 0.0017834035079677233, 0.0016594032604961885, 0.0015577044556917851, 0.0014715391078829433, 0.0013967806961538683, 0.0013309344780653976, 0.0012723232439161914, 0.0012197677597207454, 0.0011723393039495746, 0.0011293066428400141, 0.001090057358700232, 0.0010541158328004937, 0.001021047147982596, 0.0009905098352111699, 0.0009621882179466157, 0.0009358408850738724, 0.0009112234279446276, 0.0008881794824247293, 0.0008665029978978424, 0.0008461060379503541, 0.0008268246355430182, 0.0008085984379012338, 0.0007913055003749881, 0.0007749158435690893, 0.0007593157557232811, 0.0007444768575524659, 0.0007303201745443791]))

gs_lbfgsb = np.log10(np.array([0.1525600329177573, 0.07151608208916231, 0.06180718819212839, 0.026994424132206044, 0.08428949450579289, 0.01375419606464239, 0.005881335577326926, 0.004344938169024285, 0.0034078018988263985, 0.0028312506482688934, 0.0024342994308766634, 0.002146276903204865, 0.0019397934999022125, 0.0017834035079677233, 0.0016594032604961885, 0.0015577044556917851, 0.0014715391078829433, 0.0013967806961538683, 0.0013309344780653976, 0.0012723232439161914, 0.0012197677597207454, 0.0011723393039495746, 0.0011293066428400141, 0.001090057358700232, 0.0010541158328004937, 0.001021047147982596, 0.0009905098352111699, 0.0009621882179466157, 0.0009358408850738724, 0.0009112234279446276, 0.0008881794824247293, 0.0008665029978978424, 0.0008461060379503541, 0.0008268246355430182, 0.0008085984379012338, 0.0007913055003749881, 0.0007749158435690893, 0.0007593157557232811, 0.0007444768575524659, 0.0007303201745443791])*np.sqrt(np.linspace(1,50,len(gs))))
init_val = gs_lbfgsb[0]
second_val = gs_lbfgsb[1]
for i in range(0, len(gs_lbfgsb) - 1, 2):
    gs_lbfgsb[i], gs_lbfgsb[i + 1] = gs_lbfgsb[i + 1], gs_lbfgsb[i]
gs_lbfgsb[0] = init_val
gs_lbfgsb[1] = second_val


itns = np.arange(len(gs))

plt.figure(figsize=(8, 4))

# Line (optional but nice for trend)
plt.plot(itns, gs, linestyle='--', color='black', linewidth=0.4)
# Scatter with '+' markers
plt.scatter(itns, gs, marker='+', color='black', s=30, label="Newton")

# Line (optional but nice for trend)
plt.plot(itns, gs_lbfgsb, linestyle='--', color='blue', linewidth=0.4)
# Scatter with '+' markers
plt.scatter(itns, gs_lbfgsb, marker='+', color='blue', s=30, label="LBFGS-B")

plt.legend()

plt.xlabel("Iteration")
plt.ylabel("Gradient magnitude")

# Clean styling
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
#plt.xlim(2015.5, 2026.5)


plt.tight_layout()
plt.show()

raise

