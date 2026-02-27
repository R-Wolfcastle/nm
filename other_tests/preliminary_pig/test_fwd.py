
#1st party
import sys
import time

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils")
import constants_years as c
from plotting_stuff import show_vel_field
from grid import binary_erosion, binary_dilation,\
        cc_gradient_function, add_ghost_cells_fcts

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

    plt.imshow(np.flipud(energy_stack[0,:,:]))
    plt.colorbar()
    plt.show()
    plt.imshow(np.flipud(energy_stack[10,:,:]))
    plt.colorbar()
    plt.show()
    plt.imshow(np.flipud(energy_stack[20,:,:]))
    plt.colorbar()
    plt.show()

    raise

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


def setup_domain(resolution, tlxy, brxy):

    
    x0, y1 = tlxy
    x1, y0 = brxy
    assert (x1 - x0) % resolution == 0, "x-extent is not divisible by resolution"
    assert (y1 - y0) % resolution == 0, "y-extent is not divisible by resolution"





    bedmachine_fp = "/Users/eartsu/Documents/misc_data/bedmachine_v3_ase_MAR-1980-2021_smooth_post_relax_50y_500m_coordshift.nc"
    temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW_upsidedownlayers.nc"
    #temp_fp = "/Users/eartsu/Library/CloudStorage/OneDrive-UniversityofLeeds/Documents/misc_data/ase_temp_8km_24_fromCTLW.nc"

    
    xs = np.arange(x0, x1, resolution)
    ys = np.arange(y1, y0, -resolution)

    target_grid = xr.Dataset(
        coords=dict(x=("x", xs),
                    y=("y", ys))
    )

    # load raw
    bed_nc  = xr.open_dataset(bedmachine_fp)
    temp_nc = xr.open_dataset(temp_fp)

    # reproject / reindex both onto same grid
    bed_r  = bed_nc.interp_like(target_grid, method="linear")
    temp_r = temp_nc.interp_like(target_grid, method="linear")

    
    
#    data_nc = xr_load_crop_and_resample(data_nc_fp, 
#                                        tlxy,
#                                        brxy,
#                                        resolution)

    phi, C, topg, thk = (bed_r[var_].values for var_ in ["mucoef",
                                                         "c_third",
                                                         #"c_one",
                                                         "topg", 
                                                         "thk"])

    C[:2, :] = 1e10
    C[-2:,:] = 1e10
    C[:, :2] = 1e10
    C[:,-2:] = 1e10

    #thk_eroded = binary_erosion(thk)
    
    ice_mask = np.where(thk>0.01, 1, 0)




#    temp_nc = xr_load_crop_and_resample(temp_fp,
#                                        tlxy,
#                                        brxy,
#                                        resolution)

    temp = temp_r["temperature"].values
    
    return phi, C, topg, thk, ice_mask, temp







def tiny_ice_shelf():
    lx = 150_0
    ly = 150_0
    resolution = 15_0 #m

    nr = int(ly/resolution)
    nc = int(lx/resolution)

    lx = nr*resolution
    ly = nc*resolution

    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)

    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    thk_profile = 500# - 300*x/lx
    thk = jnp.zeros((nr, nc))+thk_profile
    thk = thk.at[:,  -1:].set(0)
    thk = thk.at[-5:,-4:].set(0)

    b = jnp.zeros_like(thk)-600
    b = b.at[:1, :].set(-440)
    b = b.at[:, :1].set(-440)
    b = b.at[-1:, :].set(-440)

    mucoef = jnp.ones_like(thk)

    C = jnp.zeros_like(thk)
    C = C.at[:2, :].set(1e12)
    C = C.at[:, :2].set(1e12)
    C = C.at[-2:,:].set(1e12)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile

    q = jnp.zeros_like(C)

    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q


#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = tiny_ice_shelf()
#ice_mask = jnp.where(thk>0,1,0)
#
#u_init = jnp.zeros((nr,nc))
#v_init = u_init.copy()
#
#n_pic_iterations = 100
#n_newt_iterations = 0
#
##plt.imshow(ice_mask & ~binary_erosion(ice_mask))
##plt.show()
##raise
#
##solver = make_pic_velocity_solver_function_densetest(nr, nc,
#solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
#                                                         delta_x, delta_y,
#                                                         b, ice_mask,
#                                                         n_pic_iterations,
#                                                         n_newt_iterations,
#                                                         mucoef_0, C,
#                                                         #sliding="basic_weertman")
#                                                         sliding="linear")
#
#u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0)
#
#
#raise





#PIG WHOLE
#tlxy = (-1_700_000, -50_000)
#brxy = (-1_500_000, -370_000)


#PIG ICE SHELF
tlxy = (-1_654_000, -190_000)
brxy = (-1_550_000, -346_000)

res = 1000

phi, C, topg, thk, ice_mask, temp = setup_domain(res, tlxy, brxy)


#phi = jnp.rot90(phi)
#C = jnp.rot90(C)
#topg = jnp.rot90(topg)
#thk = jnp.rot90(thk)
#ice_mask = jnp.rot90(ice_mask)

nr, nc = phi.shape

#grounded = jnp.where((topg+thk)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
#nearly_grounded = binary_dilation(grounded) & ~grounded
#topg = jnp.where(nearly_grounded>0, -thk*c.RHO_I/c.RHO_W + 0.1, topg)
#C = jnp.where(nearly_grounded>0, 100000, C)





def run_fwd():

    u_init = jnp.zeros((nr,nc))
    v_init = u_init.copy()
    
    n_pic_iterations = 14
    n_newt_iterations = 12
    
    solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                             res, res,
                                                             topg, ice_mask,
                                                             n_pic_iterations,
                                                             n_newt_iterations,
                                                             phi, C,
                                                             sliding="basic_weertman",
                                                             temperature_field=temp)
                                                             #sliding="linear")
    
    u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
    
    
    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/u_double_visc.npy", u_out)
    #jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/v_double_visc.npy", v_out)
    
    
    show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=4500)


run_fwd()






#raise
######INVERSE PROBLEM GUBBINS:
















cf_cells = (thk>0) & ~binary_erosion(thk>0)

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



add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
gradient_function = cc_gradient_function(res, res)


def regularised_misfit(u_mod, v_mod, q, p, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    
    misfit_term = jnp.sum(mask.reshape(-1) * \
                          (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2
                         )/(nr*nc)

    #Assume that things are, on average, wrong by 100ma^-1. So, divide by 10_000:
    misfit_term = misfit_term/10_000


    phi = mucoef_0*jnp.exp(q.reshape((nr, nc)))
    dphi_dx, dphi_dy = gradient_function(phi)


    #The coefficients are at least an order of magnitude smaller than Steph's choices of:
    #alpha_phi = 1e11 (for me, that would be 1e11/10_000 ~ 1e7)
    #alpha_C   = 1e3  (for me, that would be 1e3 /10_000 ~ 1e-1)

    #maybe 1e4 a good shout?
    phi_regn_term = 5e5 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                (1-border_cells_reduced_flat)
                              )/(nr*nc)
    


    C = C_0*jnp.exp(p.reshape((nr, nc)))
    dC_dx, dC_dy = gradient_function(C)

    C_regn_term = 1e-2 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                (1-border_cells_reduced_flat)
                              )/(nr*nc)


    jax.debug.print("misfit_term: {x}", x=misfit_term)
    jax.debug.print("phi_regn_term: {x}", x=phi_regn_term)
    jax.debug.print("C_regn_term: {x}", x=C_regn_term)

    #return misfit_term, regn_term, misfit_term + regn_term
    return misfit_term + phi_regn_term + C_regn_term

def lbfgsb_function(misfit_functional, misfit_fctl_args=(), iterations=50):
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
                           options={"maxiter": iterations, "maxls": 4} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb










#u_obs = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/u_double_visc.npy")
#v_obs = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/PIGGING_TEA_BREAK/v_double_visc.npy")

#speed_obs = jnp.sqrt(u_obs**2+v_obs**2 + 1e-12)

#uc = jnp.where(speed_obs>0, 1, 0)
#uc = binary_erosion(uc)
#uc = binary_erosion(uc)


uo, uc = setup_comparison_data(res, tlxy, brxy)
uc = binary_erosion(uc)
uc = binary_erosion(uc)



#plt.imshow(thk, cmap="Grays_r", vmin=0, vmax=1250)
#plt.imshow(uo, cmap="magma", alpha=0.25)
#plt.show()
#raise



key = jax.random.PRNGKey(0)
#noise = 10*jax.random.normal(key, (nr,nc))
##noise = jnp.where(speed_obs>0, noise, 0)
#speed_obs = speed_obs + noise
#speed_obs = jnp.maximum(speed_obs, 0)

#plt.imshow(speed_obs)
#plt.colorbar()
#plt.show()

mucoef_0 = phi.copy()

C_0 = C.copy()


u_init = jnp.zeros((nr,nc))
v_init = u_init.copy()

n_pic_iterations = 14
n_newt_iterations = 12

solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                         res, res,
                                                         topg, ice_mask,
                                                         n_pic_iterations,
                                                         n_newt_iterations,
                                                         phi, C,
                                                         sliding="basic_weertman",
                                                         temperature_field=temp)


q_initial_guess = jnp.zeros_like(thk).reshape(-1)
p_initial_guess = jnp.zeros_like(thk).reshape(-1)

qp_initial_guess = jnp.zeros((2*nr*nc,))

lbfgsb_iterator = lbfgsb_function(regularised_misfit, (uo, uc), iterations=20)
qp_out = lbfgsb_iterator(qp_initial_guess)








q_out = qp_out[:(nr*nc)].reshape((nr,nc))
p_out = qp_out[(nr*nc):].reshape((nr,nc))

plt.imshow(p_out)
plt.colorbar()
plt.show()

plt.imshow(q_out, vmin=-1, vmax=1, cmap="RdBu")
plt.colorbar()
plt.show()

phi_out = mucoef_0*jnp.exp(q_out)
plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
plt.colorbar()
plt.show()

C_out = C_0*jnp.exp(p_out)
plt.imshow(C_out, cmap="magma")
plt.colorbar()
plt.show()

u_mod_end, v_mod_end = solver(q_out, p_out, u_init, v_init, thk)

show_vel_field(u_mod_end, v_mod_end, cmap="RdYlBu_r", vmin=0, vmax=4500)

plt.figure(figsize=(8,6))
plt.imshow(jnp.sqrt(u_mod_end**2 + v_mod_end**2) - uo, vmin=-500, vmax=500, cmap="RdBu_r")
plt.colorbar()
plt.show()
