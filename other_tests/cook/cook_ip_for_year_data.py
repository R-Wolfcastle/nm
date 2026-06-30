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
from osgeo import gdal, osr
from scipy import ndimage as ndi


year = sys.argv[1]

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
    datafile = gdal.Rasterize('/vsimem/dataOddGeoFcts.tif', gdal.OpenEx(in_shp), options=options)
    data_ = datafile.ReadAsArray()
    datafile=None
    gdal.Unlink('/vsimem/dataOddGeoFcts.tif')
    return data_

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

def smooth_gaussian(array, sigma=1.5):
    return ndi.gaussian_filter(array, sigma=sigma)

def smooth_gaussian_nan(array, sigma=1.5):
    kernel = Gaussian2DKernel(x_stddev=sigma)
    return convolve(array, kernel, preserve_nan=True, boundary="extend")

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

def level_set_gl_normal(thickness, bed, add_scalar_ghost_cells,
                        gradient_function):
    sg = bed+thickness
    sf = thickness*(1-c.RHO_I/c.RHO_W)

    sg = add_scalar_ghost_cells(sg)
    sf = add_scalar_ghost_cells(sf)

    varphi = smooth_gaussian(sg-sf, sigma=3)

    dp_dx, dp_dy = gradient_function(-varphi)
    dp_dx_n = dp_dx/jnp.sqrt(dp_dx**2 + dp_dy**2 + 1e-12)
    dp_dy_n = dp_dy/jnp.sqrt(dp_dx**2 + dp_dy**2 + 1e-12)
    
    return dp_dx_n, dp_dy_n


def grounding_line_flux(gl_normal_x, gl_normal_y, u, v, h):
    return jnp.sum(h*(u*gl_normal_x+v*gl_normal_y))
    

def gl_flux_functional(gl_normal_x, gl_normal_y, h, solver):

    def gl_flux(q, p):
        u, v = solver(q, p, jnp.zeros_like(h), jnp.zeros_like(h), h)
        return grounding_line_flux(gl_normal_x, gl_normal_y, u, v, h)
        
    return gl_flux

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

def run_ip(year, res):

    print(f"IP FOR YEAR {year}")

    print("Extracting things from ncdf")
    nc_file = xr.open_dataset(f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_data_wpp/{res}m_res/{year}.nc")

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

    #plt.imshow(speed_obs, vmin=0, vmax=1000, cmap="RdYlBu_r")
    #plt.imshow(uc, cmap="Grays_r", alpha=0.1)
    #plt.colorbar()
    #plt.show()

    #raise


    print("defining misfit")
    
    reg_phigrad   = 1e6
    reg_cgrad     = 2e-1
    reg_phi       = 1e0
    reg_c         = 1e-6
    lambda_       = 2e-3*((res/500)**2)
    ip_iterations = 40

    
    outdir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/{reg_phigrad}_{reg_cgrad}_{reg_phi}_{reg_c}_lambda{lambda_}_{ip_iterations}its/{res}m_res/"
    os.system(f"mkdir -p {outdir}")


    
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
        phi_regn_term = reg_phigrad * jnp.sum( mask.reshape(-1) *\
                                    (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
        
    
    
        C = C_0*jnp.exp(p.reshape((nr, nc)))
        dC_dx, dC_dy = left_top_centred_gradient_function(C)
    
        C_regn_term = reg_cgrad * jnp.sum( mask.reshape(-1) *\
                                    (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
    

        #phi_box_constraint = 1e0 * jnp.sum(
        #    jax.nn.softplus(5*(phi - 4))**2 +
        #    jax.nn.softplus(10*(0.1 - phi))**2
        #) / (nr*nc)
        phi_box_constraint = reg_phi * jnp.sum(
            (phi - phi_0)**2
        ) / (nr*nc)
        c_box_constraint = reg_c * jnp.sum( 
            (C - C_0)**2
        ) / (nr*nc)
        
        #+\
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
        jax.debug.print("c_box_constraint: {x}", x=c_box_constraint)
        jax.debug.print("Total cost: {x}", x=misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint)
   

        #return misfit_term, regn_term, misfit_term + regn_term
        return misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint
   

    print("defining newton optimiser")
   
    def newton_function(misfit_functional, solver,  misfit_fctl_args=(), iterations=50, lambda_=0):

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
            
            damping = lambda_

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
    
    
    newton_iterator = newton_function(regularised_misfit, solver, (speed_obs, uc), iterations=ip_iterations, lambda_=lambda_)

    print("solving optimisation problem")
    qp_out = newton_iterator(qp_initial_guess)


    q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    p_out = qp_out[(nr*nc):].reshape((nr,nc))
    

    print(f"SAVING OUTPUT TO {outdir}")

    geocode_array_1(q_out, (tlxy[0], tlxy[1]), res, f"{outdir}/q_out_{year}.tiff")
    geocode_array_1(p_out, (tlxy[0], tlxy[1]), res, f"{outdir}/p_out_{year}.tiff")


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
    #plt.imshow(phi_0, vmin=0, vmax=1, cmap="cubehelix")
    #plt.colorbar()
    #plt.show()
    #
    #phi_out = phi_0*jnp.exp(q_out)
    #plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
    #plt.colorbar()
    #plt.show()
    #
    #C_out = C_0*jnp.exp(p_out)
    #plt.imshow(jnp.log(C_out), cmap="magma", vmin=0, vmax=8)
    #plt.colorbar()
    #plt.show()


    #u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)

    #show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=1000)


    #plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc, vmin=-100, vmax=100, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()

    #plt.imshow(((u_out**2 + v_out**2)**(0.5)-speed_obs) * uc/(speed_obs+1e-10), vmin=-1, vmax=1, cmap="RdBu_r")
    #plt.colorbar()
    #plt.show()




def run_ip_measures(res):

    print("Extracting things from ncdf")
    nc_file = xr.open_dataset(f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_data_wpp/{res}m_res/measures.nc")

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
  

    rst_bb_fp = "/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/resistive_stress_bb/box.shp"
    rst_bb = open_shapefile_as_mask(rst_bb_fp,
                                    (*tlxy, *brxy),
                                    res)
    

    randd_stress_function = cc_resistive_and_deviatoric_stress_tensors(nr, nc, res, res,
                                                       extrapolate_over_cf,
                                                       add_uv_ghost_cells, add_scalar_ghost_cells,
                                                       gradient_function, phi_0)
    
    



    print("defining misfit")

    reg_phigrad   = 1e6
    reg_cgrad     = 2e-1
    reg_phi       = 1e0
    reg_c         = 1e-6
    lambda_       = 2e-3
    ip_iterations = 40


    outdir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/{reg_phigrad}_{reg_cgrad}_{reg_phi}_{reg_c}_lambda{lambda_}_{ip_iterations}its_INITIAL_MEASURES/{res}m_res/"
    os.system(f"mkdir -p {outdir}")


    
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
        phi_regn_term = reg_phigrad * jnp.sum( mask.reshape(-1) *\
                                    (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
        
    
    
        C = C_0*jnp.exp(p.reshape((nr, nc)))
        dC_dx, dC_dy = left_top_centred_gradient_function(C)
    
        C_regn_term = reg_cgrad * jnp.sum( mask.reshape(-1) *\
                                    (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
    

        #phi_box_constraint = 1e0 * jnp.sum(
        #    jax.nn.softplus(5*(phi - 4))**2 +
        #    jax.nn.softplus(10*(0.1 - phi))**2
        #) / (nr*nc)
        phi_box_constraint = reg_phi * jnp.sum(
            (phi - phi_0)**2
        ) / (nr*nc)
        c_box_constraint = reg_c * jnp.sum( 
            (C - C_0)**2
        ) / (nr*nc)
        
        #+\
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
        jax.debug.print("c_box_constraint: {x}", x=c_box_constraint)
        jax.debug.print("Total cost: {x}", x=misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint)
   

        #return misfit_term, regn_term, misfit_term + regn_term
        return misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint
   

    print("defining newton optimiser")
   
    def newton_function(misfit_functional, solver,  misfit_fctl_args=(), iterations=50, lambda_=0):

        def reduced_functional(qp):
            q = qp[:(nr*nc)]
            p = qp[(nr*nc):]
            u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
            return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)
    
        def newton(initial_guess):

            qp = initial_guess
            
            damping = lambda_

            cost = jnp.inf

            
            g_old = jnp.inf

            qp_old = qp.copy()

            itns_since_damping_reduced = 0

            get_grad = jax.grad(reduced_functional)

            for itn in range(iterations):

                g = get_grad(qp)
                
                _, vjp_grad = jax.vjp(get_grad, qp)

                g_np = np.array(g)
            
                gnorm = np.linalg.norm(g_np)
                print(f"iter {itn}, ||g|| = {gnorm}")
                
                if itn == 0:
                    first_gnorm = gnorm.copy()
                

                def matvec(v_np):
                    v = jnp.array(v_np, dtype=qp.dtype)
            
                    (Hv,) = vjp_grad(v)
            
                    return np.array(Hv + damping * v)
            


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
                
                qp = qp + dqp
                qp = jnp.concatenate( (
                                       jnp.minimum( qp[:(nr*nc)], 1 ),
                                       jnp.minimum( qp[(nr*nc):], 3.5 )
                                      )
                                    )

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
    
    
    newton_iterator = newton_function(regularised_misfit, solver, (speed_obs, uc), iterations=ip_iterations, lambda_=lambda_)

    print("solving optimisation problem")
    qp_out = newton_iterator(qp_initial_guess)


    q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    p_out = qp_out[(nr*nc):].reshape((nr,nc))
    

    print(f"SAVING OUTPUT TO {outdir}")

    geocode_array_1(q_out, (tlxy[0], tlxy[1]), res, f"{outdir}/q_out_measures.tiff")
    geocode_array_1(p_out, (tlxy[0], tlxy[1]), res, f"{outdir}/p_out_measures.tiff")


    phi_out = phi_0*jnp.exp(q_out)
    C_out = C_0*jnp.exp(p_out)
    geocode_array_1(phi_out, (tlxy[0], tlxy[1]), res, f"{outdir}/Phi_measures.tiff")
    geocode_array_1(C_out, (tlxy[0], tlxy[1]), res, f"{outdir}/C_measures.tiff")
    
   

    u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)
    
    geocode_array_1(u_out, (tlxy[0], tlxy[1]), res, f"{outdir}/u_out_measures.tiff")
    geocode_array_1(v_out, (tlxy[0], tlxy[1]), res, f"{outdir}/v_out_measures.tiff")
   
    




    print("computing RST")
    rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
    rst = rst*ice_mask[:,:,None,None]
    dst = dst*ice_mask[:,:,None,None]
    
    tau_1, tau_2, tau_shear = principal_and_max_shear(dst)
    
    s1, s2, shear = principal_and_max_shear(rst)
    
    sigma_1 = 2*tau_1 + tau_2
    sigma_2 = 2*tau_2 + tau_1
    
    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    

    geocode_array_1(s1, (tlxy[0], tlxy[1]), res, f"{outdir}/s1_measures.tiff")
    geocode_array_1(s2, (tlxy[0], tlxy[1]), res, f"{outdir}/s2_measures.tiff")
    geocode_array_1(shear, (tlxy[0], tlxy[1]), res, f"{outdir}/shear_measures.tiff")

   

    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    
    geocode_array_1(aligned_rst[:,:,0,0], (tlxy[0], tlxy[1]), res, f"{outdir}/sxx_measures.tiff")
    
    mean_s1 = jnp.nanmean(jnp.where(rst_bb==1,
                                    s1,
                                    jnp.nan)
                                  )

    print(f"!!!!!!!!! MEASURES MEAN S1: {mean_s1}")


    q_flat = q_out.reshape(-1)

    def mean_rst_and_uncertainty():

        def rst_fctl(q_flattened):
            q_out = q_flattened.reshape((nr, nc))
            u_out, v_out = solver(q_out, p_out,
                                  u_init, v_init,
                                  thk)
            rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
            s1, s2, shear = principal_and_max_shear(rst)
            mean_s1 = jnp.nanmean(jnp.where(rst_bb==1,
                                            s1,
                                            jnp.nan)
                                          )
    
            return mean_s1


        def reduced_misfit_functional(q_flattened):
            u_out, v_out = solver(q_flattened.reshape(nr, nc), p_out, u_init, v_init, thk)
            return regularised_misfit(u_out, v_out, q_flat, p_out.reshape(-1), speed_obs, uc)


        get_grad = jax.grad(rst_fctl)
        grad_functional = np.array(get_grad(q_flat))


        get_grad_cost = jax.grad(reduced_misfit_functional)
        _, vjp_grad = jax.vjp(get_grad_cost, q_flat)

        def matvec(v_np):
            v = jnp.array(v_np, dtype=q_flat.dtype)
        
            (Hv,) = vjp_grad(v)
        
            return np.array(Hv + lambda_ * v)
        

        ########## PETSc VERSION #################
        petsc_solver = create_petsc_operator_solver(matvec,
                                                    size=q_flat.size,
                                                    ksp_type="gmres",
                                                    preconditioner=None,
                                                    ksp_max_iter=200,
                                                    monitor_ksp=True)
        right_product = petsc_solver(grad_functional)

        uncertainty = np.dot(grad_functional, right_product)
        ########## END PETSc VERSION #############


        print(f"######## MEASURES Delta S1: {uncertainty}")

        
    mean_rst_and_uncertainty()


    #grounded = jnp.where((thk+topg)>(thk*(1-c.RHO_I/c.RHO_W)), 1, 0)
    #grounded = clean_mask_scipy(grounded).astype(int)
    #grounded = add_scalar_ghost_cells(grounded).astype(int)
    #gl_main  = (grounded & ~binary_erosion(grounded))[1:-1,1:-1]
    #grounded = grounded[1:-1, 1:-1]
    #
    #gradient_function = cc_gradient_function(res, res)
    #gl_normal_x, gl_normal_y = level_set_gl_normal(thk, topg, add_scalar_ghost_cells,
    #                                               gradient_function)
    #gl_normal_x = gl_normal_x*gl_main
    #gl_normal_y = gl_normal_y*gl_main
    #
    #gl_flux_function = gl_flux_functional(gl_normal_x, gl_normal_y, thk, solver)
    #gl_flux_gradient_fct = jax.grad(gl_flux_function, argnums=0)
    #    
    #dFdq = gl_flux_gradient_fct(q_out, p_out)*ice_mask*(1-grounded)

    #geocode_array_1(dFdq, (tlxy[0], tlxy[1]), res, f"{outdir}/dFdq_{year}.tiff")




def run_ip_measures_prior(year, res):

    print(f"IP FOR YEAR {year}")

    print("Extracting things from ncdf")
    nc_file = xr.open_dataset(f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_data_wpp/{res}m_res/{year}.nc")

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

    measures_ctrl_dir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/1000000.0_0.2_1.0_1e-06_lambda0.002_40its_INITIAL_MEASURES/{res}m_res/"
    #q_m = gdal.Open(measures_ctrl_dir+"q_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()
    p_m = gdal.Open(measures_ctrl_dir+"p_out_measures.tiff", gdal.GA_ReadOnly).ReadAsArray()

    #phi_0 = phi_0*jnp.exp(q_m)
    C_0   = C_0 * jnp.exp(p_m)


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
  

    rst_bb_fp = "/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/resistive_stress_bb/box.shp"
    rst_bb = open_shapefile_as_mask(rst_bb_fp,
                                    (*tlxy, *brxy),
                                    res)
    

    randd_stress_function = cc_resistive_and_deviatoric_stress_tensors(nr, nc, res, res,
                                                       extrapolate_over_cf,
                                                       add_uv_ghost_cells, add_scalar_ghost_cells,
                                                       gradient_function, phi_0)
    
    



    print("defining misfit")
    
    reg_phigrad   = 3e4
    reg_cgrad     = 2e-1
    reg_phi       = 2e-3
    reg_c         = 1e-4
    lambda_       = 8e-4*((res/500)**2)
    ip_iterations = 50

    
    outdir = f"/uolstore/Research/b/b0133/eartsu/new_model_misc/cook_study/annual_ip_out_wpp/{reg_phigrad}_{reg_cgrad}_{reg_phi}_{reg_c}_lambda{lambda_}_{ip_iterations}its_measuresCprior/{res}m_res/"
    os.system(f"mkdir -p {outdir}")


    
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
        phi_regn_term = reg_phigrad * jnp.sum( mask.reshape(-1) *\
                                    (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
        
    
    
        C = C_0*jnp.exp(p.reshape((nr, nc)))
        dC_dx, dC_dy = left_top_centred_gradient_function(C)
    
        C_regn_term = reg_cgrad * jnp.sum( mask.reshape(-1) *\
                                    (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2) *\
                                    (1-border_cells_flat)
                                  )/(nr*nc)
    

        #phi_box_constraint = 1e0 * jnp.sum(
        #    jax.nn.softplus(5*(phi - 4))**2 +
        #    jax.nn.softplus(10*(0.1 - phi))**2
        #) / (nr*nc)
        phi_box_constraint = reg_phi * jnp.sum(
            (phi - phi_0)**2
        ) / (nr*nc)
        c_box_constraint = reg_c * jnp.sum( 
            (C - C_0)**2
        ) / (nr*nc)
        
        #+\
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
        jax.debug.print("c_box_constraint: {x}", x=c_box_constraint)
        jax.debug.print("Total cost: {x}", x=misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint)
   

        #return misfit_term, regn_term, misfit_term + regn_term
        return misfit_term + phi_regn_term + C_regn_term + phi_box_constraint + c_box_constraint
   

    print("defining newton optimiser")
   
    def newton_function(misfit_functional, solver,  misfit_fctl_args=(), iterations=50, lambda_=0):

        def reduced_functional(qp):
            q = qp[:(nr*nc)]
            p = qp[(nr*nc):]
            u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
            return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)
    
        def newton(initial_guess):

            qp = initial_guess
            
            damping = lambda_

            cost = jnp.inf

            
            g_old = jnp.inf

            qp_old = qp.copy()

            itns_since_damping_reduced = 0

            get_grad = jax.grad(reduced_functional)

            for itn in range(iterations):

                g = get_grad(qp)
                
                _, vjp_grad = jax.vjp(get_grad, qp)

                g_np = np.array(g)
            
                gnorm = np.linalg.norm(g_np)
                print(f"iter {itn}, ||g|| = {gnorm}")
                
                if itn == 0:
                    first_gnorm = gnorm.copy()
                

                def matvec(v_np):
                    v = jnp.array(v_np, dtype=qp.dtype)
            
                    (Hv,) = vjp_grad(v)
            
                    return np.array(Hv + damping * v)
            


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
                
                qp = qp + dqp
                qp = jnp.concatenate( (
                                       jnp.minimum( qp[:(nr*nc)], 1 ),
                                       jnp.minimum( qp[(nr*nc):], 3.5 )
                                      )
                                    )

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
    
    
    newton_iterator = newton_function(regularised_misfit, solver, (speed_obs, uc), iterations=ip_iterations, lambda_=lambda_)

    print("solving optimisation problem")
    qp_out = newton_iterator(qp_initial_guess)


    q_out = qp_out[:(nr*nc)].reshape((nr,nc))
    p_out = qp_out[(nr*nc):].reshape((nr,nc))
    

    print(f"SAVING OUTPUT TO {outdir}")

    geocode_array_1(q_out, (tlxy[0], tlxy[1]), res, f"{outdir}/q_out_{year}.tiff")
    geocode_array_1(p_out, (tlxy[0], tlxy[1]), res, f"{outdir}/p_out_{year}.tiff")


    phi_out = phi_0*jnp.exp(q_out)
    C_out = C_0*jnp.exp(p_out)
    geocode_array_1(phi_out, (tlxy[0], tlxy[1]), res, f"{outdir}/Phi_{year}.tiff")
    geocode_array_1(C_out, (tlxy[0], tlxy[1]), res, f"{outdir}/C_{year}.tiff")
    
   

    u_out, v_out = solver(q_out, p_out, u_init, v_init, thk)
    
    geocode_array_1(u_out, (tlxy[0], tlxy[1]), res, f"{outdir}/u_out_{year}.tiff")
    geocode_array_1(v_out, (tlxy[0], tlxy[1]), res, f"{outdir}/v_out_{year}.tiff")
   



    print("computing RST")
    rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
    rst = rst*ice_mask[:,:,None,None]
    dst = dst*ice_mask[:,:,None,None]
    
    tau_1, tau_2, tau_shear = principal_and_max_shear(dst)
    
    s1, s2, shear = principal_and_max_shear(rst)
    
    sigma_1 = 2*tau_1 + tau_2
    sigma_2 = 2*tau_2 + tau_1
    
    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    

    geocode_array_1(s1, (tlxy[0], tlxy[1]), res, f"{outdir}/s1_{year}.tiff")
    geocode_array_1(s2, (tlxy[0], tlxy[1]), res, f"{outdir}/s2_{year}.tiff")
    geocode_array_1(shear, (tlxy[0], tlxy[1]), res, f"{outdir}/shear_{year}.tiff")

   

    aligned_rst = align_tensor_with_flow(u_out, v_out, rst)
    
    geocode_array_1(aligned_rst[:,:,0,0], (tlxy[0], tlxy[1]), res, f"{outdir}/sxx_{year}.tiff")
    
    mean_s1 = jnp.nanmean(jnp.where(rst_bb==1,
                                    s1,
                                    jnp.nan)
                                  )

    print(f"!!!!!!!!! {year} MEAN S1: {mean_s1}")


    q_flat = q_out.reshape(-1)

    def mean_rst_and_uncertainty():

        def rst_fctl(q_flattened):
            q_out = q_flattened.reshape((nr, nc))
            u_out, v_out = solver(q_out, p_out,
                                  u_init, v_init,
                                  thk)
            rst, dst = randd_stress_function(q_out, u_out, v_out, thk)
            s1, s2, shear = principal_and_max_shear(rst)
            mean_s1 = jnp.nanmean(jnp.where(rst_bb==1,
                                            s1,
                                            jnp.nan)
                                          )
    
            return mean_s1


        def reduced_misfit_functional(q_flattened):
            u_out, v_out = solver(q_flattened.reshape(nr, nc), p_out, u_init, v_init, thk)
            return regularised_misfit(u_out, v_out, q_flat, p_out.reshape(-1), speed_obs, uc)


        get_grad = jax.grad(rst_fctl)
        grad_functional = np.array(get_grad(q_flat))


        get_grad_cost = jax.grad(reduced_misfit_functional)
        _, vjp_grad = jax.vjp(get_grad_cost, q_flat)

        def matvec(v_np):
            v = jnp.array(v_np, dtype=q_flat.dtype)
        
            (Hv,) = vjp_grad(v)
        
            return np.array(Hv + lambda_ * v)
        

        ########## PETSc VERSION #################
        petsc_solver = create_petsc_operator_solver(matvec,
                                                    size=q_flat.size,
                                                    ksp_type="gmres",
                                                    preconditioner=None,
                                                    ksp_max_iter=200,
                                                    monitor_ksp=True)
        right_product = petsc_solver(grad_functional)

        uncertainty = np.dot(grad_functional, right_product)
        ########## END PETSc VERSION #############


        print(f"######## {year} Delta S1: {uncertainty}")

        
    mean_rst_and_uncertainty()

    



    geocode_array_1(jnp.sqrt(u_out**2 + v_out**2), (tlxy[0], tlxy[1]), res, f"{outdir}/speed_out_{year}.tiff")

    geocode_array_1((jnp.sqrt(u_out**2 + v_out**2)-speed_obs)*uc, (tlxy[0], tlxy[1]), res, f"{outdir}/misfit_{year}.tiff")





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

    geocode_array_1(dFdq, (tlxy[0], tlxy[1]), res, f"{outdir}/dFdq_{year}.tiff")


res = 250

tlxy = (1_020_000,  -2_035_000)
brxy = (1_154_000, -2_148_000)

##run_ip(year, res)
run_ip_measures_prior(year, res)
#run_ip_measures(res)












