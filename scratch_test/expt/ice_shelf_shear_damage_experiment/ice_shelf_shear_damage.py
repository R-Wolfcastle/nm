#1st party
from pathlib import Path
import sys
import time


#local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants as c
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps


#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy
from scipy.optimize import minimize as scinimize

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import xarray as xr


#np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
np.set_printoptions(precision=10, suppress=False, linewidth=np.inf, threshold=np.inf)


def create_sparse_petsc_la_solver_with_custom_vjp(coordinates, jac_shape,\
                                    ksp_type='gmres', preconditioner='hypre',\
                                    precondition_only=False, monitor_ksp=False):

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()
    
    def construct_ab(values, b, transpose):
        if transpose:
            iptr, j, values = scipy_coo_to_csr(values, coordinates[::-1,:],\
                                           jac_shape, return_decomposition=True)
        else:
            iptr, j, values = scipy_coo_to_csr(values, coordinates,\
                                           jac_shape, return_decomposition=True)
        #rows_local = int(jac_shape[0] / size)

        A = PETSc.Mat().createAIJ(size=jac_shape, \
                                  csr=(iptr.astype(np.int32), j.astype(np.int32), values),\
                                  comm=comm)
        
        b = PETSc.Vec().createWithArray(b, comm=comm)
        return A, b

    
    def create_solver_object(A):
        
        #set ksp iterations
        opts = PETSc.Options()
        opts['ksp_max_it'] = 20
        if monitor_ksp:
            opts['ksp_monitor'] = None
        opts['ksp_rtol'] = 1e-20
        
        # Create a linear solver
        ksp = PETSc.KSP().create()
        ksp.setType(ksp_type)

        ksp.setOperators(A)
        ksp.setFromOptions()
        
        if preconditioner is not None:
            #assessing if preconditioner is doing anything:
            #print((A*x - b).norm())

            if preconditioner == 'hypre':
                pc = ksp.getPC()
                pc.setType('hypre')
                pc.setHYPREType('boomeramg')
            else:
                pc = ksp.getPC()
                pc.setType(preconditioner)
        
            return ksp, pc
        else:
            return ksp, None


    @custom_vjp
    def petsc_sparse_la_solver(values, b, transpose=False):
    
        A, b = construct_ab(values, b, transpose)
        x = b.duplicate()
        
        ksp, pc = create_solver_object(A)

        if precondition_only:
            pc.apply(b, x)
        else:
            ksp.solve(b, x)
        
        x_jnp = jnp.array(x.getArray())

        return x_jnp

    def la_solver_fwd(values, b):
        solution = petsc_sparse_la_solver(values, b)
        return solution, (values, b, solution)

    def linear_solve_bwd(res, x_bar):
        values, b, x = res

        lambda_ = petsc_sparse_la_solver(values, -x_bar, transpose=True)

        b_bar = -lambda_

        #sparse version of jnp.outer(x,lambda_)
        values_bar = x[coordinates[0]] * lambda_[coordinates[1]]

        return values_bar, b_bar


    petsc_sparse_la_solver.defvjp(la_solver_fwd, linear_solve_bwd)

    return petsc_sparse_la_solver




def interp_cc_to_fc_function(ny, nx):

    def interp_cc_to_fc(var):
        
        var_ew = jnp.zeros((ny, nx+1))
        var_ew = var_ew.at[:, 1:-1].set(0.5*(var[:, 1:]+var[:, :-1]))
        var_ew = var_ew.at[:, 0].set(var[:, 0])
        var_ew = var_ew.at[:, -1].set(var[:, -1])

        var_ns = jnp.zeros((ny+1, nx))
        var_ns = var_ns.at[1:-1, :].set(0.5*(var[:-1, :]+var[1:, :]))
        var_ns = var_ns.at[0, :].set(var[0, :])
        var_ns = var_ns.at[-1, :].set(var[-1, :])

        return var_ew, var_ns

    return interp_cc_to_fc


def interp_fc_to_cc_function(ny, nx):
    def interp_fc_to_cc(var_ew, var_ns):
        var_cc = 0.25*(var_ew[:,:-1]+var_ew[:,1:]+var_ns[1:,:]+var_ns[:-1,:])
        return var_cc
    return interp_fc_to_cc


def cc_gradient_function(dy, dx):

    def cc_gradient(var):

        dvar_dx = (0.5/dx) * (var[1:-1, 2:] - var[1:-1,:-2])
        dvar_dy = (0.5/dy) * (var[:-2,1:-1] - var[2:, 1:-1])

        return dvar_dx, dvar_dy

    return cc_gradient


def fc_gradient_functions(dy, dx):

    def ew_face_gradient(var):
        
        dvar_dx_ew = (var[1:-1, 1:] - var[1:-1, :-1])/dx

        dvar_dy_ew = (var[:-2, 1:] + var[:-2, :-1] - var[2:, 1:] - var[2:, :-1])/(4*dy)
        
        return dvar_dx_ew, dvar_dy_ew
    
    def ns_face_gradient(var):
        
        dvar_dy_ns = (var[:-1, 1:-1]-var[1:, 1:-1])/dy

        dvar_dx_ns = (var[:-1, 2:] + var[1:, 2:] - var[:-1, :-2] - var[1:, :-2])/(4*dx)
        
        return dvar_dx_ns, dvar_dy_ns
    
    return ew_face_gradient, ns_face_gradient


def add_ghost_cells_fcts(ny, nx):

    def add_reflection_ghost_cells(u_int, v_int):

        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #edges
        u = u.at[0, 1:-1].set( u[1, 1:-1])
        u = u.at[-1,1:-1].set( u[-2,1:-1])
        u = u.at[1:-1, 0].set(-u[1:-1, 1])
        u = u.at[1:-1,-1].set(-u[1:-1,-2])
        #corner points
        u = u.at[0,0].set(-u[1,1])
        u = u.at[-1,-1].set(-u[-2,-2])
        u = u.at[-1,0].set(-u[-2,1])
        u = u.at[0,-1].set(-u[1,-2])

        
        v = jnp.zeros((ny+2, nx+2))
        v = v.at[1:-1,1:-1].set(v_int)
        #edges
        v = v.at[0, 1:-1].set(-v[1, 1:-1])
        v = v.at[-1,1:-1].set(-v[-2,1:-1])
        v = v.at[1:-1, 0].set( v[1:-1, 1])
        v = v.at[1:-1,-1].set( v[1:-1,-2])
        #corner points
        v = v.at[0,0].set(-v[1,1])
        v = v.at[-1,-1].set(-v[-2,-2])
        v = v.at[-1,0].set(-v[-2,1])
        v = v.at[0,-1].set(-v[1,-2])

        return u, v

    def add_continuation_ghost_cells(h_int):

        h = jnp.zeros((ny+2, nx+2))
        h = h.at[1:-1,1:-1].set(h_int)
        #edges
        h = h.at[0, 1:-1].set(h[1, 1:-1])
        h = h.at[-1,1:-1].set(h[-2,1:-1])
        h = h.at[1:-1, 0].set(h[1:-1, 1])
        h = h.at[1:-1,-1].set(h[1:-1,-2])
        #corner points
        h = h.at[0,0].set(h[1,1])
        h = h.at[-1,0].set(h[-2,1])
        h = h.at[0,-1].set(h[1,-2])
        h = h.at[-1,-1].set(h[-2,-2])

        return h

    return add_reflection_ghost_cells, add_continuation_ghost_cells


def binary_erosion(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    #kernel = jnp.array([[1,1,1],
    #                    [1,1,1],
    #                    [1,1,1]], dtype=jnp.bool_)
    kernel = jnp.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float32)

    def erode_once(mask_float):
        out = jax.lax.conv_general_dilated(
            #shape (batch_size,channels,H,W)
            mask_float[None, None, :, :],
            #shape (out_chan,in_chan,H,W)
            kernel[None, None, :, :],
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NCHW','OIHW','NCHW')
        )
        return (out[0,0] > 3).astype(jnp.bool_)

    return erode_once(boolean_array.astype(jnp.float32))

def binary_dilation(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    kernel = jnp.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float32)

    def dilate_once(mask_float):
        out = jax.lax.conv_general_dilated(
            #shape (batch_size,channels,H,W)
            mask_float[None, None, :, :],
            #shape (out_chan,in_chan,H,W)
            kernel[None, None, :, :],
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NCHW','OIHW','NCHW')
        )
        return (out[0,0] > 0).astype(jnp.bool_)

    return dilate_once(boolean_array.astype(jnp.float32))

def extrapolate_over_cf_function(thk):

    cf_adjacent_zero_ice_cells = (thk<0) & binary_dilation(thk>0)

    ice_mask = (thk>0)

    ice_mask_shift_up    = jnp.roll(ice_mask, -1, axis=0)
    ice_mask_shift_down  = jnp.roll(ice_mask,  1, axis=0)
    ice_mask_shift_left  = jnp.roll(ice_mask, -1, axis=1)
    ice_mask_shift_right = jnp.roll(ice_mask,  1, axis=1)
    

    def extrapolate_over_cf(cc_field):

        u_shift_up    = jnp.roll(cc_field, -1, axis=0)
        u_shift_down  = jnp.roll(cc_field,  1, axis=0)
        u_shift_left  = jnp.roll(cc_field, -1, axis=1)
        u_shift_right = jnp.roll(cc_field,  1, axis=1)
        
        neighbour_values = jnp.stack([
            jnp.where(ice_mask_shift_up   ==1, u_shift_up,    0),
            jnp.where(ice_mask_shift_down ==1, u_shift_down,  0),
            jnp.where(ice_mask_shift_left ==1, u_shift_left,  0),
            jnp.where(ice_mask_shift_right==1, u_shift_right, 0),
        ])
        
        neighbour_counts = jnp.stack([
            (ice_mask_shift_up   ==1).astype(int),
            (ice_mask_shift_down ==1).astype(int),
            (ice_mask_shift_left ==1).astype(int),
            (ice_mask_shift_right==1).astype(int),
        ])
        
        u_extrap_boundary = neighbour_values.sum(axis=0) / neighbour_counts.sum(axis=0)

        return cc_field + u_extrap_boundary*cf_adjacent_zero_ice_cells.astype(jnp.float32)

    return extrapolate_over_cf

def compute_u_v_residuals_function_damcoef_ip(ny, nx, dy, dx,
                                   h_1d, mucoef,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_rflc_ghost_cells,
                                   add_cont_ghost_cells,
                                   extrp_over_cf):


    def compute_beta(u, v, sliding_coef):
        return sliding_coef*(jnp.sqrt(u**2+v**2+1e-15)**(-2/3))

    def compute_damage(u, v, h, damcoef):
        #damcoef = 5e-6

        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #interpolate things onto face-cenres
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        #damage. note it lives on faces
        damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew))/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)))
        damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns))/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)))
        #damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2)/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2))
        #damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2)/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2))
        #damage_ew = damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)
        #damage_ns = damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)
        damage_ew = jnp.minimum(damage_ew, 0.75)
        damage_ns = jnp.minimum(damage_ns, 0.75)

        return 0.25*(damage_ns[1:,:]+damage_ns[:-1,:]+damage_ew[:,1:]+damage_ew[:,:-1])
        

    def compute_u_v_residuals(u_1d, v_1d, damcoef, sliding_coef):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        dsdx, dsdy = cc_gradient(s)
        beta = compute_beta(u, v, sliding_coef)


        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #TODO: set bespoke conditions at the front for dvel_dx! otherwise
        #over-estimating the viscosity on the faces near the front!
        #u, v = extrapolate_over_cf(u, v)


        #momentum_term
        #quickly extrapolate velocity over calving front
        #NOTE: I'm not sure it's even really necessary to do this you know...
        #in the y-aligned calving front it only affects the ddx_ns derivatives.
        #u = extrp_over_cf(u)
        #v = extrp_over_cf(v)
        #and add the ghost cells in
        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        ##cc_derivatives, e.g. for viscosity calculation
        #dudx_cc, dudy_cc = cc_gradient(u)
        #dvdx_cc, dvdy_cc = cc_gradient(v)

        #interpolate things onto face-cenres
        h_ew, h_ns = interp_cc_to_fc(h)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)

        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        #damage. note it lives on faces
        damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew))/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)))
        damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns))/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)))
        #damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2)/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2))
        #damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2)/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2))
        #damage_ew = damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)
        #damage_ns = damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)
        damage_ew = jnp.minimum(damage_ew, 0.75)
        damage_ns = jnp.minimum(damage_ns, 0.75)

#        print(damage_ew)
#        raise

        #mu_ns = mu_ns.at[:, -2].set(0) #screws everything up for some reason!
        #mu_ns = mu_ns*0

        visc_x = 2 * (1-damage_ew[:, 1:]) * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * (1-damage_ew[:,:-1]) * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * (1-damage_ns[:-1,:]) * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * (1-damage_ns[1:, :]) * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx


        visc_y = 2 * (1-damage_ew[:, 1:]) * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * (1-damage_ew[:,:-1]) * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * (1-damage_ns[:-1,:]) * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * (1-damage_ns[1:, :]) * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        ##removing the thickness makes speeds look better!
        #visc_x = mu_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
        #         mu_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
        #         mu_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
        #         mu_ns[1:, :]*(dudy_ns[1:,:] + dvdx_ns[1:,:])*0.5*dx


        #visc_y = mu_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
        #         mu_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
        #         mu_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
        #         mu_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx


        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals), compute_damage


def compute_u_v_residuals_function_with_damage(ny, nx, dy, dx,
                                   h_1d,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_rflc_ghost_cells,
                                   add_cont_ghost_cells,
                                   extrp_over_cf):

    #damcoef = 5e-6
    #damcoef = 1.3e3

    def compute_beta(u, v, sliding_coef):
        return sliding_coef*(jnp.sqrt(u**2+v**2+1e-15)**(-2/3))

    def compute_damage(u, v, mucoef, h, damcoef):
        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #interpolate things onto face-cenres
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        #damage. note it lives on faces
        damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew))/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)))
        damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns))/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)))
        #damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2)/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2))
        #damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2)/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2))
        #damage_ew = damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)
        #damage_ns = damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)
        damage_ew = jnp.minimum(damage_ew, 0.75)
        damage_ns = jnp.minimum(damage_ns, 0.75)

        return 0.25*(damage_ns[1:,:]+damage_ns[:-1,:]+damage_ew[:,1:]+damage_ew[:,:-1])
        

    def compute_u_v_residuals(u_1d, v_1d, mucoef, sliding_coef, damcoef):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        dsdx, dsdy = cc_gradient(s)
        beta = compute_beta(u, v, sliding_coef)


        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #TODO: set bespoke conditions at the front for dvel_dx! otherwise
        #over-estimating the viscosity on the faces near the front!
        #u, v = extrapolate_over_cf(u, v)


        #momentum_term
        #quickly extrapolate velocity over calving front
        #NOTE: I'm not sure it's even really necessary to do this you know...
        #in the y-aligned calving front it only affects the ddx_ns derivatives.
        #u = extrp_over_cf(u)
        #v = extrp_over_cf(v)
        #and add the ghost cells in
        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        ##cc_derivatives, e.g. for viscosity calculation
        #dudx_cc, dudy_cc = cc_gradient(u)
        #dvdx_cc, dvdy_cc = cc_gradient(v)

        #interpolate things onto face-cenres
        h_ew, h_ns = interp_cc_to_fc(h)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)

        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        #damage. note it lives on faces
        damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew))/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)))
        damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns))/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)))
        #damage_ew = (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2)/(1+ (damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)**2))
        #damage_ns = (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2)/(1+ (damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)**2))
        #damage_ew = damcoef*mu_ew*0.5*jnp.abs(dudy_ew + dvdx_ew)
        #damage_ns = damcoef*mu_ns*0.5*jnp.abs(dudy_ns + dvdx_ns)
        damage_ew = jnp.minimum(damage_ew, 0.75)
        damage_ns = jnp.minimum(damage_ns, 0.75)

#        print(damage_ew)
#        raise

        #mu_ns = mu_ns.at[:, -2].set(0) #screws everything up for some reason!
        #mu_ns = mu_ns*0

        visc_x = 2 * (1-damage_ew[:, 1:]) * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * (1-damage_ew[:,:-1]) * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * (1-damage_ns[:-1,:]) * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * (1-damage_ns[1:, :]) * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx


        visc_y = 2 * (1-damage_ew[:, 1:]) * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * (1-damage_ew[:,:-1]) * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * (1-damage_ns[:-1,:]) * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * (1-damage_ns[1:, :]) * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        ##removing the thickness makes speeds look better!
        #visc_x = mu_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
        #         mu_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
        #         mu_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
        #         mu_ns[1:, :]*(dudy_ns[1:,:] + dvdx_ns[1:,:])*0.5*dx


        #visc_y = mu_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
        #         mu_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
        #         mu_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
        #         mu_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx


        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals), compute_damage



def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   h_1d,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_rflc_ghost_cells,\
                                   add_cont_ghost_cells,\
                                   extrp_over_cf):

    def compute_beta(u, v, sliding_coef):
        return sliding_coef*(jnp.sqrt(u**2+v**2+1e-15)**(-2/3))

    def compute_u_v_residuals(u_1d, v_1d, mucoef, sliding_coef):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        dsdx, dsdy = cc_gradient(s)
        beta = compute_beta(u, v, sliding_coef)


        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #TODO: set bespoke conditions at the front for dvel_dx! otherwise
        #over-estimating the viscosity on the faces near the front!
        #u, v = extrapolate_over_cf(u, v)


        #momentum_term
        #quickly extrapolate velocity over calving front
        #NOTE: I'm not sure it's even really necessary to do this you know...
        #in the y-aligned calving front it only affects the ddx_ns derivatives.
        #u = extrp_over_cf(u)
        #v = extrp_over_cf(v)
        #and add the ghost cells in
        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        ##cc_derivatives, e.g. for viscosity calculation
        #dudx_cc, dudy_cc = cc_gradient(u)
        #dvdx_cc, dvdy_cc = cc_gradient(v)

        #interpolate things onto face-cenres
        h_ew, h_ns = interp_cc_to_fc(h)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)

        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))


        #mu_ns = mu_ns.at[:, -2].set(0) #screws everything up for some reason!
        #mu_ns = mu_ns*0

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx


        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        ##removing the thickness makes speeds look better!
        #visc_x = mu_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
        #         mu_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
        #         mu_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
        #         mu_ns[1:, :]*(dudy_ns[1:,:] + dvdx_ns[1:,:])*0.5*dx


        #visc_y = mu_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
        #         mu_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
        #         mu_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
        #         mu_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx



        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals)



def print_residual_things(residual, rhs, init_residual, i):
    old_residual = residual
    residual = jnp.max(jnp.abs(-rhs))

    if i==0:
        init_residual = residual.copy()
        print("Initial residual: {}".format(residual))
    else:
        print("residual: {}".format(residual))
        print("Residual reduction factor: {}".format(old_residual/residual))
    print("------")

    return old_residual, residual, init_residual



def make_newton_velocity_solver_function_custom_vjp(ny, nx, dy, dx,\
                                                    h, n_iterations):

    h_1d = h.reshape(-1)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)
    extrapolate_over_cf                        = extrapolate_over_cf_function(h)

    get_u_v_residuals, compute_damage = compute_u_v_residuals_function_with_damage(ny, nx, dy, dx,\
                                                       h_1d,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_rflc_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf)
    #get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
    #                                                   h_1d,\
    #                                                   interp_cc_to_fc,\
    #                                                   ew_gradient, ns_gradient,\
    #                                                   cc_gradient,\
    #                                                   add_rflc_ghost_cells,\
    #                                                   add_cont_ghost_cells,\
    #                                                   extrapolate_over_cf)


    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))
    mask = (i_coordinate_sets>=0)


    sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
                                                        basis_vectors,\
                                                        i_coordinate_sets,\
                                                        j_coordinate_sets,\
                                                        mask,\
                                                        2,
                                                        active_indices=(0,1)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    #############

    coords = jnp.stack([
                    jnp.concatenate(
                                [i_coordinate_sets,         i_coordinate_sets,\
                                 i_coordinate_sets+(ny*nx), i_coordinate_sets+(ny*nx)]
                                   ),\
                    jnp.concatenate(
                                [j_coordinate_sets, j_coordinate_sets+(ny*nx),\
                                 j_coordinate_sets, j_coordinate_sets+(ny*nx)]
                                   )
                       ])

   


    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
                                                              ksp_type="bcgs",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False)



    def _solver(mucoef, u_trial, v_trial, sliding_coef, damcoef):
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                           (u_1d, v_1d, mucoef, sliding_coef, damcoef)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, mucoef, sliding_coef, damcoef))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, mucoef, sliding_coef, damcoef)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    
    @custom_vjp
    def solver(mucoef, u_trial, v_trial, sliding_coef, damcoef):
        # primal call uses the plain solver
        return _newton_solve(mucoef, u_trial, v_trial, sliding_coef, damcoef)




    def solver_fwd(mucoef, u_trial, v_trial, sliding_coef, damcoef):
        u, v = _solver(mucoef, u_trial, v_trial, sliding_coef, damcoef)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1),\
                                                        mucoef, sliding_coef, damcoef)
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])


        fwd_residuals = (u, v, dJ_dvel_nz_values, mucoef, sliding_coef, damcoef)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, mucoef, sliding_coef, damcoef = res
        u_bar, v_bar = cotangent

        lambda_ = la_solver(dJ_dvel_nz_values,
                            -jnp.concatenate([u_bar, v_bar]),
                            transpose=True)

        lambda_u = lambda_[:(ny*nx)]
        lambda_v = lambda_[(ny*nx):]


        _, pullback_function = jax.vjp(get_u_v_residuals,
                                         u.reshape(-1), v.reshape(-1),
                                         mucoef, sliding_coef, damcoef
                                      )

        #_, _, mu_bar, _, damcoef_bar = pullback_function((lambda_u, lambda_v))
        _, _, _, _, damcoef_bar = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        #return (mu_bar.reshape((ny, nx)), None, None, None, damcoef_bar)
        return (None, None, None, None, damcoef_bar)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver, compute_damage



def make_misfit_function(u_obs, v_obs, reg_param, solver, misfit_only=False):

    def misfit_function(mucoef_internal, u_trial, v_trial):
        u_mod, v_mod = solver(mucoef_internal, u_trial, v_trial)

        misfit = jnp.sum((u_obs-u_mod)**2 + (v_obs-v_mod)**2)
        regularisation = reg_param * jnp.sum(mucoef_internal**2)
        # print(regularisation)

        #In order for grad to work, the function has to return a scalar
        #value as the first argument and then other things packaged together as the second.
        #Then you have to specify has_aux=True so that it knows there's stuff in the
        #second argument and to leave it alone.
        return misfit + regularisation, (u_mod, v_mod)

    return misfit_function


def make_misfit_function_speed(u_obs, u_c, reg_param, solver, misfit_only=False):

    def misfit_function(mucoef_internal, u_trial, v_trial):
        u_mod, v_mod = solver(mucoef_internal, u_trial, v_trial)


        #plt.imshow(np.array(v_mod.copy()))
        #plt.show()
    
        #plt.imshow(uc)
        #plt.show()

        misfit = jnp.sum(uc * (u_obs - jnp.sqrt(u_mod**2 + v_mod**2 + 1e-12))**2)

        regularisation = reg_param * jnp.sum((1-mucoef_internal)**2)
        # print(regularisation)

        #In order for grad to work, the function has to return a scalar
        #value as the first argument and then other things packaged together as the second.
        #Then you have to specify has_aux=True so that it knows there's stuff in the
        #second argument and to leave it alone.
        return misfit + regularisation, (u_mod, v_mod)

    return misfit_function



def plotcontrol(field):
    plt.figure(figsize=(5,5))
    plt.imshow(field, vmin=0, vmax=1, cmap="cubehelix")
    plt.colorbar()
    plt.show()


def gradient_descent_function(misfit_function, iterations=400, step_size=1e7):
    def gradient_descent(initial_guess, u_init_initial, v_init_initial):
        #instead of grad, value_and_grad returns value too
        #so we can keep track
        get_grad = jax.value_and_grad(misfit_function, has_aux=True)
        #Note the has_aux as per the above comment.

        ctrl_i = initial_guess
        u_i = u_init_initial
        v_i = v_init_initial
        ctrls = [ctrl_i]
        for i in range(iterations):
            print(i)
            #note that grad by default takes gradient wrt first arg
            (misfit, (u_i, v_i)), grad = get_grad(ctrl_i, u_i, v_i) 
            print(misfit)

            #print(grad)

            grad = jnp.clip(grad, a_min=-1e-4, a_max=1e-4)
            
            print(jnp.min(grad))
            print(jnp.max(grad))

            #plt.imshow(grad, cmap="hsv")
            #plt.show()
            #raise

            #print(jnp.min(grad))
            #print(jnp.min(grad)==0)
            #raise
            ctrl_i = ctrl_i.at[:,:].set(ctrl_i - step_size*grad)
            ctrl_i = jnp.clip(ctrl_i, a_min=0.1, a_max=2)

            ctrls.append(ctrl_i[::-1, :])

#        print("making gif")
#        make_gif(ctrls, filename="../../../bits_of_data/ice_shelf_ip/larsen_c_ip.gif",
#                 cmap="cubehelix", vmin=0, vmax=1)

        return ctrl_i, u_i, v_i
    return gradient_descent




def make_misfit_function_speed_basic(u_obs, u_c, reg_param, solver, u_trial, v_trial):

    def misfit_function(mucoef_internal):

        u_mod, v_mod = solver(mucoef_internal.reshape(u_trial.shape), u_trial, v_trial)

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



def make_misfit_function_speed_for_damcoef(u_obs, u_c, reg_param, solver, mucoef, sliding_coef):

    def misfit_function(damcoef_internal, u_trial, v_trial):

        #print(damcoef_internal)

        u_mod, v_mod = solver(mucoef, u_trial, v_trial, sliding_coef, damcoef_internal)

        u_mod = u_mod
        v_mod = v_mod

        misfit = jnp.sum(u_c * (u_obs - jnp.sqrt(u_mod**2 + v_mod**2 + 1e-12))**2)/(u_mod.size)

        #regularisation = reg_param * jnp.sum((1-mucoef_internal)**2)/(u_mod.size)
        regularisation = reg_param * jnp.square(damcoef_internal)

        return misfit + regularisation, (u_mod, v_mod)

    return misfit_function



def gradient_descent_function_for_damcoef(misfit_function, iterations=400, step_size=1e-12):
    def gradient_descent(initial_guess, u_init_initial, v_init_initial):
        #instead of grad, value_and_grad returns value too
        #so we can keep track
        get_grad = jax.value_and_grad(misfit_function, has_aux=True)
        #Note the has_aux as per the above comment.

        ctrl_i = initial_guess
        u_i = u_init_initial
        v_i = v_init_initial
        ctrls = [ctrl_i]
        for i in range(iterations):
            #note that grad by default takes gradient wrt first arg
            #(misfit, (u_i, v_i)), grad = get_grad(ctrl_i, u_i, v_i)
            (misfit, (u_i, v_i)), grad = get_grad(ctrl_i, jnp.zeros_like(u_i), jnp.zeros_like(v_i))
            print(ctrl_i)
            print(misfit)
            print(grad)

            
            jnp.save("../../../bits_of_data/damage_coefficient_ip/test_1/ip/u_{}.npy".format(i), u_i)
            jnp.save("../../../bits_of_data/damage_coefficient_ip/test_1/ip/v_{}.npy".format(i), v_i)

            #grad = jnp.clip(grad, a_min=-1e-4, a_max=1e-4) 
            #print(jnp.min(grad))
            #print(jnp.max(grad))


            #print(jnp.min(grad))
            #print(jnp.min(grad)==0)
            #raise
            ctrl_i = ctrl_i - step_size*grad
            ctrl_i = jnp.clip(ctrl_i, a_min=0, a_max=1e-5)

            #print(ctrls)

            ctrls.append(ctrl_i)

        print(ctrls)

        return ctrl_i, u_i, v_i
    return gradient_descent





Lx = 100_000
Ly = 100_000

resolution = 1000

x = jnp.arange(0, Lx, resolution)
y = jnp.arange(0, Ly, resolution)

nx = int(Lx/resolution)
ny = int(Ly/resolution)

b = jnp.zeros((ny,nx))-1000

#thk_profile = 4000 - (3500*x/Lx)
#NOTE: this is totally fucked. The above expression doesn't work,
#but this one does:
thk_profile = 1000*(1-(500/1000)*x/Lx)
thk = jnp.zeros((ny, nx))+thk_profile[None,:]
thk = thk.at[:, -1].set(0)

A = 6.338e-25
#B = 0.5 * (A**(-1/c.GLEN_N))
B = A**(-1/c.GLEN_N)
m = 3

C = jnp.zeros_like(thk)+3.16e6
C = C.at[15:-15, 1:].set(0)
C = C.at[:, -20:].set(0)



mucoef = jnp.ones_like(thk)


def run_fwd_prob(damage_coefficient, save=False):

    mucoef = jnp.ones_like(thk)
    
    u_init = jnp.zeros_like(thk)
    v_init = jnp.zeros_like(thk)
    n_iterations = 15
    solver, compute_damage = make_newton_velocity_solver_function_custom_vjp(ny, nx, resolution, resolution, thk,\
                                                             n_iterations)
    
    u_out, v_out = solver(mucoef, u_init, v_init, C, damage_coefficient)
    
    
    #show_vel_field(u_out*c.S_PER_YEAR, v_out*c.S_PER_YEAR, vmin=0, vmax=100_000, showcbar=False)
    
    d = compute_damage(u_out, v_out, mucoef, thk, damage_coefficient)
   
    plt.figure(figsize=(5,5))
    plt.imshow(d, vmin=0, vmax=0.75, cmap="cubehelix_r")
    plt.colorbar()
    plt.show()
    
    show_vel_field(u_out*c.S_PER_YEAR, v_out*c.S_PER_YEAR, vmin=0, vmax=1000)
    
    if save:
        jnp.save("../../../bits_of_data/damage_coefficient_ip/test_1/u.npy", u_out)
        jnp.save("../../../bits_of_data/damage_coefficient_ip/test_1/v.npy", v_out)
        jnp.save("../../../bits_of_data/damage_coefficient_ip/test_1/damage.npy", d)
    


def run_inv_prob():
    uo = jnp.load("../../../bits_of_data/damage_coefficient_ip/test_1/u.npy")
    vo = jnp.load("../../../bits_of_data/damage_coefficient_ip/test_1/v.npy")
    
    speed_o = jnp.sqrt(uo**2 + vo**2)
    uc = jnp.ones_like(uo)

    reg_param = 0

    
    u_init = jnp.zeros_like(thk)
    v_init = jnp.zeros_like(thk)
    n_iterations = 8
    solver, compute_damage = make_newton_velocity_solver_function_custom_vjp(ny, nx,
                                                            resolution, resolution, thk,\
                                                            n_iterations)

    misfit_function = make_misfit_function_speed_for_damcoef(speed_o, uc, reg_param, solver, mucoef, C)

    #gd_function = gradient_descent_function_for_damcoef(misfit_function, iterations=2, step_size=1e-5)
    gd_function = gradient_descent_function_for_damcoef(misfit_function, iterations=60, step_size=2e-2)

    #damcoef_initial_guess = 4e-6
    damcoef_initial_guess = 1e-10
#    damcoef_initial_guess = 2.2017598e-06
    u_ig = jnp.zeros_like(thk)
    v_ig = jnp.zeros_like(thk)

    damcoef_out, u_out, v_out = gd_function(damcoef_initial_guess, u_ig, v_ig)

    print(damcoef_out)

    pass

dcs = [1e-10, 1.2421302e-07, 2.4784572e-07, 3.7091746e-07, 4.9336654e-07, 6.1514845e-07, 7.3609976e-07, 8.5629654e-07, 9.755597e-07, 1.0937932e-06, 1.2109094e-06, 1.3268461e-06, 1.4414945e-06, 1.5547799e-06, 1.6666309e-06, 1.7769751e-06, 1.8857198e-06, 1.992773e-06, 2.0980658e-06, 2.2017598e-06, 2.3033065e-06, 2.40276e-06, 2.5009285e-06, 2.5961624e-06, 2.6888058e-06, 2.7799701e-06, 2.868898e-06, 2.9555474e-06, 3.0399028e-06, 3.1219236e-06, 3.201623e-06, 3.2789446e-06, 3.3538872e-06, 3.426499e-06, 3.4967538e-06, 3.5646635e-06, 3.630188e-06, 3.693454e-06, 3.7545105e-06, 3.8131454e-06, 3.869722e-06, 3.9238535e-06, 3.976093e-06, 4.026014e-06, 4.074076e-06, 4.1199755e-06, 4.1638436e-06, 4.205777e-06, 4.245922e-06, 4.284201e-06, 4.3207733e-06, 4.355593e-06, 4.3888103e-06, 4.4204767e-06, 4.4505687e-06, 4.479258e-06, 4.506361e-06, 4.53225e-06, 4.556801e-06, 4.579987e-06, 4.6021255e-06]

def make_a_plot_or_two():
    speedpaths = []
    damagepaths = []
    
    _, compute_damage = make_newton_velocity_solver_function_custom_vjp(ny, nx,
                                                            resolution, resolution, thk,\
                                                            0)

    for i in range(60):
    #    u = jnp.load("../../../bits_of_data/damage_coefficient_ip/test_1/ip/u_{}.npy".format(i))
    #    v = jnp.load("../../../bits_of_data/damage_coefficient_ip/test_1/ip/v_{}.npy".format(i))
    #    damcoef = dcs[i]
    #    d = compute_damage(u, v, mucoef, thk, damcoef)

    #    show_vel_field(u*c.S_PER_YEAR, v*c.S_PER_YEAR, vmin=0, vmax=800,
    #       savepath= "../../../bits_of_data/damage_coefficient_ip/test_1/ip/speed{}.png".format(i),
    #       show=False)

        speedpaths.append("../../../bits_of_data/damage_coefficient_ip/test_1/ip/speed{}.png".format(i))

    #    show_damage_field(d, 
    #       savepath= "../../../bits_of_data/damage_coefficient_ip/test_1/ip/damage{}.png".format(i),
    #       show=False)

        damagepaths.append("../../../bits_of_data/damage_coefficient_ip/test_1/ip/damage{}.png".format(i))

    create_high_quality_gif_from_pngfps(speedpaths, "../../../bits_of_data/damage_coefficient_ip/test_1/ip/speed_gif.gif")
    create_high_quality_gif_from_pngfps(damagepaths, "../../../bits_of_data/damage_coefficient_ip/test_1/ip/damage_gif.gif")
    #create_gif_from_png_fps(speedpaths, "../../../bits_of_data/damage_coefficient_ip/test_1/ip/speed_gif.gif")
    #create_gif_from_png_fps(damagepaths, "../../../bits_of_data/damage_coefficient_ip/test_1/ip/damage_gif.gif")




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatter, AutoMinorLocator

def plot_pretty_series(values, title="Series", y_label="Value", x_label="Index",
                       save_to=None, logy=True, dpi=240):
    v = np.asarray(values, dtype=float)
    x = np.arange(v.size)

    with plt.rc_context({
        "figure.figsize": (7.2, 4.4),
        "axes.facecolor": "#fcfcff",
        "axes.edgecolor": "#2f2f2f",
        "axes.grid": True,
        "grid.color": "#d0d7de",
        "grid.linestyle": "-",
        "grid.alpha": 0.6,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "font.size": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(dpi=dpi, constrained_layout=True)

        mark_every = max(1, v.size // 60)
        ax.plot(x, v, color="#0b84a5", lw=2.2, marker="o", markersize=4,
                markevery=mark_every, alpha=0.95)

        if logy:
            ax.set_yscale("log", nonpositive="clip")
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=9))
            ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=90))
            ax.yaxis.set_major_formatter(LogFormatter())
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # Highlight min/max with annotations
        imin = int(np.nanargmin(v))
        imax = int(np.nanargmax(v))
        ax.scatter([x[imin], x[imax]], [v[imin], v[imax]], s=60, zorder=3,
                   color="#f66d44", edgecolor="white", linewidth=0.8)
        ax.annotate(f"min = {v[imin]:.2e}", (x[imin], v[imin]),
                    xytext=(-10, 14), textcoords="offset points",
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#f66d44", lw=0.8),
                    arrowprops=dict(arrowstyle="->", color="#f66d44", lw=0.8))
        ax.annotate(f"max = {v[imax]:.2e}", (x[imax], v[imax]),
                    xytext=(10, -14), textcoords="offset points",
                    ha="left", va="top", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#f66d44", lw=0.8),
                    arrowprops=dict(arrowstyle="->", color="#f66d44", lw=0.8))

        if save_to:
            fig.savefig(save_to, dpi=dpi)

        return fig, ax




#run_fwd_prob(5e-6)
#run_inv_prob()
#make_a_plot_or_two()

plot_pretty_series(dcs, save_to="../../../bits_of_data/damage_coefficient_ip/test_1/ip/dcs.png", dpi=200)

raise
