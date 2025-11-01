#1st party
from pathlib import Path
import sys
import time
from functools import partial

##local apps
sys.path.insert(1, "../../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants_years as c
#import constants as c
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,\
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette



#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp, custom_jvp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy
from scipy.optimize import minimize as scinimize

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)

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


    @partial(jax.custom_vjp, nondiff_argnums=(2,))
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

    
    def la_solver_fwd(values, b, transpose=False):
        solution = petsc_sparse_la_solver(values, b, transpose=transpose)
        return solution, (values, b, solution)


    #NOTE: the nondiff_argnums=(2,) thing shunts the transpose ragument to the front.
    def linear_solve_bwd(transpose, res, x_bar):
        #NOTE: The sign convention here is correct, despite what people say...
        #It just follows the documentation rather than "textbook" versions.
        values, b, x = res

        lambda_ = petsc_sparse_la_solver(values, -x_bar, transpose=True)

        b_bar = -lambda_.reshape(b.shape) #ensure same shape as input b.

        #sparse version of jnp.outer(x,lambda_)
        #TODO: CHECK WHICH WAY ROUND THESE COORDS GO. SHOULD BE RIGHT IF THEY ARE IJ!
        values_bar = x[coordinates[1]] * lambda_[coordinates[0]]

        return values_bar, b_bar

    petsc_sparse_la_solver.defvjp(la_solver_fwd, linear_solve_bwd)

    return petsc_sparse_la_solver

def create_sparse_petsc_la_solver_test(jac_shape, ksp_type='gmres', preconditioner='hypre',\
                                    precondition_only=False, monitor_ksp=False):

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()
    
    def construct_ab(values, b, coordinates, transpose):
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
        opts['ksp_max_it'] = 50
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


    @partial(jax.custom_vjp, nondiff_argnums=(2,3))
    def petsc_sparse_la_solver(values, b, coordinates, transpose=False):
    
        A, b = construct_ab(values, b, coordinates, transpose)
        x = b.duplicate()
        
        ksp, pc = create_solver_object(A)

        if precondition_only:
            pc.apply(b, x)
        else:
            ksp.solve(b, x)
        
        x_jnp = jnp.array(x.getArray())

        return x_jnp

    
    def la_solver_fwd(values, b, coordinates, transpose=False):
        solution = petsc_sparse_la_solver(values, b, coordinates, transpose=transpose)
        return solution, (values, b, solution)


    #NOTE: the nondiff_argnums=(2,) thing shunts the transpose ragument to the front.
    def linear_solve_bwd(transpose, coordinates, res, x_bar):
        #NOTE: The sign convention here is correct, despite what people say...
        #It just follows the documentation rather than "textbook" versions.
        values, b, x = res

        lambda_ = petsc_sparse_la_solver(values, -x_bar, coordinates, transpose=True)

        b_bar = -lambda_.reshape(b.shape) #ensure same shape as input b.

        #sparse version of jnp.outer(x,lambda_)
        #TODO: CHECK WHICH WAY ROUND THESE COORDS GO. SHOULD BE RIGHT IF THEY ARE IJ!
        values_bar = x[coordinates[1]] * lambda_[coordinates[0]]

        return values_bar, b_bar

    petsc_sparse_la_solver.defvjp(la_solver_fwd, linear_solve_bwd)

    return petsc_sparse_la_solver

def interp_cc_with_ghosts_to_fc_function(ny, nx):
    def interp_cc_to_fc(var):
        
        var_ew = 0.5*(var[1:-1, 1:]+var[1:-1, :-1])
        var_ns = 0.5*(var[:-1, 1:-1]+var[1:, 1:-1])

        return var_ew, var_ns
    return jax.jit(interp_cc_to_fc)


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

    return jax.jit(interp_cc_to_fc)


def cc_gradient_function(dy, dx):

    def cc_gradient(var):

        dvar_dx = (0.5/dx) * (var[1:-1, 2:] - var[1:-1,:-2])
        dvar_dy = (0.5/dy) * (var[:-2,1:-1] - var[2:, 1:-1])

        return dvar_dx, dvar_dy

    return jax.jit(cc_gradient)

def fc_gradient_functions(dy, dx):

    def ew_face_gradient(var):
        
        dvar_dx_ew = (var[1:-1, 1:] - var[1:-1, :-1])/dx

        dvar_dy_ew = (var[:-2, 1:] + var[:-2, :-1] - var[2:, 1:] - var[2:, :-1])/(4*dy)
        
        return dvar_dx_ew, dvar_dy_ew
    
    def ns_face_gradient(var):
        
        dvar_dy_ns = (var[:-1, 1:-1]-var[1:, 1:-1])/dy

        dvar_dx_ns = (var[:-1, 2:] + var[1:, 2:] - var[:-1, :-2] - var[1:, :-2])/(4*dx)
        
        return dvar_dx_ns, dvar_dy_ns
    
    return jax.jit(ew_face_gradient), jax.jit(ns_face_gradient)


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
    
    return jax.jit(add_reflection_ghost_cells), jax.jit(add_continuation_ghost_cells)
    
def add_ghost_cells_periodic_dirichlet_function(ny, nx):
    def add_periodic_ghost_cells_x_dirchlet_ghost_cells_y(u_int):

        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #left/right edges: periodic bcs
        u = u.at[:, 0].set(u[:,-2])
        u = u.at[:,-1].set(u[:, 1])
        #top/bottom edges: dirchlet bcs
        u = u.at[0, :].set(-u[1, :])
        u = u.at[-1,:].set(-u[-2,:])

        return u
    return jax.jit(add_periodic_ghost_cells_x_dirchlet_ghost_cells_y)

def binary_erosion(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    #kernel = jnp.array([[1,1,1],
    #                    [1,1,1],
    #                    [1,1,1]], dtype=jnp.bool_)
    kernel = jnp.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float64)

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

    return erode_once(boolean_array.astype(jnp.float64))

def binary_dilation(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    kernel = jnp.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float64)

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

    return dilate_once(boolean_array.astype(jnp.float64))


#NOTE: this is not actually used...

def extrapolate_over_cf_function(thk):
    def extrapolate_over_cf(cc_field):
        return cc_field
    return jax.jit(extrapolate_over_cf)

def extrapolate_over_cf_function_nonuniform_thk(thk):
    
    cf_adjacent_zero_ice_cells = (thk==0) & binary_dilation(thk>0)

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

        return cc_field + u_extrap_boundary*cf_adjacent_zero_ice_cells.astype(jnp.float64)

    return extrapolate_over_cf

@jax.jit
def double_dot_contraction(A, B):
    return A[:,:,0,0]*B[:,:,0,0] + A[:,:,1,0]*B[:,:,0,1] +\
           A[:,:,0,1]*B[:,:,1,0] + A[:,:,1,1]*B[:,:,1,1]


def cc_vector_field_gradient_function(ny, nx, dy, dx, cc_grad,
                                      extrp_over_cf,
                                      add_rb_ghost_cells):
    def cc_vector_field_gradient(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u = extrp_over_cf(u)
        v = extrp_over_cf(v)

        u, v = add_rb_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)

        #dudx = jnp.where(thk>0, dudx, 0)
        #dvdx = jnp.where(thk>0, dvdx, 0)
        #dudy = jnp.where(thk>0, dudy, 0)
        #dvdy = jnp.where(thk>0, dvdy, 0)

        grad_vf = jnp.zeros((ny, nx, 2, 2))

        grad_vf = grad_vf.at[:,:,0,0].set(dudx)
        grad_vf = grad_vf.at[:,:,0,1].set(dudy)
        grad_vf = grad_vf.at[:,:,1,0].set(dvdx)
        grad_vf = grad_vf.at[:,:,1,1].set(dvdy)

        return grad_vf

    return jax.jit(cc_vector_field_gradient)


def membrane_strain_rate_function(ny, nx, dy, dx, 
                                  cc_grad,
                                  extrapolate_over_cf,
                                  add_rb_ghost_cells):

    def membrane_sr_tensor(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u = extrapolate_over_cf(u)
        v = extrapolate_over_cf(v)

        u, v = add_rb_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)
        
        #dudx = jnp.where(thk>0, dudx, 0)
        #dvdx = jnp.where(thk>0, dvdx, 0)
        #dudy = jnp.where(thk>0, dudy, 0)
        #dvdy = jnp.where(thk>0, dvdy, 0)

        msr_tensor = jnp.zeros((ny, nx, 2, 2))


        #TODO: CHECK THAT FACTOR OF 4 AND 2! ARE THEY 2 AND 1 REALLY?
        #I think this is right.
        msr_tensor = msr_tensor.at[:,:,0,0].set(4*dudx + 2*dvdy)
        msr_tensor = msr_tensor.at[:,:,0,1].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,0].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,1].set(4*dvdy + 2*dudx)

        return msr_tensor

    return jax.jit(membrane_sr_tensor)


def divergence_of_tensor_field_function(ny, nx, dy, dx):
    def div_tensor_field(tf):
        #these have to be 2d scalar fields, of course
        tf_xx = tf[...,0,0]
        tf_xy = tf[...,0,1]
        tf_yx = tf[...,1,0]
        tf_yy = tf[...,1,1]

        shape_0, shape_1 = tf_xx.shape


        #NOTE: This is done assuming basically everything of interest and its
        #gradient is zero at the boundaries

        dx_tf_xx = jnp.zeros((shape_0, shape_1))
        dx_tf_xx = dx_tf_xx.at[:,1:-1].set((tf_xx[:,2:] - tf_xx[:,:-2])/(2*dx))
        
        dy_tf_yx = jnp.zeros((shape_0, shape_1))
        dy_tf_yx = dy_tf_yx.at[1:-1,:].set((tf_yx[:-2,:] - tf_yx[2:,:])/(2*dy))

        dx_tf_xy = jnp.zeros((shape_0, shape_1))
        dx_tf_xy = dx_tf_xy.at[:,1:-1].set((tf_xy[:,2:] - tf_xy[:,:-2])/(2*dx))
        
        dy_tf_yy = jnp.zeros((shape_0, shape_1))
        dy_tf_yy = dy_tf_yy.at[1:-1,:].set((tf_yy[:-2,:] - tf_yy[2:,:])/(2*dy))

        return dx_tf_xx+dy_tf_yx, dx_tf_xy+dy_tf_yy
    return jax.jit(div_tensor_field)


def compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx, \
                                          h_1d, beta,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_h_ghost_cells,\
                                          extrapolate_over_cf):

    def compute_linear_ssa_residuals(u_1d, v_1d, mu_ew, mu_ns, cc_rhs):

        cc_rhs_x = cc_rhs[:nx*ny].reshape((ny, nx))
        cc_rhs_y = cc_rhs[nx*ny:].reshape((ny, nx))

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        #volume_term
        volume_x = - ( beta * u + cc_rhs_x ) * dx * dy
        volume_y = - ( beta * v + cc_rhs_y ) * dy * dx

        #momentum_term
        u, v = add_uv_ghost_cells(u, v)

        #get thickness on the faces
        h = add_h_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        
        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_linear_ssa_residuals)

def compute_linear_ssa_residuals_function_fc_visc_no_rhs(ny, nx, dy, dx, \
                                          h_1d, beta,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_s_ghost_cells,\
                                          extrapolate_over_cf):

    def compute_linear_ssa_residuals(u_1d, v_1d, mu_ew, mu_ns):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        
        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)
        #jax.debug.print("s: {x}",x=s)

        dsdx, dsdy = cc_gradient(s)
        #jax.debug.print("dsdx: {x}",x=dsdx)
        #sneakily fudge this:
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])


        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx
        
        #momentum_term
        u = add_uv_ghost_cells(u)
        v = add_uv_ghost_cells(v)

        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        
        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_linear_ssa_residuals)

def compute_u_v_residuals_function_linear_mu(ny, nx, dy, dx, \
                                   h_1d, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf):
    
    def compute_u_v_residuals(u_1d, v_1d, mu_ew, mu_ns):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))


        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)
        #jax.debug.print("s: {x}",x=s)

        dsdx, dsdy = cc_gradient(s)
        #jax.debug.print("dsdx: {x}",x=dsdx)
        #sneakily fudge this:
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #obvs not going to do anything in the no-cf case
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        #momentum_term
        u = add_uv_ghost_cells(u)
        v = add_uv_ghost_cells(v)

        #interpolate things onto face-cenres
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        #remove those ghost cells again!
        h = h[1:-1,1:-1]
        #jax.debug.print("h_ew = {x}",x=h_ew)
        #jax.debug.print("h = {x}",x=h)

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))
        
        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx

        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals)

def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   h_1d, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf):
    
    def compute_u_v_residuals(u_1d, v_1d, q):

        mucoef = mucoef_0*jnp.exp(q)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))


        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)
        #jax.debug.print("s: {x}",x=s)

        dsdx, dsdy = cc_gradient(s)
        #jax.debug.print("dsdx: {x}",x=dsdx)
        #sneakily fudge this:
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #obvs not going to do anything in the no-cf case
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        #momentum_term
        u = add_uv_ghost_cells(u)
        v = add_uv_ghost_cells(v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #jax.debug.print("dudxew = {x}",x=dudx_ew)

        #interpolate things onto face-cenres
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        #remove those ghost cells again!
        h = h[1:-1,1:-1]
        #jax.debug.print("h_ew = {x}",x=h_ew)
        #jax.debug.print("h = {x}",x=h)

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #jax.debug.print("mucoef_ew = {x}",x=mucoef_ew)

        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))
        

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx

        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals)


def cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient):
    def cc_viscosity(q, u, v):

        vfg = cc_vector_field_gradient(u, v)
        
        mu = B * mucoef_0 * jnp.exp(q) * (vfg[:,:,0,0]**2 + vfg[:,:,1,1]**2 + vfg[:,:,0,0]*vfg[:,:,1,1] + \
                           0.25*(vfg[:,:,0,1] + vfg[:,:,1,0])**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

        #NOTE: the effective viscosity isn't actually set to zero here in the ice-free regions,
        #but wherever it's used, it should be multiplied by a zero thickness there...

        return mu
    return jax.jit(cc_viscosity)


def fc_viscosity_function(ny, nx, dy, dx, extrp_over_cf, add_uv_ghost_cells,
                          add_mucoef_ghost_cells,
                          interp_cc_to_fc, ew_gradient, ns_gradient, h_1d):
    def fc_viscosity(q, u, v):
        mucoef = mucoef_0*jnp.exp(q)
        mucoef = add_mucoef_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        #and add the ghost cells in
        u = add_uv_ghost_cells(u)
        v = add_uv_ghost_cells(v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)
        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        return mu_ew, mu_ns
    return jax.jit(fc_viscosity)


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
                                                    h, C,\
                                                    n_iterations):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)
    extrapolate_over_cf                        = extrapolate_over_cf_function(h)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_rflc_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf)


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



    @custom_vjp
    def solver(q, u_trial, v_trial):
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                           (u_1d, v_1d, q)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])


            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, q))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, q)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, u_trial, v_trial):
        u, v = solver(q, u_trial, v_trial)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1), q)
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])


        fwd_residuals = (u, v, dJ_dvel_nz_values, q)
#        fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q = res
        #u, v, q = res

        u_bar, v_bar = cotangent
        

        #dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
        #                                               (u.reshape(-1), v.reshape(-1), q)
        #                                              )
        #dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
        #                                     dJv_du[mask], dJv_dv[mask]])


        lambda_ = la_solver(dJ_dvel_nz_values,
                            -jnp.concatenate([u_bar, v_bar]),
                            transpose=True)

        #NOTE: Ok, so if you stop the gradients through lambda, it gets rid of the
        #massive gradients. But it also gets rid of the interesting information...
        #lambda_ = jax.lax.stop_gradient(lambda_)

        lambda_u = lambda_[:(ny*nx)]
        lambda_v = lambda_[(ny*nx):]


        #phi_bar = (dG/dphi)^T lambda
        _, pullback_function = jax.vjp(get_u_v_residuals,
                                         u.reshape(-1), v.reshape(-1), q
                                      )
        _, _, mu_bar = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        return (mu_bar.reshape((ny, nx)), None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver



def generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver):

    def solver(u_trial, v_trial, residuals_function, n_iterations, residuals_fct_args, coordinates):

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        
        residual = jnp.inf
        init_res = 0


        test_la_solver = create_sparse_petsc_la_solver_test((2*ny*nx, 2*ny*nx), monitor_ksp=True)


        for i in range(n_iterations):
            #dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(residuals_function,
            #                                               (u_1d, v_1d,
            #                                                *residuals_fct_args)
            #                                              )

            #nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
            #                                 dJv_du[mask], dJv_dv[mask]])




            rhs = -jnp.concatenate(residuals_function(u_1d, v_1d,
                                                      *residuals_fct_args))

            ##jax.debug.print("nz jac vals: {x}", x=nz_jac_values)
            #dense_jac = jnp.zeros((2*ny*nx, 2*ny*nx))
            #dense_jac = dense_jac.at[coordinates[0], coordinates[1]].set(nz_jac_values)
            ##jax.debug.print("{x}", x=(nz_jac_values.shape, coordinates.shape))
            ##raise

            #jax.debug.print("{x}", x=dense_jac)
            ##jax.debug.print("rhs: {x}", x=rhs)

            #dense_jac_2 = jax.jacrev(residuals_function, argnums=(0,1))(u_1d, v_1d, q)
            #jax.debug.print("{x}", x=dense_jac_2)
            #raise



            ####NOTE: DENSE JACOBIAN COMPUTATION TO ISOLATE BUG
            ((Juu, Jvu), (Juv, Jvv)) = jax.jacrev(residuals_function, argnums=(0,1))(u_1d, v_1d,
                                                                                     *residuals_fct_args)
            
            top = jnp.concatenate([Juu, Juv], axis=1)
            bot = jnp.concatenate([Jvu, Jvv], axis=1)
            J_full = jnp.concatenate([top, bot], axis=0)
            

            mask = J_full != 0.0
            rows, cols = jnp.where(mask)
            rows = rows.astype(jnp.int32)
            cols = cols.astype(jnp.int32)
            coords = jnp.stack([rows, cols], axis=0)
            vals = J_full[cols, rows]


            #########################


            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)

            #lin_res = J_full @ du - rhs
            #print("‖J du - rhs‖_inf =", np.linalg.norm(lin_res, ord=np.inf))


            #du = la_solver(nz_jac_values, rhs)
            du = test_la_solver(vals, rhs, coords)
            #du = jnp.linalg.solve(J_full, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    residuals_function(u_1d, v_1d, *residuals_fct_args)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver


def forward_adjoint_and_second_order_adjoint_solvers(ny, nx, dy, dx, h, C, n_iterations):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)
    
    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)
    add_uv_ghost_cells                         = add_ghost_cells_periodic_dirichlet_function(ny,nx)
    extrapolate_over_cf                        = extrapolate_over_cf_function(h)
    cc_vector_field_gradient                   = cc_vector_field_gradient_function(ny, nx, dy,
                                                                                   dx, cc_gradient, 
                                                                                   extrapolate_over_cf,
                                                                                   add_rflc_ghost_cells)
    membrane_strain_rate                       = membrane_strain_rate_function(ny, nx, dy, dx,
                                                                               cc_gradient,
                                                                               extrapolate_over_cf,
                                                                               add_rflc_ghost_cells)
    div_tensor_field                           = divergence_of_tensor_field_function(ny, nx, dy, dx)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient)
    fc_viscosity = fc_viscosity_function(ny, nx, dy, dx, extrapolate_over_cf, add_uv_ghost_cells,
                                         add_cont_ghost_cells,
                                         interp_cc_to_fc, ew_gradient, ns_gradient, h_1d)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf)
    
    #get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
    #                                                   h_1d, beta_eff,\
    #                                                   interp_cc_to_fc,\
    #                                                   ew_gradient, ns_gradient,\
    #                                                   cc_gradient,\
    #                                                   add_uv_ghost_cells,\
    #                                                   add_cont_ghost_cells,\
    #                                                   extrapolate_over_cf)

    linear_ssa_residuals = compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_rflc_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_no_rhs = compute_linear_ssa_residuals_function_fc_visc_no_rhs(
                                                       ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf
                                                                                      )

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
                                                              precondition_only=False,
                                                              monitor_ksp=False)


    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)

    #picard_solver = make_picard_solver(ny, nx, sparse_jacrev, mask, la_solver, get_u_v_residuals, fc_visc)

    def solve_fwd_problem(q, u_trial, v_trial):
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,), coords)
        return u.reshape((ny,nx)), v.reshape((ny,nx))


    def solve_fwd_problem_picard(q, u, v, n_pic_its=16):
        mu_bar_ew = jnp.zeros((ny,nx+1))+3e5
        mu_bar_ns = jnp.zeros((ny+1,nx))+3e5
        for i in range(n_pic_its):
            #calculate viscosity

            #plt.imshow(mu_bar_ns)
            #plt.colorbar()
            #plt.show()

            #solve adjoint problem
            u_new, v_new = newton_solver(u.reshape(-1), v.reshape(-1),
                                 linear_ssa_residuals_no_rhs, 1, (mu_bar_ew, mu_bar_ns), coords)

            #u = 0.9*u + 0.1*u_new
            #v = 0.9*v + 0.1*v_new
            u = u_new
            v = v_new
            
            mu_bar_ew, mu_bar_ns = fc_viscosity(q, u, v)

        return u.reshape((ny,nx)), v.reshape((ny,nx))


    def solve_adjoint_problem(q, u, v, lx_trial, ly_trial,
                              functional:callable, additional_fctl_args=None):
        #calculate viscosity
        mu_bar_ew, mu_bar_ns = fc_viscosity(q, u, v)

        #right-hand-side (\partial_u J)
        if additional_fctl_args is None:
            argz = (u.reshape(-1), v.reshape(-1), q,)
        else:
            argz = (q, *additional_fctl_args)

        dJdu, dJdv = jax.grad(functional, argnums=(0,1))(*argz)
        rhs = - jnp.concatenate([dJdu, dJdv])


        #solve adjoint problem
        lx, ly = newton_solver(lx_trial.reshape(-1), ly_trial.reshape(-1),
                               linear_ssa_residuals, 1, (mu_bar_ew, mu_bar_ns, rhs))

        mu_bar = cc_viscosity(q, u, v)
        #calculate gradient
        dJdq = jax.grad(functional, argnums=2)(*argz) \
               - mu_bar * h * double_dot_contraction(cc_vector_field_gradient(lx.reshape(-1), ly.reshape(-1)),
                                                           membrane_strain_rate(u.reshape(-1), v.reshape(-1))
                                                    )

        return lx.reshape((ny,nx)), ly.reshape((ny,nx)), dJdq


    def solve_soa_problem(q, u, v, lx, ly, perturbation_direction,
                          functional:callable, 
                          additional_fctl_args=None):
        #calculate viscosity
        mu_bar = cc_viscosity(q, u, v)
        
        #right-hand-side (\partial_u J)
        if additional_fctl_args is None:
            argz = (u.reshape(-1), v.reshape(-1), q)
        else:
            argz = (u.reshape(-1), v.reshape(-1), q, *additional_fctl_args)

        
        #solve first equation for mu
        rhs_x, rhs_y = div_tensor_field((h * mu_bar * perturbation_direction)[...,None,None] *\
                                        membrane_strain_rate(u.reshape(-1), v.reshape(-1)))
        rhs = - jnp.concatenate([rhs_x.reshape(-1), rhs_y.reshape(-1)])

        x_trial = jnp.zeros_like(mu_bar).reshape(-1)
        y_trial = jnp.zeros_like(x_trial)


        mu_x, mu_y = newton_solver(x_trial, y_trial, linear_ssa_residuals,
                                   1, (mu_bar, rhs))


        #solve second equation for beta
        #NOTE: make functional essentially a function just of u,v to avoid the
        #difficulties with what to do with q gradients
        functional_fixed_q = lambda u, v: functional(u, v, q)
        gradient_j = jax.grad(functional_fixed_q, argnums=(0,1))
        direct_hvp_x, direct_hvp_y = jax.jvp(gradient_j,
                                             (u.reshape(-1), v.reshape(-1)),
                                             (mu_x.reshape(-1), mu_y.reshape(-1)))[1]
        rhs_1_x, rhs_1_y = div_tensor_field((h * mu_bar * perturbation_direction)[...,None,None] *\
                                        membrane_strain_rate(lx.reshape(-1), ly.reshape(-1))
                                           )

        rhs = - jnp.concatenate([(rhs_1_x.reshape(-1) + direct_hvp_x),
                                 (rhs_1_y.reshape(-1) + direct_hvp_y)])

        beta_x, beta_y = newton_solver(x_trial, y_trial, linear_ssa_residuals,
                                       1, (mu_bar, rhs))


        #calculate hessian-vector-product
        functional_fixed_vel = lambda q: functional(u.reshape(-1), v.reshape(-1), q)
        direct_hvp_part = jax.jvp(jax.grad(functional_fixed_vel, argnums=0),
                                  (q,),
                                  (perturbation_direction,))[1]
        #NOTE: the first term in the brackets is zero if we're thinking about q rather than q
        hvp = direct_hvp_part - mu_bar * h * (\
              perturbation_direction * double_dot_contraction(
                                              cc_vector_field_gradient(lx.reshape(-1), ly.reshape(-1)),
                                              membrane_strain_rate(u.reshape(-1), v.reshape(-1))
                                                             ) +\
              double_dot_contraction(
                            cc_vector_field_gradient(lx.reshape(-1), ly.reshape(-1)),
                            membrane_strain_rate(mu_x.reshape(-1), mu_y.reshape(-1))
                                    ) +\
              double_dot_contraction(
                            cc_vector_field_gradient(beta_x.reshape(-1), beta_y.reshape(-1)),
                            membrane_strain_rate(u.reshape(-1), v.reshape(-1))
                                    )
                                              )

        return hvp

    #return solve_fwd_problem, solve_adjoint_problem, solve_soa_problem
    return solve_fwd_problem_picard, solve_adjoint_problem, solve_soa_problem






##NOTE: make everything linear by changing to 1
nvisc = c.GLEN_N
#nvisc = 1.001

A = c.A_COLD
B = 0.5 * (A**(-1/nvisc))


#lx = 128_000
#ly = 32_000

#lx = 16_384
#ly = 4_096
#resolution = 256 #m

lx = 150_000
ly = 150_000
resolution = 3_000 #m


nr = int(ly/resolution)
nc = int(lx/resolution)

lx = nc*resolution
ly = nr*resolution


#nr, nc = 64, 64
#nr, nc = 96*2, 64*2


x = jnp.linspace(0, lx, nc)
y = jnp.linspace(0, ly, nr)


delta_x = x[1]-x[0]
delta_y = y[1]-y[0]

thk = jnp.zeros((nr,nc)) + 1000

#want a slope of 0.5 degrees.
sine_0pt5 = 0.0087265

#b_profile = 1000 - 500*x/lx
b_profile = 1000 - sine_0pt5*x
b = jnp.zeros((nr, nc)) + b_profile

#C = jnp.zeros_like(b)#5#+5e3
#C = C.at[:5,:].set(1e16)
#C = C.at[-5:,:].set(1e16)
#C = C.at[:4, :].set(1e16)
#C = C.at[:, :4].set(1e16)
#C = C.at[-4:,:].set(1e16)
#C = jnp.where(thk==0, 1, C)


xs, ys = jnp.meshgrid(x,y)
R = 160_000
m = 0.25
C = 1e5 * (1 + 5e-5 + jnp.sin(2*jnp.pi*ys/R + m*jnp.sin(2*jnp.pi*xs/R)))
C = jnp.flipud(C)

#plt.imshow(C)
#plt.show()
#raise

#plt.imshow(jnp.log10(C))
#plt.show()

u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 15

mucoef = jnp.ones_like(C)
mucoef_0 = jnp.ones_like(C)
q = jnp.zeros_like(C)

mask = jnp.ones_like(b)

def functional(v_field_x, v_field_y, q):
    #NOTE: for things like this, you need to ensure the value isn't
    #zero where the argument is zero, because JAX can't differentiate
    #through the square-root otherwise. Silly JAX.
    return jnp.sum(mask.reshape(-1) * jnp.sqrt(v_field_x**2 + v_field_y**2 + 1e-10).reshape(-1))


def calculate_gradient_via_foa():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                                             nr, nc, delta_y, delta_x, thk, C, n_iterations
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    #show_vel_field(u_out, v_out)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    return gradient

def calculate_gradient_via_ad():

    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, C,
                                                             n_iterations)

    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    gradient = get_grad(q)

    return gradient

#g_foa = calculate_gradient_via_foa()
#g_ad = calculate_gradient_via_ad()
#
#plt.imshow((2*g_foa-g_ad)/g_foa, vmin=-3, vmax=3)
#plt.colorbar()
#plt.show()
#
#raise

def calculate_hvp_via_soa():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                                             nr, nc, delta_y, delta_x, thk, C, n_iterations
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    show_vel_field(u_out, v_out)
    raise
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    
    
    ##NOTE: Calving front stiffness dominating the gradient calculation
    ##Should investigate. For now, cutting out to visualise
    plt.imshow(gradient[:,:])
    plt.title("gradient via adjoint")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    plt.plot(gradient[25,:])
    plt.ylim((-600,600))
    plt.show()
    
    print("solving second-order adjoint problem:")
    pert_dir = gradient.copy()/(jnp.linalg.norm(gradient)*10)
    hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir, functional)
    
    plt.imshow(hvp[:,:], vmin=-50, vmax=50, cmap="twilight_shifted")
    plt.title("hvp via soa")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()
    
    plt.plot(hvp[25,:])
    plt.ylim((-10,10))
    plt.show()
    

def calculate_hvp_via_ad():

    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, C,
                                                             n_iterations)

#    u_out, v_out = solver(q, u_init, v_init)
#    plt.imshow(u_out)
#    plt.show()
    
    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    gradient = get_grad(q)
    
    plt.imshow(gradient[:,:])
    plt.title("gradient via ad")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    plt.plot(gradient[25,:])
    plt.ylim((-600,600))
    plt.show()

    
    pert_dir = gradient / (jnp.linalg.norm(gradient)*10)


    ##finite diff hvp for comparison
    ##plt.imshow(p)
    ##plt.colorbar()
    ##plt.show()
    ##raise
    eps = 0.01
    fd_hvp = (get_grad(q + eps*pert_dir) - get_grad(q)) / eps
    plt.imshow(fd_hvp[:,:], vmin=-50, vmax=50, cmap="twilight_shifted")
    plt.title("hvp via fd")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    plt.plot(fd_hvp[25,:])
    plt.ylim((-10,10))
    plt.show()
    ##raise

    #I'd have imagined it's ok to do the following:
    #      hvp = jax.vjp(get_grad, (q,), (pert_dir,))
    #However, JAX seemingly doesn't support reverse mode AD for functions
    #that have custom vjps defined within them. IDK why!!
    #But people seem to reverse the order of the dot product and derivative computation.
    #That's actually Lemma 2 in the original Adjoints document I wrote so
    #that, at least, seems at least to be true!

    #def hessian(q_prime):
    #    return jax.jacrev(get_grad)(q_prime)

    #hess = hessian(q)

    #plt.imshow(hess)
    #plt.show()
    #raise

    def hess_vec_product(q, perturbation):
        return jax.grad(lambda m: jnp.vdot(get_grad(m), perturbation))(q)


    #hvp = hess_vec_product(q, pert_dir)
    #hvp = jax.jvp(get_grad, (q,), (pert_dir,))[1]
    #NOTE: This contradicts the above comment, but I'm not sure I got that right first time.
    _, vjp_grad = jax.vjp(get_grad, q)
    (hvp,) = vjp_grad(pert_dir)


    #plt.imshow((hvp/jnp.linalg.norm(hvp))[:,:])
    #plt.colorbar()
    #plt.show()


    #plt.imshow((gradient/jnp.linalg.norm(gradient))[:,:])
    #plt.colorbar()
    #plt.show()


    #plt.imshow((hvp/jnp.linalg.norm(hvp) - gradient/jnp.linalg.norm(gradient))[:,:35], cmap="Spectral_r")
    #plt.colorbar()
    #plt.show()

    
    plt.imshow(hvp[:,:], vmin=-50, vmax=50, cmap="twilight_shifted")
    plt.title("hvp via ad")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()
    
    plt.plot(hvp[25,:])
    plt.ylim((-10,10))
    plt.show()
    
    #plt.imshow(jnp.where(jnp.abs(hvp)>1, ((fd_hvp - hvp)/fd_hvp[:,:]), jnp.nan), vmin=-1, vmax=1, cmap="RdBu_r")
    #plt.title("fd-ad percentage difference")
    #plt.colorbar()
    #plt.show()



#TODO: get full Hessian rather than HVP once and see
#TODO: write linear version.

calculate_hvp_via_soa()
#calculate_hvp_via_ad()





