#1st party
from pathlib import Path
import sys
import time
from functools import partial

##local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants as c
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


jax.config.update("jax_enable_x64", True)

def create_sparse_petsc_la_solver_with_custom_jvp(coordinates, jac_shape,\
                                    ksp_type='gmres', preconditioner='hypre',\
                                    precondition_only=False, monitor_ksp=False):

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()
    
    def construct_ab(values, b):
        iptr, j, values = scipy_coo_to_csr(values, coordinates,\
                                           jac_shape, return_decomposition=True)

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


    def petsc_sparse_la_solver(values, b):
    
        A, b = construct_ab(values, b)
        x = b.duplicate()
        
        ksp, pc = create_solver_object(A)

        if precondition_only:
            pc.apply(b, x)
        else:
            ksp.solve(b, x)
        
        x_jnp = jnp.array(x.getArray())

        return x_jnp


    @custom_jvp
    def la_solver_callback(values, b):
        out_spec = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return jax.pure_callback(petsc_sparse_la_solver, out_spec, values, b)

    @la_solver_callback.defjvp
    def solver_jvp(primals, tangents):
        vals, b = primals
        vals_dot, b_dot = tangents

        x = la_solver_callback(vals, b)

        #This is only possible because the j coords are just ordered normally
        #because of how we construct the basis vectors. Otherwise would have to
        #think about this more carefully...
        rhs = b_dot - jnp.einsum('ij,j->i', vals_dot, x)

        x_dot = la_solver_callback(vals, rhs)

        return x, x_dot

    return la_solver_callback



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


    @partial(jax.custom_vjp, nondiff_argnums=(2,))
    def la_solver_callback(values, b, transpose=False):
        out_spec = jax.ShapeDtypeStruct(b.shape, b.dtype)
        return jax.pure_callback(petsc_sparse_la_solver, out_spec, values, b)


    def la_solver_fwd(values, b, transpose=False):
        solution = la_solver_callback(values, b, transpose)
        return solution, (values, b, solution)


    #NOTE: the nondiff_argnums=(2,) thing shunts the transpose ragument to the front.
    def linear_solve_bwd(transpose, res, x_bar):
        #NOTE: The sign convention here is correct, despite what people say...
        #It just follows the documentation rather than "textbook" versions.
        values, b, x = res

        lambda_ = la_solver_callback(values, -x_bar, transpose=True)

        b_bar = -lambda_.reshape(b.shape) #ensure same shape as input b.

        #sparse version of jnp.outer(x,lambda_)
        #TODO: CHECK WHICH WAY ROUND THESE COORDS GO. SHOULD BE RIGHT IF THEY ARE IJ!
        values_bar = x[coordinates[1]] * lambda_[coordinates[0]]

        return values_bar, b_bar
    

    la_solver_callback.defvjp(la_solver_fwd, linear_solve_bwd)

    return la_solver_callback



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
#NOTE: this is not actually used...
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

@jax.jit
def double_dot_contraction(A, B):
    return A[:,:,0,0]*B[:,:,0,0] + A[:,:,1,0]*B[:,:,0,1] +\
           A[:,:,0,1]*B[:,:,1,0] + A[:,:,1,1]*B[:,:,1,1]


def cc_vector_field_gradient_function(ny, nx, dy, dx, cc_grad,
                                      add_rb_ghost_cells):
    def cc_vector_field_gradient(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u, v = add_rb_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)

        grad_vf = jnp.zeros((ny, nx, 2, 2))

        grad_vf = grad_vf.at[:,:,0,0].set(dudx)
        grad_vf = grad_vf.at[:,:,0,1].set(dudy)
        grad_vf = grad_vf.at[:,:,1,0].set(dvdx)
        grad_vf = grad_vf.at[:,:,1,1].set(dvdy)

        return grad_vf

    return jax.jit(cc_vector_field_gradient)


def membrane_strain_rate_function(ny, nx, dy, dx, 
                                  cc_grad,
                                  add_rb_ghost_cells):

    def membrane_sr_tensor(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u, v = add_rb_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)

        msr_tensor = jnp.zeros((ny, nx, 2, 2))

        msr_tensor = msr_tensor.at[:,:,0,0].set(2*dudx + dvdy)
        msr_tensor = msr_tensor.at[:,:,0,1].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,0].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,1].set(2*dvdy + dudx)

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


def compute_linear_ssa_residuals_function(ny, nx, dy, dx, \
                                          h_1d, beta,\
                                          interp_cc_to_fc,\
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_rflc_ghost_cells,\
                                          add_cont_ghost_cells,\
                                          extrp_over_cf):

    def compute_linear_ssa_residuals(u_1d, v_1d, mu_bar, cc_rhs):

        cc_rhs_x = cc_rhs[:nx*ny].reshape((ny, nx))
        cc_rhs_y = cc_rhs[nx*ny:].reshape((ny, nx))

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        volume_x = - ( beta * u + cc_rhs_x ) * dx * dy
        volume_y = - ( beta * v + cc_rhs_y ) * dy * dx

        #momentum_term
        u, v = add_rflc_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #interpolate things onto face-cenres
        h_ew, h_ns = interp_cc_to_fc(h)
        mu_ew, mu_ns = interp_cc_to_fc(mu_bar)
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
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

    return jax.jit(compute_linear_ssa_residuals)


def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   h_1d, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_rflc_ghost_cells,\
                                   add_cont_ghost_cells,\
                                   extrp_over_cf):

    
    def compute_u_v_residuals(u_1d, v_1d, q):

        mucoef = mucoef_0*jnp.exp(q)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        dsdx, dsdy = cc_gradient(s)

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
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

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


def cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient):
    def cc_viscosity(q, u, v):
        
        vfg = cc_vector_field_gradient(u, v)
        
        mu = B * mucoef_0 * jnp.exp(q) * (vfg[:,:,0,0]**2 + vfg[:,:,1,1]**2 + vfg[:,:,0,0]*vfg[:,:,1,1] + \
                           0.25*(vfg[:,:,0,1] + vfg[:,:,1,0])**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

        return mu
    return jax.jit(cc_viscosity)




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


def make_newton_velocity_solver_function_custom_jvp(ny, nx, dy, dx,\
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

   


    la_solver = create_sparse_petsc_la_solver_with_custom_jvp(coords, (ny*nx*2, ny*nx*2),\
                                                              ksp_type="bcgs",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False)

    @custom_jvp
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

    @solver.defjvp
    def solver_jvp(primals, tangents):
        q, u0, v0 = primals
        q_dot,_,_ = tangents #who gives a shit about u0_dot and v0_dot
        
        #build the sparse matrix for the lhs
        u, v = solver(q, u0, v0)
        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1), q)
                                                      )
        nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                         dJv_du[mask], dJv_dv[mask]])

        
        #build the rhs
        rhs = - jnp.concatenate(jax.jvp(get_u_v_residuals, 
                        (u.reshape(-1), v.reshape(-1), q), 
                        (jnp.zeros_like(u.reshape(-1)), jnp.zeros_like(u.reshape(-1)), q_dot)
                       )[1]
                               )
        #the output of jax.jvp applied to the tangents is the same structure as the output of
        #get_u_v_residuals, so have to concatenate these into the right shape!


        #computing tangents
        u_dot = la_solver(nz_jac_values, rhs)

        return (u, v), (u_dot[:(ny*nx)].reshape(ny,nx), u_dot[(ny*nx):].reshape(ny,nx))

    return solver



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

    def solver(u_trial, v_trial, residuals_function, n_iterations, residuals_fct_args):

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        
        residual = jnp.inf
        init_res = 0


        for i in range(n_iterations):
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(residuals_function,
                                                           (u_1d, v_1d,
                                                            *residuals_fct_args)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(residuals_function(u_1d, v_1d,
                                                      *residuals_fct_args))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

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
    interp_cc_to_fc                            = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)
    extrapolate_over_cf                        = extrapolate_over_cf_function(h)
    cc_vector_field_gradient                   = cc_vector_field_gradient_function(ny, nx, dy,
                                                                                   dx, cc_gradient,
                                                                                   add_rflc_ghost_cells)
    membrane_strain_rate                       = membrane_strain_rate_function(ny, nx, dy, dx,
                                                                               cc_gradient,
                                                                               add_rflc_ghost_cells)
    div_tensor_field                           = divergence_of_tensor_field_function(ny, nx, dy, dx)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_rflc_ghost_cells,\
                                                       add_cont_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals = compute_linear_ssa_residuals_function(ny, nx, dy, dx,
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
                                                              precondition_only=False,
                                                              monitor_ksp=False)


    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)


    def solve_fwd_problem(q, u_trial, v_trial):
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,))
        return u.reshape((ny,nx)), v.reshape((ny,nx))


    def solve_adjoint_problem(q, u, v, lx_trial, ly_trial,
                              functional:callable, additional_fctl_args=None):
        #calculate viscosity
        mu_bar = cc_viscosity(q, u, v)

        #right-hand-side (\partial_u J)
        if additional_fctl_args is None:
            argz = (u.reshape(-1), v.reshape(-1), q)
        else:
            argz = (q, *additional_fctl_args)

        dJdu, dJdv = jax.grad(functional, argnums=(0,1))(*argz)
        rhs = - jnp.concatenate([dJdu, dJdv])


        #solve adjoint problem
        lx, ly = newton_solver(lx_trial.reshape(-1), ly_trial.reshape(-1),
                               linear_ssa_residuals, 1, (mu_bar, rhs))

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

    return solve_fwd_problem, solve_adjoint_problem, solve_soa_problem









##NOTE: make everything linear by changing to 1
nvisc = c.GLEN_N
#nvisc = 1.001

A = c.A_COLD
B = 0.5 * (A**(-1/nvisc))




lx = 150_000
ly = 200_000

resolution = 4000 #m

nr = int(ly/resolution)
nc = int(lx/resolution)

lx = nr*resolution
ly = nc*resolution


#nr, nc = 64, 64
#nr, nc = 96*2, 64*2


x = jnp.linspace(0, lx, nc)
y = jnp.linspace(0, ly, nr)


delta_x = x[1]-x[0]
delta_y = y[1]-y[0]

thk_profile = 500# - 300*x/lx
thk = jnp.zeros((nr, nc))+thk_profile
thk = thk.at[:, -1:].set(0)

b = jnp.zeros_like(thk)-600

mucoef = jnp.ones_like(thk)

C = jnp.zeros_like(thk)
C = C.at[:4, :].set(1e16)
C = C.at[:, :4].set(1e16)
C = C.at[-4:,:].set(1e16)
C = jnp.where(thk==0, 1, C)


#plt.imshow(jnp.log10(C))
#plt.show()


u_init = jnp.zeros_like(thk)
v_init = jnp.zeros_like(thk)

n_iterations = 15



mucoef = jnp.ones_like(C)

mucoef_0 = jnp.ones_like(C)

q = jnp.zeros_like(C)


mask = jnp.zeros_like(C)
mask = mask.at[6:-6,6:-6].set(1)


def functional(v_field_x, v_field_y, q):
    #NOTE: for things like this, you need to ensure the value isn't
    #zero where the argument is zero, because JAX can't differentiate
    #through the square-root otherwise. Silly JAX.
    return jnp.sum(mask.reshape(-1) * jnp.sqrt(v_field_x**2 + v_field_y**2 + 1e-10).reshape(-1)) * c.S_PER_YEAR




def calculate_hvp_via_soa():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                                                 nr, nc, delta_y, delta_x, thk, C, n_iterations
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    show_vel_field(u_out*c.S_PER_YEAR, v_out*c.S_PER_YEAR)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    
    
    ###NOTE: Calving front stiffness dominating the gradient calculation
    ###Should investigate. For now, cutting out to visualise
    #plt.imshow(gradient[:,:35])
    #plt.colorbar()
    #plt.show()
    
    print("solving second-order adjoint problem:")
    pert_dir = gradient.copy()/(jnp.linalg.norm(gradient)*10)
    hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir, functional)
    
    #plt.imshow(-hvp[:,:35], vmin=-35, vmax=2)
    plt.imshow(hvp[:,:35])
    plt.colorbar()
    plt.show()
    
    #plt.imshow(jnp.log(hvp[6:-6,6:35]))
    #plt.colorbar()
    #plt.show()


def calculate_hvp_via_ad():

    solver = make_newton_velocity_solver_function_custom_jvp(nr, nc,
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
    
    plt.imshow(gradient[:,:35])
    plt.colorbar()
    plt.show()

    
    pert_dir = gradient / (jnp.linalg.norm(gradient)*10)


    #finite diff hvp for comparison
    #plt.imshow(p)
    #plt.colorbar()
    #plt.show()
    #raise
    eps = 4e-11
    fd_hvp = (get_grad(q + eps*pert_dir) - get_grad(q)) / eps
    plt.imshow(fd_hvp[:,:35])
    plt.title("hvp via fd")
    plt.colorbar()
    plt.show()
    #raise

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

    
    plt.imshow(hvp[:,:29])
    plt.title("hvp via ad")
    plt.colorbar()
    plt.show()
    
    #plt.imshow((fd_hvp - hvp)[:,:35])
    #plt.title("fd-ad difference")
    #plt.colorbar()
    #plt.show()



#TODO: get full Hessian rather than HVP once and see
#TODO: write linear version.

#calculate_hvp_via_soa()
calculate_hvp_via_ad()


raise




# ---- Drop-in test for the linear-solve custom VJP (PETSc wrapper) ----
import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import numpy as np

def test_linear_solver_vjp(
    n=200,
    alpha=2.0,
    key_seed=0,
    ksp_type="cg",
    preconditioner="hypre",
    print_details=True,
    fd_eps=1e-6,
    rel_tol_ok=5e-8,
):
    """
    Builds a symmetric positive definite 1D Laplacian + alpha*I system in COO,
    instantiates your PETSc linear solver wrapper (la), and verifies:
      1) VJP wrt b (scalar FD),
      2) VJP wrt values (directional FD),
      3) Sign/index convention by comparing to explicit formula.

    Parameters
    ----------
    n : int
        Problem size.
    alpha : float
        Shift to make Laplacian SPD: A = Laplacian + alpha*I.
    key_seed : int
        RNG seed for reproducibility.
    ksp_type : str
        PETSc KSP type (e.g., "cg", "gmres", "bcgs").
    preconditioner : str or None
        PETSc preconditioner (e.g., "hypre", "ilu", or None).
    fd_eps : float
        Finite-difference epsilon.
    rel_tol_ok : float
        Relative error threshold for tests.
    """

    # --- Build SPD matrix A in COO: 1D Laplacian + alpha*I ---
    # rows, cols, vals in 64-bit
    row_list, col_list, val_list = [], [], []
    for i in range(n):
        row_list.append(i); col_list.append(i); val_list.append(2.0 + alpha)
        if i > 0:     row_list.append(i); col_list.append(i-1); val_list.append(-1.0)
        if i < n - 1: row_list.append(i); col_list.append(i+1); val_list.append(-1.0)

    rows = jnp.array(row_list, dtype=jnp.int32)
    cols = jnp.array(col_list, dtype=jnp.int32)
    values = jnp.array(val_list, dtype=jnp.float64)
    coords = jnp.stack([rows, cols])
    shape = (n, n)

    # --- Instantiate the PETSc solver wrapper from your code ---
    # Expected signature:
    # la = create_sparse_petsc_la_solver_with_custom_vjp(coords, shape, ksp_type, preconditioner, precondition_only=False, monitor_ksp=False)
    la = create_sparse_petsc_la_solver_with_custom_vjp(
        coords, shape, ksp_type=ksp_type, preconditioner=preconditioner, precondition_only=False
    )

    # --- Random RHS and upstream cotangent ---
    key = jax.random.PRNGKey(key_seed)
    key1, key2, key3 = jax.random.split(key, 3)
    b = jax.random.normal(key1, (n,), dtype=jnp.float64)
    x_bar = jax.random.normal(key2, (n,), dtype=jnp.float64)     # upstream cotangent
    e_vals = jax.random.normal(key3, values.shape, dtype=jnp.float64)  # random dir for values FD
    e_vals = e_vals / (jnp.linalg.norm(e_vals) + 1e-30)

    # --- Define scalar functional L for testing pullbacks ---
    # L(values, b) = < x , x_bar > with x = la(values, b)
    def L_of_b(bb):
        x = la(values, bb, transpose=False)
        return jnp.vdot(x, x_bar)

    def L_of_values(vv):
        x = la(vv, b, transpose=False)
        return jnp.vdot(x, x_bar)

    # ========== (1) Test dL/db via VJP vs FD ==========
    # VJP wrt b (keeping values fixed)
    _, vjp_b = jax.vjp(lambda bb: la(values, bb, transpose=False), b)
    (b_bar_pred,) = vjp_b(x_bar)

    # FD along random direction on b
    e_b = jax.random.normal(key1, b.shape, dtype=jnp.float64)
    e_b = e_b / (jnp.linalg.norm(e_b) + 1e-30)
    L_fd_b = (L_of_b(b + fd_eps * e_b) - L_of_b(b)) / fd_eps
    vjp_dot_e_b = jnp.vdot(b_bar_pred, e_b)
    rel_err_db = float(abs(L_fd_b - vjp_dot_e_b) / (abs(L_fd_b) + 1e-30))

    # ========== (2) Test dL/dvalues via VJP vs FD (directional) ==========
    _, vjp_vals = jax.vjp(lambda vv: la(vv, b, transpose=False), values)
    (values_bar_pred,) = vjp_vals(x_bar)

    L_fd_vals = (L_of_values(values + fd_eps * e_vals) - L_of_values(values)) / fd_eps
    vjp_dot_e_vals = jnp.vdot(values_bar_pred, e_vals)
    rel_err_dvalues = float(abs(L_fd_vals - vjp_dot_e_vals) / (abs(L_fd_vals) + 1e-30))

    # ========== (3) Cross-check sign/index convention explicitly ==========
    # Forward solution x:
    x = la(values, b, transpose=False)

    # Build explicit expressions for both sign conventions:
    # Option (+): A^T λ = +x_bar  => b_bar_exp = +λ,  values_bar_exp = -(λ[rows]*x[cols])
    lambda_plus = la(values, x_bar, transpose=True)
    b_bar_exp_plus = lambda_plus
    values_bar_exp_plus = -(lambda_plus[rows] * x[cols])

    # Option (−): A^T λ = −x_bar  => b_bar_exp = −λ, values_bar_exp = +(λ[rows]*x[cols])
    lambda_minus = la(values, -x_bar, transpose=True)
    b_bar_exp_minus = -lambda_minus
    values_bar_exp_minus = +(lambda_minus[rows] * x[cols])

    # Compare to your VJP outputs:
    err_plus_b = float(jnp.linalg.norm(b_bar_pred - b_bar_exp_plus) / (jnp.linalg.norm(b_bar_pred) + 1e-30))
    err_plus_v = float(jnp.linalg.norm(values_bar_pred - values_bar_exp_plus) / (jnp.linalg.norm(values_bar_pred) + 1e-30))
    err_minus_b = float(jnp.linalg.norm(b_bar_pred - b_bar_exp_minus) / (jnp.linalg.norm(b_bar_pred) + 1e-30))
    err_minus_v = float(jnp.linalg.norm(values_bar_pred - values_bar_exp_minus) / (jnp.linalg.norm(values_bar_pred) + 1e-30))

    if print_details:
        print("\n--- Linear-solve VJP checks ---")
        print(f"(1) dL/db:  rel err (FD vs VJP·e)     = {rel_err_db: .3e}")
        print(f"(2) dL/dA:  rel err (FD-dir vs VJP·e) = {rel_err_dvalues: .3e}")
        print("Convention check (smaller is better):")
        print(f"  Option (+) A^T λ = +x_bar :  ||b_bar_vjp - b_bar_exp||/||·|| = {err_plus_b: .3e},  "
              f"||values_bar_vjp - values_bar_exp||/||·|| = {err_plus_v: .3e}")
        print(f"  Option (−) A^T λ = −x_bar :  ||b_bar_vjp - b_bar_exp||/||·|| = {err_minus_b: .3e}, "
              f"||values_bar_vjp - values_bar_exp||/||·|| = {err_minus_v: .3e}")

    # Decide which convention matches your implementation best
    sum_plus = err_plus_b + err_plus_v
    sum_minus = err_minus_b + err_minus_v
    convention = "+ (A^T λ = +x_bar)" if sum_plus < sum_minus else "− (A^T λ = −x_bar)"
    if print_details:
        print(f"=> Your custom VJP matches **Option {convention}**")

    # --- Assertions (soft) ---
    ok = (rel_err_db < rel_tol_ok) and (rel_err_dvalues < rel_tol_ok)
    if not ok and print_details:
        print("WARNING: Relative errors exceed tolerance. "
              "Consider tightening PETSc KSP tolerances and ensuring 64-bit throughout.")
    return {
        "rel_err_db": rel_err_db,
        "rel_err_dvalues": rel_err_dvalues,
        "err_plus_b": err_plus_b,
        "err_plus_v": err_plus_v,
        "err_minus_b": err_minus_b,
        "err_minus_v": err_minus_v,
        "convention": convention,
    }

# Example usage (uncomment to run):
results = test_linear_solver_vjp()
print(results)








raise






# --- Force 64-bit for the test ---
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# Build SPD matrix: 1D Laplacian + alpha*I in COO
n = 200
alpha = 2.0
rows = []
cols = []
vals = []
for i in range(n):
    rows.append(i); cols.append(i);   vals.append(2.0 + alpha)
    if i > 0:     rows.append(i); cols.append(i-1); vals.append(-1.0)
    if i < n-1:   rows.append(i); cols.append(i+1); vals.append(-1.0)
rows   = jnp.array(rows, dtype=jnp.int32)
cols   = jnp.array(cols, dtype=jnp.int32)
values = jnp.array(vals, dtype=jnp.float64)
coords = jnp.stack([rows, cols])

# IMPORTANT: In create_solver_object(A) set:
# ksp.setTolerances(rtol=1e-12, atol=0.0, max_it=500)
# and use ksp_type="cg", preconditioner="hypre" or None.

la = create_sparse_petsc_la_solver_with_custom_vjp(
    coords, (n, n), ksp_type="cg", preconditioner="hypre"
)

key = jax.random.PRNGKey(0)
b     = jax.random.normal(key, (n,), dtype=jnp.float64)
x_bar = jax.random.normal(jax.random.split(key, 2)[1], (n,), dtype=jnp.float64)

def L_of_b(b):
    x = la(values, b, transpose=False)       # Ax = b
    return jnp.vdot(x, x_bar)

# ---- Test dL/db ----
_, vjp_fn = jax.vjp(lambda bb: la(values, bb, transpose=False), b)
(b_bar_pred,) = vjp_fn(x_bar)

eps = 1e-6
e   = jax.random.normal(jax.random.split(key, 3)[2], (n,), dtype=jnp.float64)
L_fd = (L_of_b(b + eps*e) - L_of_b(b)) / eps
rel_err_db = float(abs(L_fd - jnp.vdot(b_bar_pred, e)) / (abs(L_fd) + 1e-30))
print("rel err dL/db:", rel_err_db)

# ---- Test a single matrix entry dL/dA_ij (Option A) ----
def L_of_values(vals):
    x = la(vals, b, transpose=False)
    return jnp.vdot(x, x_bar)

x          = la(values, b, transpose=False)
lambda_opA = la(values, -x_bar, transpose=True)   # Option A: RHS = -x_bar
k = 0
i, j = int(coords[0, k]), int(coords[1, k])
Abar_pred = lambda_opA[i] * x[j]                  # Option A formula

eps = 1e-6
vals_fd = values.at[k].add(eps)
Abar_fd = (L_of_values(vals_fd) - L_of_values(values)) / eps
rel_err_dA = float(abs(Abar_fd - Abar_pred) / (abs(Abar_fd) + 1e-30))
print("rel err dL/dA_ij:", rel_err_dA)


raise








