#1st party
from pathlib import Path
import sys
import time



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
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy
from scipy.optimize import minimize as scinimize

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm





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


def double_dot_contraction(A, B):
    return jnp.dot(A[0,0], B[0,0]) + jnp.dot(A[1,0], B[0,1])\
        +  jnp.dot(A[0,1], B[1,0]) + jnp.dot(A[1,1], B[1,1])


def cc_vector_field_gradient_function(ny, nx, dy, dx, cc_grad,
                                      add_rb_ghost_cells):
    def cc_vector_field_gradient(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u, v = add_rb_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)

        grad_vf = jnp.zeros((ny, nx, 2, 2))

        grad_vf[:,:,0,0] = dudx
        grad_vf[:,:,0,1] = dudy
        grad_vf[:,:,1,0] = dvdx
        grad_vf[:,:,1,1] = dvdy

        return grad_vf

    return cc_vector_field_gradient


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

        msr_tensor[:,:,0,0] = 2*dudx + dvdy
        msr_tensor[:,:,0,1] = dudy + dvdx
        msr_tensor[:,:,1,0] = dudy + dvdx
        msr_tensor[:,:,1,1] = 2*dvdy + dudx

        return msr_tensor

    return membrane_sr_tensor


def divergence_of_tensor_field_function(ny, nx, dy, dx):
    def div_tensor_field(tf):
        #these have to be 2d scalar fields, of course
        tf_xx = tf[0,0]
        tf_xy = tf[0,1]
        tf_yx = tf[1,0]
        tf_yy = tf[1,1]

        shape_0, shape_1 = tf_xx.shape

        dx_tf_xx = jnp.zeros((shape_0, shape_1))
        dx_tf_xx = dx_tf_xx.at[:,1:-1](tf_xx[:,2:] - tf_xx[:,:-2])/(2*dx)
        
        dy_tf_yx = jnp.zeros((shape_0, shape_1))
        dy_tf_yx = dy_tf_yx.at[1:-1,:](tf_xx[:-2] - tf_xx[2:,:])/(2*dy)

        dx_tf_xy = jnp.zeros((shape_0, shape_1))
        dx_tf_xy = dx_tf_xy.at[:,1:-1](tf_xy[:,2:] - tf_xy[:,:-2])/(2*dx)
        
        dy_tf_yy = jnp.zeros((shape_0, shape_1))
        dy_tf_yy = dy_tf_yy.at[1:-1,:](tf_yy[:-2] - tf_yy[2:,:])/(2*dy)

        return dx_tf_xx+dy_tf_yx, dx_tf_xy+dy_tf_yy
    return div_tensor_field


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

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        volume_x = - ( beta * u + cc_rhs ) * dx * dy
        volume_y = - ( beta * v + cc_rhs ) * dy * dx

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

    return jax.jit(compute_u_v_residuals)


def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   h_1d, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_rflc_ghost_cells,\
                                   add_cont_ghost_cells,\
                                   extrp_over_cf):

    
    def compute_u_v_residuals(u_1d, v_1d, mucoef):

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


def cc_viscosity_function(ny, nx, dy, dx):

    def effective_strain_rate(u, v, cc_vector_field_gradient, add_rflc_ghost_cells):
        vector_field_gradient = cc_vector_field_gradient(u, v)
        #TODO: finish

        return epsilon_ii

    def cc_viscosity(mucoef, u, v, cc_vector_field_gradient, add_rflc_ghost_cells):

        effective_sr = effective_strain_rate(u, v, cc_vector_field_gradient, add_rflc_ghost_cells)

        mu_va = B * effective_sr**(-1/3)

        return mu_va
    return cc_viscosity


def make_solver_of_linear_viscous_problem(ny, nx, dy, dx, h, C, rhs):
    
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
    
    def solver(u_trial, v_trial, mu_bar):
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0


        ###LINEAR SOLVE######:
        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u_1d, v_1d, mucoef)
                                                      )

        nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                         dJv_du[mask], dJv_dv[mask]])

        rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, mucoef))

        old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


        du = la_solver(nz_jac_values, rhs)

        u_1d = u_1d+du[:(ny*nx)]
        v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, mucoef)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))






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
    def solver(mucoef, u_trial, v_trial):
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                           (u_1d, v_1d, mucoef)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, mucoef))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, mucoef)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(mucoef, u_trial, v_trial):
        u, v = solver(mucoef, u_trial, v_trial)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1), mucoef)
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])


        fwd_residuals = (u, v, dJ_dvel_nz_values, mucoef)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, mucoef = res
        u_bar, v_bar = cotangent

        lambda_ = la_solver(dJ_dvel_nz_values,
                            -jnp.concatenate([u_bar, v_bar]),
                            transpose=True)

        lambda_u = lambda_[:(ny*nx)]
        lambda_v = lambda_[(ny*nx):]


        _, pullback_function = jax.vjp(get_u_v_residuals,
                                         u.reshape(-1), v.reshape(-1), mucoef
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

    def solver(u_trial, v_trial, residuals_function, n_iterations,
               additional_residuals_fct_args):

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
                                    residuals_function(*residuals_fct_args)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver


    





def solve_forward_adjoint_and_second_order_adjoint_problems(ny, nx, dy, dx,
                                                            h, C,
                                                            n_iterations):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)
    extrapolate_over_cf                        = extrapolate_over_cf_function(h)
    cc_vector_field_gradient                   = cc_vector_field_gradient_function(ny, nx, dy,
                                                                                   dy, cc_gradient,
                                                                                   add_rflc_ghost_cells)
    membrane_strain_rate                       = membrane_strain_rate_function(ny, nx, dy, dx,
                                                                               cc_grad,
                                                                               add_rflc_ghost_cells)
    div_tensor_field                           = divergence_of_tensor_field_function(ny, nx, dy, dx)

    #calculate cell-centred viscosity based on velocity and mucoef
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx)

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
                                                              precondition_only=False)


    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)


    def solve_fwd_problem(mucoef, u_trial, v_trial):
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (mucoef))
        return u, v


    def solve_adjoint_problem(mucoef, u, v, lx_trial, ly_trial, functional:callable, additional_fctl_args=(None)):
        #calculate viscosity
        mu_bar = cc_viscosity(mucoef, u, v)

        #right-hand-side (\partial_u J)
        dJdu, dJdv = jax.grad(functional, argnums=(0,1))(u.reshape(-1), v.reshape(-1), *additional_fctl_args)
        rhs = - jnp.concatenate([dJdu, dJdv])

        #solve adjoint problem
        lx, ly = newton_solver(lx_trial, ly_trial, linear_ssa_residuals, 1, (mu_bar, rhs))

        #calculate gradient
        dJdq = - mu_bar * h * double_dot_contraction(cc_vector_field_gradient(lx.reshape(-1), ly.reshape(-1)),
                                                     membrane_strain_rate(u.reshape(-1), v.reshape(-1))
                                                    )

        return lx, ly, dJdq

    def solve_soa_problem(mucoef, u, v, lx, ly, functional:callable, 
                          additional_fctl_args, perturbation_direction):
        #calculate viscosity
        mu_bar = cc_viscosity(mucoef, u, v)

        #rhs of first equation
        rhs_x, rhs_y = div_tensor_field(h * mu_bar * perturbation_direction *\
                                        membrane_strain_rate(u.reshape(-1), v.reshape(-1)))
        rhs = - jnp.concatenate([rhs_x, rhs_y])

        #solve second-order adjoint equations

        #calculate hessian-vector-product





