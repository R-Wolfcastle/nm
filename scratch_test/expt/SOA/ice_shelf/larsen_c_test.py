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
from grid import *

sys.path.insert(1, "../../../../solvers/")
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp

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

def compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx, \
                                          h_1d, beta,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_s_ghost_cells,\
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
        u, v = add_uv_ghost_cells(u, v)

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
        u, v = add_uv_ghost_cells(u, v)

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
        #jax.debug.print("{x}",x=u[:,-3:])

        v = extrp_over_cf(v)
        #momentum_term
        u, v = add_uv_ghost_cells(u, v)

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
        u, v = add_uv_ghost_cells(u, v)

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
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx)
    #add_uv_ghost_cells                         = apply_scalar_ghost_cells_to_vector(
    #                                                add_ghost_cells_periodic_dirichlet_function(ny,nx)
    #                                             )
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(h)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf)
    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=False)

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
                                                              ksp_type="gmres",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False)



    @custom_vjp
    def solver(q, u_trial, v_trial):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

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

           
            #nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values)

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
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, u_trial, v_trial)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1), q)
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q)
        #fwd_residuals = (u, v, q)

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

            
            nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values.reshape((ny,nx)))
            #raise


            rhs = -jnp.concatenate(residuals_function(u_1d, v_1d,
                                                      *residuals_fct_args))


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
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny,nx)
    #add_uv_ghost_cells                         = apply_scalar_ghost_cells_to_vector(
    #                                                add_ghost_cells_periodic_dirichlet_function(ny,nx)
    #                                             )
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(h)
    cc_vector_field_gradient                   = cc_vector_field_gradient_function(ny, nx, dy,
                                                                                   dx, cc_gradient, 
                                                                                   extrapolate_over_cf,
                                                                                   add_uv_ghost_cells)
    membrane_strain_rate                       = membrane_strain_rate_function(ny, nx, dy, dx,
                                                                               cc_gradient,
                                                                               extrapolate_over_cf,
                                                                               add_uv_ghost_cells)
    div_tensor_field                           = divergence_of_tensor_field_function(ny, nx, dy, dx,
                                                                                     periodic_x=False)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient)
    fc_viscosity = fc_viscosity_function(ny, nx, dy, dx, extrapolate_over_cf, add_uv_ghost_cells,
                                         add_scalar_ghost_cells,
                                         interp_cc_to_fc, ew_gradient, ns_gradient, h_1d)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf)
    
    linear_ssa_residuals = compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_no_rhs = compute_linear_ssa_residuals_function_fc_visc_no_rhs(
                                                       ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf
                                                                                      )

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=False)

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
                                                              ksp_type="gmres",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,
                                                              monitor_ksp=True)


    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)

    #picard_solver = make_picard_solver(ny, nx, sparse_jacrev, mask, la_solver, get_u_v_residuals, fc_visc)

    def solve_fwd_problem(q, u_trial, v_trial):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        #u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,), coords)
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,))
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
            argz = (u.reshape(-1), v.reshape(-1), q, *additional_fctl_args)

        dJdu, dJdv = jax.grad(functional, argnums=(0,1))(*argz)

        
        rhs = - jnp.concatenate([dJdu, dJdv])


        #solve adjoint problem
        lx, ly = newton_solver(lx_trial.reshape(-1), ly_trial.reshape(-1),
                               linear_ssa_residuals, 1, (mu_bar_ew, mu_bar_ns, rhs))

        mu_bar = cc_viscosity(q, u, v)

        #plt.imshow(mu_bar)
        #plt.colorbar()
        #plt.show()

        #plt.imshow(mu_bar_ns)
        #plt.colorbar()
        #plt.show()


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
        mu_bar_ew, mu_bar_ns = fc_viscosity(q, u, v)
        
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


        print("solving for mu")
        mu_x, mu_y = newton_solver(x_trial, y_trial, linear_ssa_residuals,
                                   1, (mu_bar_ew, mu_bar_ns, rhs))


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

        print("solving for beta")
        beta_x, beta_y = newton_solver(x_trial, y_trial, linear_ssa_residuals,
                                       1, (mu_bar_ew, mu_bar_ns, rhs))


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
    #return solve_fwd_problem_picard, solve_adjoint_problem, solve_soa_problem






##NOTE: make everything linear by changing to 1
nvisc = c.GLEN_N
#nvisc = 1.001

A = c.A_COLD
B = 0.5 * (A**(-1/nvisc))


def ice_shelf():
    lx = 150_000
    ly = 200_000
    
    resolution = 4_000 #m
    
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
    thk = thk.at[:, -2:].set(0)
    
    b = jnp.zeros_like(thk)-600
    
    mucoef = jnp.ones_like(thk)
    
    C = jnp.zeros_like(thk)
    C = C.at[:4, :].set(1e8)
    C = C.at[:, :4].set(1e8)
    C = C.at[-4:,:].set(1e8)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q


lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = ice_shelf()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10

mask = jnp.where(thk>0,1,0)
mask = binary_erosion(mask)
mask = binary_erosion(mask)
mask = binary_erosion(mask)
mask = mask.astype(int)
#print(mask)
#raise

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
    
    #show_vel_field(u_out, v_out)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    
    
    plt.imshow(gradient[:,:])
    plt.title("gradient via adjoint")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    print("solving second-order adjoint problem:")
    pert_dir = gradient.copy()/(jnp.linalg.norm(gradient)*10)
    #pert_dir = pert_dir.at[:,-4:].set(0)
    hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir, functional)
    
    #plt.imshow(hvp[:,:], vmin=-3, vmax=3, cmap="twilight_shifted")
    plt.imshow(hvp[:,:], vmin=-15, vmax=15, cmap="twilight_shifted")
    plt.title("hvp via soa")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()
    

def calculate_hvp_via_ad():

    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, C,
                                                             n_iterations)

    u_out, v_out = solver(q, u_init, v_init)
    #show_vel_field(u_out, v_out)
    
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

    #plt.plot(gradient[25,:])
    #plt.ylim((-600,600))
    #plt.show()

    
    pert_dir = gradient.copy() / (jnp.linalg.norm(gradient)*10)
    #pert_dir = pert_dir.at[:,-4:].set(0)

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

    #plt.plot(fd_hvp[25,:])
    #plt.ylim((-10,10))
    #plt.show()
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
    
    #plt.plot(hvp[25,:])
    #plt.ylim((-10,10))
    #plt.show()
    
    #plt.imshow(jnp.where(jnp.abs(hvp)>1, ((fd_hvp - hvp)/fd_hvp[:,:]), jnp.nan), vmin=-1, vmax=1, cmap="RdBu_r")
    #plt.title("fd-ad percentage difference")
    #plt.colorbar()
    #plt.show()




#calculate_hvp_via_ad()
#calculate_hvp_via_soa()


#raise


solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                         delta_y,
                                                         delta_x,
                                                         thk, C,
                                                         n_iterations)

#u_out, v_out = solver(q, u_init, v_init)
#show_vel_field(u_out, v_out)
#raise
#jnp.save("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/u_big.npy", u_out)
#jnp.save("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/v_big.npy", v_out)
#raise

#u_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/u_big.npy")
#v_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/v_big.npy")

u_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/u.npy")
v_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/v.npy")

#show_vel_field(u_data, v_data)

def misfit_functional(v_field_x, v_field_y, q):

    return jnp.sum(mask.reshape(-1) * \
                   jnp.sqrt((v_field_x.reshape(-1)-u_data.reshape(-1))**2 +\
                            (v_field_y.reshape(-1)-v_data.reshape(-1))**2 +\
                            1e-10)
                  )




def make_hvp_ad_fct():
    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, C,
                                                             n_iterations)
    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return misfit_functional(u_out, v_out, q)
        #return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    _, vjp_grad = jax.vjp(get_grad, q)
    
    def hessian_vector_product(pert_dir):
        (hvp,) = vjp_grad(pert_dir.reshape((nr,nc)))
        return hvp

    return hessian_vector_product


def make_hvp_soa_fct():
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
                                      misfit_functional)
    
    
    def hessian_vector_product(pert_dir):
        hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir.reshape((nr,nc)), misfit_functional)
        return hvp

    return hessian_vector_product
    

#function for computing hvp from perturbation direction
#hessian_vector_product = make_hvp_ad_fct()
hessian_vector_product = make_hvp_soa_fct()





import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

n = nr * nc
dtype = np.float64

# JIT-compile and warm up once to avoid recompiles in the loop
#HVP_jit = jax.jit(lambda x: HVP(x.reshape(nr, nc)).reshape(-1))
#_ = HVP_jit(np.zeros(n))   # warmup

H = LinearOperator(
    shape=(n, n), dtype=dtype,
    matvec=lambda x: np.array(hessian_vector_product(x))  # ensure ndarray to avoid device transfers
)

# Largest algebraic eigenvalues (most positive)
k = 20  # or 1000
w, V = eigsh(H, k=k, which='LA', tol=1e-6, maxiter=None)  # V[:, i] is eigenvector of w[i]

# If you actually want largest magnitude (could pick big negative too), use which='LM'.
# Normalize or reshape as needed:
eigvals = w
eigvecs = V.reshape(nr, nc, k)

for i in range(k):
    plt.imshow(eigvecs[...,i])
    plt.colorbar()
    plt.title("lambda={}".format(eigvals[i]))
    plt.savefig("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/ts_evec_soa_small_{}.png".format(i))
    plt.close()

plt.plot(eigvals[::-1])
plt.show()





