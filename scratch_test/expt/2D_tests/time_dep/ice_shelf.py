
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
from grid import *

sys.path.insert(1, "../../../../solvers/")
from residuals import make_fo_upwind_residual_function
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp

                    
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





def make_implicit_advect_scalar_field_fou_residuals_function(nx, ny, dx, dy, vel_bcs="rflc"):
    if vel_bcs!="rflc":
        raise "Add the 2 lines of code to enable periodic bcs you lazy bastard, Trys."

    cc_to_fc = interp_cc_with_ghosts_to_fc_function(ny, nx)
    add_reflection_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)

    def implicit_advect_scalar_field_fou_residuals(u, v, scalar_field,
                                                   scalar_field_current,
                                                   source, delta_t, thickness):

        u = linear_extrapolate_over_cf_dynamic_thickness(u, thickness)
        v = linear_extrapolate_over_cf_dynamic_thickness(v, thickness)
        h = linear_extrapolate_over_cf_dynamic_thickness(scalar_field_current, thickness)
        
        u, v = add_reflection_ghost_cells(u,v)
        h = add_cont_ghost_cells(h)
     
        u_fc_ew, _ = cc_to_fc(u)
        _, v_fc_ns = cc_to_fc(v)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)

        h_fc_fou_ew = jnp.where(u_fc_ew>0, h[1:-1,:-1], h[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h[1:, 1:-1], h[-1:,1:-1])

        ew_flux = u_fc_ew*h_fc_fou_ew
        ns_flux = v_fc_ns*h_fc_fou_ns

        #remove those ghost cells again!
        h = h[1:-1,1:-1]

        return ((scalar_field - h)/delta_t - source)*dx*dy +\
                (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy +\
                (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx

    return implicit_advect_scalar_field_fou_residuals


def make_advect_scalar_field_fou_function(nx, ny, dx, dy, vel_bcs="rflc"):
    if vel_bcs!="rflc":
        raise "Add the 2 lines of code to enable periodic bcs you lazy bastard, Trys."

    cc_to_fc = interp_cc_with_ghosts_to_fc_function(ny, nx)
    add_reflection_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)

    def advect_scalar_field_fou(u, v, scalar_field, source, delta_t, thickness):

        u = linear_extrapolate_over_cf_dynamic_thickness(u, thickness)
        v = linear_extrapolate_over_cf_dynamic_thickness(v, thickness)
        h = linear_extrapolate_over_cf_dynamic_thickness(scalar_field, thickness)
        
        u, v = add_reflection_ghost_cells(u,v)
        h = add_cont_ghost_cells(h)
     
        u_fc_ew, _ = cc_to_fc(u)
        _, v_fc_ns = cc_to_fc(v)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)

        h_fc_fou_ew = jnp.where(u_fc_ew>0, h[1:-1,:-1], h[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h[1:, 1:-1], h[-1:,1:-1])

        ew_flux = u_fc_ew*h_fc_fou_ew
        ns_flux = v_fc_ns*h_fc_fou_ns

        #remove those ghost cells again!
        h = h[1:-1,1:-1]

        h_new = h + delta_t * (
                    source - \
                    (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])/dx -\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])/dy
                )

        return jnp.where(thickness>1e-2, h_new, 0)
    return advect_scalar_field_fou



def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf):
    
    def compute_u_v_residuals(u_1d, v_1d, q, h_1d):

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
        u = extrp_over_cf(u, h)
        v = extrp_over_cf(v, h)
        #momentum_term
        u, v = add_uv_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)


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


def compute_uvh_residuals_function_lr(ny, nx, dy, dx,
                                      beta,
                                      interp_cc_to_fc,
                                      ew_gradient,
                                      ns_gradient,
                                      cc_gradient,
                                      add_uv_ghost_cells,
                                      add_s_ghost_cells):
    

    def compute_uvh_residuals(u, v, h, viscosity, h_t, source, delta_t):
        
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))
        h = h.reshape((ny, nx))
        h_t = h_t.reshape((ny, nx))


        ######### MOMENTUM RESIDUALS #############################

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

        mu_ew, mu_ns = viscosity

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


        #################################################################







        ################ ADVECTION RESIDUALS #############################


        h = linear_extrapolate_over_cf_dynamic_thickness(h, h)
        u = linear_extrapolate_over_cf_dynamic_thickness(u, h)
        v = linear_extrapolate_over_cf_dynamic_thickness(v, h)
        
        

        u_fc_ew, _ = interp_cc_to_fc(u)
        _, v_fc_ns = interp_cc_to_fc(v)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        #face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, h[1:-1,:-1], h[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h[1:, 1:-1], h[-1:,1:-1])

        ew_flux = u_fc_ew*h_fc_fou_ew
        ns_flux = v_fc_ns*h_fc_fou_ns

        #remove those ghost cells again!
        h = h[1:-1,1:-1]

        adv_residual =  ((h - h_t)/delta_t - source)*dx*dy +\
                (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy +\
                (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx


        #################################################################



        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1), adv_residual.reshape(-1) 

    return compute_uvh_residuals


def compute_uvh_residuals_function(ny, nx, dy, dx,
                                   beta,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells):
    

    def compute_uvh_residuals(u, v, h, q, h_t, source, delta_t):

        
        mucoef = mucoef_0*jnp.exp(q)

        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))
        h = h.reshape((ny, nx))
        h_t = h_t.reshape((ny, nx))


        ######### MOMENTUM RESIDUALS #############################

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
        u = linear_extrapolate_over_cf_dynamic_thickness(u, h)
        v = linear_extrapolate_over_cf_dynamic_thickness(v, h)
        #momentum_term
        u, v = add_uv_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #jax.debug.print("dudxew = {x}",x=dudx_ew)

        #interpolate things onto face-centres
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)


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
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h[1:-1,1:-1]==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h[1:-1,1:-1]==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h[1:-1,1:-1]==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h[1:-1,1:-1]==0, 0, mu_ns[:-1,:]))
        

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


        #################################################################







        ################ ADVECTION RESIDUALS #############################


        h = linear_extrapolate_over_cf_dynamic_thickness(h, h)
        

        u_fc_ew, _ = interp_cc_to_fc(u)
        _, v_fc_ns = interp_cc_to_fc(v)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        #face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, h[1:-1,:-1], h[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h[1:, 1:-1], h[-1:,1:-1])

        ew_flux = u_fc_ew*h_fc_fou_ew
        ns_flux = v_fc_ns*h_fc_fou_ns

        #remove those ghost cells again!
        h = h[1:-1,1:-1]

        adv_residual =  ((h - h_t)/delta_t - source)*dx*dy +\
                (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy +\
                (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx


        #################################################################



        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1), adv_residual.reshape(-1) 

    return compute_uvh_residuals


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


def make_newton_coupled_solver_function(ny, nx, dy, dx, C, n_iterations):

    beta_eff = C.copy()

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx)
    get_uvh_residuals = compute_uvh_residuals_function(ny, nx, dy, dx,\
                                                       beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells)

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
                                                        3,
                                                        active_indices=(0,1,2)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    #############

    coords = jnp.stack([
        jnp.concatenate(
           [i_coordinate_sets,           i_coordinate_sets,           i_coordinate_sets,\
            i_coordinate_sets+(ny*nx),   i_coordinate_sets+(ny*nx),   i_coordinate_sets+(ny*nx),\
            i_coordinate_sets+(2*ny*nx), i_coordinate_sets+(2*ny*nx), i_coordinate_sets+(2*ny*nx)]
                       ),\
        jnp.concatenate(
           [j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx),\
            j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx),\
            j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx)]
                       )
                       ])

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*3, ny*nx*3),\
                                                              ksp_type="bcgs",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,\
                                                              monitor_ksp=False)



    #@custom_vjp
    def solver(q, u_trial, v_trial, h_now, accm, delta_t):
        u_trial = jnp.where(h_now>1e-10, u_trial, 0)
        v_trial = jnp.where(h_now>1e-10, v_trial, 0)

        h_1d = h_now.copy().reshape(-1)
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            dRu_du, dRv_du, dRh_du, \
            dRu_dv, dRv_dv, dRh_dv, \
            dRu_dh, dRv_dh, dRh_dh = sparse_jacrev(get_uvh_residuals, \
                                                  (u_1d, v_1d, h_1d, q, h_now, accm, delta_t)
                                                          )

            nz_jac_values = jnp.concatenate([dRu_du[mask], dRu_dv[mask], dRu_dh[mask],\
                                             dRv_du[mask], dRv_dv[mask], dRv_dh[mask],\
                                             dRh_du[mask], dRh_dv[mask], dRh_dh[mask]])

          
            #full_jac = jnp.zeros((ny*nx*3, ny*nx*3))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #print(full_jac)
            #raise


            rhs = -jnp.concatenate(get_uvh_residuals(u_1d, v_1d, h_1d, q, h_now, accm, delta_t))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):(2*ny*nx)]
            h_1d = h_1d+du[(2*ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_uvh_residuals(u_1d, v_1d, h_1d,  q, h_now, accm, delta_t)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx)), h_1d.reshape((ny,nx))

    return solver

def make_quasi_newton_coupled_solver_function(ny, nx, dy, dx, C, n_iterations):

    beta_eff = C.copy()

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx)
    get_uvh_residuals = compute_uvh_residuals_function_lr(ny, nx, dy, dx,\
                                                          beta_eff,\
                                                          interp_cc_to_fc,\
                                                          ew_gradient, ns_gradient,\
                                                          cc_gradient,\
                                                          add_uv_ghost_cells,\
                                                          add_scalar_ghost_cells)

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
                                                        3,
                                                        active_indices=(0,1,2)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    #############

    coords = jnp.stack([
        jnp.concatenate(
           [i_coordinate_sets,           i_coordinate_sets,           i_coordinate_sets,\
            i_coordinate_sets+(ny*nx),   i_coordinate_sets+(ny*nx),   i_coordinate_sets+(ny*nx),\
            i_coordinate_sets+(2*ny*nx), i_coordinate_sets+(2*ny*nx), i_coordinate_sets+(2*ny*nx)]
                       ),\
        jnp.concatenate(
           [j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx),\
            j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx),\
            j_coordinate_sets, j_coordinate_sets+(ny*nx), j_coordinate_sets+(2*ny*nx)]
                       )
                       ])

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*3, ny*nx*3),\
                                                              ksp_type="bcgs",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,\
                                                              monitor_ksp=False)



    #@custom_vjp
    def solver(q, u_trial, v_trial, h_now, accm, delta_t):
        u_trial = jnp.where(h_now>1e-10, u_trial, 0)
        v_trial = jnp.where(h_now>1e-10, v_trial, 0)

        h_1d = h_now.copy().reshape(-1)
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0


        viscosity = fc_viscosity(u_trial, v_trial, q, h_now)


        for i in range(n_iterations):

            dRu_du, dRv_du, dRh_du, \
            dRu_dv, dRv_dv, dRh_dv, \
            dRu_dh, dRv_dh, dRh_dh = sparse_jacrev(get_uvh_residuals_lr, \
                                        (u_1d, v_1d, h_1d, viscosity, h_now, accm, delta_t)
                                                          )

            nz_jac_values = jnp.concatenate([dRu_du[mask], dRu_dv[mask], dRu_dh[mask],\
                                             dRv_du[mask], dRv_dv[mask], dRv_dh[mask],\
                                             dRh_du[mask], dRh_dv[mask], dRh_dh[mask]])

          
            #full_jac = jnp.zeros((ny*nx*3, ny*nx*3))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #print(full_jac)
            #raise


            rhs = -jnp.concatenate(get_uvh_residuals_lr(u_1d, v_1d, h_1d,
                                                        viscosity, h_now, accm, delta_t
                                                        )
                                   )

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):(2*ny*nx)]
            h_1d = h_1d+du[(2*ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                get_uvh_residuals_lr(u_1d, v_1d, h_1d, 
                                                     viscosity, h_now, accm, delta_t
                                                    )
                                                   )
                                   )
                           )

        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx)), h_1d.reshape((ny,nx))

    return solver




def make_newton_velocity_solver_function_custom_vjp(ny, nx, dy, dx,\
                                                    C, n_iterations):

    beta_eff = C.copy()

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx)
    #add_uv_ghost_cells                         = apply_scalar_ghost_cells_to_vector(
    #                                                add_ghost_cells_periodic_dirichlet_function(ny,nx)
    #                                             )
    extrapolate_over_cf                        = linear_extrapolate_over_cf_dynamic_thickness

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       beta_eff,\
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
                                                              ksp_type="bcgs",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False)



    @custom_vjp
    def solver(q, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        h_1d = h.copy().reshape(-1)
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                           (u_1d, v_1d, q, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            #full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #print(full_jac)
            #raise


            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, q, h_1d))

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(residual, rhs, init_res, i)


            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, q, h_1d)
                                                   )
                                   )
                           )
        print("----------")
        print("Final residual: {}".format(res_final))
        print("Total residual reduction factor: {}".format(init_res/res_final))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                                       (u.reshape(-1), v.reshape(-1), q, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])


        fwd_residuals = (u, v, dJ_dvel_nz_values, q, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, h = res
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
                                         u.reshape(-1), v.reshape(-1), q, h.reshape(-1)
                                      )
        _, _, mu_bar = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar, v_trial_bar, h_bar...
        return (mu_bar.reshape((ny, nx)), None, None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver



##NOTE: make everything linear by changing to 1
nvisc = c.GLEN_N
#nvisc = 1.001

A = c.A_COLD
B = 0.5 * (A**(-1/nvisc))


def ice_shelf():
    lx = 150_000
    ly = 200_000
    
    resolution = 4000 #m
    
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
    C = C.at[:4, :].set(1e16)
    C = C.at[:, :4].set(1e16)
    C = C.at[-4:,:].set(1e16)
    C = jnp.where(thk==0, 1, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q


def tiny_ice_shelf():
    lx = 150_0
    ly = 150_0
    resolution = 50_0 #m
    
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
    thk = thk.at[:, -1:].set(0)
    
    b = jnp.zeros_like(thk)-600
    
    mucoef = jnp.ones_like(thk)
    
    C = jnp.zeros_like(thk)
    C = C.at[:1, :].set(1e8)
    C = C.at[:, :1].set(1e8)
    C = C.at[-1:,:].set(1e8)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q


#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = tiny_ice_shelf()
lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = ice_shelf()


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10


timestep = 1 #year



def expl_ts():
    vel_solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             C,
                                                             n_iterations)
    
    thickness_update = make_advect_scalar_field_fou_function(nc, nr, 
                                                             delta_x, delta_y,
                                                             vel_bcs="rflc")
    
    
    
    ui = u_init.copy()
    vi = v_init.copy()
    hi = thk.copy()
    for i in range(10):
        ui, vi = vel_solver(q, ui, vi, hi)
        #show_vel_field(u_out, v_out)
        hi = thickness_update(ui, vi, hi, 0, timestep, hi)
    #plt.plot(hi[10,:])
    #plt.show()
    plt.imshow(hi, vmin=0, vmax=500)
    plt.show()


def impl_ts():
    uvh_solver = make_newton_coupled_solver_function(nr, nc,
                                                     delta_y,
                                                     delta_x,
                                                     C,
                                                     n_iterations)
    
    vel_solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             C,
                                                             n_iterations)
   
    ui = u_init.copy()
    vi = v_init.copy()

    ui, vi = vel_solver(q, ui, vi, thk)
    #show_vel_field(ui,vi)

    hi = thk.copy()

    for i in range(10):
        ui, vi, hi = uvh_solver(q, ui, vi, hi, 0, timestep)

        show_vel_field(ui,vi)

        plt.imshow(hi, vmin=0, vmax=500)
        plt.show()

expl_ts()
#impl_ts()










