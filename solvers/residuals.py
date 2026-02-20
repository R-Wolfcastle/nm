#1st party
from pathlib import Path
import sys
import time
from functools import partial

##local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants_years as c
from grid import *
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


def compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_s_ghost_cells,\
                                          extrapolate_over_cf):

    def compute_linear_ssa_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)

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
        u = extrapolate_over_cf(u)
        v = extrapolate_over_cf(v)

        u, v = add_uv_ghost_cells(u, v)
        #don't need to exrapolate over cf as mu on those faces is set to zero
        #to prevent momentum flux out of the cell
        #NOTE: THIS WAS TOTAL BULLSHIT!!!! YOU DEFINITELY NEEDED IT!!!
        #You need it because it is used to calculate the gradients in the direction
        #perp to the CF on the faces perp to the CF. If you don't include it, you get
        #weird high-frequency noise in the gradients parallel to the CF.


        #Ok, it's kind of nuts though. If I allow extrapolation over cf, then this is
        #NO LONGER A LINEAR FUNCTION OF U.



        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        
        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        u = u[1:-1,1:-1]
        v = v[1:-1,1:-1]
        u = u*ice_mask
        v = v*ice_mask


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
                                          h_1d, b, beta,\
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


def make_fo_upwind_residual_function(nx, ny, dx, dy, vel_bcs="rflc"):
    if vel_bcs!="rflc":
        raise "Add the 2 lines of code to enable periodic bcs you lazy bastard, Trys."

    cc_to_fc = interp_cc_to_fc_function(ny, nx)
    add_reflection_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)

    def fo_upwind_residual_function(u, v, scalar_field, scalar_field_old, source, delta_t):

        u, v = add_reflection_ghost_cells(u,v)
        h = add_cont_ghost_cells(scalar_field)
        
        u_fc_ew, _ = cc_to_fc(u)
        _, v_fc_ns = cc_to_fc(v)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)

        h_fc_fou_ew = jnp.where(u_fc_ew>0, h[1:-1,:-1], h[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ew>0, h[1:, 1:-1], h[-1:,1:-1])

        volume_term = ( (scalar_field - scalar_field_old)/delta_t - source ) * dx * dy
        x_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy
        y_term = (u_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - u_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx


        #To stop the calving front advancing, need to ensure no flux into ice-free cells
        #might as well kill all components there tbh...
        return jnp.where(thk>1e-2, volume_term - x_term - y_term, 0)
    return fo_upwind_residual_function


def compute_ssa_uv_residuals_function_pnotC(ny, nx, dy, dx, b,
                                   beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0, C_0):
    
    def compute_uv_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)

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


        beta = beta_fct(C, u, v, h)

        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx


        #obvs not going to do anything in the no-cf case
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
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

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #jax.debug.print("mucoef_ew = {x}",x=mucoef_ew)

        #calculate face-centred viscosity:
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(ice_mask==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(ice_mask==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(ice_mask==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(ice_mask==0, 0, mu_ns[:-1,:]))
        

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

    return jax.jit(compute_uv_residuals)

def compute_ssa_uv_residuals_function(ny, nx, dy, dx, b,
                                   beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0):
    
    def compute_uv_residuals(u_1d, v_1d, q, C, h_1d):

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


        beta = beta_fct(C, u, v, h)

        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx


        #obvs not going to do anything in the no-cf case
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
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

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #jax.debug.print("mucoef_ew = {x}",x=mucoef_ew)

        #calculate face-centred viscosity:
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(ice_mask==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(ice_mask==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(ice_mask==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(ice_mask==0, 0, mu_ns[:-1,:]))
        

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

    return jax.jit(compute_uv_residuals)


def compute_uv_residuals_function_dynamic_thk_anisotropic(ny, nx, dy, dx,
                                   b, beta,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0):
    
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





        
#        trace_sr_ew = dudx_ew + dvdy_ew
#        trace_sr_ns = dudx_ns + dvdy_ns
#
#        isotropic_sr_ew = jnp.zeros((2,2,dudx_ew.shape[0], dudx_ew.shape[1]))
#        isotropic_sr_ew = isotropic_sr_ew.at[0,0,:,:].set(trace_sr_ew)
#        isotropic_sr_ew = isotropic_sr_ew.at[1,1,:,:].set(trace_sr_ew)
#        isotropic_sr_ns = jnp.zeros((2,2,dudx_ns.shape[0], dudx_ns.shape[1]))
#        isotropic_sr_ns = isotropic_sr_ns.at[0,0,:,:].set(trace_sr_ns)
#        isotropic_sr_ns = isotropic_sr_ns.at[1,1,:,:].set(trace_sr_ns)
#
#        shear_sr_ew = jnp.zeros_like(isotropic_sr_ew)
#        shear_sr_ew = shear_sr_ew.at[0,0,:,:].set(-dvdy_ew)
#        shear_sr_ew = shear_sr_ew.at[0,1,:,:].set(0.5*(dudy_ew + dvdx_ew))
#        shear_sr_ew = shear_sr_ew.at[1,0,:,:].set(0.5*(dudy_ew + dvdx_ew))
#        shear_sr_ew = shear_sr_ew.at[1,1,:,:].set(-dudx_ew)
#
#        shear_sr_ns = jnp.zeros_like(isotropic_sr_ns)
#        shear_sr_ns = shear_sr_ns.at[0,0,:,:].set(-dvdy_ns)
#        shear_sr_ns = shear_sr_ns.at[0,1,:,:].set(0.5*(dudy_ns + dvdx_ns))
#        shear_sr_ns = shear_sr_ns.at[1,0,:,:].set(0.5*(dudy_ns + dvdx_ns))
#        shear_sr_ns = shear_sr_ns.at[1,1,:,:].set(-dudx_ns)
#
#        #shear_sr_ew = shear_sr_ew + 1e-10
#        #shear_sr_ns = shear_sr_ns + 1e-10
#        #isotropic_sr_ew = isotropic_sr_ew + 1e-10
#        #isotropic_sr_ns = isotropic_sr_ns + 1e-10
#
#        
#        
#        #calculate face-centred viscosity:
#        mu_ew_iso = c.B_COLD * mucoef_ew * (isotropic_sr_ew[0,0,:,:]**2 + isotropic_sr_ew[1,1,:,:]**2 + isotropic_sr_ew[0,0,:,:]*isotropic_sr_ew[1,1,:,:] +\
#                    0.25*(isotropic_sr_ew[0,1,:,:]+isotropic_sr_ew[1,0,:,:])**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#        mu_ns_iso = c.B_COLD * mucoef_ns * (isotropic_sr_ns[0,0,:,:]**2 + isotropic_sr_ns[1,1,:,:]**2 + isotropic_sr_ns[0,0,:,:]*isotropic_sr_ns[1,1,:,:] +\
#                    0.25*(isotropic_sr_ns[0,1,:,:]+isotropic_sr_ns[1,0,:,:])**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#        
#        mu_ew_s = c.B_COLD * mucoef_ew * (shear_sr_ew[0,0,:,:]**2 + shear_sr_ew[1,1,:,:]**2 + shear_sr_ew[0,0,:,:]*shear_sr_ew[1,1,:,:] +\
#                    0.25*(shear_sr_ew[0,1,:,:]+shear_sr_ew[1,0,:,:])**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#        mu_ns_s = c.B_COLD * mucoef_ns * (shear_sr_ns[0,0,:,:]**2 + shear_sr_ns[1,1,:,:]**2 + shear_sr_ns[0,0,:,:]*shear_sr_ns[1,1,:,:] +\
#                    0.25*(shear_sr_ns[0,1,:,:]+shear_sr_ns[1,0,:,:])**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#
#
#        
#        #to account for calving front boundary condition, set effective viscosities
#        #of faces of all cells with zero thickness to zero:
#        #Again, shouldn't do owt when there's no calving front
#        mu_ew_iso = mu_ew_iso.at[:, 1:].set(jnp.where(h==0, 0, mu_ew_iso[:, 1:]))
#        mu_ew_iso = mu_ew_iso.at[:,:-1].set(jnp.where(h==0, 0, mu_ew_iso[:,:-1]))
#        mu_ns_iso = mu_ns_iso.at[1:, :].set(jnp.where(h==0, 0, mu_ns_iso[1:, :]))
#        mu_ns_iso = mu_ns_iso.at[:-1,:].set(jnp.where(h==0, 0, mu_ns_iso[:-1,:]))
#
#        mu_ew_s = mu_ew_s.at[:, 1:].set(jnp.where(h==0, 0, mu_ew_s[:, 1:]))
#        mu_ew_s = mu_ew_s.at[:,:-1].set(jnp.where(h==0, 0, mu_ew_s[:,:-1]))
#        mu_ns_s = mu_ns_s.at[1:, :].set(jnp.where(h==0, 0, mu_ns_s[1:, :]))
#        mu_ns_s = mu_ns_s.at[:-1,:].set(jnp.where(h==0, 0, mu_ns_s[:-1,:]))
#
#
#
#        phi_i = 1
#        phi_s = 0
#
#
#        htau_ew = 2 * h_ew * ( phi_i * mu_ew_iso * isotropic_sr_ew + phi_s * mu_ew_s * shear_sr_ew )
#        htau_ns = 2 * h_ns * ( phi_i * mu_ns_iso * isotropic_sr_ns + phi_s * mu_ns_s * shear_sr_ns )
#
#
#        rst_ew = jnp.zeros_like(htau_ew)
#        rst_ew = rst_ew.at[0,0,:,:].set(2*htau_ew[0,0,:,:]+htau_ew[1,1,:,:])
#        rst_ew = rst_ew.at[1,1,:,:].set(2*htau_ew[1,1,:,:]+htau_ew[0,0,:,:])
#        rst_ew = rst_ew.at[1,0,:,:].set(htau_ew[1,0,:,:])
#        rst_ew = rst_ew.at[0,1,:,:].set(htau_ew[0,1,:,:])
#
#        rst_ns = jnp.zeros_like(htau_ns)
#        rst_ns = rst_ns.at[0,0,:,:].set(2*htau_ns[0,0,:,:]+htau_ns[1,1,:,:])
#        rst_ns = rst_ns.at[1,1,:,:].set(2*htau_ns[1,1,:,:]+htau_ns[0,0,:,:])
#        rst_ns = rst_ns.at[1,0,:,:].set(htau_ns[1,0,:,:])
#        rst_ns = rst_ns.at[0,1,:,:].set(htau_ns[0,1,:,:])
#
#
#
#        visc_x = rst_ew[0,0,:, 1:] * dy -\
#                 rst_ew[0,0,:,:-1] * dy +\
#                 rst_ns[0,1,:-1,:] * dx -\
#                 rst_ns[0,1,1:, :] * dx
#        
#        visc_y = rst_ns[1,1,:-1,:] * dx -\
#                 rst_ns[1,1,1:, :] * dx +\
#                 rst_ew[1,0,:, 1:] * dy -\
#                 rst_ew[1,0,:,:-1] * dy







        ##############################################################
        #Another dodgy method of splitting things up

#        #calculate face-centred viscosity:
#        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
#                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
#                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
#        
#        #to account for calving front boundary condition, set effective viscosities
#        #of faces of all cells with zero thickness to zero:
#        #Again, shouldn't do owt when there's no calving front
#        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
#        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
#        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
#        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))
#      
#
#
#
#
#
#        trace_sr_ew = dudx_ew + dvdy_ew
#        trace_sr_ns = dudx_ns + dvdy_ns
#
#        isotropic_sr_ew = jnp.zeros((2,2,dudx_ew.shape[0], dudx_ew.shape[1]))
#        isotropic_sr_ew = isotropic_sr_ew.at[0,0,:,:].set(trace_sr_ew)
#        isotropic_sr_ew = isotropic_sr_ew.at[1,1,:,:].set(trace_sr_ew)
#        isotropic_sr_ns = jnp.zeros((2,2,dudx_ns.shape[0], dudx_ns.shape[1]))
#        isotropic_sr_ns = isotropic_sr_ns.at[0,0,:,:].set(trace_sr_ns)
#        isotropic_sr_ns = isotropic_sr_ns.at[1,1,:,:].set(trace_sr_ns)
#
#        shear_sr_ew = jnp.zeros_like(isotropic_sr_ew)
#        shear_sr_ew = shear_sr_ew.at[0,0,:,:].set(-dvdy_ew)
#        shear_sr_ew = shear_sr_ew.at[0,1,:,:].set(0.5*(dudy_ew + dvdx_ew))
#        shear_sr_ew = shear_sr_ew.at[1,0,:,:].set(0.5*(dudy_ew + dvdx_ew))
#        shear_sr_ew = shear_sr_ew.at[1,1,:,:].set(-dudx_ew)
#
#        shear_sr_ns = jnp.zeros_like(isotropic_sr_ns)
#        shear_sr_ns = shear_sr_ns.at[0,0,:,:].set(-dvdy_ns)
#        shear_sr_ns = shear_sr_ns.at[0,1,:,:].set(0.5*(dudy_ns + dvdx_ns))
#        shear_sr_ns = shear_sr_ns.at[1,0,:,:].set(0.5*(dudy_ns + dvdx_ns))
#        shear_sr_ns = shear_sr_ns.at[1,1,:,:].set(-dudx_ns)
#        
#
#        phi_i = 1
#        phi_s = 0
#
#
#        htau_ew = 2 * h_ew * mu_ew * ( phi_i * isotropic_sr_ew + phi_s * shear_sr_ew )
#        htau_ns = 2 * h_ns * mu_ns * ( phi_i * isotropic_sr_ns + phi_s * shear_sr_ns )
#
#
#        rst_ew = jnp.zeros_like(htau_ew)
#        rst_ew = rst_ew.at[0,0,:,:].set(2*htau_ew[0,0,:,:]+htau_ew[1,1,:,:])
#        rst_ew = rst_ew.at[1,1,:,:].set(2*htau_ew[1,1,:,:]+htau_ew[0,0,:,:])
#        rst_ew = rst_ew.at[1,0,:,:].set(htau_ew[1,0,:,:])
#        rst_ew = rst_ew.at[0,1,:,:].set(htau_ew[0,1,:,:])
#
#        rst_ns = jnp.zeros_like(htau_ns)
#        rst_ns = rst_ns.at[0,0,:,:].set(2*htau_ns[0,0,:,:]+htau_ns[1,1,:,:])
#        rst_ns = rst_ns.at[1,1,:,:].set(2*htau_ns[1,1,:,:]+htau_ns[0,0,:,:])
#        rst_ns = rst_ns.at[1,0,:,:].set(htau_ns[1,0,:,:])
#        rst_ns = rst_ns.at[0,1,:,:].set(htau_ns[0,1,:,:])
#
#
#
#        visc_x = rst_ew[0,0,:, 1:] * dy -\
#                 rst_ew[0,0,:,:-1] * dy +\
#                 rst_ns[0,1,:-1,:] * dx -\
#                 rst_ns[0,1,1:, :] * dx
#        
#        visc_y = rst_ns[1,1,:-1,:] * dx -\
#                 rst_ns[1,1,1:, :] * dx +\
#                 rst_ew[1,0,:, 1:] * dy -\
#                 rst_ew[1,0,:,:-1] * dy

        ##############################################################
        

        

        #################################
        #Sort of dodgy just-apply-to-resistive-stress method
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        trace_resistive_st_ew = 3 * mu_ew * h_ew * (dudx_ew + dvdy_ew)
        trace_resistive_st_ns = 3 * mu_ns * h_ns * (dudx_ns + dvdy_ns)

        
        phi_n = jnp.ones_like(mucoef)
        phi_n = phi_n.at[10:20, :].set(1.9)
        phi_n = phi_n.at[(-20):(-10), :].set(1.9)
        
        phi_s = jnp.ones_like(mucoef)
        phi_s = phi_s.at[10:20, :].set(0.1)
        phi_s = phi_s.at[(-20):(-10), :].set(0.1)

        phi_n_ew, phi_n_ns = interp_cc_to_fc(phi_n)
        phi_s_ew, phi_s_ns = interp_cc_to_fc(phi_s)


        visc_x = ((phi_n_ew - phi_s_ew)[:, 1:] * trace_resistive_st_ew[:, 1:] -\
                  (phi_n_ew - phi_s_ew)[:, :-1] * trace_resistive_st_ew[:, :-1] )*dy +\
                 phi_s_ew[:, 1:]*mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 phi_s_ew[:,:-1]*mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 phi_s_ns[:-1,:]*mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 phi_s_ns[1:, :]*mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx

        #visc_y = (phi_n_ew - phi_s_ew)*( trace_resistive_st_ns[:-1, :] - trace_resistive_st_ns[1:, :] )*dx +\
        visc_y = ((phi_n_ns - phi_s_ns)[:-1,:] * trace_resistive_st_ns[:-1,:] -\
                  (phi_n_ns - phi_s_ns)[1:, :] * trace_resistive_st_ns[1:, :] )*dx +\
                 phi_s_ew[:, 1:]*mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 phi_s_ew[:,:-1]*mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 phi_s_ns[:-1,:]*mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 phi_s_ns[1:, :]*mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx

        ################################

        x_mom_residual = 2 * visc_x + volume_x
        y_mom_residual = 2 * visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals)

def compute_u_v_residuals_function_dynamic_thk(ny, nx, dy, dx, \
                                   b, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0):
    
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
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
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
                                   b,\
                                   h_1d, beta,\
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0):
    
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

        #speed = jnp.sqrt(u**2 + v**2 + 1e-16)

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
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
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


#This only differs from what's below by the fact that beta is calculated inside
def compute_uvh_residuals_function_fully_nonlinear(ny, nx, dy, dx,
                                   b, beta_function, 
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells, mucoef_0):
    

    def compute_uvh_residuals(u_1d, v_1d, h_1d, q, C, h_t, source, delta_t):

        h_static = add_s_ghost_cells(h_t)

        #print(h_static)
        
        mucoef = mucoef_0*jnp.exp(q)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        #h_t = h_t.reshape((ny, nx))


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

        #NOTE: big question here!!! Do we use h_static and ignore subgrid
        #grounding line effects..?? Maybe have to use h
        #beta = beta_function(C, u, v, h)
        beta = beta_function(C, u, v, h_static)
        #I suppose it depends on what we're using this for... If it's only
        #for computing gradients, then perhaps it doesn't matter much either
        #way?


        volume_x = - (beta * u + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * h * dsdy) * dy * dx

        #momentum_term
        u_ghost, v_ghost = add_uv_ghost_cells(u, v)
        u_full = linear_extrapolate_over_cf_dynamic_thickness(u_ghost, h_static)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(v_ghost, h_static)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u_full)
        dvdx_ew, dvdy_ew = ew_gradient(v_full)
        dudx_ns, dudy_ns = ns_gradient(u_full)
        dvdx_ns, dvdy_ns = ns_gradient(v_full)

        #jax.debug.print("dudxew = {x}",x=dudx_ew)

        #interpolate things onto face-centres
        h_ghost = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_ghost)
        

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #jax.debug.print("mucoef_ew = {x}",x=mucoef_ew)

        #calculate face-centred viscosity:
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h_t<1e-2, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h_t<1e-2, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h_t<1e-2, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h_t<1e-2, 0, mu_ns[:-1,:]))
        

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


        hh = linear_extrapolate_over_cf_dynamic_thickness(h_ghost, h_static)

        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        ##face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, hh[1:-1,:-1], hh[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, hh[1:, 1:-1], hh[-1:,1:-1])



        flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t
        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
        flux_term = jnp.where(h_t>1e-2, flux_term, 0)


        adv_residual =  ((h - h_t) - source * delta_t)*dx*dy + flux_term

        ##################################################################



        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1), adv_residual.reshape(-1)

    return compute_uvh_residuals


def compute_uvh_residuals_function(ny, nx, dy, dx,
                                   b, beta,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells, mucoef_0):
    

    def compute_uvh_residuals(u_1d, v_1d, h_1d, q, h_t, source, delta_t):

        h_static = add_s_ghost_cells(h_t)

        #print(h_static)
        
        mucoef = mucoef_0*jnp.exp(q)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        #h_t = h_t.reshape((ny, nx))


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

        #momentum_term
        u_ghost, v_ghost = add_uv_ghost_cells(u, v)
        u_full = linear_extrapolate_over_cf_dynamic_thickness(u_ghost, h_static)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(v_ghost, h_static)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u_full)
        dvdx_ew, dvdy_ew = ew_gradient(v_full)
        dudx_ns, dudy_ns = ns_gradient(u_full)
        dvdx_ns, dvdy_ns = ns_gradient(v_full)

        #jax.debug.print("dudxew = {x}",x=dudx_ew)

        #interpolate things onto face-centres
        h_ghost = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_ghost)
        

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #jax.debug.print("mucoef_ew = {x}",x=mucoef_ew)

        #calculate face-centred viscosity:
        mu_ew = c.B_COLD * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = c.B_COLD * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h_t<1e-2, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h_t<1e-2, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h_t<1e-2, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h_t<1e-2, 0, mu_ns[:-1,:]))
        

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


        hh = linear_extrapolate_over_cf_dynamic_thickness(h_ghost, h_static)

        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        ##face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, hh[1:-1,:-1], hh[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, hh[1:, 1:-1], hh[-1:,1:-1])



        flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t
        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
        flux_term = jnp.where(h_t>1e-2, flux_term, 0)


        adv_residual =  ((h - h_t) - source * delta_t)*dx*dy + flux_term

        ##################################################################



        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1), adv_residual.reshape(-1)

    return compute_uvh_residuals




def compute_uvh_linear_ssa_residuals_function(ny, nx, dy, dx, b,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_s_ghost_cells,\
                                          extrapolate_over_cf):

    def compute_uvh_residuals(u_1d, v_1d, h_1d,
                              mu_ew, mu_ns, beta,
                              h_t, source, delta_t):

        h_static = add_s_ghost_cells(h_t)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        


        ######### MOMENTUM RESIDUALS #############################
        s_gnd = h + b
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
        u = extrapolate_over_cf(u)
        v = extrapolate_over_cf(v)
        u_full, v_full = add_uv_ghost_cells(u, v)

        #get thickness on the faces
        h_extrp = extrapolate_over_cf(h)
        h_full = add_s_ghost_cells(h_extrp)
        h_ew, h_ns = interp_cc_to_fc(h_full)
        
        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u_full)
        dvdx_ew, dvdy_ew = ew_gradient(v_full)
        dudx_ns, dudy_ns = ns_gradient(u_full)
        dvdx_ns, dvdy_ns = ns_gradient(v_full)

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

        


        ################ ADVECTION RESIDUALS #############################


        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        ##face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, h_full[1:-1,:-1], h_full[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h_full[1:, 1:-1], h_full[-1:,1:-1])



        flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t
        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
        flux_term = jnp.where(h_t>1e-2, flux_term, 0)


        adv_residual =  ((h - h_t) - source * delta_t)*dx*dy + flux_term

        ##################################################################



        return (x_mom_residual.reshape(-1),
                y_mom_residual.reshape(-1),
                adv_residual.reshape(-1))

    return jax.jit(compute_uvh_residuals)




