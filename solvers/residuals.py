#1st party
from pathlib import Path
import os
import sys
import time
from functools import partial

##local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
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
from thermo import B_from_T

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

        #u = u[1:-1,1:-1]
        #v = v[1:-1,1:-1]
        #u = u*ice_mask
        #v = v*ice_mask


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

def compute_linear_ssa_residuals_function_fc_visc_gl_aware(ny, nx, dy, dx, b,\
                                          interp_cc_to_fc,
                                          ew_gradient,\
                                          ns_gradient,\
                                          cc_gradient,\
                                          add_uv_ghost_cells,\
                                          add_s_ghost_cells,\
                                          extrapolate_over_cf,\
                                          hgrads_fct):

    def compute_linear_ssa_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        #s = add_s_ghost_cells(s)
        #jax.debug.print("s: {x}",x=s)

        hdsdx, hdsdy = hgrads_fct(s, h, (s_gnd>s_flt).astype(int))

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx



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

        #u = u[1:-1,1:-1]
        #v = v[1:-1,1:-1]
        #u = u*ice_mask
        #v = v*ice_mask


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

def compute_nonlinear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc):
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask = 

    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)


        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
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





        ########### momentum_term ###########
        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

        
        #various face-centred derivatives
        dudx_ew, dudy_ew,\
        dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns,\
        dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)
        
        #calculate face-centred viscosity:
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        #Again, shouldn't do owt when there's no calving front
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(ice_mask==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(ice_mask==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(ice_mask==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(ice_mask==0, 0, mu_ns[:-1,:]))

       
        
        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        #calculate node-centred viscosity:
        mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
                    nc_ice_mask



        mu_ew = mu_ew * h_ew
        mu_ns = mu_ns * h_ns
        mu_nc = mu_nc * h_nc

        mu_e  = mu_ew[:,  1:]
        mu_w  = mu_ew[:, :-1]
        mu_n  = mu_ns[:-1, :]
        mu_s  = mu_ns[1:,  :]
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        u_padded, v_padded = add_uv_ghost_cells(u, v)
        u_P  = u_padded[1:-1, 1:-1]
        u_N  = u_padded[:-2,  1:-1]
        u_S  = u_padded[2:,   1:-1]
        u_W  = u_padded[1:-1,  :-2]
        u_E  = u_padded[1:-1,   2:]
        u_NW = u_padded[:-2,   :-2]
        u_NE = u_padded[:-2,    2:]
        u_SW = u_padded[2:,    :-2]
        u_SE = u_padded[2:,     2:]
        v_P  = v_padded[1:-1, 1:-1]
        v_N  = v_padded[:-2,  1:-1]
        v_S  = v_padded[2:,   1:-1]
        v_W  = v_padded[1:-1,  :-2]
        v_E  = v_padded[1:-1,   2:]
        v_NW = v_padded[:-2,   :-2]
        v_NE = v_padded[:-2,    2:]
        v_SW = v_padded[2:,    :-2]
        v_SE = v_padded[2:,     2:]
       
        visc_x =     2 * ( mu_e*(u_E - u_P) - mu_w*(u_P - u_W) ) +\
                   0.5 * ( mu_n*(u_N - u_P) - mu_s*(u_P - u_S) ) +\
                 0.125 * ( mu_se*(3*v_P + v_E - v_S - 3*v_SE) +\
                           mu_ne*(3*v_NE - v_E + v_N - 3*v_P) +\
                           mu_sw*(3*v_SW - v_W + v_S - 3*v_P) +\
                           mu_nw*(3*v_P - v_N + v_W - 3*v_NW)
                         )
        
        visc_y =     2 * ( mu_n*(v_N - v_P) - mu_s*(v_P - v_S) ) +\
                   0.5 * ( mu_e*(v_E - v_P) - mu_w*(v_P - v_W) ) +\
                 0.125 * ( mu_se*(3*u_P + u_S - u_E - 3*u_SE) +\
                           mu_ne*(3*u_NE - u_N + u_E - 3*u_P) +\
                           mu_sw*(3*u_SW - u_S + u_W - 3*u_P) +\
                           mu_nw*(3*u_P - u_W + u_N - 3*u_NW)
                         )

        x_mom_residual = 2*visc_x + volume_x
        y_mom_residual = 2*visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)


def compute_linear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          fc_vel_gradient,
                                          cc_gradient,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells):

    def compute_linear_ssa_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)

        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
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

        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
        
        mu_ew = mu_ew * h_ew
        mu_ns = mu_ns * h_ns
        mu_nc = mu_nc * h_nc

        mu_e  = mu_ew[:,  1:]
        mu_w  = mu_ew[:, :-1]
        mu_n  = mu_ns[:-1, :]
        mu_s  = mu_ns[1:,  :]
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        u_padded, v_padded = add_uv_ghost_cells(u, v)
        u_P  = u_padded[1:-1, 1:-1]
        u_N  = u_padded[:-2,  1:-1]
        u_S  = u_padded[2:,   1:-1]
        u_W  = u_padded[1:-1,  :-2]
        u_E  = u_padded[1:-1,   2:]
        u_NW = u_padded[:-2,   :-2]
        u_NE = u_padded[:-2,    2:]
        u_SW = u_padded[2:,    :-2]
        u_SE = u_padded[2:,     2:]
        v_P  = v_padded[1:-1, 1:-1]
        v_N  = v_padded[:-2,  1:-1]
        v_S  = v_padded[2:,   1:-1]
        v_W  = v_padded[1:-1,  :-2]
        v_E  = v_padded[1:-1,   2:]
        v_NW = v_padded[:-2,   :-2]
        v_NE = v_padded[:-2,    2:]
        v_SW = v_padded[2:,    :-2]
        v_SE = v_padded[2:,     2:]
       
        visc_x =     2 * ( mu_e*(u_E - u_P) - mu_w*(u_P - u_W) ) +\
                   0.5 * ( mu_n*(u_N - u_P) - mu_s*(u_P - u_S) ) +\
                 0.125 * ( mu_se*(3*v_P + v_E - v_S - 3*v_SE) +\
                           mu_ne*(3*v_NE - v_E + v_N - 3*v_P) +\
                           mu_sw*(3*v_SW - v_W + v_S - 3*v_P) +\
                           mu_nw*(3*v_P - v_N + v_W - 3*v_NW)
                         )
        
        visc_y =     2 * ( mu_n*(v_N - v_P) - mu_s*(v_P - v_S) ) +\
                   0.5 * ( mu_e*(v_E - v_P) - mu_w*(v_P - v_W) ) +\
                 0.125 * ( mu_se*(3*u_P + u_S - u_E - 3*u_SE) +\
                           mu_ne*(3*u_NE - u_N + u_E - 3*u_P) +\
                           mu_sw*(3*u_SW - u_S + u_W - 3*u_P) +\
                           mu_nw*(3*u_P - u_W + u_N - 3*u_NW)
                         )

        x_mom_residual = 2*visc_x + volume_x
        y_mom_residual = 2*visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_linear_ssa_residuals)



def laplacian_stabilization_residual(
        u, v,
        eps,
        dx, dy,
        add_uv_ghost_cells):

    u_g, v_g = add_uv_ghost_cells(u, v)

    # symmetric 5-point Laplacian
    lap_u = (
          u_g[1:-1,2:]
        + u_g[1:-1,:-2]
        + u_g[2:,1:-1]
        + u_g[:-2,1:-1]
        - 4.0*u
    ) / dx**2

    lap_v = (
          v_g[1:-1,2:]
        + v_g[1:-1,:-2]
        + v_g[2:,1:-1]
        + v_g[:-2,1:-1]
        - 4.0*v
    ) / dy**2

    # residual contribution = -eps * Laplacian
    stab_x = -eps * lap_u * dx * dy
    stab_y = -eps * lap_v * dx * dy

    return stab_x, stab_y

def compute_nonlinear_ssa_residuals_function_variational_visc_an_option(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask =


    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        
        #calving_front = jnp.zeros_like(ice_mask)
        #calving_front = calving_front.at[:,-3].set(1)
        #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))

        #calving_front = jnp.where(
        #    jnp.concatenate([jnp.zeros((ny,1)), jnp.where(h[:,1:]>0, 0, 1)], axis=1) > 0,
        #    1, 0
        #)  # shape (ny, nx)

        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #C = add_s_ghost_cells(C)
        #beta_linear_nc = interp_cc_to_nc(C)
        #
        #u_g, v_g = add_uv_ghost_cells(u, v)
        #beta_u_nc = beta_linear_nc * interp_cc_to_nc(u_g)
        #beta_v_nc = beta_linear_nc * interp_cc_to_nc(v_g)

        #linear_sliding_term_x = 0.25 * (beta_u_nc[:-1, :-1] + beta_u_nc[1:, :-1] + beta_u_nc[:-1, 1:] + beta_u_nc[1:, 1:]) * ice_mask + u * (1-ice_mask)
        #linear_sliding_term_y = 0.25 * (beta_v_nc[:-1, :-1] + beta_v_nc[1:, :-1] + beta_v_nc[:-1, 1:] + beta_v_nc[1:, 1:]) * ice_mask + v * (1-ice_mask)


        #jax.debug.print("{x}", x=(h.shape, dsdx.shape, ice_mask.shape))

        volume_x = -(beta * u - c.RHO_I * c.g * h * dsdx * ice_mask) * dx * dy
        volume_y = -(beta * v - c.RHO_I * c.g * h * dsdy * ice_mask) * dy * dx
        #volume_x = -(linear_sliding_term_x - 1*c.RHO_I * c.g * h * dsdx) * dx * dy
        #volume_y = -(linear_sliding_term_y - 1*c.RHO_I * c.g * h * dsdy) * dy * dx


        #jax.debug.print("{x}", x=jnp.count_nonzero(calving_front))
        #terminus_stress = 0.5 * c.RHO_I * c.g * h**2 * (1 - c.RHO_W/c.RHO_I)  # shape (ny, nx)
        
        #jax.debug.print("{x}", x=jnp.sum(calving_front * terminus_stress * dy))
        
        #volume_x = volume_x + calving_front * terminus_stress * dy


        ############ momentum_term ###########
        #get thickness on the faces
        #h = add_s_ghost_cells(extrap_over_cf(h))
        #NOTE: for some reason, it prefers the version without extrapolation at the cf...
        h = add_s_ghost_cells(h)
        #h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        #mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

       
        #u = extrap_over_cf(u)
        #v = extrap_over_cf(v)
        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        #calculate node-centred viscosity:
        mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
                    nc_ice_mask

        #jax.debug.print("{x}", x=jnp.max(dudx_nc))
        mu_nc = mu_nc * h_nc
        
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        dudx_nw = dudx_nc[:-1, :-1]
        dudx_ne = dudx_nc[:-1,  1:]
        dudx_se = dudx_nc[1:,   1:]
        dudx_sw = dudx_nc[1:,  :-1]
        
        dvdx_nw = dvdx_nc[:-1, :-1]
        dvdx_ne = dvdx_nc[:-1,  1:]
        dvdx_se = dvdx_nc[1:,   1:]
        dvdx_sw = dvdx_nc[1:,  :-1]

        dudy_nw = dudy_nc[:-1, :-1]
        dudy_ne = dudy_nc[:-1,  1:]
        dudy_se = dudy_nc[1:,   1:]
        dudy_sw = dudy_nc[1:,  :-1]

        dvdy_nw = dvdy_nc[:-1, :-1]
        dvdy_ne = dvdy_nc[:-1,  1:]
        dvdy_se = dvdy_nc[1:,   1:]
        dvdy_sw = dvdy_nc[1:,  :-1]


        #NOTE: Those factors of 0.5 might be wrong.... CCHECK! NOTE: Checked. They're right.
        #NOTE: This only works if dx=dy. Otherwise, faff around with some factors of dx and dy.
        visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx
        
        visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx

        
        #visc_x = 0.5 * dy * visc_x
        #visc_y = 0.5 * dx * visc_y

        stab_x, stab_y = laplacian_stabilization_residual(
            u, v,
            1e10,
            dx, dy,
            add_uv_ghost_cells
        )

        #jax.debug.print("{x}", x=(jnp.max(visc_x), jnp.max(volume_x)))
        
        jax.debug.print("{x}", x=jnp.sum(visc_x*ice_mask))
        #jax.debug.print("{x}", x=jnp.sum(c.RHO_I * c.g * h * dsdx * dx * dy * ice_mask))
        
        x_mom_residual = visc_x*ice_mask + volume_x #+ #stab_x*ice_mask
        y_mom_residual = visc_y*ice_mask + volume_y #+ stab_y*ice_mask

        #x_mom_residual = visc_x*ice_mask + volume_x
        #y_mom_residual = visc_y*ice_mask + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)

def compute_nonlinear_ssa_residuals_function_variational_visc_diagnosis(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask =


    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        calving_front = jnp.zeros_like(ice_mask)
        calving_front = calving_front.at[:,-3].set(1)
        #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))

        #calving_front = jnp.where(
        #    jnp.concatenate([jnp.zeros((ny,1)), jnp.where(h[:,1:]>0, 0, 1)], axis=1) > 0,
        #    1, 0
        #)  # shape (ny, nx)

        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #C = add_s_ghost_cells(C)
        #beta_linear_nc = interp_cc_to_nc(C)
        #
        #u_g, v_g = add_uv_ghost_cells(u, v)
        #beta_u_nc = beta_linear_nc * interp_cc_to_nc(u_g)
        #beta_v_nc = beta_linear_nc * interp_cc_to_nc(v_g)

        #linear_sliding_term_x = 0.25 * (beta_u_nc[:-1, :-1] + beta_u_nc[1:, :-1] + beta_u_nc[:-1, 1:] + beta_u_nc[1:, 1:]) * ice_mask + u * (1-ice_mask)
        #linear_sliding_term_y = 0.25 * (beta_v_nc[:-1, :-1] + beta_v_nc[1:, :-1] + beta_v_nc[:-1, 1:] + beta_v_nc[1:, 1:]) * ice_mask + v * (1-ice_mask)


        #jax.debug.print("{x}", x=(h.shape, dsdx.shape, ice_mask.shape))

        volume_x = -(beta * u - c.RHO_I * c.g * h * dsdx * ice_mask * 0) * dx * dy
        volume_y = -(beta * v - c.RHO_I * c.g * h * dsdy * ice_mask * 0) * dy * dx
        #volume_x = -(linear_sliding_term_x - 1*c.RHO_I * c.g * h * dsdx) * dx * dy
        #volume_y = -(linear_sliding_term_y - 1*c.RHO_I * c.g * h * dsdy) * dy * dx


        #jax.debug.print("{x}", x=jnp.count_nonzero(calving_front))
        terminus_stress = 0.5 * c.RHO_I * c.g * h**2 * (1 - c.RHO_W/c.RHO_I)  # shape (ny, nx)
        
        jax.debug.print("{x}", x=jnp.sum(calving_front * terminus_stress * dy))
        
        volume_x = volume_x + calving_front * terminus_stress * dy


        ############ momentum_term ###########
        #get thickness on the faces
        h = add_s_ghost_cells(extrap_over_cf(h))
        #h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        #mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

       
        u = extrap_over_cf(u)
        v = extrap_over_cf(v)
        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        #calculate node-centred viscosity:
        mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
                    1#nc_ice_mask

        plt.imshow(mu_nc)
        plt.colorbar()
        plt.show()

        #jax.debug.print("{x}", x=jnp.max(dudx_nc))
        mu_nc = mu_nc * h_nc
        
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        dudx_nw = dudx_nc[:-1, :-1]
        dudx_ne = dudx_nc[:-1,  1:]
        dudx_se = dudx_nc[1:,   1:]
        dudx_sw = dudx_nc[1:,  :-1]
        
        dvdx_nw = dvdx_nc[:-1, :-1]
        dvdx_ne = dvdx_nc[:-1,  1:]
        dvdx_se = dvdx_nc[1:,   1:]
        dvdx_sw = dvdx_nc[1:,  :-1]

        dudy_nw = dudy_nc[:-1, :-1]
        dudy_ne = dudy_nc[:-1,  1:]
        dudy_se = dudy_nc[1:,   1:]
        dudy_sw = dudy_nc[1:,  :-1]

        dvdy_nw = dvdy_nc[:-1, :-1]
        dvdy_ne = dvdy_nc[:-1,  1:]
        dvdy_se = dvdy_nc[1:,   1:]
        dvdy_sw = dvdy_nc[1:,  :-1]


        #NOTE: Those factors of 0.5 might be wrong.... CCHECK! NOTE: Checked. They're right.
        #NOTE: This only works if dx=dy. Otherwise, faff around with some factors of dx and dy.
        visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx
        
        visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx

        
        #visc_x = 0.5 * dy * visc_x
        #visc_y = 0.5 * dx * visc_y

        stab_x, stab_y = laplacian_stabilization_residual(
            u, v,
            1e10,
            dx, dy,
            add_uv_ghost_cells
        )

        #jax.debug.print("{x}", x=(jnp.max(visc_x), jnp.max(volume_x)))
        
        jax.debug.print("{x}", x=jnp.sum(visc_x*ice_mask))
        #jax.debug.print("{x}", x=jnp.sum(c.RHO_I * c.g * h * dsdx * dx * dy * ice_mask))
        
        x_mom_residual = visc_x*ice_mask + volume_x #+ #stab_x*ice_mask
        y_mom_residual = visc_y*ice_mask + volume_y #+ stab_y*ice_mask

        #x_mom_residual = visc_x*ice_mask + volume_x
        #y_mom_residual = visc_y*ice_mask + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return compute_nl_ssa_residuals

def compute_nonlinear_ssa_residuals_function_variational_visc_messing_round_no_cf(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):
    #temp_cc = temp_cc
    B_cc = B_from_T(temp_cc)
    B_cc = add_s_ghost_cells(B_cc)
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        
        s = h + b
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        volume_x = -(beta * u - c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = -(beta * v - c.RHO_I * c.g * h * dsdy) * dy * dx


        ############ momentum_term ###########
        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        mu_nc = B_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))


        #jax.debug.print("{x}", x=jnp.max(dudx_nc))
        mu_nc = mu_nc * h_nc
        
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        dudx_nw = dudx_nc[:-1, :-1]
        dudx_ne = dudx_nc[:-1,  1:]
        dudx_se = dudx_nc[1:,   1:]
        dudx_sw = dudx_nc[1:,  :-1]
        
        dvdx_nw = dvdx_nc[:-1, :-1]
        dvdx_ne = dvdx_nc[:-1,  1:]
        dvdx_se = dvdx_nc[1:,   1:]
        dvdx_sw = dvdx_nc[1:,  :-1]

        dudy_nw = dudy_nc[:-1, :-1]
        dudy_ne = dudy_nc[:-1,  1:]
        dudy_se = dudy_nc[1:,   1:]
        dudy_sw = dudy_nc[1:,  :-1]

        dvdy_nw = dvdy_nc[:-1, :-1]
        dvdy_ne = dvdy_nc[:-1,  1:]
        dvdy_se = dvdy_nc[1:,   1:]
        dvdy_sw = dvdy_nc[1:,  :-1]

        #NOTE: Those factors of 0.5 might be wrong.... CCHECK! NOTE: Checked. They're right.
        #NOTE: This only works if dx=dy. Otherwise, faff around with some factors of dx and dy.
        visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx
        
        visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx

        
        #visc_x = 0.5 * dy * visc_x
        #visc_y = 0.5 * dx * visc_y

        stab_x, stab_y = laplacian_stabilization_residual(
            u, v,
            1e5,
            dx, dy,
            add_uv_ghost_cells
        )

        #jax.debug.print("{x}", x=(jnp.max(visc_x), jnp.max(volume_x)))
        
        jax.debug.print("viscs_term: {x}", x=jnp.sum(visc_x*ice_mask))
        #jax.debug.print("{x}", x=jnp.sum(c.RHO_I * c.g * h * dsdx * dx * dy * ice_mask))
        
        x_mom_residual = visc_x*ice_mask + volume_x #+ stab_x
        y_mom_residual = visc_y*ice_mask + volume_y #+ stab_y

        #x_mom_residual = visc_x*ice_mask + volume_x
        #y_mom_residual = visc_y*ice_mask + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)

def compute_nonlinear_ssa_residuals_function_variational_visc_messing_round(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):
    #temp_cc = temp_cc
    B_cc = B_from_T(temp_cc)
    B_cc = add_s_ghost_cells(extrap_over_cf(B_cc))
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask =


    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        calving_front = jnp.zeros_like(ice_mask)
        calving_front = calving_front.at[:,-3].set(1)
        #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))

        #calving_front = jnp.where(
        #    jnp.concatenate([jnp.zeros((ny,1)), jnp.where(h[:,1:]>0, 0, 1)], axis=1) > 0,
        #    1, 0
        #)  # shape (ny, nx)

        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #C = add_s_ghost_cells(C)
        #beta_linear_nc = interp_cc_to_nc(C)
        #
        #u_g, v_g = add_uv_ghost_cells(u, v)
        #beta_u_nc = beta_linear_nc * interp_cc_to_nc(u_g)
        #beta_v_nc = beta_linear_nc * interp_cc_to_nc(v_g)

        #linear_sliding_term_x = 0.25 * (beta_u_nc[:-1, :-1] + beta_u_nc[1:, :-1] + beta_u_nc[:-1, 1:] + beta_u_nc[1:, 1:]) * ice_mask + u * (1-ice_mask)
        #linear_sliding_term_y = 0.25 * (beta_v_nc[:-1, :-1] + beta_v_nc[1:, :-1] + beta_v_nc[:-1, 1:] + beta_v_nc[1:, 1:]) * ice_mask + v * (1-ice_mask)


        #jax.debug.print("{x}", x=(h.shape, dsdx.shape, ice_mask.shape))

        volume_x = -(beta * u - c.RHO_I * c.g * h * dsdx * ice_mask) * dx * dy
        volume_y = -(beta * v - c.RHO_I * c.g * h * dsdy * ice_mask) * dy * dx
        #volume_x = -(linear_sliding_term_x - 1*c.RHO_I * c.g * h * dsdx) * dx * dy
        #volume_y = -(linear_sliding_term_y - 1*c.RHO_I * c.g * h * dsdy) * dy * dx


        terminus_stress =  - 0.5 * c.RHO_TILDE * c.g * h**2 # shape (ny, nx)
        #jax.debug.print("term term: {x}", x=jnp.sum(calving_front * terminus_stress * dy))
        volume_x = volume_x + calving_front * terminus_stress * dy


        ############ momentum_term ###########
        #get thickness on the faces
        h = add_s_ghost_cells(extrap_over_cf(h))
        #h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        #mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

        #NOTE: turning this off masiively increases size of solution sometimes,
        #but sometimes decreases it.
        u = extrap_over_cf(u)
        v = extrap_over_cf(v)
        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        #calculate node-centred viscosity:
        #mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
        #            0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
        #            1#nc_ice_mask
        mu_nc = B_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
                    nc_ice_mask


        #jax.debug.print("{x}", x=jnp.max(dudx_nc))
        mu_nc = mu_nc * h_nc
        
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        dudx_nw = dudx_nc[:-1, :-1]
        dudx_ne = dudx_nc[:-1,  1:]
        dudx_se = dudx_nc[1:,   1:]
        dudx_sw = dudx_nc[1:,  :-1]
        
        dvdx_nw = dvdx_nc[:-1, :-1]
        dvdx_ne = dvdx_nc[:-1,  1:]
        dvdx_se = dvdx_nc[1:,   1:]
        dvdx_sw = dvdx_nc[1:,  :-1]

        dudy_nw = dudy_nc[:-1, :-1]
        dudy_ne = dudy_nc[:-1,  1:]
        dudy_se = dudy_nc[1:,   1:]
        dudy_sw = dudy_nc[1:,  :-1]

        dvdy_nw = dvdy_nc[:-1, :-1]
        dvdy_ne = dvdy_nc[:-1,  1:]
        dvdy_se = dvdy_nc[1:,   1:]
        dvdy_sw = dvdy_nc[1:,  :-1]



        ####KIND OF A FINITE VOLUME FLAVOUR TO THINGS:

        #au_sw = mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) * 0.5 * dx
        #au_nw = mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) * 0.5 * dx
        #au_ne = mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) * 0.5 * dx
        #au_se = mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) * 0.5 * dx

        #av_sw = mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) * 0.5 * dx
        #av_nw = mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) * 0.5 * dx
        #av_ne = mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) * 0.5 * dx
        #av_se = mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) * 0.5 * dx


        #fu_n = 0.5 * ( au_nw + au_ne )
        #fu_e = 0.5 * ( au_ne + au_se )
        #fu_s = 0.5 * ( au_sw + au_se )
        #fu_w = 0.5 * ( au_nw + au_sw )

        #fv_n = 0.5 * ( av_nw + av_ne )
        #fv_e = 0.5 * ( av_ne + av_se )
        #fv_s = 0.5 * ( av_sw + av_se )
        #fv_w = 0.5 * ( av_nw + av_sw )


        #fu_e = fu_e*(1-calving_front)
        #fv_e = fv_e*(1-calving_front)

        #visc_x = fu_n + fu_e + fu_s + fu_w
        #visc_y = fv_n + fv_e + fv_s + fv_w

        #visc_x = au_sw + au_nw + au_ne + au_se
        #visc_y = av_sw + av_nw + av_ne + av_se

        ###################################################


       
        #NOTE: Those factors of 0.5 might be wrong.... CCHECK! NOTE: Checked. They're right.
        #NOTE: This only works if dx=dy. Otherwise, faff around with some factors of dx and dy.
        visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx
        
        visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx

        
        #visc_x = 0.5 * dy * visc_x
        #visc_y = 0.5 * dx * visc_y

        #stab_x, stab_y = laplacian_stabilization_residual(
        #    u, v,
        #    1e10,
        #    dx, dy,
        #    add_uv_ghost_cells
        #)

        #jax.debug.print("{x}", x=(jnp.max(visc_x), jnp.max(volume_x)))
        
        jax.debug.print("viscs_term: {x}", x=jnp.sum(visc_x*ice_mask))
        #jax.debug.print("{x}", x=jnp.sum(c.RHO_I * c.g * h * dsdx * dx * dy * ice_mask))
        
        x_mom_residual = visc_x*ice_mask + volume_x #+ #stab_x*ice_mask
        y_mom_residual = visc_y*ice_mask + volume_y #+ stab_y*ice_mask

        #x_mom_residual = visc_x*ice_mask + volume_x
        #y_mom_residual = visc_y*ice_mask + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)

def compute_nonlinear_ssa_residuals_function_variational_visc_messing_round_wcf(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):
    #temp_cc = temp_cc
    B_cc = B_from_T(temp_cc)
    B_cc = add_s_ghost_cells(extrap_over_cf(B_cc))
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask =


    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        calving_front = jnp.zeros_like(ice_mask)
        calving_front = calving_front.at[:,-3].set(1)
        #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))

        #calving_front = jnp.where(
        #    jnp.concatenate([jnp.zeros((ny,1)), jnp.where(h[:,1:]>0, 0, 1)], axis=1) > 0,
        #    1, 0
        #)  # shape (ny, nx)

        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #C = add_s_ghost_cells(C)
        #beta_linear_nc = interp_cc_to_nc(C)
        #
        #u_g, v_g = add_uv_ghost_cells(u, v)
        #beta_u_nc = beta_linear_nc * interp_cc_to_nc(u_g)
        #beta_v_nc = beta_linear_nc * interp_cc_to_nc(v_g)

        #linear_sliding_term_x = 0.25 * (beta_u_nc[:-1, :-1] + beta_u_nc[1:, :-1] + beta_u_nc[:-1, 1:] + beta_u_nc[1:, 1:]) * ice_mask + u * (1-ice_mask)
        #linear_sliding_term_y = 0.25 * (beta_v_nc[:-1, :-1] + beta_v_nc[1:, :-1] + beta_v_nc[:-1, 1:] + beta_v_nc[1:, 1:]) * ice_mask + v * (1-ice_mask)


        #jax.debug.print("{x}", x=(h.shape, dsdx.shape, ice_mask.shape))

        volume_x = -(beta * u - c.RHO_I * c.g * h * dsdx * ice_mask) * dx * dy
        volume_y = -(beta * v - c.RHO_I * c.g * h * dsdy * ice_mask) * dy * dx
        #volume_x = -(linear_sliding_term_x - 1*c.RHO_I * c.g * h * dsdx) * dx * dy
        #volume_y = -(linear_sliding_term_y - 1*c.RHO_I * c.g * h * dsdy) * dy * dx


        terminus_stress =  - 0.5 * c.RHO_TILDE * c.g * h**2 # shape (ny, nx)
        #jax.debug.print("term term: {x}", x=jnp.sum(calving_front * terminus_stress * dy))
        volume_x = volume_x + calving_front * terminus_stress * dy


        ############ momentum_term ###########
        #get thickness on the faces
        h = add_s_ghost_cells(extrap_over_cf(h))
        #h_ew, h_ns = interp_cc_to_fc(h)
        h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        mucoef = add_s_ghost_cells(mucoef)
        #mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        mucoef_nc = interp_cc_to_nc(mucoef)

        #NOTE: turning this off masiively increases size of solution sometimes,
        #but sometimes decreases it.
        u = extrap_over_cf(u)
        v = extrap_over_cf(v)
        #various node-centred derivatives
        dudx_nc, dudy_nc,\
        dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)


        #AT CALVING FRONT:

    

        #calculate node-centred viscosity:
        #mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
        #            0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
        #            1#nc_ice_mask
        mu_nc = B_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                    0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
                    nc_ice_mask


        #jax.debug.print("{x}", x=jnp.max(dudx_nc))
        mu_nc = mu_nc * h_nc
        
        mu_nw = mu_nc[:-1, :-1]
        mu_ne = mu_nc[:-1,  1:]
        mu_se = mu_nc[1:,   1:]
        mu_sw = mu_nc[1:,  :-1]

        dudx_nw = dudx_nc[:-1, :-1]
        dudx_ne = dudx_nc[:-1,  1:]
        dudx_se = dudx_nc[1:,   1:]
        dudx_sw = dudx_nc[1:,  :-1]
        
        dvdx_nw = dvdx_nc[:-1, :-1]
        dvdx_ne = dvdx_nc[:-1,  1:]
        dvdx_se = dvdx_nc[1:,   1:]
        dvdx_sw = dvdx_nc[1:,  :-1]

        dudy_nw = dudy_nc[:-1, :-1]
        dudy_ne = dudy_nc[:-1,  1:]
        dudy_se = dudy_nc[1:,   1:]
        dudy_sw = dudy_nc[1:,  :-1]

        dvdy_nw = dvdy_nc[:-1, :-1]
        dvdy_ne = dvdy_nc[:-1,  1:]
        dvdy_se = dvdy_nc[1:,   1:]
        dvdy_sw = dvdy_nc[1:,  :-1]



        #AT CALVING FRONT:




        ####KIND OF A FINITE VOLUME FLAVOUR TO THINGS:

        #au_sw = mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) * 0.5 * dx
        #au_nw = mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) * 0.5 * dx
        #au_ne = mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) * 0.5 * dx
        #au_se = mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) * 0.5 * dx

        #av_sw = mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) * 0.5 * dx
        #av_nw = mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) * 0.5 * dx
        #av_ne = mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) * 0.5 * dx
        #av_se = mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) * 0.5 * dx


        #fu_n = 0.5 * ( au_nw + au_ne )
        #fu_e = 0.5 * ( au_ne + au_se )
        #fu_s = 0.5 * ( au_sw + au_se )
        #fu_w = 0.5 * ( au_nw + au_sw )

        #fv_n = 0.5 * ( av_nw + av_ne )
        #fv_e = 0.5 * ( av_ne + av_se )
        #fv_s = 0.5 * ( av_sw + av_se )
        #fv_w = 0.5 * ( av_nw + av_sw )


        #fu_e = fu_e*(1-calving_front)
        #fv_e = fv_e*(1-calving_front)

        #visc_x = fu_n + fu_e + fu_s + fu_w
        #visc_y = fv_n + fv_e + fv_s + fv_w

        #visc_x = au_sw + au_nw + au_ne + au_se
        #visc_y = av_sw + av_nw + av_ne + av_se

        ###################################################


       
        #NOTE: Those factors of 0.5 might be wrong.... CCHECK! NOTE: Checked. They're right.
        #NOTE: This only works if dx=dy. Otherwise, faff around with some factors of dx and dy.
        visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx
        
        visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
                   mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
                   mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
                   mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) ) * 0.5 * dx

        visc_x = visc_x.at[-4, :].set(visc_x)

        
        #visc_x = 0.5 * dy * visc_x
        #visc_y = 0.5 * dx * visc_y

        #stab_x, stab_y = laplacian_stabilization_residual(
        #    u, v,
        #    1e10,
        #    dx, dy,
        #    add_uv_ghost_cells
        #)

        #jax.debug.print("{x}", x=(jnp.max(visc_x), jnp.max(volume_x)))
        
        jax.debug.print("viscs_term: {x}", x=jnp.sum(visc_x*ice_mask))
        #jax.debug.print("{x}", x=jnp.sum(c.RHO_I * c.g * h * dsdx * dx * dy * ice_mask))
        
        x_mom_residual = visc_x*ice_mask + volume_x #+ #stab_x*ice_mask
        y_mom_residual = visc_y*ice_mask + volume_y #+ stab_y*ice_mask

        #x_mom_residual = visc_x*ice_mask + volume_x
        #y_mom_residual = visc_y*ice_mask + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)


def compute_nonlinear_ssa_residuals_function_variational_visc(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc):
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_nc = interp_cc_to_nc(B_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)
    
    #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
    #TODO: make that better, do same for fc_ew and fc_ns:
    #fc_ice_mask = 

    def compute_nl_ssa_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        #nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))



        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
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


        ## calving front BC — add after volume_x/volume_y are computed
        ## identifies cells where the cell to the right has no ice
        #calving_front = jnp.where(
        #    jnp.concatenate([jnp.zeros((ny,1)), jnp.where(h[:,1:]>0, 0, 1)], axis=1) > 0,
        #    1, 0
        #)  # shape (ny, nx)
       
        ##jax.debug.print("{x}", x=jnp.count_nonzero(calving_front))
        #terminus_stress = 0.5 * c.RHO_I * c.g * h**2 * (1 - c.RHO_I/c.RHO_W)  # shape (ny, nx)
        #
        #volume_x = volume_x + 10000*calving_front * terminus_stress * dy


        ############# momentum_term ###########
        ##get thickness on the faces
        #h = add_s_ghost_cells(h)
        #h_ew, h_ns = interp_cc_to_fc(h)
        #h_nc = interp_cc_to_nc(h) #Note: interp_cc_to_nc is not made in a function factory
       
        #mucoef = add_s_ghost_cells(mucoef)
        ##mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        #mucoef_nc = interp_cc_to_nc(mucoef)

        #
        ##various node-centred derivatives
        #dudx_nc, dudy_nc,\
        #dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)
    

        ##calculate node-centred viscosity:
        #mu_nc = B_nc * mucoef_nc * (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
        #            0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)) *\
        #            nc_ice_mask

        #mu_nc = mu_nc * h_nc
        #
        #mu_nw = mu_nc[:-1, :-1]
        #mu_ne = mu_nc[:-1,  1:]
        #mu_se = mu_nc[1:,   1:]
        #mu_sw = mu_nc[1:,  :-1]

        #dudx_nw = dudx_nc[:-1, :-1]
        #dudx_ne = dudx_nc[:-1,  1:]
        #dudx_se = dudx_nc[1:,   1:]
        #dudx_sw = dudx_nc[1:,  :-1]
        #
        #dvdx_nw = dvdx_nc[:-1, :-1]
        #dvdx_ne = dvdx_nc[:-1,  1:]
        #dvdx_se = dvdx_nc[1:,   1:]
        #dvdx_sw = dvdx_nc[1:,  :-1]

        #dudy_nw = dudy_nc[:-1, :-1]
        #dudy_ne = dudy_nc[:-1,  1:]
        #dudy_se = dudy_nc[1:,   1:]
        #dudy_sw = dudy_nc[1:,  :-1]

        #dvdy_nw = dvdy_nc[:-1, :-1]
        #dvdy_ne = dvdy_nc[:-1,  1:]
        #dvdy_se = dvdy_nc[1:,   1:]
        #dvdy_sw = dvdy_nc[1:,  :-1]


        #NOTE: Those factors of 0.5 might be wrong.... CCHECK!!!!!
        #visc_x = ( mu_sw * ( 2 * dudx_sw + dvdy_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\ 
        #           mu_nw * ( 2 * dudx_nw + dvdy_nw - 0.5 * (dvdx_nw + dudy_nw) ) +\
        #           mu_ne * (-2 * dudx_ne - dvdy_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
        #           mu_se * (-2 * dudx_se - dvdy_se + 0.5 * (dvdx_se + dudy_se) ) )
        #
        #visc_y = ( mu_sw * ( 2 * dvdy_sw + dudx_sw + 0.5 * (dvdx_sw + dudy_sw) ) +\
        #           mu_nw * (-2 * dvdy_nw - dudx_nw + 0.5 * (dvdx_nw + dudy_nw) ) +\
        #           mu_ne * (-2 * dvdy_ne - dudx_ne - 0.5 * (dvdx_ne + dudy_ne) ) +\
        #           mu_se * ( 2 * dvdy_se + dudx_se - 0.5 * (dvdx_se + dudy_se) ) )

        #
        #visc_x = 0.5 * dx * visc_x
        #visc_y = 0.5 * dy * visc_y


        ##jax.debug.print("visc_x: {x}", x=visc_x)
        ##jax.debug.print("volm_x: {x}", x=volume_x)

        #x_mom_residual = visc_x + volume_x
        #y_mom_residual = visc_y + volume_y






        ############ momentum_term again ###########

        #cell-centred grid of shape (ny+2, nx+2)
        u_g, v_g = add_uv_ghost_cells(u, v)
        h = add_s_ghost_cells(h)
        mucoef = add_s_ghost_cells(mucoef)
        
        h_nc = interp_cc_to_nc(h)
        mucoef_nc = interp_cc_to_nc(mucoef)
        
        u_ur = u_g[:-1, 1:]
        u_lr = u_g[1:,  1:]
        u_ll = u_g[1:, :-1]
        u_ul = u_g[:-1,:-1]
        
        v_ur = v_g[:-1, 1:]
        v_lr = v_g[1:,  1:]
        v_ll = v_g[1:, :-1]
        v_ul = v_g[:-1,:-1]
       

        #node-centred grid of shape (ny+1, nx+1)

        A = (u_ur + u_lr - u_ul - u_ll) / (2*dx)
        B = (v_ur + v_ul - v_ll - v_lr) / (2*dy)
        C = (v_ur + v_lr - v_ul - v_ll) / (2*dx)
        D = (u_ur + u_ul - u_ll - u_lr) / (2*dy)
        
        S = A*A + B*B + A*B + 0.25*(C + D)**2 + c.EPSILON_VISC**2
        
        factor = 0.5 *  B_nc * mucoef_nc * h_nc * (S**(0.5*(1/c.GLEN_N - 1)))
        
        factor = factor * nc_ice_mask
        
        common = 0.5 * (C + D)
     

        #back to cell-centred grid of shape (ny, nx)
        
        Ru_ll_cotribution = factor[1:, :-1] * ( 2*A[1:, :-1] + B[1:, :-1] + common[1:, :-1] ) / (2*dx)
        Ru_ul_cotribution = factor[:-1,:-1] * ( 2*A[:-1,:-1] + B[:-1,:-1] - common[:-1,:-1] ) / (2*dx)
        Ru_ur_cotribution = factor[:-1, 1:] * (-2*A[:-1, 1:] - B[:-1, 1:] - common[:-1, 1:] ) / (2*dx)
        Ru_lr_cotribution = factor[1:,  1:] * (-2*A[1:,  1:] - B[1:,  1:] + common[1:,  1:] ) / (2*dx)

        Rv_ll_cotribution = factor[1:, :-1] * ( 2*B[1:, :-1] + A[1:, :-1] + common[1:, :-1] ) / (2*dx)
        Rv_ul_cotribution = factor[:-1,:-1] * (-2*B[:-1,:-1] - A[:-1,:-1] + common[:-1,:-1] ) / (2*dx)
        Rv_ur_cotribution = factor[:-1, 1:] * (-2*B[:-1, 1:] - A[:-1, 1:] - common[:-1, 1:] ) / (2*dx)
        Rv_lr_cotribution = factor[1:,  1:] * ( 2*B[1:,  1:] + A[1:,  1:] - common[1:,  1:] ) / (2*dx)

        visc_x = Ru_ll_cotribution + Ru_ul_cotribution + Ru_ur_cotribution + Ru_lr_cotribution
        visc_y = Rv_ll_cotribution + Rv_ul_cotribution + Rv_ur_cotribution + Rv_lr_cotribution
        
        
        visc_x = visc_x * dx * dy
        visc_y = visc_y * dx * dy


        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        
        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_nl_ssa_residuals)



def compute_linear_ssa_residuals_function_fc_visc_new_noextrap(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          fc_vel_gradient,
                                          cc_gradient,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells):

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

        #get thickness on the faces
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        
        #various face-centred derivatives
        dudx_ew, dudy_ew,\
        dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns,\
        dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)

        
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

def compute_linear_ssa_residuals_function_fc_visc_new_TEMP(ny, nx, dy, dx, b,\
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


        #dsdx = jnp.where(ice_mask & ~binary_erosion(ice_mask), 0.005*dsdx, dsdx)
        #dsdy = jnp.where(ice_mask & ~binary_erosion(ice_mask), 0.005*dsdy, dsdy)


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

def compute_linear_ssa_residuals_function_fc_visc_nl_fudge(ny, nx, dy, dx, \
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
        
        x_mom_residual = (1/c.GLEN_N) * visc_x + volume_x
        y_mom_residual = (1/c.GLEN_N) * visc_y + volume_y

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


def compute_ssa_uv_residuals_function_wextrap(ny, nx, dy, dx, b,
                                   beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0,
                                   C_0, temp_cc,
                                   hgrads_fct):
  
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)

    def compute_uv_residuals(u_1d, v_1d, q, p, h_1d):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))


        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        hdsdx, hdsdy = hgrads_fct(s, h, (s_gnd>s_flt).astype(int))
        
        beta = beta_fct(C, u, v, h)

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx


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
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
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


def compute_ssa_uv_residuals_function_pnotC_givenT(ny, nx, dy, dx, b,
                                   beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   ew_gradient,
                                   ns_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   extrp_over_cf, mucoef_0,
                                   C_0, temp_cc):
  
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)

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
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
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


def compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(ny, nx, dy, dx, b,
                                   beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   fc_vel_gradient,
                                   cc_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   mucoef_0,
                                   C_0, temp_cc):
  
    temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)
    B_ew, B_ns = interp_cc_to_fc(B_cc)

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


        #various face-centred derivatives
        dudx_ew, dudy_ew,\
        dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns,\
        dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)


        #interpolate things onto face-cenres
        h = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h)
        #remove those ghost cells again!
        h = h[1:-1,1:-1]

        mucoef = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)

        #calculate face-centred viscosity:
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        
        #jax.debug.print("max dudx_ns = {x}",x=jnp.max(dudx_ns))
        #jax.debug.print("max dudx_ew = {x}",x=jnp.max(dudx_ew))

        
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

def node_centred_action_functional_function_no_cf(ny, nx, dy, dx, b,
                                          nc_vel_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc):

    B_cc = B_from_T(temp_cc)
    B_cc = add_s_ghost_cells(B_cc)
    B_nc = interp_cc_to_nc(B_cc)

    def action_functional(u_1d, v_1d, q, p, h_1d):
        
        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)

        mucoef_nc = 1

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        #s = h + b
        #
        #s = add_s_ghost_cells(s)

        #dsdx, dsdy = cc_gradient(s)
        #dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        #dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        #dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        #dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        #dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        #dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        #dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        #dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        
        dsdx_nc = -0.00872653549
        dsdy_nc = 0

        #beta = beta_fct(C, u, v, h)
        beta_nc = interp_cc_to_nc(add_s_ghost_cells(C))

        #Viscous term
        ug, vg = add_uv_ghost_cells(u, v)
        u_nc = interp_cc_to_nc(ug)
        v_nc = interp_cc_to_nc(vg)

        h_nc = 1000

        dudx_nc, dudy_nc, dvdx_nc, dvdy_nc = nc_vel_gradient(u, v)

        mask = jnp.ones_like(dudx_nc)
        mask = mask.at[:,-1].set(0.5)

        visc_term = 2*(c.GLEN_N/(c.GLEN_N + 1)) *\
                        jnp.sum(mask * h_nc * B_nc * mucoef_nc * \
                                 (dudx_nc**2 + dvdy_nc**2 + dudx_nc*dvdy_nc +\
                                  0.25*(dudy_nc+dvdx_nc)**2 + c.EPSILON_VISC**2
                                 )**(0.5/c.GLEN_N+0.5)
                               )

        #Frictional term
        #fric_term = jnp.sum(0.5 * beta_nc * (u_nc**2 + v_nc**2))
        fric_term = jnp.sum(0.5 * C * (u**2 + v**2))

        #Gravitational term
        #grav_term = c.RHO_I * c.g * jnp.sum( h_nc * (dsdx_nc * u_nc + dsdy_nc * v_nc) )
        grav_term = c.RHO_I * c.g * jnp.sum( h * (dsdx_nc * u + dsdy_nc * v) )


        #Boundary term
        boundary_term = 0


        jax.debug.print("visc: {x}", x=visc_term)
        jax.debug.print("fric: {x}", x=fric_term)
        jax.debug.print("grav: {x}", x=grav_term)


        return visc_term + fric_term - grav_term

    return action_functional

def node_centred_action_functional_function(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          interp_cc_to_nc,
                                          fc_vel_gradient,
                                          nc_vel_gradient,
                                          cc_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          mucoef_0, C_0,
                                          temp_cc, extrap_over_cf):

    B_cc = B_from_T(temp_cc)
    B_cc = add_s_ghost_cells(extrap_over_cf(B_cc))
    B_nc = interp_cc_to_nc(B_cc)

    def action_functional(u_1d, v_1d, q, p, h_1d):
        
        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        calving_front = jnp.zeros_like(ice_mask)
        calving_front = calving_front.at[:,-3].set(1)
        #nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)
        nc_ice_mask = interp_cc_to_nc(add_s_ghost_cells(jnp.where(h>0, 1, 0)))


        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #Viscous term
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        dudx, dudy, dvdx, dvdy = cc_vel_gradient(u, v)

        visc_term = (c.GLEN_N/(c.GLEN_N + 1)) *\
                        jnp.sum(h * B_cc * mucoef * \
                                 (dudx**2 + dvdy**2 + dudx*dvdy +\
                                  0.25*(dudy+dvdx)**2 + c.EPSILON_VISC**2
                                 )**(0.5/c.GLEN_N+0.5)
                               )

        #Frictional term
        fric_term = jnp.sum(0.5 * beta * (u**2 + v**2))


        #Gravitational term
        grav_term = c.RHO_I * c.g * jnp.sum( h * (dsdx * u + dsdy * v) )


        #Boundary term
        boundary_term = 0


        jax.debug.print("visc: {x}", x=visc_term)
        jax.debug.print("fric: {x}", x=fric_term)
        jax.debug.print("grav: {x}", x=grav_term)



        return visc_term + fric_term - grav_term

    return action_functional

def action_functional_function(ny, nx, dy, dx, b,
                              cc_gradient,
                              cc_vel_gradient,
                              beta_fct,
                              add_uv_ghost_cells,
                              add_s_ghost_cells,
                              extrp_over_cf,
                              mucoef_0, C_0,
                              temp_cc):

    #temp_cc = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc)

    def action_functional(u_1d, v_1d, q, p, h_1d):
        
        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)


        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ice_mask = jnp.where(h.copy()>0, 1, 0)
        nc_ice_mask = jnp.where(interp_cc_to_nc(add_s_ghost_cells(ice_mask))>0.999, 1, 0)


        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)
        
        s = add_s_ghost_cells(s)

        dsdx, dsdy = cc_gradient(s)
        dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        beta = beta_fct(C, u, v, h)

        #Viscous term
        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        dudx, dudy, dvdx, dvdy = cc_vel_gradient(u, v)

        visc_term = (c.GLEN_N/(c.GLEN_N + 1)) *\
                        jnp.sum(h * B_cc * mucoef * \
                                 (dudx**2 + dvdy**2 + dudx*dvdy +\
                                  0.25*(dudy+dvdx)**2 + c.EPSILON_VISC**2
                                 )**(0.5/c.GLEN_N+0.5)
                               )

        #Frictional term
        fric_term = jnp.sum(0.5 * beta * (u**2 + v**2))


        #Gravitational term
        grav_term = c.RHO_I * c.g * jnp.sum( h * (dsdx * u + dsdy * v) )


        #Boundary term
        boundary_term = 0


        jax.debug.print("visc: {x}", x=visc_term)
        jax.debug.print("fric: {x}", x=fric_term)
        jax.debug.print("grav: {x}", x=grav_term)



        return visc_term + fric_term - grav_term

    return action_functional


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

def compute_uvh_linear_ssa_residuals_function_fc_visc_noextrap(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          ice_mask,
                                          fc_vel_gradient,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          hgrads_fct):

    """
    Linear (frozen-coefficient) coupled u,v,h residual for the Picard/quasi-Newton
    phase of a coupled momentum+advection solve, with NO calving-front
    extrapolation anywhere. Velocity gradients at faces are computed with
    fc_vel_gradient (the same cf-safe gradient function used to build mu_ew/mu_ns),
    so the linearised viscous term and the actual viscosity closure see exactly
    the same treatment of the ice edge -- the last ice-filled cells simply see
    zero velocity/thickness across the edge, equivalent to those cells being
    strongly damaged, rather than having their state linearly extrapolated
    past the edge.

    The calving front is not allowed to move within a single implicit solve:
    flux is zeroed for any cell that was ice-free at the start of the step
    (h_t<=0), so the ice extent is fixed for the duration of the solve.
    """

    def compute_uvh_residuals(u_1d, v_1d, h_1d,
                              mu_ew, mu_ns, beta,
                              h_t, source, delta_t):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ######### MOMENTUM RESIDUALS #############################
        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        hdsdx, hdsdy = hgrads_fct(h, s, (s_gnd>s_flt).astype(int))

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx

        #momentum_term -- no extrapolation over the calving front
        u_full, v_full = add_uv_ghost_cells(u, v)

        #get thickness on the faces
        h_full = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_full)

        #various face-centred derivatives, computed with the same cf-safe
        #gradient function used in the viscosity closure
        dudx_ew, dudy_ew,\
        dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns,\
        dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)

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
        

        ##On the calving front faces, the interpolated thickness will be half what it should be
        #so ice builds up there in the advection bit. To fudge that, multiply the face-centred
        #For the purposes of the advection, the velocity should also be doubled there, I guess.
        mask_ew, mask_ns = interp_cc_to_fc(add_s_ghost_cells(ice_mask))
        ff_ew = jnp.nan_to_num(1/mask_ew, nan=0.0, posinf=0.0, neginf=0.0)
        ff_ns = jnp.nan_to_num(1/mask_ns, nan=0.0, posinf=0.0, neginf=0.0)
        u_fc_ew, v_fc_ns = u_fc_ew*ff_ew, v_fc_ns*ff_ns


        ##face-centred values according to first-order upwinding
        h_fc_fou_ew = jnp.where(u_fc_ew>0, h_full[1:-1,:-1], h_full[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h_full[1:, 1:-1], h_full[-1:,1:-1])

        flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t
        #calving front is fixed within an implicit solve: no flux into or out
        #of cells that were ice-free at the start of the timestep
        flux_term = jnp.where(h_t>1e-2, flux_term, 0)

        adv_residual = ((h - h_t) - source * delta_t)*dx*dy + flux_term

        ##################################################################

        return (x_mom_residual.reshape(-1),
                y_mom_residual.reshape(-1),
                adv_residual.reshape(-1))

    return jax.jit(compute_uvh_residuals)


def compute_uvh_residuals_function_fully_nonlinear_givenT_noextrap(ny, nx, dy, dx,
                                   b, beta_fct, ice_mask,
                                   interp_cc_to_fc,
                                   fc_vel_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   hgrads_fct,
                                   mucoef_0, C_0, temp_cc):
    """
    Fully nonlinear coupled u,v,h residual (viscosity and beta evaluated at the
    current iterate rather than frozen), with NO calving-front extrapolation,
    for use in the Newton phase of a coupled Picard-Newton (picnewton) solve
    and for adjoint linearisation. Parameterised by q (log mucoef) and p (log C)
    to match the rest of the coupled solver stack.
    """

    temp_cc_ghost = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc_ghost)
    B_ew, B_ns = interp_cc_to_fc(B_cc)

    def compute_uvh_residuals(u_1d, v_1d, h_1d, q, p, h_t, source, delta_t):

        mucoef = mucoef_0*jnp.exp(q)
        C = C_0*jnp.exp(p)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        ######### MOMENTUM RESIDUALS #############################
        s_gnd = h + b
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        hdsdx, hdsdy = hgrads_fct(h, s, (s_gnd>s_flt).astype(int))

        beta = beta_fct(C, u, v, h)

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx

        #various face-centred derivatives, no extrapolation over the cf
        dudx_ew, dudy_ew,\
        dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns,\
        dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)

        #interpolate thickness and mucoef onto face-centres
        h_full = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_full)
        
        mucoef_full = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef_full)

        #calculate face-centred viscosity using the given temperature field:
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #calving front is fixed for the duration of this solve, so we use the
        #(fixed) ice_mask captured at solver construction time to zero the
        #viscosity of faces belonging to ice-free cells:
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

        ################ ADVECTION RESIDUALS #############################

        u_full, v_full = add_uv_ghost_cells(u, v)

        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        ##On the calving front faces, the interpolated thickness will be half what it should be
        #so ice builds up there in the advection bit. To fudge that, multiply the face-centred
        #For the purposes of the advection, the velocity should also be doubled there, I guess.
        mask_ew, mask_ns = interp_cc_to_fc(add_s_ghost_cells(ice_mask))
        ff_ew = jnp.nan_to_num(1/mask_ew, nan=0.0, posinf=0.0, neginf=0.0)
        ff_ns = jnp.nan_to_num(1/mask_ns, nan=0.0, posinf=0.0, neginf=0.0)
        u_fc_ew, v_fc_ns = u_fc_ew*ff_ew, v_fc_ns*ff_ns

        h_fc_fou_ew = jnp.where(u_fc_ew>0, h_full[1:-1,:-1], h_full[1:-1, 1:])
        h_fc_fou_ns = jnp.where(v_fc_ns>0, h_full[1:, 1:-1], h_full[-1:,1:-1])

        flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t
        #calving front is fixed within an implicit solve
        flux_term = jnp.where(h_t>1e-2, flux_term, 0)

        adv_residual = ((h - h_t) - source * delta_t)*dx*dy + flux_term

        ##################################################################

        return (x_mom_residual.reshape(-1),
                y_mom_residual.reshape(-1),
                adv_residual.reshape(-1))

    return compute_uvh_residuals


def compute_uvh_linear_ssa_residuals_function_noextrap(ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          ew_gradient,
                                          ns_gradient,
                                          cc_gradient,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          extrapolate_over_cf,
                                          hgrads_fct):

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
        
        #s = add_s_ghost_cells(s)
        ##jax.debug.print("s: {x}",x=s)

        #dsdx, dsdy = cc_gradient(s)
        ##jax.debug.print("dsdx: {x}",x=dsdx)
        ##sneakily fudge this:
        #dsdx = dsdx.at[-1,:].set(dsdx[-2,:])
        #dsdx = dsdx.at[0, :].set(dsdx[1 ,:])
        #dsdx = dsdx.at[:, 0].set(dsdx[:, 1])
        #dsdx = dsdx.at[:,-1].set(dsdx[:,-2])
        #dsdy = dsdy.at[-1,:].set(dsdy[-2,:])
        #dsdy = dsdy.at[0, :].set(dsdy[1 ,:])
        #dsdy = dsdy.at[:, 0].set(dsdy[:, 1])
        #dsdy = dsdy.at[:,-1].set(dsdy[:,-2])

        hdsdx, hdsdy = hgrads_fct(h, s, (s_gnd>s_flt).astype(int))

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx


        #momentum_term
        #u = extrapolate_over_cf(u)
        #v = extrapolate_over_cf(v)
        u_full, v_full = add_uv_ghost_cells(u, v)

        #get thickness on the faces
        #h_extrp = extrapolate_over_cf(h)
        h_full = add_s_ghost_cells(h)
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



