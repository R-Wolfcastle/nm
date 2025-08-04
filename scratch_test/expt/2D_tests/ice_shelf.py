#1st party
from pathlib import Path
import sys
import time


#local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new
import constants as c


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

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=1, suppress=False, linewidth=np.inf, threshold=np.inf)



def interp_cc_to_fc_function(ny, nx):

    def interp_cc_to_fc(var):
        
        var_ew = jnp.zeros((ny, nx+1))
        var_ew = var_ew.at[:, 1:-1].set(0.5*(var[:, 1:]+var[:, -1:]))
        var_ew = var_ew.at[:, 0].set(var[:, 0])
        var_ew = var_ew.at[:, -1].set(var[:, -1])

        var_ns = jnp.zeros((ny+1, nx))
        var_ns = var_ns.at[1:-1, :].set(0.5*(var[:-1, :]+var[1:, :]))
        var_ns = var_ns.at[0, :].set(var[0, :])
        var_ns = var_ns.at[-1, :].set(var[-1, :])

        return var_ew, var_ns

    return interp_cc_to_fc


def cc_gradient_function(ny, nx):
    pass


def cc_gradient_function(ny, nx, dy, dx):

    def cc_gradient(var):
        dvar_dx = jnp.zeros((ny, nx))
        dvar_dy = jnp.zeros((ny, nx))
        
        dvar_dx = dvar_dx.at[:, 1:-1].set((0.5/dx) * (var[:,2:] - var[:,:-2]))
        dvar_dx = dvar_dx.at[:, 0].set((0.5/dx) * (var[:, 1] - var[:, 0])) #using reflection bc
        dvar_dx = dvar_dx.at[:,-1].set((0.5/dx) * (-var[:,-2]))
        
        dvar_dy = dvar_dy.at[1:-1, :].set((0.5/dy) * (var[:-2,:] - var[2:,:]))
        dvar_dy = dvar_dy.at[0, :].set((0.5/dy) * (var[0, :] - var[1, :]))
        dvar_dy = dvar_dy.at[-1,:].set((0.5/dy) * (2*var[-2,:]))

        return dvar_dx, dvar_dy

    return cc_gradient



def fc_gradient_functions(ny, nx, dy, dx):

    def ew_face_gradient(var):
        dvar_dx_ew = jnp.zeros((ny, nx+1))
        dvar_dy_ew = jnp.zeros((ny, nx+1))


        dvar_dx_ew = dvar_dx_ew.at[:, 1:-1].set((var[:,1:]-var[:,:-1])/dx)
        dvar_dx_ew = dvar_dx_ew.at[:,0].set(2*var[:,0]/dx)
        dvar_dx_ew = dvar_dx_ew.at[:,-1].set(2*var[:,-1]/dx)

        
        dvar_dy_ew = dvar_dy_ew.at[1:-1, 1:-1].set((var[:-1, :-1] +\
                                                    var[:-1, 1:]  -\
                                                    var[1:, :-1]  -\
                                                    var[1:, 1:]
                                                   )/(4*dx))
        dvar_dy_ew = dvar_dy_ew.at[0, 1:-1].set(  -(var[0, 1:]  +\
                                                    var[0, :-1] +\
                                                    var[1, 1:]  +\
                                                    var[1, :-1]
                                                   )/(4*dx))
        dvar_dy_ew = dvar_dy_ew.at[-1, 1:-1].set(  (var[-2, 1:]  +\
                                                    var[-2, :-1] +\
                                                    var[-1, 1:]  +\
                                                    var[-1, :-1]
                                                   )/(4*dx))
        #due to reflection bcs, dudy_ew is 0 on left and right boundaries

        return dvar_dx_ew, dvar_dy_ew

    def ns_face_gradient(var):
        dvar_dx_ns = jnp.zeros((ny+1, nx))
        dvar_dy_ns = jnp.zeros((ny+1, nx))

        dvar_dy_ns = dvar_dy_ns.at[1:-1,:].set((var[:-1,:]-var[1:,:])/dy)
        dvar_dy_ns = dvar_dy_ns.at[0,:].set(-2*var[0,:]/dy)
        dvar_dy_ns = dvar_dy_ns.at[-1,:].set(2*var[-1,:]/dy)


        dvar_dx_ns = dvar_dx_ns.at[1:-1, 1:-1].set((var[:-1, 2:] +\
                                                    var[1:, 2:]  -\
                                                    var[:-1,:-2] -\
                                                    var[1:, :-2]
                                                   )/(4*dx))
        dvar_dx_ns = dvar_dx_ns.at[1:-1, 0].set(   (var[:-1, 1] +\
                                                    var[1:, 1]  +\
                                                    var[:-1, 0] +\
                                                    var[1:, 0]
                                                   )/(4*dx))
        dvar_dx_ns = dvar_dx_ns.at[1:-1, -1].set( -(var[:-1, -1] +\
                                                    var[1:, -1]  +\
                                                    var[:-1, -2] +\
                                                    var[1:, -2]
                                                   )/(4*dx))
        #due to rbcs, ddx_ns is 0 on upper and lower boundaries

        return dvar_dx_ns, dvar_dy_ns

    
    return ew_face_gradient, ns_face_gradient


def compute_u_v_residuals_function(ny, nx, dy, dx):

    interp_cc_to_fc = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient = fc_gradient_functions(ny, nx, dy, dx)
    cc_gradient = cc_gradient_function(ny, nx, dy, dx)

    def compute_u_v_residuals(u_1d, v_1d, h_1d, mu_bar):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)


        #volume_term
        dsdx, dsdy = cc_gradient(s)
        volume_x = -rho * g * h * dsdx
        volume_y = -rho * g * h * dsdy


        #momentum_term
        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        ##cc_derivatives, e.g. for viscosity calculation
        #dudx_cc, dudy_cc = cc_gradient(u)
        #dvdx_cc, dvdy_cc = cc_gradient(v)

        #interpolate things onto face-cenres
        mu_ew, mu_ns = interp_cc_to_fc(mu_bar)
        h_ew, h_ns = interp_cc_to_fc(h)


        visc_x = mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx

        visc_y = mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx


        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual, y_mom_residual

    return compute_u_v_residuals



def qn_velocity_solver_function(ny, nx, dy, dx):
    
    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx)


    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    mask = (i_coordinate_sets>=0)

    sparse_jacrev_func, _ = make_sparse_jacrev_fct_new(basis_vectors,\
                                             i_coordinate_sets,\
                                             j_coordinate_sets,\
                                             mask)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    coords = jnp.stack([i_coordinate_sets, j_coordinate_sets])
    #############

    
    def new_viscosity(u, v, h):
        dudx, dudy = cc_gradient(u)
        dvdx, dvdy = cc_gradient(v)

        return B * (dudx**2 + dvdy**2 + dudx*dvdy +\
                    0.25*(dudy+dvdx)**2 + c.EPSILON_VISC)**(0.5*(1/c.GLEN_N - 1))


    def solver(u_trial, v_trial):
        u = u_trial.copy()
        v = v_trial.copy()

        for i in range(n_iterations):
















A = 5e-25
B = 2 * (A**(-1/3))

#epsilon_visc = 1e-5/(3.15e7)
epsilon_visc = 3e-13


nr, nc = 64, 64
mucoef = jnp.ones((nr, nc))

cr_func = make_compute_u_v_residuals_function(mucoef)

cr_func()



