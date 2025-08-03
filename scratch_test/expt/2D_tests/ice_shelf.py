#1st party
from pathlib import Path
import sys
import time


#local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new


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




def make_compute_u_v_residuals_function(mu_coef):

    ny, nx = mu_coef.shape

    def compute_u_v_residuals(u, v, h):

        u_2d = u.reshape((ny, nx))
        v_2d = v.reshape((ny, nx))
        h_2d = h.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)




        #volume_term
        dsdx = jnp.zeros((ny, nx))
        dsdx = dsdx.at[:, 1:-1].set((0.5/dx) * (s_2d[:,2:] - s_2d[:,:-2]))
        dsdx = dsdx.at[:, 0].set((0.5/dx) * (s_2d[:, 1] - s_2d[:, 0])) #using reflection bc
        dsdx = dsdx.at[:,-1].set((0.5/dx) * (-s_2d[:,-2]))
        
        dsdy = jnp.zeros((ny, nx))
        dsdy = dsdy.at[1:-1, :].set((0.5/dy) * (s_2d[:-2,:] - s_2d[2:,:]))
        dsdy = dsdy.at[0, :].set((0.5/dy) * (s_2d[0, :] - s_2d[1, :]))
        dsdy = dsdy.at[-1,:].set((0.5/dy) * (2*s_2d[-2,:]))

        volume_x = -rho * g * h * dsdx
        volume_y = -rho * g * h * dsdy




        #momentum_term
        #various face-centred derivatives
        dudx_ew = jnp.zeros((ny, nx+1))
        dudy_ew = jnp.zeros((ny, nx+1))
        dvdx_ew = jnp.zeros((ny, nx+1))
        dvdy_ew = jnp.zeros((ny, nx+1))

        dudx_ns = jnp.zeros((ny+1, nx))
        dudy_ns = jnp.zeros((ny+1, nx))
        dvdx_ns = jnp.zeros((ny+1, nx))
        dvdy_ns = jnp.zeros((ny+1, nx))


        #1
        dudx_ew = dudx_ew.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dx)
        dudx_ew = dudx_ew.at[:,0].set(2*u[:,0]/dx)
        dudx_ew = dudx_ew.at[:,-1].set(2*u[:,-1]/dx) #doesn't matter \
        #as won't enter flux on this face as it's where the cf will be
        
        #2
        #order in brackets is: north, northeast, south, southeast
        dudy_ew = dudy_ew.at[1:-1, 1:-1].set((u[:-1, :-1] +
                                              u[:-1, 1:]  -
                                              u[1:, :-1]  -
                                              u[1:, 1:]
                                             )/(4*dx))
        dudy_ew = dudy_ew.at[0, 1:-1].set(  -(u[0, 1:]  +
                                              u[0, :-1] +
                                              u[1, 1:] +
                                              u[1, :-1]
                                             )/(4*dx))
        dudy_ew = dudy_ew.at[-1, 1:-1].set(  (u[-2, 1:]  +
                                              u[-2, :-1] +
                                              u[-1, 1:] +
                                              u[-1, :-1]
                                             )/(4*dx))
        #due to reflection bcs, dudy_ew is 0 on left and right boundaries
        
        #3
        dvdx_ew = dvdx_ew.at[:, 1:-1].set((v[:,1:]-v[:,:-1])/dx)
        dudx_ew = dudx_ew.at[:,0].set(2*v[:,0]/dx)
        dudx_ew = dudx_ew.at[:,-1].set(2*v[:,-1]/dx)

        #4
        dvdy_ew = dvdy_ew.at[1:-1, 1:-1].set((v[:-1, :-1] +
                                              v[:-1, 1:]  -
                                              v[1:, :-1]  -
                                              v[1:, 1:]
                                             )/(4*dx))
        dvdy_ew = dvdy_ew.at[0, 1:-1].set(  -(v[0, 1:]  +
                                              v[0, :-1] +
                                              v[1, 1:] +
                                              v[1, :-1]
                                             )/(4*dx))
        dvdy_ew = dvdy_ew.at[-1, 1:-1].set(  (v[-2, 1:]  +
                                              v[-2, :-1] +
                                              v[-1, 1:] +
                                              v[-1, :-1]
                                             )/(4*dx))
        #due to reflection bcs, dvdy_ew is 0 on right and left boundaries

        #5
        dudx_ns = dudx_ns.at[1:-1, 1:-1].set((u[] +\
                                              u[] -\
                                              u[] -\
                                              u[]
                                             )/(4*dx))


        #6
        dudy_ns = dudy_ns.at[1:-1, :].set((u[:-1, :]-u[1:, :])/dy)
        dudy_ns = dudy_ns.at[0,:].set(-2*u[0,:]/dy)
        dudy_ns = dudy_ns.at[-1,:].set(2*u[-1,:]/dy)

        #7
        dvdy_ns = dvdy_ns.at[1:-1, :].set((v[:-1, :]-v[1:, :])/dy)
        dudy_ns = dudy_ns.at[0,:].set(-2*v[0,:]/dy)
        dudy_ns = dudy_ns.at[-1,:].set(2*v[-1,:]/dy)


        #8
        dvdx_ns = dvdx_ns.at[1:-1, 1:-1].set(()/(4*dx))



        #cc derivatives
        dudx = jnp.zeros((ny, nx))
        dudx = dudx.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dx)
        dudx = dudx.at[:, 0].set((2*u[:,0])/dx)
        dudx = dudx.at[:, -1].set((2*u[:,-1])/dx)

        dudy = dudy.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dy)
        dudy = dudy.at[:, 0].set((2*u[:,0])/dy)
        dudy = dudy.at[:, -1].set((2*u[:,-1])/dy)




        return volume_term



ny, nx = 64, 64
mucoef = jnp.ones((ny, nx))

cr_func = make_compute_u_v_residuals_function(mucoef)

cr_func()



