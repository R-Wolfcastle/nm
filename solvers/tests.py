#1st party
import sys
import time

#local apps
sys.path.insert(1, "/users/eartsu/new_model/testing/nm/utils/")
from sparsity_utils import basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new
from cg import make_sparse_matvec, sparse_dpcg_solver

#3rd party
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
#import matplotlib.pyplot as plt
#import jax.scipy.linalg as lalg
#from jax.scipy.optimize import minimize
from jax import custom_vjp
#from jax.experimental.sparse import BCOO

import numpy as np
#import scipy

#from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=1, suppress=False, linewidth=np.inf, threshold=np.inf)


def make_residual_function(C, f):
    ny, nx = C.shape

    def residual_function(u_1d):

        u_2d = u_1d.reshape((ny, nx))

        x_flux_diffs = jnp.zeros((ny, nx))
        y_flux_diffs = jnp.zeros((ny, nx))

        x_flux_diffs = x_flux_diffs.at[:, 1:-1].set((dy/dx) * (u_2d[:,2:] + u_2d[:,:-2] - 2*u_2d[:,1:-1]))
        x_flux_diffs = x_flux_diffs.at[:, 0].set((dy/dx) * (-2*u_2d[:, 0]))
        x_flux_diffs = x_flux_diffs.at[:,-1].set((dy/dx) * (-2*u_2d[:,-1]))

        y_flux_diffs = y_flux_diffs.at[1:-1, :].set((dx/dy) * (u_2d[2:,:] + u_2d[:-2,:] - 2*u_2d[1:-1,:]))
        y_flux_diffs = y_flux_diffs.at[0, :].set((dx/dy) * (-2*u_2d[0, :]))
        y_flux_diffs = y_flux_diffs.at[-1,:].set((dx/dy) * (-2*u_2d[-1,:]))


        volume_term = (-C*u_2d + f)*dx*dy

        return (-x_flux_diffs - y_flux_diffs - volume_term).reshape(-1)

    return residual_function

def make_residual_function_face_centred_boundary(C, f):
    ny, nx = C.shape

    def residual_function(u_1d):

        u_2d = u_1d.reshape((ny, nx))

        x_flux_diffs = jnp.zeros((ny, nx))
        y_flux_diffs = jnp.zeros((ny, nx))

        x_flux_diffs = x_flux_diffs.at[:, 1:-1].set((dy/dx) * (u_2d[:,2:] + u_2d[:,:-2] - 2*u_2d[:,1:-1]))
        x_flux_diffs = x_flux_diffs.at[:, 0].set((dy/dx) * (u_2d[:, 1] - 3*u_2d[:, 0]))
        x_flux_diffs = x_flux_diffs.at[:,-1].set((dy/dx) * (u_2d[:,-2] - 3*u_2d[:,-1]))

        y_flux_diffs = y_flux_diffs.at[1:-1, :].set((dx/dy) * (u_2d[2:,:] + u_2d[:-2,:] - 2*u_2d[1:-1,:]))
        y_flux_diffs = y_flux_diffs.at[0, :].set((dx/dy) * (u_2d[1, :] - 3*u_2d[0, :]))
        y_flux_diffs = y_flux_diffs.at[-1,:].set((dx/dy) * (u_2d[-2,:] - 3*u_2d[-1,:]))

        volume_term = (-C*u_2d + f)*dx*dy

        return (-x_flux_diffs - y_flux_diffs - volume_term).reshape(-1)

    return residual_function

def make_newton_solver_sparse_jac(C, f, n_iterations):


    #residual_func = make_residual_function(C, f)
    residual_func = make_residual_function_face_centred_boundary(C, f)
    #residual_func = make_residual_function_advectiony(C, f)

    basis_vectors, i_coordinate_sets\
            = basis_vectors_and_coords_2d_square_stencil(nr, nc, 1)

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



    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(nr*nc, coords)



    def solver(u_trial):
        
        u = u_trial.copy()
        du = jnp.zeros_like(u)

        #interesting to see that it doesn't go that well if I do only one iteration
        #even though the problem is linear... suggests it's not solving it all that well..
        for i in range(n_iterations):
            print(jnp.max(jnp.abs(residual_func(u))))

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            rhs = -residual_func(u)

            #residual_jac_dense = jnp.zeros((nr*nc, nr*nc))
            #residual_jac_dense = residual_jac_dense.at[coords[0,:], coords[1,:]].set(residual_jac_sparse[mask])
            #print(residual_jac_dense)
            #print(rhs)
            #eigvals = jnp.linalg.eigvalsh(residual_jac_dense)
            #print(eigvals)
            #raise
            
            print(residual_jac_sparse[mask].shape)
            print(coords.shape)


            #t0 = time.time()
            
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="scipy-umfpack")
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="jax-native")
            #du, res = sparse_linear_solve(residual_jac_sparse[mask], coords, (nr*nc, nr*nc), rhs, mode="jax-scipy-bicgstab")
            
            #NOTE: Weird issue when we get to nr, nc = 5000
            #print(coords.max())
            #raise

            #du = solve_petsc_sparse(residual_jac_sparse[mask],\
            #                        coords, (nr*nc, nr*nc), rhs,\
            #                        ksp_type="bcgs",\
            #                        preconditioner="hypre",\
            #                        precondition_only=False)
           

            du = sparse_dpcg_solver(sparse_matvec,
                                    extract_inverse_diagonal,
                                    residual_jac_sparse[mask],
                                    rhs, du, iterations=1000)


            #t1 = time.time()
            #print("Linear solve time: {}s".format(t1-t0))
            

            u += du

        print(jnp.max(jnp.abs(residual_func(u))))

        return u

    return solver



def spherical_wave(nr, nc, amplitude=1, frequency=10):
    y = jnp.linspace(0, 1, nr)
    x = jnp.linspace(0, 1, nc)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cy, cx = (0.5, 0.5)
    r = jnp.sqrt((yy - cy)**2 + (xx - cx)**2)

    wave = amplitude * (1 + jnp.sin(2 * jnp.pi * frequency * r))

    return wave

nr = int(2**9.5)
nc = int(2**9.5)
#nr = int(2**2.5)
#nc = int(2**2.5)
#nr = int(2**12)
#nc = int(2**12)
dy = 1/nr
dx = 1/nc

C = spherical_wave(nr, nc, frequency=60, amplitude=500)
#C = 199


#plt.imshow(C)
#plt.colorbar()
#plt.show()
#raise

f = 1
#f = jnp.zeros((nr, nc))
#f = f.at[int(nr/4):int(nr/2), int(nc/4):int(nc/2)].set(1)


u_init = jnp.ones((nr, nc)).reshape(-1)

#1 should be enough as the problem is linear but seemingly benefits from another
n_iterations = 1
#solver = make_newton_solver(C, f, n_iterations)
solver = make_newton_solver_sparse_jac(C, f, n_iterations)

t0 = time.time()
u_final = solver(u_init)
t1 = time.time()
print("Solver time with nr={}: {}s".format(nr, t1-t0))

plt.imshow(u_final.reshape((nr,nc)), cmap="gnuplot2", vmin=0)
plt.colorbar()
plt.show()

