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

def make_residual_function_advectiony(C, f):
    ny, nx = f.shape

    def residual_function(u_1d):

        u_2d = u_1d.reshape((ny, nx))



        fd_flux_diffs_x = jnp.zeros((ny, nx))
        fd_flux_diffs_y = jnp.zeros((ny, nx))

        fd_flux_diffs_x = fd_flux_diffs_x.at[:, 1:].set((u_2d[:, 1:] - u_2d[:, :-1])*dy)
        fd_flux_diffs_x = fd_flux_diffs_x.at[:, 0].set((2*u_2d[:, 0])*dy)

        fd_flux_diffs_y = fd_flux_diffs_y.at[:-1,:].set((u_2d[:-1, :] - u_2d[1:, :])*dx)
        fd_flux_diffs_y = fd_flux_diffs_y.at[-1, :].set((2*u_2d[-1,:])*dx)



        x_flux_diffs = jnp.zeros((ny, nx))
        y_flux_diffs = jnp.zeros((ny, nx))

        x_flux_diffs = x_flux_diffs.at[:, 1:-1].set((dy/dx) * (u_2d[:,2:] + u_2d[:,:-2] - 2*u_2d[:,1:-1]))
        x_flux_diffs = x_flux_diffs.at[:, 0].set((dy/dx) * (-2*u_2d[:, 0]))
        x_flux_diffs = x_flux_diffs.at[:,-1].set((dy/dx) * (-2*u_2d[:,-1]))

        y_flux_diffs = y_flux_diffs.at[1:-1, :].set((dx/dy) * (u_2d[2:,:] + u_2d[:-2,:] - 2*u_2d[1:-1,:]))
        y_flux_diffs = y_flux_diffs.at[0, :].set((dx/dy) * (-2*u_2d[0, :]))
        y_flux_diffs = y_flux_diffs.at[-1,:].set((dx/dy) * (-2*u_2d[-1,:]))


        #volume_term = (-C*u_2d + f)*dx*dy

        #return (-x_flux_diffs - y_flux_diffs + C*(fd_flux_diffs_x + fd_flux_diffs_y) - f).reshape(-1)
        return (C*(fd_flux_diffs_x + fd_flux_diffs_y) - f).reshape(-1)

    return residual_function


def make_residual_function(C, f):
    ny, nx = C.shape

    def residual_function(u_1d):

        u_2d = u_1d.reshape((ny, nx))

        #dudx = jnp.zeros((ny, nx+1))
        #dudy = jnp.zeros((ny+1,nx))

        #dudx = dudx.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dx)
        #dudx = dudx.at[:, 0].set((2*u[:,0])/dx)
        #dudx = dudx.at[:, -1].set((2*u[:,-1])/dx)

        #dudy = dudy.at[:, 1:-1].set((u[:,1:]-u[:,:-1])/dy)
        #dudy = dudy.at[:, 0].set((2*u[:,0])/dy)
        #dudy = dudy.at[:, -1].set((2*u[:,-1])/dy)


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


def make_newton_solver(C, f, n_iterations):

    residual_func = make_residual_function(C, f)

    basis_vectors, i_coordinate_sets\
            = basis_vectors_and_coords_2d_square_stencil(nr, nc, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    mask = (i_coordinate_sets>=0).astype(jnp.int8)

    sparse_jacrev_func, densify_func = make_sparse_jacrev_fct_new(basis_vectors,\
                                             i_coordinate_sets,\
                                             j_coordinate_sets,\
                                             mask)


    def solver(u_trial):
        
        u = u_trial.copy()

        #interesting to see that it doesn't go that well if I do only one iteration
        #even though the problem is linear... wtf.
        for i in range(n_iterations):
            print(jnp.max(jnp.abs(residual_func(u))))

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            residual_jac_dense = densify_func(residual_jac_sparse, nr*nc)

            rhs = -residual_func(u)

            #print(residual_jac_dense)
            #print(rhs)

            u += lalg.solve(residual_jac_dense, rhs)

        print(jnp.max(jnp.abs(residual_func(u))))

        return u

    return solver


def solve_petsc_sparse(values, coordinates, jac_shape,\
                       b, ksp_type='gmres', preconditioner='hypre',\
                       precondition_only=False):
    
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)

    #rows_local = int(jac_shape[0] / size)

    #A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), bsize=[rows_local, jac_shape], comm=comm)
    A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr.astype(np.int32), j.astype(np.int32), values), comm=comm)
    
    b = PETSc.Vec().createWithArray(b, comm=comm)
    
    x = b.duplicate()
    
    #set ksp iterations
    opts = PETSc.Options()
    opts['ksp_max_it'] = 20
    opts['ksp_monitor'] = None
    opts['ksp_rtol'] = 1e-10
    
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

        #pc.apply(b, x)
        #print((A*x - b).norm())
        #raise


    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()
    

    x_jnp = jnp.array(x.getArray())

    return x_jnp


def sparse_linear_solve(values, coordinates, jac_shape, b, x0=None, mode="scipy-umfpack"):
    print("solver-mode: {}".format(mode))

    match mode:
        case "jax-scipy-bicgstab":
            coordinates = coordinates.T

            if x0 is None:
                x0 = jnp.zeros_like(b)

            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
            A = BCOO((values, coordinates.astype(jnp.int32)), shape=jac_shape)

            #If you don't include this preconditioner then things really go to shit
            diag_indices = jnp.where(coordinates[:, 0] == coordinates[:, 1])[0]
            jacobi_values = values[diag_indices]
            jacobi_indices = coordinates[diag_indices, :]
            M = BCOO((1.0 / jacobi_values, jacobi_indices.astype(jnp.int32)), shape=jac_shape)
            preconditioner = lambda x: M @ x

            x, info = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x0, M=preconditioner,
                                                      tol=1e-10, atol=1e-10,
                                                      maxiter=10)

            # print(x)

            # Verify convergence
            residual = np.linalg.norm(b - A @ x)

        case "jax-native":
            iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)
            x = jax.experimental.sparse.linalg.spsolve(values, j, iptr, b)
            residual = None

        case "scipy-umfpack":
            csr_array = scipy_coo_to_csr(values, coordinates, jac_shape)
            x = scipy.sparse.linalg.spsolve(csr_array, np.array(b), use_umfpack=True)

            residual = np.linalg.norm(b - csr_array @ x)

    return x, residual


def make_newton_solver_sparse_jac(C, f, n_iterations):

    #residual_func = make_residual_function(C, f)
    residual_func = make_residual_function_advectiony(C, f)

    basis_vectors, i_coordinate_sets\
            = basis_vectors_and_coords_2d_square_stencil(nr, nc, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    #mask = ~jnp.isnan(i_coordinate_sets)
    mask = (i_coordinate_sets>=0)


    sparse_jacrev_func, _ = make_sparse_jacrev_fct_new(basis_vectors,\
                                             i_coordinate_sets,\
                                             j_coordinate_sets,\
                                             mask)


    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]
    coords = jnp.stack([i_coordinate_sets, j_coordinate_sets])

    def solver(u_trial):
        
        u = u_trial.copy()

        #interesting to see that it doesn't go that well if I do only one iteration
        #even though the problem is linear... suggests it's not solving it all that well..
        for i in range(n_iterations):
            print(jnp.max(jnp.abs(residual_func(u))))

            residual_jac_sparse = sparse_jacrev_func(residual_func, (u,))

            rhs = -residual_func(u)

            #print(residual_jac_dense)
            #print(rhs)
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

            du = solve_petsc_sparse(residual_jac_sparse[mask],\
                                    coords, (nr*nc, nr*nc), rhs,\
                                    ksp_type="bcgs",\
                                    preconditioner="hypre",\
                                    precondition_only=False)
            
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
dy = 1/nr
dx = 1/nc

#C = spherical_wave(nr, nc, frequency=20, amplitude=500)
C = 199


#plt.imshow(C)
#plt.colorbar()
#plt.show()
#raise

#f = 1
f = jnp.zeros((nr, nc))
f = f.at[int(nr/4):int(nr/2), int(nc/4):int(nc/2)].set(1)


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

#plt.figure(figsize=(10, 4))
#plt.plot(u_final.reshape((nr,nc))[1250,:])
#plt.show()
raise


def plot_with_diagonal_grid(x, y, title="X vs Y with Diagonal Grid", xlabel="x", ylabel="y"):
    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    # Plot the data
    plt.plot(x, y, marker='o', linestyle='-', color='royalblue', label='Data')

    # Get actual data limits (including padding)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Expand limits a bit for aesthetics
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Compute diagonal gridline offsets (c in y = x + c) that intersect visible region
    c_min = ymin - xmax
    c_max = ymax - xmin
    offsets = np.arange(np.floor(c_min), np.ceil(c_max) + 1)

    for c in offsets:
        # y = x + c
        xs = np.array([xmin, xmax])
        ys = xs + c
        ax.plot(xs, ys, color='lightgray', linestyle='--', linewidth=1, zorder=0)

        # y = 2x + c
        xs = np.array([xmin, xmax])
        ys = 2*(xs + c)
        ax.plot(xs, ys, color='lightgray', linestyle=':', linewidth=1, zorder=0)


    # Standard grid, labels, etc.
    #plt.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    #plt.legend()
    plt.tight_layout()
    plt.show()


#dofs_log2 = np.array([ 6.,  8., 10., 12., 14., 16., 18., 19., 20., 21., 22., 23., 24., 25., 26])
#times_log2 = np.array([1.52356196, 1.36120689, 1.39999135, 1.41792001, 1.49005662,
#              1.67671874, 2.12465899, 2.60715274, 3.26828467, 3.98713893,
#              5.23817542, 6.20045727, 7.50049934, 9.926, np.log2(796)])


#getting rid of my 2**25 one as I was running out of memory a bit...
dofs_log2 = np.array([ 6.,  8., 10., 12., 14., 16., 18., 19., 20., 21., 22., 23., 24., 26])
times_log2 = np.array([1.52356196, 1.36120689, 1.39999135, 1.41792001, 1.49005662,
              1.67671874, 2.12465899, 2.60715274, 3.26828467, 3.98713893,
              5.23817542, 6.20045727, 7.50049934, np.log2(796)])

#dofs_log2 = np.array([ 6.,  8., 10., 12., 14., 16., 18., 19., 20., 21., 22., 23., 24.])
#times_log2 = np.array([1.52356196, 1.36120689, 1.39999135, 1.41792001, 1.49005662,
#              1.67671874, 2.12465899, 2.60715274, 3.26828467, 3.98713893,
#              5.23817542, 6.20045727, 7.50049934])

plot_with_diagonal_grid(dofs_log2, times_log2, title="", xlabel="log2(n_dofs)", ylabel="log2(time)")




