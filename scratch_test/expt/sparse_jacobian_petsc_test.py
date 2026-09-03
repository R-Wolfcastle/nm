

#1st party
import sys

#local apps
#sys.path.insert(1, '../')
from sparsity_utils import basis_vectors_etc, make_sparse_jacrev_fct, dodgy_coo_to_csr

#3rd party
from petsc4py import PETSc
from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm




def make_vto(s):

    def vto(u, mu):
        mu_longer = jnp.zeros((n+1,))
        mu_longer = mu_longer.at[:n].set(mu)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        dudx = dudx.at[0].set(2*u[0]/dx)

        mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()

        flux = mu_nl * dudx

        sgrad = jnp.zeros((n,))
        sgrad = sgrad.at[-1].set(-s)

        return flux[1:n+1] - flux[:n] - sgrad

    return vto



def solve_petsc_dense(A, b, ksp_type='gmres', preconditioner=None, precondition_only=False):
    #ksp_type could be 'cg', 'bicg', 'bcgs', ...

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    print(f"Number of processes available: {size}")

    A = PETSc.Mat().createDense(comm=comm, size=A.shape, array=A)
    b = PETSc.Vec().createWithArray(b, comm=comm)

    #A = PETSc.DECIDE

    x = b.duplicate()

    # Create a linear solver
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)

    ksp.setOperators(A)
    #ksp.setFromOptions()
    
    
    if preconditioner == 'hypre':
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')

    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()

    x_jnp = jnp.array(x.getArray())

    return x_jnp



def solve_petsc_sparse(values, coordinates, jac_shape, b, ksp_type='gmres', preconditioner='hypre', precondition_only=False):
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    iptr, j, values = dodgy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)

    

    #rows_local = int(jac_shape[0] / size)

    #A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), bsize=[rows_local, jac_shape], comm=comm)
    A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), comm=comm)
    
    b = PETSc.Vec().createWithArray(b, comm=comm)
    
    x = b.duplicate()
    
    
    # Create a linear solver
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)

    ksp.setOperators(A)
    #ksp.setFromOptions()
    
    
    if preconditioner == 'hypre':
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')
    else:
        pc = ksp.getPC()
        pc.setType(preconditioner)

    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()

    x_jnp = jnp.array(x.getArray())

    return x_jnp



def sparse_linear_solve(values, coordinates, jac_shape, b, x0, mode="jax-native"):

    match mode:
        case "jax_bicgstab":
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_array.html
            A = BCOO((values, coordinates), shape=jac_shape)

            #If you don't include this preconditioner then things really go to shit
            diag_indices = jnp.where(coordinates[:, 0] == coordinates[:, 1])[0]
            jacobi_values = values[diag_indices]
            jacobi_indices = coordinates[diag_indices, :]
            M = BCOO((1.0 / jacobi_values, jacobi_indices), shape=jac_shape)
            preconditioner = lambda x: M @ x

            x, info = jax.scipy.sparse.linalg.bicgstab(A, b, x0=x0, M=preconditioner,
                                                      tol=1e-10, atol=1e-10,
                                                      maxiter=10000)

            # print(x)

            # Verify convergence
            residual = np.linalg.norm(b - A @ x)

        case "jax-native":
            iptr, j, values = dodgy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)
            x = jax.experimental.sparse.linalg.spsolve(values, j, iptr, b)
            residual = None

        case "scipy-umfpack":
            csr_array = dodgy_coo_to_csr(values, coordinates, jac_shape)
            x = scipy.sparse.linalg.spsolve(csr_array, np.array(b))

            residual = np.linalg.norm(b - csr_array @ x)

    return x, residual



def make_solver(u_trial, intermediates=False):

    def newton_solve(mu):
        vto = make_vto(s)

        vto_jac = jacfwd(vto)

        # vto_jac = jacrev(vto)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            jac = vto_jac(u, mu)
            rhs = -vto(u, mu)
            
            #du = lalg.solve(jac, rhs)
            du = solve_petsc_dense(jac, rhs, preconditioner='hypre')
            # du = linear_solve(jac, rhs) #making this change alone makes no difference to the hessian computation

            u = u.at[:].set(u+du)
            if intermediates:
              us.append(u)

        if intermediates:
          return u, us
        else:
          return u

    return newton_solve


#moved to sparsity utils
#def make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets):
#    # This can be made significantly more general, but this is just to
#    # see whether the basics work and reduce demands on memory
#
#
#    def sparse_jacrev(fun_, primals):
#        y, jvp_fun = jax.vjp(fun_, *primals)
#        rows = []
#        for bv in basis_vectors:
#            row, _ = jvp_fun(bv)
#            rows.append(row)
#        rows = jnp.concatenate(rows)
#
#        # print(rows)
#        return rows
#
#    def densify_sparse_jac(jacrows_vec):
#        jac = jnp.zeros((n, n))
#
#        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
#            # jac = jac.at[bv_is, bv_js].set(jacrow)
#
#        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)
#
#        return jac
#
#    return sparse_jacrev, densify_sparse_jac


#solve:
def make_solver_sparse_jvp(u_trial, intermediates=False):

    def newton_solve(mu):
        vto = make_vto(s)

        basis_vectors, i_coord_sets, j_coord_sets = basis_vectors_etc(n)

        sparse_jacrev, densify_op = make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            print("computing jacobian and solving linear problem")

            jacrows = sparse_jacrev(vto, (u, mu))
            # jac = densify_op(jacrows)

            rhs = -vto(u, mu)

            # du = lalg.solve(jac, rhs)

            coordinates = jnp.column_stack([i_coord_sets, j_coord_sets])

            # du = linear_solve(jac, rhs) #making this change alone makes no difference to the hessian computation

            # du = sparse_linear_solve(jacrows, coordinates, (n,n), rhs, jnp.zeros(n), mode="scipy-umfpack")[0]
            # du = sparse_linear_solve(jacrows, coordinates, (n,n), rhs, jnp.zeros(n), mode="jax-native")[0]
            # du = sparse_linear_solve(jacrows, coordinates, (n,n), rhs, jnp.zeros(n), mode="jax-bicgstab")[0]

            du = solve_petsc_sparse(jacrows, coordinates, (n,n), rhs)

            u = u.at[:].set(u+jnp.array(du))
            if intermediates:
              us.append(u)

        if intermediates:
          return u, us
        else:
          return u

    return newton_solve







lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)
s = 1

epsilon = 1e-10

mu = jnp.zeros((n,)) + 1
mu = mu.at[:].set(1-(x/2))
mu = mu.at[int(2*n/3)].set(0.25)



u_trial = jnp.exp(x)-1



#newton_solve_with_intermediates = make_solver(u_trial, intermediates=True)
newton_solve_with_intermediates = make_solver_sparse_jvp(u_trial, intermediates=True)

u_end, us = newton_solve_with_intermediates(mu)




plt.figure(figsize=(10,5))
plt.plot(x, mu, c='k', linewidth=0.5, marker='x')
plt.gca().set_ylim(0, 1.2)
for axis in ['bottom','left']:
    plt.gca().spines[axis].set_linewidth(2.5)
for axis in ['top','right']:
    plt.gca().spines[axis].set_linewidth(0)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("../misc/input_to_solve_native.png", dpi=100)


fig, ax = plt.subplots(figsize=(10,5))
colors = cm.coolwarm(jnp.linspace(0, 1, len(us)))
for i, u in enumerate(np.array(us) / 3.5):
    plt.plot(x, u, color=colors[i], label=str(i))
# change all spines
for axis in ['bottom','left']:
    plt.gca().spines[axis].set_linewidth(2.5)
for axis in ['top','right']:
    plt.gca().spines[axis].set_linewidth(0)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig("../misc/solve_native.png", dpi=100)








