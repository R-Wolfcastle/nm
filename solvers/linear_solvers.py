"""A set of linear solvers for the linear system Ax = b."""

#1st party
from pathlib import Path
import sys
from functools import partial

#3rd party
import jax
import jax.numpy as jnp
import numpy as np
from petsc4py import PETSc

#native
import sys

#local apps
sys.path.insert(1, '/Users/eartsu/new_model/testing/utils/')
from sparsity_utils import scipy_coo_to_csr,\
                           dodgy_coo_to_csr,\
                           jax_coo_to_csr


jax.config.update("jax_enable_x64", True)

def create_sparse_petsc_la_solver_with_custom_vjp(coordinates, jac_shape,\
                                    ksp_type='gmres', preconditioner='hypre',\
                                    precondition_only=False, monitor_ksp=False,\
                                    ksp_max_iter=40):

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    #NOTE: INITIALISE THIS HERE AS IT IS EXPENSIVE AND THEN FILL LATER
    #A = PETSc.Mat().createAIJ(size=jac_shape,\
    #        csr=(iptr.astype(np.int32), j.astype(np.int32), values),\
    #        comm=comm)
    
    def construct_ab(values, b, transpose):
        if transpose:
            iptr, j, values, _ = jax_coo_to_csr(values, coordinates[::-1,:], jac_shape)
        else:
            iptr, j, values, _ = jax_coo_to_csr(values, coordinates, jac_shape)
        #rows_local = int(jac_shape[0] / size)

        A = PETSc.Mat().createAIJ(size=jac_shape, \
                                  csr=(iptr.astype(np.int32), j.astype(np.int32), values),\
                                  comm=comm)
        
        b = PETSc.Vec().createWithArray(b, comm=comm)
        return A, b

    
    def create_solver_object(A):
        
        #set ksp iterations
        opts = PETSc.Options()
        opts['ksp_max_it'] = ksp_max_iter

        opts['ksp_norm_type'] = "unpreconditioned"
        #opts['ksp_norm_type'] = "preconditioned"

        if monitor_ksp:
            opts['ksp_monitor'] = None
            #opts['ksp_converged_reason'] = None
            #opts['ksp_view'] = None
        opts['ksp_rtol'] = 1e-20
        
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
        
            return ksp, pc
        else:
            return ksp, None


    @partial(jax.custom_vjp, nondiff_argnums=(2,))
    def petsc_sparse_la_solver(values, b, transpose=False):
    
        A, b = construct_ab(values, b, transpose)
        x = b.duplicate()
        #x.set(0)

        ksp, pc = create_solver_object(A)

        if precondition_only:
            pc.apply(b, x)
        else:
            ksp.solve(b, x)
      
        
        reason = ksp.getConvergedReason()
        print("KSP converged reason:", reason)


        x_jnp = jnp.array(x.getArray())

        return x_jnp

    
    def la_solver_fwd(values, b, transpose=False):
        solution = petsc_sparse_la_solver(values, b, transpose=transpose)
        return solution, (values, b, solution)


    #NOTE: the nondiff_argnums=(2,) thing shunts the transpose ragument to the front.
    def linear_solve_bwd(transpose, res, x_bar):
        #NOTE: The sign convention here is correct, despite what people say...
        #It just follows the documentation rather than "textbook" versions.
        values, b, x = res

        lambda_ = petsc_sparse_la_solver(values, -x_bar, transpose=True)

        b_bar = -lambda_.reshape(b.shape) #ensure same shape as input b.

        #sparse version of jnp.outer(x,lambda_)
        #TODO: CHECK WHICH WAY ROUND THESE COORDS GO. SHOULD BE RIGHT IF THEY ARE IJ!
        values_bar = x[coordinates[1]] * lambda_[coordinates[0]]

        return values_bar, b_bar

    petsc_sparse_la_solver.defvjp(la_solver_fwd, linear_solve_bwd)

    return petsc_sparse_la_solver


def create_sparse_petsc_la_solver_with_custom_vjp_given_csr(coordinates, jac_shape,\
                                    ksp_type='gmres', preconditioner='hypre',\
                                    precondition_only=False, monitor_ksp=False,\
                                    ksp_max_iter=40):

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()





    #construct empty AIJ Matrix
    nnz = coordinates[0].size

    iptr, indices, _, order = jax_coo_to_csr(np.zeros(nnz, dtype=np.float64),
                                      coordinates, jac_shape)
    iptr = np.array(iptr)
    indices = np.array(indices)

    A = PETSc.Mat().createAIJ(
            size=jac_shape,
            csr=(iptr.astype(np.int32),
                 indices.astype(np.int32),
                 np.zeros(nnz, dtype=np.float64)),
            comm=comm
                             )
    A.setOption(PETSc.Mat.Option.KEEP_NONZERO_PATTERN, True)
    A.assemble()

    #create a virtual transposed version...
    AT_virtual = PETSc.Mat().createTranspose(A)






    # create_solver_object:
        
    #set ksp iterations
    opts = PETSc.Options()
    opts['ksp_max_it'] = ksp_max_iter
    opts['ksp_rtol'] = 1e-20
    opts['ksp_norm_type'] = "unpreconditioned"
    #opts['ksp_norm_type'] = "preconditioned"

    if monitor_ksp:
        opts['ksp_monitor'] = None
        #opts['ksp_converged_reason'] = None
        #opts['ksp_view'] = None
    
    # Create a linear solver
    ksp = PETSc.KSP().create(comm=comm)
    ksp.setType(ksp_type)

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
    else:
        pc = None
        





    @partial(jax.custom_vjp, nondiff_argnums=(2,))
    def petsc_sparse_la_solver(values, b, transpose=False):

        #_,_, vals = jax_coo_to_csr(values, coordinates, jac_shape)
        values = values[order]


        A.setValuesCSR(iptr, indices, np.array(values))
        A.assemblyBegin(); A.assemblyEnd()
        
        #A_use = A if not transpose else AT_virtual

        b = PETSc.Vec().createWithArray(np.array(b), comm=comm)
       
        if not transpose:
            ksp.setOperators(A)
        else:
            ksp.setOperators(AT_virtual)
      

        x = b.duplicate()


        if precondition_only:
            pc.apply(b, x)
        else:
            ksp.solve(b, x)
      
        
        reason = ksp.getConvergedReason()
        print("KSP converged reason:", reason)


        x_jnp = jnp.array(x.getArray())

        return x_jnp

    
    def la_solver_fwd(values, b, transpose=False):
        solution = petsc_sparse_la_solver(values, b, transpose=transpose)
        return solution, (values, b, solution)


    #NOTE: the nondiff_argnums=(2,) thing shunts the transpose ragument to the front.
    def linear_solve_bwd(transpose, res, x_bar):
        #NOTE: The sign convention here is correct, despite what people say...
        #It just follows the documentation rather than "textbook" versions.
        values, b, x = res

        lambda_ = petsc_sparse_la_solver(values, -x_bar, transpose=True)

        b_bar = -lambda_.reshape(b.shape) #ensure same shape as input b.

        #sparse version of jnp.outer(x,lambda_)
        #TODO: CHECK WHICH WAY ROUND THESE COORDS GO. SHOULD BE RIGHT IF THEY ARE IJ!
        values_bar = x[coordinates[1]] * lambda_[coordinates[0]]

        return values_bar, b_bar

    petsc_sparse_la_solver.defvjp(la_solver_fwd, linear_solve_bwd)

    return petsc_sparse_la_solver



def basic_sparse_linear_solve(values, coordinates, jac_shape, b, x0, mode="jax-native"):

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


