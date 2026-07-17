#1st Party
import os
import sys
import time
import psutil


#3rd Party
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

#local apps
nm_home = os.environ['NM_HOME']   

if nm_home is None:
    raise RuntimeError("NM_HOME is not set")


sys.path.insert(1, os.path.join(nm_home, 'utils'))
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis,\
                           make_sparse_jacrev_fct_shared_basis_new,\
                           assemble_sparse_2x2_block_jacobian_function,\
                           assemble_sparse_2x2_block_jacobian_function_nl,\
                           assemble_sparse_2x2_block_jacobian_function_general
import constants_years as c
from grid import *
from vertical_grid import define_z_coordinates
from diva_functions import *

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
import residuals as rdl
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp,\
                           create_sparse_petsc_la_solver_with_custom_vjp_given_csr
from residuals import *
from cg import make_sparse_matvec, make_sparse_dpgc_solver_comp,\
               make_sparse_damped_jacobi_solver,\
               make_sparse_bicgstab_solver, make_point_sor_preconditioner,\
               make_multicoloured_relaxation,\
               make_sparse_gs_precond_bicgstab_solver,\
               make_sparse_dpcg_solver_jsp_comp,\
               make_sparse_dpcg_solver_jsp_comp_fori


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


def make_newton_coupled_solver_function(ny, nx, dy, dx, C, b, n_iterations, mucoef_0, periodic=False):

    beta_eff = C.copy()

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    get_uvh_residuals = rdl.compute_uvh_residuals_function(ny, nx, dy, dx,\
                                                       b,\
                                                       beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells, mucoef_0)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                         periodic_x=periodic)

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
                                                              ksp_type="gmres",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,\
                                                              monitor_ksp=False)



    comm = PETSc.COMM_WORLD
    size = comm.Get_size()
    
    #@custom_vjp
    def solver(q, u_trial, v_trial, h_now, accm, delta_t):
        u_trial = jnp.where(h_now>1e-10, u_trial, 0)
        v_trial = jnp.where(h_now>1e-10, v_trial, 0)

        h_1d = h_now.copy().reshape(-1)
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        old_residual = jnp.inf

        #get_uvh_residuals(u_1d, v_1d, h_1d, q, h_now, accm, delta_t)

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



            #residual = jnp.sqrt(jnp.sum(rhs**2))
            #print(residual)
            #print(old_residual/residual)
            #old_residual = residual.copy()

            #print(f"!!! {jnp.max(jnp.abs(du))}")
            #

            #iptr, j, values = scipy_coo_to_csr(nz_jac_values, coords,\
            #                               (ny*nx*3, ny*nx*3), return_decomposition=True)

            #A = PETSc.Mat().createAIJ(size=(ny*nx*3, ny*nx*3), \
            #                          csr=(iptr.astype(np.int32), j.astype(np.int32), values),\
            #                          comm=comm)
            #
            #b = PETSc.Vec().createWithArray(du, comm=comm)

            #jdu = jnp.zeros_like(du)
            #pjdu = PETSc.Vec().createWithArray(jdu, comm=comm)
            #A.mult(b, pjdu)

            #print(f"Jdeltau = {jnp.max(jnp.abs(jdu))}")

            #
            #print(f"R = {jnp.max(jnp.abs(jdu - rhs))}")







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

def make_newton_velocity_solver_function_custom_vjp_dynamic_thk(ny, nx, dy, dx,\
                                    C, b, n_iterations, mucoef_0, periodic=False):

    beta_eff = C.copy()

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_dynamic_thickness

    get_u_v_residuals = compute_u_v_residuals_function_dynamic_thk(ny, nx, dy, dx,\
    #get_u_v_residuals = compute_uv_residuals_function_dynamic_thk_anisotropic(ny, nx, dy, dx,\
                                                       b,\
                                                       beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf, mucoef_0)
    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
    #                                                          ksp_type="bcgs",\
    #                                                          preconditioner="hypre",\
    #                                                          precondition_only=False)
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=True)



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

def make_newton_velocity_solver_function_custom_vjp(ny, nx, dy, dx,\
                                                    h, b, C,\
                                                    n_iterations,\
                                                    mucoef_0,\
                                                    periodic=False):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)

    ice_mask = jnp.where(h>1e-10, 1, 0)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       b,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_scalar_ghost_cells,\
                                                       extrapolate_over_cf, mucoef_0)
    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
    #                                                          ksp_type="gmres",\
    #                                                          preconditioner="hypre",\
    #                                                          precondition_only=False,\
    #                                                          ksp_max_iter=40)

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False)
 



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

            u_1d = (u_1d+du[:(ny*nx)])*ice_mask.reshape(-1)
            v_1d = (v_1d+du[(ny*nx):])*ice_mask.reshape(-1)


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, q)
                                                   )
                                   )
                           )
        #plt.imshow(get_u_v_residuals(u_1d, v_1d, q)[1].reshape((ny,nx)))
        #plt.colorbar()
        #plt.show()
        #raise

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


def make_coupled_quasi_newton_solver_function(ny, nx, dy, dx,
                                             b, ice_mask, n_iterations,
                                             mucoef_0, sliding="linear",
                                             periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uvh_residuals = compute_uvh_linear_ssa_residuals_function(
                                                       ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    
    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                         periodic_x=periodic)

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


    i_coord_sets = i_coordinate_sets[mask]
    j_coord_sets = j_coordinate_sets[mask]
    #############

    coords = jnp.stack([
        jnp.concatenate(
           [i_coord_sets,           i_coord_sets,           i_coord_sets,\
            i_coord_sets+(ny*nx),   i_coord_sets+(ny*nx),   i_coord_sets+(ny*nx),\
            i_coord_sets+(2*ny*nx), i_coord_sets+(2*ny*nx), i_coord_sets+(2*ny*nx)]
                       ),\
        jnp.concatenate(
           [j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx),\
            j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx),\
            j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx)]
                       )
                       ])

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords,
                                                      (ny*nx*3, ny*nx*3),\
                                                      ksp_type="gmres",\
                                                      preconditioner="hypre",\
                                                      precondition_only=False,\
                                                      monitor_ksp=False)




    #@custom_vjp
    def solver(q, C, u_trial, v_trial, h, delta_t, accm=0):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        h_t = h.copy()

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            dRu_du, dRv_du, dRh_du, \
            dRu_dv, dRv_dv, dRh_dv, \
            dRu_dh, dRv_dh, dRh_dh = sparse_jacrev(
                                                   get_uvh_residuals,
                                                   (u_1d, v_1d, h_1d, 
                                                   mu_ew, mu_ns, beta, 
                                                   h_t, accm, delta_t)
                                                          )

            nz_jac_values = jnp.concatenate(
                                [dRu_du[mask], dRu_dv[mask], dRu_dh[mask],\
                                 dRv_du[mask], dRv_dv[mask], dRv_dh[mask],\
                                 dRh_du[mask], dRh_dv[mask], dRh_dh[mask]]
                                           )

          
            #full_jac = jnp.zeros((ny*nx*3, ny*nx*3))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #print(full_jac)
            #raise


            rhs = -jnp.concatenate(get_uvh_residuals(u_1d, v_1d, h_1d, 
                                                     mu_ew, mu_ns, beta, 
                                                     h_t, accm, delta_t)
                                   )

            #print(jnp.max(rhs))
            #raise

            old_residual, residual, init_res = print_residual_things(
                                                  residual, rhs, init_res, i
                                                                    )
            
            


            du = la_solver(nz_jac_values, rhs)



            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):(2*ny*nx)]
            h_1d = h_1d+du[(2*ny*nx):]
            

            rhs_new = -jnp.concatenate(get_uvh_residuals(
                                       u_1d, v_1d, h_1d, 
                                       mu_ew, mu_ns, beta, 
                                       h_t, accm, delta_t)
                                      )
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {jnp.max(jnp.abs(rhs))/jnp.max(jnp.abs(rhs_new))}")

        final_residual = jnp.max(jnp.abs(rhs_new))

        print("----------")
        print("Final residual: {}".format(final_residual))
        print("Total residual reduction factor: {}".format(initial_residual/final_residual))
        print("----------")
        
        return (u_1d.reshape((ny, nx)),
                v_1d.reshape((ny, nx)),
                h_1d.reshape((ny,nx)) )

    return solver


def make_coupled_quasi_newton_solver_function_cvjp(ny, nx, dy, dx,
                                                   b, ice_mask, n_iterations,
                                                   mucoef_0, sliding="linear",
                                                   periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uvh_residuals = compute_uvh_linear_ssa_residuals_function(
                                                       ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    get_uvh_residuals_nonlinear = compute_uvh_residuals_function_fully_nonlinear(ny, nx, dy, dx,\
                                                       b, beta_fct,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells, mucoef_0)
    
    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                         periodic_x=periodic)

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


    i_coord_sets = i_coordinate_sets[mask]
    j_coord_sets = j_coordinate_sets[mask]
    #############

    coords = jnp.stack([
        jnp.concatenate(
           [i_coord_sets,           i_coord_sets,           i_coord_sets,\
            i_coord_sets+(ny*nx),   i_coord_sets+(ny*nx),   i_coord_sets+(ny*nx),\
            i_coord_sets+(2*ny*nx), i_coord_sets+(2*ny*nx), i_coord_sets+(2*ny*nx)]
                       ),\
        jnp.concatenate(
           [j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx),\
            j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx),\
            j_coord_sets, j_coord_sets+(ny*nx), j_coord_sets+(2*ny*nx)]
                       )
                       ])

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords,
                                                      (ny*nx*3, ny*nx*3),\
                                                      ksp_type="gmres",\
                                                      preconditioner="hypre",\
                                                      precondition_only=False,\
                                                      monitor_ksp=False)




    @custom_vjp
    def solver(q, C, u_trial, v_trial, h, delta_t, accm=0):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        h_t = h.copy()

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            dRu_du, dRv_du, dRh_du, \
            dRu_dv, dRv_dv, dRh_dv, \
            dRu_dh, dRv_dh, dRh_dh = sparse_jacrev(
                                                   get_uvh_residuals,
                                                   (u_1d, v_1d, h_1d, 
                                                   mu_ew, mu_ns, beta, 
                                                   h_t, accm, delta_t)
                                                          )

            nz_jac_values = jnp.concatenate(
                                [dRu_du[mask], dRu_dv[mask], dRu_dh[mask],\
                                 dRv_du[mask], dRv_dv[mask], dRv_dh[mask],\
                                 dRh_du[mask], dRh_dv[mask], dRh_dh[mask]]
                                           )

          
            rhs = -jnp.concatenate(get_uvh_residuals(u_1d, v_1d, h_1d, 
                                                     mu_ew, mu_ns, beta, 
                                                     h_t, accm, delta_t)
                                   )

            old_residual, residual, init_res = print_residual_things(
                                                  residual, rhs, init_res, i
                                                                    )
            
            

            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):(2*ny*nx)]
            h_1d = h_1d+du[(2*ny*nx):]
            
            rhs_new = -jnp.concatenate(get_uvh_residuals(
                                       u_1d, v_1d, h_1d, 
                                       mu_ew, mu_ns, beta, 
                                       h_t, accm, delta_t)
                                      )
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {jnp.max(jnp.abs(rhs))/jnp.max(jnp.abs(rhs_new))}")

        final_residual = jnp.max(jnp.abs(rhs_new))

        print("----------")
        print("Final residual: {}".format(final_residual))
        print("Total residual reduction factor: {}".format(initial_residual/final_residual))
        print("----------")
        
        return (u_1d.reshape((ny, nx)),
                v_1d.reshape((ny, nx)),
                h_1d.reshape((ny,nx)) )
    


    def solver_fwd(q, C, u_trial, v_trial, h, delta_t, accm=0):

        h_t = h.copy()

        u, v, h = solver(q, C, u_trial, v_trial, h, delta_t, accm)

        #NOTE: NOT FINISHED CODE
            
        dRu_du, dRv_du, dRh_du, \
        dRu_dv, dRv_dv, dRh_dv, \
        dRu_dh, dRv_dh, dRh_dh = sparse_jacrev(
                                               get_uvh_residuals_nonlinear,
                                               (u_1d, v_1d, h_1d, 
                                               q, C, 
                                               h_t, accm, delta_t)
                                                      )

        nz_jac_values = jnp.concatenate(
                            [dRu_du[mask], dRu_dv[mask], dRu_dh[mask],\
                             dRv_du[mask], dRv_dv[mask], dRv_dh[mask],\
                             dRh_du[mask], dRh_dv[mask], dRh_dh[mask]]
                                       )


        fwd_residuals = (u, v, h, nz_jac_values, q, C, h_t, accm, delta_t)

        return (u, v, h), fwd_residuals

    def solver_bwd(res, cotangent):

        u_bar, v_bar, h_bar = cotangent

        u, v, h, dJduvh_nz_jac_values, q, C, h_t, accm, delta_t = res



        lambda_ = la_solver(dJduvh_nz_values,
                            -jnp.concatenate([u_bar, v_bar, h_bar]),
                            transpose=True)

        lambda_u = lambda_[:(ny*nx)]
        lambda_v = lambda_[(ny*nx):(2*ny*nx)]
        lambda_h = lambda_[(2*ny*nx):]


        #phi_bar = (dG/dphi)^T lambda
        _, pullback_function = jax.vjp(get_uvh_residuals_nonlinear,
                                       (u.reshape(-1), v.reshape(-1), h.reshape(-1),
                                        q, C, h_t, source, delta_t)
                                      )
        _,_,_, q_bar, _,_,_,_ = pullback_function((lambda_u, lambda_v, lambda_h))

        
        return (q_bar.reshape((ny, nx)), None, None, None, None, None, None)


    return solver

def make_diva3d_velocity_solver_function(ny, nx, dy, dx, n_levels,
                                         b, ice_mask, n_iterations,
                                         mucoef_0, sliding="linear",
                                         periodic=False,
                                         temperature_field=None,
                                         n_timesteps=0):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+258.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    cc_vel_gradient                            = cc_vel_gradient_function(dy, dx, add_uv_ghost_cells)
    hgrads_fct                                 = gl_aware_driving_stress_function(dy, dx) 

    #DIVA-specific functions from grid:
    diva_viscosity = diva_cc_viscosity_function(dy, dx, cc_vel_gradient, mucoef_0, temperature_field)
    beta_raw_fct   = beta_function(b, sliding)
    beta_eff_fct   = diva_beta_eff_function(beta_raw_fct)
    new_shear      = diva_vertical_shear_function()
    reconstruct_3d = diva_reconstruct_3d_velocity_function()




    advection_step = make_advection_stepper(nx, ny, dx, dy, interp_cc_to_fc,
                                            add_uv_ghost_cells, add_scalar_ghost_cells,
                                            method="PPM")



    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_gl_aware(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf,
                                                       hgrads_fct)
    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(coords, (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              ksp_type="gmres",
                                                              preconditioner="hypre",
                                                              ksp_max_iter=60,
                                                              monitor_ksp=False)

    def momentum_solver(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h > 1e-10, u_trial, 0)
        v_trial = jnp.where(h > 1e-10, v_trial, 0)

        u_vv = jnp.zeros((ny, nx, n_levels)) + u_trial[..., None] 
        v_vv = jnp.zeros((ny, nx, n_levels)) + v_trial[..., None] 

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        zs = define_z_coordinates(b, h, n_levels)
        dudz = jnp.zeros((ny, nx, n_levels))
        dvdz = jnp.zeros((ny, nx, n_levels))

        for i in range(n_iterations):

            u_va = u_1d.reshape((ny, nx))
            v_va = v_1d.reshape((ny, nx))

            #1. update 3D viscosity from current (u_va, v_va, dudz, dvdz),
            #   vertically average it
            mu_vv, mu_va = diva_viscosity(q, u_va, v_va, dudz, dvdz, zs)

            #2. Arthern F2 integral, then beta_eff
            f2 = arthern_function(mu_vv, zs, m=2)
            beta, beta_eff = beta_eff_fct(C, u_vv[..., 0], v_vv[..., 0], h, f2)

            #3. interpolate the vertically-averaged viscosity to faces
            mu_va_gc = add_scalar_ghost_cells(mu_va)
            mu_ew, mu_ns = interp_cc_to_fc(mu_va_gc)
            #to account for calving front boundary condition, set effective viscosities
            #of faces of all cells with zero thickness to zero:
            mu_ew = mu_ew.at[:, 1:].set(jnp.where(ice_mask==0, 0, mu_ew[:, 1:]))
            mu_ew = mu_ew.at[:,:-1].set(jnp.where(ice_mask==0, 0, mu_ew[:,:-1]))
            mu_ns = mu_ns.at[1:, :].set(jnp.where(ice_mask==0, 0, mu_ns[1:, :]))
            mu_ns = mu_ns.at[:-1,:].set(jnp.where(ice_mask==0, 0, mu_ns[:-1,:]))

            #4. linear solve for (u_va, v_va) from linear SSA formulation
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta_eff)
                                                          )
            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d,
                                                                mu_ew, mu_ns, beta_eff))

            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d + du[:(ny*nx)]
            v_1d = v_1d + du[(ny*nx):]

            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d,
                                                                    mu_ew, mu_ns, beta_eff))
            if i == 0:
                initial_residual = jnp.max(jnp.abs(rhs))
            print(f"linear residual reduction factor: {jnp.max(jnp.abs(rhs))/jnp.max(jnp.abs(rhs_new))}")

            #5. update dudz/dvdz for the next iteration's viscosity (Eq. 21)
            u_va = u_1d.reshape((ny, nx))
            v_va = v_1d.reshape((ny, nx))
            dudz, dvdz = new_shear(mu_vv, u_va, v_va, beta_eff, zs)
           
            #dudz *= 0
            #dvdz *= 0

            #6. reconstruct the full 3D velocity field
            u_vv, v_vv = reconstruct_3d(u_va, v_va, dudz, dvdz, beta, f2, zs)
            #weirdly, it seems that u_va does not match the average taken over the vertical
            #layers of u_vv after this function is executed... Maybe that disappears
            #as the number of vertical levels is increased...


        final_residual = jnp.max(jnp.abs(rhs_new))
        print("----------")
        print("Final residual: {}".format(final_residual))
        print("Total residual reduction factor: {}".format(initial_residual/final_residual))
        print("----------")

        return u_va, v_va, u_vv, v_vv, zs
    

    def run_model_forward(q, C, u_trial, v_trial, h_init):
        h = h_init
        u_va, v_va = u_trial, v_trial

        t_cum = 0
        if n_timesteps:
            for ts in range(n_timesteps):
                #plt.imshow(h)
                #plt.colorbar()
                #plt.show()
                
                u_va, v_va, u_vv, v_vv, zs = momentum_solver(q, C, u_va, v_va, h)

                accumulation = jnp.where(h>0, 0.3, 0)

                delta_t = 0.45*(dx/jnp.max(jnp.sqrt(u_va**2+v_va**2)))
                print(delta_t)
                
                t_cum += delta_t
                print(f"Time: {t_cum} years")

                #delta_t = jnp.maximum(delta_t, 0.06)

                h = advection_step(u_va.reshape(-1), v_va.reshape(-1),
                                   h.reshape(-1), source=accumulation,
                                   delta_t=delta_t)
        else:
            u_va, v_va, u_vv, v_vv, zs = momentum_solver(q, C, u_va, v_va, h)

        dhdt_final = ( advection_step(u_va.reshape(-1), v_va.reshape(-1),
                               h.reshape(-1), source=accumulation,
                               delta_t=delta_t)\
                      - h 
                     ) /delta_t


        return u_va, v_va, u_vv, v_vv, zs, h, dhdt_final

    return run_model_forward


def make_picard_velocity_solver_function_custom_vjp(ny, nx, dy, dx,
                                                    b, ice_mask, n_iterations,
                                                    mucoef_0, sliding="linear",
                                                    periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf, mucoef_0)

    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),
    #                                                          ksp_type="gmres",
    #                                                          preconditioner="hypre",
    #                                                          precondition_only=False,
    #                                                          ksp_max_iter=60,
    #                                                          monitor_ksp=False)
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2))




    @custom_vjp
    def solver(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0
        
        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        for i in range(n_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            #nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            
            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d + du[:(ny*nx)]
            v_1d = v_1d + du[(ny*nx):]
        
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {jnp.max(jnp.abs(rhs))/jnp.max(jnp.abs(rhs_new))}")

        final_residual = jnp.max(jnp.abs(rhs_new))

        print("----------")
        print("Final residual: {}".format(final_residual))
        print("Total residual reduction factor: {}".format(initial_residual/final_residual))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, C, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa, \
                                        (u.reshape(-1), v.reshape(-1), q, C, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, C, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, C, h = res
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
        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, C, h.reshape(-1)
                                      )
        _, _, mu_bar, _, _ = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        return (mu_bar.reshape((ny, nx)), None, None, None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver


def make_pic_velocity_solver_function_densetest(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_pic_iterations, n_newt_iterations,
                                                 mucoef_0, C_0, sliding="linear",
                                                 periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    ##############
    ##setting up bvs and coords for a single block of the jacobian
    #basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 2,
    #                                                                              periodic_x=periodic)

    #i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    #j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))
    #mask = (i_coordinate_sets>=0)


    #sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
    #                                                    basis_vectors,\
    #                                                    i_coordinate_sets,\
    #                                                    j_coordinate_sets,\
    #                                                    mask,\
    #                                                    2,
    #                                                    active_indices=(0,1)
    #                                                   )
    ##sparse_jacrev = jax.jit(sparse_jacrev)
   
    jacrev = jax.jacrev(get_uv_residuals_linear_ssa, argnums=(0,1))

    #i_coordinate_sets = i_coordinate_sets[mask]
    #j_coordinate_sets = j_coordinate_sets[mask]
    ##############

    #coords = jnp.stack([
    #                jnp.concatenate(
    #                            [i_coordinate_sets,         i_coordinate_sets,\
    #                             i_coordinate_sets+(ny*nx), i_coordinate_sets+(ny*nx)]
    #                               ),\
    #                jnp.concatenate(
    #                            [j_coordinate_sets, j_coordinate_sets+(ny*nx),\
    #                             j_coordinate_sets, j_coordinate_sets+(ny*nx)]
    #                               )
    #                   ])

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=False,
    #                                                          monitor_ksp=False)


    
    res_fct = lambda x: jnp.max(jnp.abs(x))
    #res_fct = lambda x: jnp.mean(jnp.abs(x))

    
    omega=1
    

    def solver(q, p, u_trial, v_trial, h):
        #plt.imshow(q, vmin=-2, vmax=0.5, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        #plt.imshow(p, vmin=-4, vmax=4, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        for i in range(n_pic_iterations):
            #NOTE: making this twice as large makes PIG look a little better...
            #mu_ew = 2*mu_ew
            #mu_ns = 2*mu_ns

            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(jnp.log10(mu_ew[:,1:].reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()
            #plt.imshow(jnp.log10(beta.reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()

            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            print("constructing LA problem")
            dJ_du, dJ_dv = jacrev(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
            dJu_du, dJu_dv = dJ_du
            dJv_du, dJv_dv = dJ_dv

            full_jac = jnp.block([[dJu_du, dJu_dv],
                                  [dJv_du, dJv_dv]])
            #full_jac = jnp.block([[dJu_du, dJv_du],
            #                      [dJu_dv, dJv_dv]])
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            print(full_jac[:(ny*nx),:(ny*nx)])
            print("--------------------------------")
            print(full_jac[(ny*nx):,:(ny*nx)])
            print("--------------------------------")
            print(full_jac[:(ny*nx),(ny*nx):])
            print("--------------------------------")
            print(full_jac[(ny*nx):,(ny*nx):])
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            print("--------------------------------")
            
            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            
            #plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx), (ny*nx):])).reshape((ny*nx,nx*ny)))
            #plt.colorbar()
            #plt.show()


            #plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),26])).reshape((ny,nx)))
            #plt.show()
            #plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),(ny*nx)+26])).reshape((ny,nx)))
            #plt.show()

            #nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            print("solving LA problem")
            du = jax.scipy.linalg.solve(full_jac, rhs)

            #plt.imshow(h>0, cmap="Grays")
            #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=6)
            #plt.colorbar()
            #plt.show()

            print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        
        final_residual_pic = res_fct(rhs_new)

        print("Final Picard residual: {}".format(final_residual_pic))
        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):
            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, p, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            du = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        print("Final Newton residual: {}".format(final_residual))
        print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")


        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver



#NOTE: This is extreeeeeeeeemly shit
def make_pseudotime_velocity_solver_function(ny, nx, dy, dx,
                                             b, ice_mask,
                                             n_iterations,
                                             mucoef_0, C_0,
                                             sliding="linear",
                                             res_tol=1e-10,
                                             periodic=False,
                                             B_field=None,
                                             temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    nc_velocity_gradient                       = nc_velocity_gradient_function(dy, dx,
                                                                               add_uv_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    beta = 1e-12

    def conditional(state):
        i, res, _,_ = state
        return jnp.logical_and(i<n_iterations, jnp.abs(res)>res_tol)

    def solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0


        vel_mask = jnp.ones_like(u_trial)
        #vel_mask = vel_mask.at[:2, :].set(0)
        #vel_mask = vel_mask.at[-2:, :].set(0)
        #vel_mask = vel_mask.at[:, :2].set(0)
        #vel_mask = vel_mask.reshape(-1)

        initial_state = (0, residual, u_1d, v_1d)

        def update(state):
            i, res, u_1d, v_1d = state
            
            u_1d = u_1d*vel_mask
            v_1d = v_1d*vel_mask

            #plt.imshow(u_1d.reshape((ny, nx)))
            #plt.colorbar()
            #plt.show()

            ur, vr = get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d)
            #plt.imshow(ur.reshape((ny, nx)))
            #plt.colorbar()
            #plt.show()
            res = res_fct(jnp.concatenate([ur, vr]))

            jax.debug.print("PT res: {x}", x=res)

            return i+1, res, u_1d - beta*ur, v_1d - beta*vr

        #i, res, u_1d, v_1d = jax.lax.while_loop(conditional, update, initial_state)
        i, res, u_1d, v_1d = fake_lax_while_loop(conditional, update, initial_state)

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver

def make_action_velocity_solver_function(ny, nx, dy, dx,
                                         b, ice_mask,
                                         pic_iterations,
                                         newt_iterations,
                                         mucoef_0, C_0,
                                         sliding="linear",
                                         res_tol=1e-10,
                                         periodic=False,
                                         B_field=None,
                                         temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    interp_nc_to_fc                            = interp_nc_to_fc_function(ny, nx)
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    
    cc_gradient                                = cc_gradient_function(dy, dx)
    cc_vel_gradient                            = cc_vel_gradient_function(dy, dx,
                                                               add_uv_ghost_cells)
    nc_vel_gradient                            = nc_velocity_gradient_function(dy, dx,
                                                               add_uv_ghost_cells)
    
    beta_fct = beta_function(b, sliding)

    #get_action = action_functional_function(ny, nx, dy, dx, b,
    #                                        cc_gradient,
    #                                        cc_vel_gradient,
    #                                        beta_fct,
    #                                        add_uv_ghost_cells,
    #                                        add_scalar_ghost_cells,
    #                                        extrapolate_over_cf,
    #                                        mucoef_0, C_0,
    #                                        temperature_field)
    
    get_action = node_centred_action_functional_function_no_cf(ny, nx, dy, dx, b,
                                          nc_vel_gradient,
                                          beta_fct,
                                          add_uv_ghost_cells,
                                          add_scalar_ghost_cells,
                                          mucoef_0, C_0,
                                          temperature_field)



    residuals_function = jax.grad(get_action, argnums=(0,1))
    
    #############
    #setting up bvs and coords for a single block of the jacobian
    #NOTE: Need to adjust discretisation so stencil radius can stay at 1 rather than 2!!!
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 2,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    basis_vectors = jnp.stack(basis_vectors).astype(jnp.float64)
    #sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
    #                                                    basis_vectors,\
    #                                                    2,
    #                                                    active_indices=(0,1)
    #                                                   )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   

    assemble_jacobian = assemble_sparse_2x2_block_jacobian_function_general(basis_vectors, ny*nx,
                                                                            mask,
                                                                            residuals_function)
    #assemble_jacobian_comparison = assemble_sparse_2x2_block_jacobian_function(basis_vectors, ny*nx, mask,
    #                                                                get_uv_residuals_linear_ssa)


    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    cg_solver = make_sparse_dpcg_solver_jsp_comp(coords, extract_inverse_diagonal, ny*nx*2,
                                            iterations=1000)
    #cg_solver = make_sparse_damped_jacobi_solver(sparse_matvec, extract_inverse_diagonal,
    #                                        iterations=30)


    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              ksp_max_iter=20,
                                                              monitor_ksp=False)


    


    

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    omega=1

    def picard_conditional(state):
        i, res, _,_,_,_,_ = state
        return jnp.logical_and(i<pic_iterations, jnp.abs(res)>res_tol)

    def newton_conditional(state):
        i, res, _,_,_,_ = state
        return jnp.logical_and(i<newt_iterations, jnp.abs(res)>res_tol)

    #NOTE: TODO
    #@custom_vjp 
    def picnewton_solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        #beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
        
        duv = jnp.zeros((nx*ny*2,))

        initial_state = (0, residual, u_1d, v_1d, h_1d, duv) 

        def newton_update(state):
            i, res, u_1d, v_1d, h_1d, duv = state

            #jax.debug.print("Pic res: {x}", x=res)

            #start_t = time.time()
            #nz_jac_values, rhs = assemble_jacobian_comparison(
            #    u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta
            #)
            #end_t = time.time()
            #print(f"act time: {time}")

            nz_jac_values, rhs = assemble_jacobian(
                u_1d, v_1d, [q, p, h_1d]
            )
            
            
            #full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            ##plt.imshow(full_jac)
            ##plt.colorbar()
            ##plt.show()
            ##raise
            ##
            ##print(nz_jac_values)
            ##raise

            #Ru, Rv = residuals_function(u_1d, v_1d, q, p, h_1d)

            ##plt.imshow(jnp.log10(jnp.abs(Ru).reshape((ny, nx))))
            ##plt.colorbar()
            ##plt.show()
            ###raise

            #dRdu, dRdv = jax.jacrev(residuals_function, argnums=(0,1))(u_1d, v_1d, q, p, h_1d)
            #dRu_du = dRdu[0].reshape((ny*nx, ny*nx))
            #dRu_dv = dRdu[1].reshape((ny*nx, ny*nx))
            #dRv_du = dRdv[0].reshape((ny*nx, ny*nx))
            #dRv_dv = dRdv[1].reshape((ny*nx, ny*nx))

            #dense_full_jac = jnp.block([[dRu_du, dRu_dv],
            #                            [dRv_du, dRv_dv]])

            ##plt.imshow(dense_full_jac)
            ##plt.colorbar()
            ##plt.show()
            #plt.imshow(dense_full_jac-full_jac)
            #plt.colorbar()
            #plt.show()
            ##
            ##plt.imshow(jnp.log10(jnp.abs((dense_full_jac-dense_full_jac.T)/dense_full_jac)))
            ##plt.colorbar()
            ##plt.show()
            ##raise

            #duv = 0.5 * cg_solver(nz_jac_values, rhs, duv)
            duv = 0.25 * la_solver(nz_jac_values, rhs)

#            print(duv)

            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
            #
            #mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
            #mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
            #beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            #return (i+1, res_fct(rhs), u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv)

#        def newton_update(state):
#            i, res, u_1d, v_1d, h_1d, duv = state
#            
#            jax.debug.print("Newt res: {x}", x=res)
#
#            nz_jac_values, rhs = assemble_jacobian_nonlinear(
#                u_1d, v_1d, q, p, h_1d
#            )
#            
#            full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
#            full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
#            
#            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
#            #plt.colorbar()
#            #plt.show()
#            #
#            #plt.imshow(jnp.log(jnp.abs(full_jac-jnp.transpose(full_jac))).reshape((ny*nx*2,2*nx*ny)))
#            #plt.colorbar()
#            #plt.show()
#
#            #jax.debug.print("NZ jac values: {x}", x=nz_jac_values)
#
#            duv = cg_solver(nz_jac_values, rhs, duv)

            jax.debug.print("newt res: {res}", res=res)
#
#            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
#            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
#            
            return (i+1, res_fct(rhs), u_1d, v_1d, h_1d, duv)
            
        #i, res, u_1d, v_1d, h_1d, duv = jax.lax.while_loop(newton_conditional, newton_update, initial_state)
        i, res, u_1d, v_1d, h_1d, duv = fake_lax_while_loop(newton_conditional, newton_update, initial_state)
        
#        newt_initial_state = (0, res, u_1d, v_1d, h_1d, duv)

#        #j, res, u_1d, v_1d, h_1d, duv = jax.lax.while_loop(newton_conditional, newton_update, newt_initial_state)
#        j, res, u_1d, v_1d, h_1d, duv = fake_lax_while_loop(newton_conditional, newton_update, newt_initial_state)


        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    #NOTE: TODO
    #def solver_fwd(q, p, u_trial, v_trial, h):
    #    u, v = solver(q, p, u_trial, v_trial, h)

    return picnewton_solver

def make_picnewton_velocity_solver_function_acrobatic(ny, nx, dy, dx,
                                              b, ice_mask,
                                              pic_iterations,
                                              newt_iterations,
                                              mucoef_0, C_0,
                                              sliding="linear",
                                              res_tol=1e-10,
                                              periodic=False,
                                              B_field=None,
                                              temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    interp_nc_to_fc                            = interp_nc_to_fc_function(ny, nx)
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    nc_velocity_gradient                       = nc_velocity_gradient_function(dy, dx,
                                                                               add_uv_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    
    fc_viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    
    nc_viscosity_fct = node_centred_viscosity_function(ny, nx, dy, dx,
                                                   add_scalar_ghost_cells,
                                                   nc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)


    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    #get_uv_residuals_nonlinear_ssa = compute_nonlinear_ssa_residuals_function_acrobatic(ny, nx,
    #get_uv_residuals_nonlinear_ssa = compute_nonlinear_ssa_residuals_function_acrobatic_symmetric(ny, nx,
    #                                                   dy, dx, b,
    #                                                   interp_cc_to_fc,
    #                                                   interp_cc_to_nc,
    #                                                   interp_nc_to_fc,
    #                                                   interp_fc_to_nc,
    #                                                   fc_velocity_gradient,
    #                                                   nc_velocity_gradient,
    #                                                   cc_gradient,
    #                                                   beta_fct,
    #                                                   add_uv_ghost_cells,
    #                                                   add_scalar_ghost_cells,
    #                                                   mucoef_0, C_0,
    #                                                   temperature_field)

    get_uv_residuals_nonlinear_ssa = compute_nonlinear_ssa_residuals_function_variational_visc_messing_round_no_cf(ny, nx,
    #get_uv_residuals_nonlinear_ssa = compute_nonlinear_ssa_residuals_function_variational_visc_an_option(ny, nx,
                                                       dy, dx, b,
                                                       interp_cc_to_fc,
                                                       interp_cc_to_nc,
                                                       fc_velocity_gradient,
                                                       nc_velocity_gradient,
                                                       cc_gradient,
                                                       beta_fct,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field,
                                                       extrapolate_over_cf)


    get_uv_residuals_nonlinear_ssa_diagnosis = compute_nonlinear_ssa_residuals_function_variational_visc_diagnosis(ny, nx,
                                                       dy, dx, b,
                                                       interp_cc_to_fc,
                                                       interp_cc_to_nc,
                                                       fc_velocity_gradient,
                                                       nc_velocity_gradient,
                                                       cc_gradient,
                                                       beta_fct,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field,
                                                       extrapolate_over_cf)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    basis_vectors = jnp.stack(basis_vectors).astype(jnp.float64)
    #sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
    #                                                    basis_vectors,\
    #                                                    2,
    #                                                    active_indices=(0,1)
    #                                                   )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   


    assemble_jacobian_linear = assemble_sparse_2x2_block_jacobian_function(basis_vectors, ny*nx, mask,
                                                                    get_uv_residuals_linear_ssa)
    assemble_jacobian_nonlinear = assemble_sparse_2x2_block_jacobian_function_nl(basis_vectors, ny*nx, mask,
                                                                    get_uv_residuals_nonlinear_ssa)

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              ksp_max_iter=20,
                                                              monitor_ksp=False)

    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    cg_solver = make_sparse_dpcg_solver_jsp_comp(coords, extract_inverse_diagonal, ny*nx*2,
                                            iterations=1000)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    omega=1


    def picard_conditional(state):
        i, res, _,_,_,_,_,_,_,_ = state
        return jnp.logical_and(i<pic_iterations, jnp.abs(res)>res_tol)

    def newton_conditional(state):
        i, res, _,_,_,_ = state
        return jnp.logical_and(i<newt_iterations, jnp.abs(res)>res_tol)

    #NOTE: TODO
    #@custom_vjp 
    def picnewton_solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
        mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
        
        duv = jnp.zeros((nx*ny*2,))

        initial_state = (0, residual, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv) 

        def pic_update(state):
            i, res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = state

            jax.debug.print("Pic res: {x}", x=res)

            nz_jac_values, rhs = assemble_jacobian_linear(
                u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta
            )

            #jax.debug.print("NZ jac values: {x}", x=nz_jac_values)

            #duv = cg_solver(nz_jac_values, rhs, duv)
            duv = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
            mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            return (i+1, res_fct(rhs), u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv)

        def newton_update(state):
            i, res, u_1d, v_1d, h_1d, duv = state
            
            jax.debug.print("Newt res: {x}", x=res)

            #get_uv_residuals_nonlinear_ssa_diagnosis(u_1d, v_1d, q, p, h_1d)

            nz_jac_values, rhs = assemble_jacobian_nonlinear(
                u_1d, v_1d, q, p, h_1d
            )

            #plt.imshow(rhs[:(nx*ny)].reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()
            #plt.imshow(rhs[(nx*ny):].reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()
            
            #full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            
            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            #
            #plt.imshow(jnp.log(jnp.abs(full_jac-jnp.transpose(full_jac))).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()

            #jax.debug.print("NZ jac values: {x}", x=nz_jac_values)

            #duv = cg_solver(nz_jac_values, rhs, jnp.zeros_like(duv))
            duv = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
            
            return (i+1, res_fct(rhs), u_1d, v_1d, h_1d, duv)
            
        #i, res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = jax.lax.while_loop(picard_conditional, pic_update, initial_state)
        i, res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = fake_lax_while_loop(picard_conditional, pic_update, initial_state)
        
        newt_initial_state = (0, res, u_1d, v_1d, h_1d, duv)

        #j, res, u_1d, v_1d, h_1d, duv = jax.lax.while_loop(newton_conditional, newton_update, newt_initial_state)
        j, res, u_1d, v_1d, h_1d, duv = fake_lax_while_loop(newton_conditional, newton_update, newt_initial_state)


        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    #NOTE: TODO
    #def solver_fwd(q, p, u_trial, v_trial, h):
    #    u, v = solver(q, p, u_trial, v_trial, h)

    return picnewton_solver


def make_pic_velocity_solver_function_gpusafe(ny, nx, dy, dx,
                                              b, ice_mask,
                                              n_iterations,
                                              mucoef_0, C_0,
                                              sliding="linear",
                                              res_tol=1e-10,
                                              periodic=False,
                                              B_field=None,
                                              temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    nc_velocity_gradient                       = nc_velocity_gradient_function(dy, dx,
                                                                               add_uv_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    
    fc_viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    
    nc_viscosity_fct = node_centred_viscosity_function(ny, nx, dy, dx,
                                                   add_scalar_ghost_cells,
                                                   nc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)


    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    basis_vectors = jnp.stack(basis_vectors).astype(jnp.float64)
    #sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
    #                                                    basis_vectors,\
    #                                                    2,
    #                                                    active_indices=(0,1)
    #                                                   )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   


    assemble_jacobian = assemble_sparse_2x2_block_jacobian_function(basis_vectors, ny*nx, mask,
                                                                    get_uv_residuals_linear_ssa)


    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    #cg_solver = make_sparse_dpcg_solver_jsp_comp(coords, extract_inverse_diagonal, ny*nx*2,
    #                                        iterations=100)
    cg_solver = make_sparse_damped_jacobi_solver(sparse_matvec, extract_inverse_diagonal,
                                            iterations=30)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    omega=1


    def conditional(state):
        i, res, _,_,_,_,_,_,_,_ = state
        return jnp.logical_and(i<n_iterations, jnp.abs(res)>res_tol)

    #NOTE: TODO
    #@custom_vjp 
    def solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
        mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
        
        duv = jnp.zeros((nx*ny*2,))

        initial_state = (0, residual, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv) 

        def update(state):
            i, res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = state

            jax.debug.print("Pic res: {x}", x=res)

            nz_jac_values, rhs = assemble_jacobian(
                u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta
            )

            #jax.debug.print("NZ jac values: {x}", x=nz_jac_values)

            duv = cg_solver(nz_jac_values, rhs, duv)

            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
            mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            return (i+1, res_fct(rhs), u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv)


        i, res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = jax.lax.while_loop(conditional, update, initial_state)

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    #NOTE: TODO
    #def solver_fwd(q, p, u_trial, v_trial, h):
    #    u, v = solver(q, p, u_trial, v_trial, h)

    return solver



def make_advection_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
                           add_uv_ghost_cells, add_s_ghost_cells,
                           method="PPM"):

    def advection_step(u_1d, v_1d, h_1d, source=0, delta_t=0.08):
        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        u_full, v_full = add_uv_ghost_cells(u, v)
        h_full = add_s_ghost_cells(h)

        u_full = linear_extrapolate_over_cf_dynamic_thickness(u_full, h_full)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(v_full, h_full)
        h_full = linear_extrapolate_over_cf_dynamic_thickness(h_full, h_full)

        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)


        if method=="FOU": 
            u_signs = jnp.where(u_fc_ew>0, 1, -1)
            v_signs = jnp.where(v_fc_ns>0, 1, -1)

            ##face-centred values according to first-order upwinding
            h_fc_fou_ew = jnp.where(u_fc_ew>0, h_full[1:-1,:-1], h_full[1:-1, 1:])
            h_fc_fou_ns = jnp.where(v_fc_ns>0, h_full[1:, 1:-1], h_full[-1:,1:-1])

            flux_term = (u_fc_ew[:,1:]*h_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*h_fc_fou_ew[:,:-1])*dy*delta_t +\
                        (v_fc_ns[:-1,:]*h_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*h_fc_fou_ns[1:,:])*dx*delta_t

        elif method=="PPM":
            h_fc_fou_ew = jnp.where(
                u_fc_ew > 0,
                h_full[1:-1, :-1],
                h_full[1:-1, 1:]
            )

            h_fc_fou_ns = jnp.where(
                v_fc_ns > 0,
                h_full[1:, 1:-1],
                h_full[:-1, 1:-1]
            )

            flux_term = - (
                (u_fc_ew[:,:-1] * h_fc_fou_ew[:,:-1]
                 - u_fc_ew[:,1:] * h_fc_fou_ew[:,1:])
                * dy * delta_t
                +
                (v_fc_ns[1:,:] * h_fc_fou_ns[1:,:]
                 - v_fc_ns[:-1,:] * h_fc_fou_ns[:-1,:])
                * dx * delta_t
            )

        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
        flux_term = jnp.where(h>0, flux_term, 0)

        return h + source*delta_t - flux_term/(dy*dx)

    return jax.jit(advection_step)



def make_advsrc_damage_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
                           add_uv_ghost_cells, add_s_ghost_cells,
                               mucoef_0, prs_function):

    def advection_step(u_1d, v_1d, h_1d, D_1d, delta_t=0.08, ts=1):
        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        D = D_1d.reshape((ny, nx))

        q = jnp.log((1-D)/(mucoef_0+1e-10))


        u_full, v_full = add_uv_ghost_cells(u, v)
        h_full = add_s_ghost_cells(h)
        D_full = add_s_ghost_cells(D)

        u_full = linear_extrapolate_over_cf_dynamic_thickness(u_full, h_full)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(v_full, h_full)
        h_full = linear_extrapolate_over_cf_dynamic_thickness(h_full, h_full)
        D_full = linear_extrapolate_over_cf_dynamic_thickness(D_full, h_full)
        



        ### ADVECTION TERM ###########

        u_fc_ew, _ = interp_cc_to_fc(u_full)
        _, v_fc_ns = interp_cc_to_fc(v_full)

        u_signs = jnp.where(u_fc_ew>0, 1, -1)
        v_signs = jnp.where(v_fc_ns>0, 1, -1)


        ##face-centred values according to first-order upwinding
        D_fc_fou_ew = jnp.where(u_fc_ew>0, D_full[1:-1,:-1], D_full[1:-1, 1:])
        D_fc_fou_ns = jnp.where(v_fc_ns>0, D_full[1:, 1:-1], D_full[-1:,1:-1])

        flux_term = (u_fc_ew[:,1:]*D_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*D_fc_fou_ew[:,:-1])*dy*delta_t +\
                    (v_fc_ns[:-1,:]*D_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*D_fc_fou_ns[1:,:])*dx*delta_t
        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
        flux_term = jnp.where(h>1e-2, flux_term, 0)

        

        ### SOURCE TERM ###########
        prs = prs_function(q, u, v, h) * ( (h>0).astype(int) )

        prs = jnp.maximum(prs, 0)


        plt.imshow(jnp.log10(prs))#, vmin=0, vmax=200_000)
        plt.colorbar()
        plt.savefig(f"{nm_home}/bits_of_data/damage_gub/prs_{ts}.png", dpi=150)
        plt.close()


        source = ((1/450_000) * prs)**8 / (1-D)**8 * jnp.where(prs>0, 1, -1)

        source_term = source * delta_t

        return D + source_term - flux_term/(dy*dx)

    #return jax.jit(advection_step)
    return advection_step

#def make_advsrc_effective_damthk_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
#                               add_uv_ghost_cells, add_s_ghost_cells,
#                               mucoef_0, prs_function):
#
#    def advection_step(u_1d, v_1d, h_1d, D_1d, delta_t=0.08, ts=1):
#        u = u_1d.reshape((ny, nx))
#        v = v_1d.reshape((ny, nx))
#        h = h_1d.reshape((ny, nx))
#        D = D_1d.reshape((ny, nx))
#        
#
#        #q = jnp.log((1-D)/(mucoef_0+1e-10))
#        q = jnp.log(1/(mucoef_0+1e-10))
#
#
#        u_full, v_full = add_uv_ghost_cells(u, v)
#        h_full = add_s_ghost_cells(h)
#        D_full = add_s_ghost_cells(D)
#
#        u_full = linear_extrapolate_over_cf_dynamic_thickness(u_full, h_full)
#        v_full = linear_extrapolate_over_cf_dynamic_thickness(v_full, h_full)
#        h_full = linear_extrapolate_over_cf_dynamic_thickness(h_full, h_full)
#        D_full = linear_extrapolate_over_cf_dynamic_thickness(D_full, h_full)
#
#
#        Dh_full = D_full*h_full
#
#
#
#        ### ADVECTION TERM ###########
#
#        u_fc_ew, _ = interp_cc_to_fc(u_full)
#        _, v_fc_ns = interp_cc_to_fc(v_full)
#
#        u_signs = jnp.where(u_fc_ew>0, 1, -1)
#        v_signs = jnp.where(v_fc_ns>0, 1, -1)
#
#
#        ##face-centred values according to first-order upwinding
#        Dh_fc_fou_ew = jnp.where(u_fc_ew>0, Dh_full[1:-1,:-1], Dh_full[1:-1, 1:])
#        Dh_fc_fou_ns = jnp.where(v_fc_ns>0, Dh_full[1:, 1:-1], Dh_full[-1:,1:-1])
#
#        flux_term = (u_fc_ew[:,1:]*Dh_fc_fou_ew[:,1:] - u_fc_ew[:,:-1]*Dh_fc_fou_ew[:,:-1])*dy*delta_t +\
#                    (v_fc_ns[:-1,:]*Dh_fc_fou_ns[:-1,:] - v_fc_ns[1:,:]*Dh_fc_fou_ns[1:,:])*dx*delta_t
#        #to keep calving front in same location, prevent any flux into or out of ice-free cells!
#        flux_term = jnp.where(h>1e-2, flux_term, 0)
#
#        
#
#        ### SOURCE TERM ###########
#        prs = prs_function(q, u, v, h) * ( (h>0).astype(int) )
#        rst, dst = 
#
#        #prs = jnp.maximum(prs, 0)
#
#        gamma = 2e-15
#
#        source = gamma * c.A_COLD * ((h * prs  - c.RHO_I*c.g*D*h)/(1-D))**4
#
#
#        plt.imshow(jnp.log10(source))#, vmin=0, vmax=200_000)
#        plt.colorbar()
#        plt.savefig(f"{nm_home}/bits_of_data/damage_gub/source_{ts}.png", dpi=150)
#        plt.close()
#       
#        plt.imshow(jnp.log10(prs))#, vmin=0, vmax=200_000)
#        plt.colorbar()
#        plt.savefig(f"{nm_home}/bits_of_data/damage_gub/prs_{ts}.png", dpi=150)
#        plt.close()
#
#
#        source_term = source * delta_t
#
#        return D + (1/(h+1e-10)) * (source_term - flux_term/(dy*dx)) * (h>0).astype(int)
#
#    #return jax.jit(advection_step)
#    return advection_step





###########SECOND PPM ATTEMPT

def ppm_reconstruct_1d(q, dx):

    dq_cd  = jnp.zeros_like(q)
    dq_lsd = jnp.zeros_like(q)
    dq_rsd = jnp.zeros_like(q)

    dq_cd  = dq_cd.at[1:-1].set((q[2:] - q[:-2]) / (2*dx))
    dq_lsd = dq_lsd.at[1:].set((q[1:] - q[:-1]) / dx)
    dq_rsd = dq_rsd.at[:-1].set((q[1:] - q[:-1]) / dx)

    same_sign = dq_lsd * dq_rsd > 0

    dq_lim = (
        jnp.sign(dq_cd)
        * jnp.minimum(
            jnp.abs(dq_cd),
            jnp.minimum(
                2.0*jnp.abs(dq_lsd),
                2.0*jnp.abs(dq_rsd)
            )
        )
    )

    dq = jnp.where(same_sign, dq_lim, 0.0)



    #dq_cd  = jnp.zeros_like(q)
    #dq_lsd = jnp.zeros_like(q)
    #dq_rsd = jnp.zeros_like(q)

    #dq_cd  = dq_cd.at[1:-1].set(
    #    (q[2:] - q[:-2])/(2*dx)
    #)

    #dq_lsd = dq_lsd.at[1:].set(
    #    (q[1:] - q[:-1])/dx
    #)

    #dq_rsd = dq_rsd.at[:-1].set(
    #    (q[1:] - q[:-1])/dx
    #)

    ## monotonic linear slopes

    #same_sign = dq_lsd * dq_rsd > 0

    #dq_lim = (
    #    jnp.sign(dq_cd)
    #    * jnp.minimum(
    #        jnp.abs(dq_cd),
    #        jnp.minimum(
    #            jnp.abs(dq_lsd),
    #            jnp.abs(dq_rsd)
    #        )
    #    )
    #)

    #dq = jnp.where(same_sign, dq_lim, 0.0)

    # fourth-order interface values

    qL = jnp.zeros_like(q)

    qL = qL.at[1:].set(
        0.5*(q[:-1] + q[1:])
        - (1.0/6.0)*(dq[1:] - dq[:-1])*dx
    )

    qR = jnp.zeros_like(q)
    qR = qR.at[:-1].set(qL[1:])

    qL = qL.at[0].set(q[0])
    qR = qR.at[-1].set(q[-1])

    # Colella-Woodward monotonicity constraints

    q6   = 6.0*(q - 0.5*(qL + qR))
    dqlr = qR - qL

    cond1 = ((qR-q)*(q-qL)) <= 0.0
    cond2 = dqlr*dqlr < dqlr*q6
    cond3 = -dqlr*dqlr > dqlr*q6

    qL = jnp.where(cond1, q, qL)
    qR = jnp.where(cond1, q, qR)

    qL = jnp.where(cond2, 3.0*q - 2.0*qR, qL)
    qR = jnp.where(cond3, 3.0*q - 2.0*qL, qR)

    return qL, qR

def ppm_flux_x(phi, u, dx, dt):

    phiL, phiR = jax.vmap(
        ppm_reconstruct_1d,
        in_axes=(0, None),
        out_axes=(0, 0)
    )(phi, dx)

    dq = phiR - phiL
    q6 = 6.0*phi - 3.0*(phiL + phiR)

    u_face = 0.5*(u[:, :-1] + u[:, 1:])

    sigma = jnp.abs(u_face) * dt / dx
    sigma = jnp.clip(sigma, 0.0, 1.0)

    left_state = (
        phiR[:, :-1]
        - 0.5*sigma*(
            dq[:, :-1]
            - (1.0 - 2.0*sigma/3.0)*q6[:, :-1]
        )
    )

    right_state = (
        phiL[:, 1:]
        + 0.5*sigma*(
            dq[:, 1:]
            + (1.0 - 2.0*sigma/3.0)*q6[:, 1:]
        )
    )

    phi_face = jnp.where(
        u_face > 0.0,
        left_state,
        right_state
    )

    return u_face * phi_face

def ppm_flux_y(phi, v, dy, dt):

    phiL, phiR = jax.vmap(
        ppm_reconstruct_1d,
        in_axes=(1, None),
        out_axes=(1, 1)
    )(phi, dy)

    dq = phiR - phiL
    q6 = 6.0*phi - 3.0*(phiL + phiR)

    v_face = 0.5*(v[:-1, :] + v[1:, :])

    sigma = jnp.abs(v_face) * dt / dy
    sigma = jnp.clip(sigma, 0.0, 1.0)

    # top cell contributes when v<0
    top_state = (
        phiR[:-1, :]
        - 0.5*sigma*(
            dq[:-1, :]
            - (1.0 - 2.0*sigma/3.0)*q6[:-1, :]
        )
    )

    # bottom cell contributes when v>0
    bottom_state = (
        phiL[1:, :]
        + 0.5*sigma*(
            dq[1:, :]
            + (1.0 - 2.0*sigma/3.0)*q6[1:, :]
        )
    )

    phi_face = jnp.where(
        v_face > 0.0,
        bottom_state,
        top_state
    )

    return v_face * phi_face


def make_advsrc_effective_damthk_stepper_threshold_ppmish(
                               nx, ny, dx, dy,
                               interp_cc_to_fc,
                               add_uv_ghost_cells,
                               add_s_ghost_cells,
                               mucoef_0,
                               prs_function,
                               advtype="PPM"):

    def advection_step(u_1d, v_1d, h_1d, D_1d,
                       delta_t=0.08,
                       ts=1, source_mask=None):
        
        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        D = D_1d.reshape((ny, nx))

        q = jnp.log((1 - D)/(mucoef_0 + 1e-10))
        #q = jnp.log((1)/(mucoef_0 + 1e-10))

        u_full, v_full = add_uv_ghost_cells(u, v)
        h_full = add_s_ghost_cells(h)
        D_full = add_s_ghost_cells(D)

        u_full = linear_extrapolate_over_cf_dynamic_thickness(
                            u_full, h_full)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(
                            v_full, h_full)
        h_full = linear_extrapolate_over_cf_dynamic_thickness(
                            h_full, h_full)
        D_full = linear_extrapolate_over_cf_dynamic_thickness(
                            D_full, h_full)

        Dh_full = D_full * h_full

        #####################################################
        # Advection of Dh
        #####################################################
       

        if advtype=="FOU":
        
            u_fc_ew, _ = interp_cc_to_fc(u_full)
            _, v_fc_ns = interp_cc_to_fc(v_full)

            Dh_fc_fou_ew = jnp.where(
                u_fc_ew > 0,
                Dh_full[1:-1, :-1],
                Dh_full[1:-1, 1:]
            )

            Dh_fc_fou_ns = jnp.where(
                v_fc_ns > 0,
                Dh_full[1:, 1:-1],
                Dh_full[:-1, 1:-1]
            )

            flux_term = (
                (u_fc_ew[:,:-1] * Dh_fc_fou_ew[:,:-1]
                 - u_fc_ew[:,1:] * Dh_fc_fou_ew[:,1:])
                * dy * delta_t
                +
                (v_fc_ns[1:,:] * Dh_fc_fou_ns[1:,:]
                 - v_fc_ns[:-1,:] * Dh_fc_fou_ns[:-1,:])
                * dx * delta_t
            )

            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        elif advtype=="PPM":
            ###PPM!!! GIVES SOME STRANGE LOOKING RESULTS...

            flux_x = ppm_flux_x(Dh_full[1:-1,:], u_full[1:-1,:], dx, delta_t)
            flux_y = ppm_flux_y(Dh_full[:,1:-1], v_full[:,1:-1], dy, delta_t)
    
            #plt.imshow(flux_x[:,:1] - flux_x[:,-1:])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(flux_y[1:,:] - flux_y[:-1,:])
            #plt.colorbar()
            #plt.show()

            #raise

            #print(flux_x.shape)
            #print(flux_y.shape)

            #print(D.shape)

            flux_term = (
                (flux_x[:,:-1] - flux_x[:,1:])
                * dy * delta_t
                +
                (flux_y[1:,:] - flux_y[:-1,:])
                * dx * delta_t
            )
            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        #####################################################
        # Source
        #####################################################

        prs = prs_function(q, u, v, h)

        prs = prs * (h > 0).astype(float)

        # membrane opening force
        N_open = h * prs

        # hydrostatic closure force
        N_close = c.RHO_I * c.g * D * h

        # stress amplification from remaining ligament
        N_eff = (N_open - N_close) / (1.0 - D + 1e-6)

        # parameters to tune
        Nc = 200_000 * (h + 1e-10)
        K  = 10     # a^-1
        m  = 4.0

        excess = jnp.maximum(N_eff / Nc - 1.0, 0.0)

        #plt.imshow(excess)
        #plt.colorbar()
        #plt.show()

        source = K * excess**m

        source_term = source * delta_t

        if source_mask is not None:
            source_term = source_term*source_mask

        return (
            D
            + (source_term + flux_term/(dy*dx))
            / (h + 1e-10)
            * (h > 0).astype(float)
        )

    return advection_step


def make_advsrc_effective_damthk_stepper_vmthres_ppmish_DNOTDH(
                               nx, ny, dx, dy,
                               interp_cc_to_fc,
                               add_uv_ghost_cells,
                               add_s_ghost_cells,
                               mucoef_0,
                               rst_dst_fct,
                               advtype="PPM",
                               vm_failure_criterion=True,
                               conservative=False,
                               upwind_source=False,
                               limit_shear_rate=False,
                               plt_dir=None):

    def advection_step(u_1d, v_1d, h_1d, D_1d,
                       delta_t=0.08,
                       ts=1, source_mask=None):
        
        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        D = D_1d.reshape((ny, nx))

        q = jnp.log((1 - D)/(mucoef_0 + 1e-10))
        #q = jnp.log((1)/(mucoef_0 + 1e-10))

        u_full, v_full = add_uv_ghost_cells(u, v)
        h_full = add_s_ghost_cells(h)
        D_full = add_s_ghost_cells(D)

        u_full = linear_extrapolate_over_cf_dynamic_thickness(
                            u_full, h_full)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(
                            v_full, h_full)
        h_full = linear_extrapolate_over_cf_dynamic_thickness(
                            h_full, h_full)
        D_full = linear_extrapolate_over_cf_dynamic_thickness(
                            D_full, h_full)

        #####################################################
        #Advection of D
        #####################################################
       
        if advtype=="FOU":
        
            u_fc_ew, _ = interp_cc_to_fc(u_full)
            _, v_fc_ns = interp_cc_to_fc(v_full)

            D_fc_fou_ew = jnp.where(
                u_fc_ew > 0,
                D_full[1:-1, :-1],
                D_full[1:-1, 1:]
            )

            D_fc_fou_ns = jnp.where(
                v_fc_ns > 0,
                D_full[1:, 1:-1],
                D_full[:-1, 1:-1]
            )

            flux_term = (
                (u_fc_ew[:,:-1] * D_fc_fou_ew[:,:-1]
                 - u_fc_ew[:,1:] * D_fc_fou_ew[:,1:])
                * dy * delta_t
                +
                (v_fc_ns[1:,:] * D_fc_fou_ns[1:,:]
                 - v_fc_ns[:-1,:] * D_fc_fou_ns[:-1,:])
                * dx * delta_t
            )

            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        elif advtype=="PPM":

            flux_x = ppm_flux_x(D_full[1:-1,:], u_full[1:-1,:], dx, delta_t)
            flux_y = ppm_flux_y(D_full[:,1:-1], v_full[:,1:-1], dy, delta_t)
    
            #plt.imshow(flux_x[:,:1] - flux_x[:,-1:])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(flux_y[1:,:] - flux_y[:-1,:])
            #plt.colorbar()
            #plt.show()

            #raise

            #print(flux_x.shape)
            #print(flux_y.shape)

            #print(D.shape)

            flux_term = (
                (flux_x[:,:-1] - flux_x[:,1:])
                * dy * delta_t
                +
                (flux_y[1:,:] - flux_y[:-1,:])
                * dx * delta_t
            )
            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        #####################################################
        # Source
        #####################################################

        #THESE ARE TRUE VALUES
        rst, _ = rst_dst_fct(q, u, v, h)

        if vm_failure_criterion: 
            #NON-TRUE VALUES, TO FIT WITH WM CRITERION
            #_, dst = rst_dst_fct(-jnp.log(1-D), u, v, h)
            #If mucoef_0 is 1, then use:
            _, dst = rst_dst_fct(jnp.zeros_like(D), u, v, h)
    
            #Threshold criterion from:
            #   Fracture criteria and tensile strength for natural glacier ice calibrated 
            #   from remote sensing observations of Antarctic ice shelves
            #   2024, by wells-moran etc.
            sig_vm = jnp.sqrt(dst[:,:,0,0]**2 +
                              dst[:,:,1,1]**2 -
                              dst[:,:,0,0]*dst[:,:,1,1] +
                              3*dst[:,:,1,0]**2 +
                              1e-10)
            sig_vm *= jnp.sqrt(3)
    
    
            critical_vm = 280_000 #Pa
            #active_crevassing = jnp.where(sig_vm>critical_vm, 1, 0)
            width = 10_000
            active_crevassing = jax.nn.sigmoid(
                (sig_vm - critical_vm) / width
            )
    
            #Might quite like to smooth this! ^^
            
    
            #plt.imshow(sig_vm, cmap="Spectral_r", vmax=250_000)
            #plt.colorbar()
            #plt.savefig(f"{plt_dir}/sig_vm_{ts}.png", dpi=150)
            #plt.close()
    
            #raise
    
            #plt.imshow(active_crevassing, vmin=0, vmax=1)
            #plt.colorbar()
            #plt.savefig(f"{plt_dir}/active_crevassing_{ts}.png", dpi=150)
            #plt.close()

        else:
            active_crevassing = jnp.ones_like(D)
        

        ###Von-Mises type thing with resistive stress tensor
        ##rst_vm = jnp.sqrt(rst[0,0,:,:]**2 +
        ##                  rst[1,1,:,:]**2 -
        ##                  rst[0,0,:,:]*rst[1,1,:,:] +
        ##                  3*rst[1,0,:,:]**2 +
        ##                  1e-10)


        
        pricipal_rs = 0.5 * (rst[:,:,0,0] + rst[:,:,1,1] + 
                             jnp.sqrt(
                                (rst[:,:,0,0] + rst[:,:,1,1])**2 -\
                                4*(rst[:,:,0,0]*rst[:,:,1,1] -
                                   rst[:,:,1,0]**2)
                             )
                            )
        #return visc_xx

        pricipal_rs = pricipal_rs * (h > 0).astype(float)

        #membrane opening force
        N_open = jnp.maximum(pricipal_rs, 0)

        #overburden closure force
        N_close = c.RHO_I * (1 - c.RHO_I/c.RHO_W) * c.g * D * h# * (1-D)


        if upwind_source:
            u_fc_ew, _ = interp_cc_to_fc(u_full)
            _, v_fc_ns = interp_cc_to_fc(v_full)
            
            D_source_x = jnp.where(
                u_fc_ew > 0,
                D_full[1:-1, :-1],
                D_full[1:-1, 1:]
            )

            D_source_y = jnp.where(
                v_fc_ns > 0,
                D_full[1:, 1:-1],
                D_full[:-1, 1:-1]
            )

            D_source = 0.5 * (D_source_x + D_source_y)
        else:
            D_source = D.copy()



        #stress diff to overburden plus amplification from remaining ligament
        N_eff = (N_open - N_close) / (1.0 - D_source + 1e-6)
        sgn_factor = jnp.where((N_open - N_close)>0, 1, -0.1)
        #N_eff = jnp.maximum(N_eff, 0)

        # parameters to tune
        m  = 4
        sigma_scale = 260_000
        gamma = ( 0.05 /( c.A_COLD * (sigma_scale + 1e-10)**m ) ) * (h > 0).astype(float)

        #effective power:
        P_eff = c.A_COLD * (N_eff**m)

        
        source = active_crevassing * gamma * P_eff * sgn_factor * 0

        #plt.imshow(source, vmin=-1, vmax=1, cmap="RdBu_r")
        #plt.colorbar()
        #plt.savefig(f"{plt_dir}/source_{ts}.png", dpi=150)
        #plt.close()

        #raise

        if limit_shear_rate:
            #Reduce rate in shear zones by a factor of 4

            #extension_metric = jnp.abs(
            #                     jnp.clip(
            #                        (R1+R2)/(R1-R2+1e-10),
            #                        -1, 1
            #                     )
            #                   )
 
            extension_metric = jnp.abs(
                                 jnp.clip(
                                    (rst[:,:,0,0] + rst[:,:,1,1])/\
                                     (jnp.sqrt((rst[:,:,0,0] - rst[:,:,1,1])**2 +\
                                      4*rst[:,:,1,0]**2 +1e-10)),
                                    -1, 1
                                 )
                               )

            #source *= (0.1 + 0.9*(extension_metric**2))
            source *= jnp.maximum(extension_metric, 0.1)

            plt.imshow(jnp.maximum(extension_metric, 0.1), vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f"{plt_dir}/extmetric_{ts}.png", dpi=150)
            plt.close()

        #source = gamma * P_eff
        
        if not conservative:
            dudx = (
                u_full[1:-1, 2:] -
                u_full[1:-1, :-2]
                   ) / (2*dx)
            
            dvdy = (
                v_full[2:, 1:-1] -
                v_full[:-2, 1:-1]
                   ) / (2*dy)
            
            divu = dudx + dvdy

            source += D * divu



        source_term = source * delta_t

        if source_mask is not None:
            source_term = source_term*source_mask
            flux_term   = flux_term*source_mask

        return (
            D
            + (source_term + flux_term/(dy*dx))
            * (h > 0).astype(float)
        )


    #return jax.jit(advection_step)
    return advection_step



def make_advsrc_effective_damthk_stepper_vmthres_ppmish(
                               nx, ny, dx, dy,
                               interp_cc_to_fc,
                               add_uv_ghost_cells,
                               add_s_ghost_cells,
                               mucoef_0,
                               rst_dst_fct,
                               advtype="PPM",
                               vm_failure_criterion=True,
                               conservative=False,
                               upwind_source=False,
                               limit_shear_rate=False,
                               plt_dir=None):

    def advection_step(u_1d, v_1d, h_1d, D_1d,
                       delta_t=0.08,
                       ts=1, source_mask=None):
        
        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))
        D = D_1d.reshape((ny, nx))

        q = jnp.log((1 - D)/(mucoef_0 + 1e-10))
        #q = jnp.log((1)/(mucoef_0 + 1e-10))

        u_full, v_full = add_uv_ghost_cells(u, v)
        h_full = add_s_ghost_cells(h)
        D_full = add_s_ghost_cells(D)

        u_full = linear_extrapolate_over_cf_dynamic_thickness(
                            u_full, h_full)
        v_full = linear_extrapolate_over_cf_dynamic_thickness(
                            v_full, h_full)
        h_full = linear_extrapolate_over_cf_dynamic_thickness(
                            h_full, h_full)
        D_full = linear_extrapolate_over_cf_dynamic_thickness(
                            D_full, h_full)

        Dh_full = D_full * h_full

        #####################################################
        # Advection of Dh
        #####################################################
       

        if advtype=="FOU":
        
            u_fc_ew, _ = interp_cc_to_fc(u_full)
            _, v_fc_ns = interp_cc_to_fc(v_full)

            Dh_fc_fou_ew = jnp.where(
                u_fc_ew > 0,
                Dh_full[1:-1, :-1],
                Dh_full[1:-1, 1:]
            )

            Dh_fc_fou_ns = jnp.where(
                v_fc_ns > 0,
                Dh_full[1:, 1:-1],
                Dh_full[:-1, 1:-1]
            )

            flux_term = (
                (u_fc_ew[:,:-1] * Dh_fc_fou_ew[:,:-1]
                 - u_fc_ew[:,1:] * Dh_fc_fou_ew[:,1:])
                * dy * delta_t
                +
                (v_fc_ns[1:,:] * Dh_fc_fou_ns[1:,:]
                 - v_fc_ns[:-1,:] * Dh_fc_fou_ns[:-1,:])
                * dx * delta_t
            )

            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        elif advtype=="PPM":

            flux_x = ppm_flux_x(Dh_full[1:-1,:], u_full[1:-1,:], dx, delta_t)
            flux_y = ppm_flux_y(Dh_full[:,1:-1], v_full[:,1:-1], dy, delta_t)
    
            #plt.imshow(flux_x[:,:1] - flux_x[:,-1:])
            #plt.colorbar()
            #plt.show()

            #plt.imshow(flux_y[1:,:] - flux_y[:-1,:])
            #plt.colorbar()
            #plt.show()

            #raise

            #print(flux_x.shape)
            #print(flux_y.shape)

            #print(D.shape)

            flux_term = (
                (flux_x[:,:-1] - flux_x[:,1:])
                * dy * delta_t
                +
                (flux_y[1:,:] - flux_y[:-1,:])
                * dx * delta_t
            )
            flux_term = jnp.where(h > 1e-2, flux_term, 0)

        #####################################################
        # Source
        #####################################################

        #THESE ARE TRUE VALUES
        rst, _ = rst_dst_fct(q, u, v, h)

        if vm_failure_criterion: 
            #NON-TRUE VALUES, TO FIT WITH WM CRITERION
            #_, dst = rst_dst_fct(-jnp.log(1-D), u, v, h)
            #If mucoef_0 is 1, then use:
            _, dst = rst_dst_fct(jnp.zeros_like(D), u, v, h)
    
            #Threshold criterion from:
            #   Fracture criteria and tensile strength for natural glacier ice calibrated 
            #   from remote sensing observations of Antarctic ice shelves
            #   2024, by wells-moran etc.
            sig_vm = jnp.sqrt(dst[:,:,0,0]**2 +
                              dst[:,:,1,1]**2 -
                              dst[:,:,0,0]*dst[:,:,1,1] +
                              3*dst[:,:,1,0]**2 +
                              1e-10)
            sig_vm *= jnp.sqrt(3)
    
    
            critical_vm = 280_000 #Pa
            #active_crevassing = jnp.where(sig_vm>critical_vm, 1, 0)
            width = 10_000
            active_crevassing = jax.nn.sigmoid(
                (sig_vm - critical_vm) / width
            )
    
            #Might quite like to smooth this! ^^
            
    
            #plt.imshow(sig_vm, cmap="Spectral_r", vmax=250_000)
            #plt.colorbar()
            #plt.savefig(f"{plt_dir}/sig_vm_{ts}.png", dpi=150)
            #plt.close()
    
            #raise
    
            plt.imshow(active_crevassing, vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f"{plt_dir}/active_crevassing_{ts}.png", dpi=150)
            plt.close()

        else:
            active_crevassing = jnp.ones_like(D)
        

        ###Von-Mises type thing with resistive stress tensor
        ##rst_vm = jnp.sqrt(rst[0,0,:,:]**2 +
        ##                  rst[1,1,:,:]**2 -
        ##                  rst[0,0,:,:]*rst[1,1,:,:] +
        ##                  3*rst[1,0,:,:]**2 +
        ##                  1e-10)


        
        pricipal_rs = 0.5 * (rst[:,:,0,0] + rst[:,:,1,1] + 
                             jnp.sqrt(
                                (rst[:,:,0,0] + rst[:,:,1,1])**2 -\
                                4*(rst[:,:,0,0]*rst[:,:,1,1] -
                                   rst[:,:,1,0]**2)
                             )
                            )
        #return visc_xx

        pricipal_rs = pricipal_rs * (h > 0).astype(float)

        #membrane opening force
        N_open = h * jnp.maximum(pricipal_rs, 0)

        #overburden closure force
        N_close = c.RHO_I * c.g * D * (1-D) * h


        if upwind_source:
            u_fc_ew, _ = interp_cc_to_fc(u_full)
            _, v_fc_ns = interp_cc_to_fc(v_full)
            
            D_source_x = jnp.where(
                u_fc_ew > 0,
                D_full[1:-1, :-1],
                D_full[1:-1, 1:]
            )

            D_source_y = jnp.where(
                v_fc_ns > 0,
                D_full[1:, 1:-1],
                D_full[:-1, 1:-1]
            )

            D_source = 0.5 * (D_source_x + D_source_y)
        else:
            D_source = D.copy()



        #stress diff to overburden plus amplification from remaining ligament
        N_eff = (N_open - N_close) / (1.0 - D_source + 1e-6)
        sgn_factor = jnp.where((N_open - N_close)>0, 1, -0.1)
        #N_eff = jnp.maximum(N_eff, 0)

        # parameters to tune
        m  = 4
        sigma_scale = 250_000
        gamma = ( 1 /( c.A_COLD * (sigma_scale * (h + 1e-10))**m ) ) * (h > 0).astype(float)

        #effective power:
        P_eff = c.A_COLD * (N_eff**m)

        #plt.imshow(excess)
        #plt.colorbar()
        #plt.show()

        source = active_crevassing * gamma * P_eff * sgn_factor


        if limit_shear_rate:
            #Reduce rate in shear zones by a factor of 4

            #extension_metric = jnp.abs(
            #                     jnp.clip(
            #                        (R1+R2)/(R1-R2+1e-10),
            #                        -1, 1
            #                     )
            #                   )
 
            extension_metric = jnp.abs(
                                 jnp.clip(
                                    (rst[:,:,0,0] + rst[:,:,1,1])/\
                                     (jnp.sqrt((rst[:,:,0,0] - rst[:,:,1,1])**2 +\
                                      4*rst[:,:,1,0]**2 +1e-10)),
                                    -1, 1
                                 )
                               )

            #plt.imshow((u**2+v**2), cmap="RdYlBu_r")
            ##plt.imshow(0.2 + 0.8*extension_metric, alpha=0.5)
            #plt.imshow(extension_metric, alpha=0.5)
            #plt.colorbar()
            #plt.show()

            #source *= (0.1 + 0.9*(extension_metric**2))
            source *= jnp.maximum(extension_metric, 0.1)

            plt.imshow(jnp.maximum(extension_metric, 0.1), vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f"{plt_dir}/extmetric_{ts}.png", dpi=150)
            plt.close()

        #source = gamma * P_eff

        if not conservative:
            dudx = (
                u_full[1:-1, 2:] -
                u_full[1:-1, :-2]
                   ) / (2*dx)
            
            dvdy = (
                v_full[2:, 1:-1] -
                v_full[:-2, 1:-1]
                   ) / (2*dy)
            
            divu = dudx + dvdy

            source += D * h * divu

        source_term = source * delta_t

        if source_mask is not None:
            source_term = source_term*source_mask
            flux_term   = flux_term*source_mask

        return (
            D
            + (source_term + flux_term/(dy*dx))
            / (h + 1e-10)
            * (h > 0).astype(float)
        )


    #return jax.jit(advection_step)
    return advection_step

#def make_advsrc_effective_damthk_stepper_threshold(
#                               nx, ny, dx, dy,
#                               interp_cc_to_fc,
#                               add_uv_ghost_cells,
#                               add_s_ghost_cells,
#                               mucoef_0,
#                               prs_function):
#
#    def advection_step(u_1d, v_1d, h_1d, D_1d,
#                       delta_t=0.08,
#                       ts=1):
#
#        u = u_1d.reshape((ny, nx))
#        v = v_1d.reshape((ny, nx))
#        h = h_1d.reshape((ny, nx))
#        D = D_1d.reshape((ny, nx))
#
#        #q = jnp.log((1 - D)/(mucoef_0 + 1e-10))
#        q = jnp.log((1)/(mucoef_0 + 1e-10))
#
#        u_full, v_full = add_uv_ghost_cells(u, v)
#        h_full = add_s_ghost_cells(h)
#        D_full = add_s_ghost_cells(D)
#
#        u_full = linear_extrapolate_over_cf_dynamic_thickness(
#                            u_full, h_full)
#        v_full = linear_extrapolate_over_cf_dynamic_thickness(
#                            v_full, h_full)
#        h_full = linear_extrapolate_over_cf_dynamic_thickness(
#                            h_full, h_full)
#        D_full = linear_extrapolate_over_cf_dynamic_thickness(
#                            D_full, h_full)
#
#        Dh_full = D_full * h_full
#
#        #####################################################
#        # Advection of Dh
#        #####################################################
#
#        u_fc_ew, _ = interp_cc_to_fc(u_full)
#        _, v_fc_ns = interp_cc_to_fc(v_full)
#
#        Dh_fc_fou_ew = jnp.where(
#            u_fc_ew > 0,
#            Dh_full[1:-1, :-1],
#            Dh_full[1:-1, 1:]
#        )
#
#        Dh_fc_fou_ns = jnp.where(
#            v_fc_ns > 0,
#            Dh_full[1:, 1:-1],
#            Dh_full[:-1, 1:-1]
#        )
#
#        flux_term = (
#            (u_fc_ew[:,1:] * Dh_fc_fou_ew[:,1:]
#             - u_fc_ew[:,:-1] * Dh_fc_fou_ew[:,:-1])
#            * dy * delta_t
#            +
#            (v_fc_ns[:-1,:] * Dh_fc_fou_ns[:-1,:]
#             - v_fc_ns[1:,:] * Dh_fc_fou_ns[1:,:])
#            * dx * delta_t
#        )
#
#        flux_term = jnp.where(h > 1e-2, flux_term, 0)
#
#        #####################################################
#        # Source
#        #####################################################
#
#        prs = prs_function(q, u, v, h)
#
#        prs = prs * (h > 0).astype(float)
#
#        # membrane opening force
#        N_open = h * prs
#
#        # hydrostatic closure force
#        N_close = c.RHO_I * c.g * D * h
#
#        # stress amplification from remaining ligament
#        N_eff = (N_open - N_close) / (1.0 - D + 1e-6)
#
#        # parameters to tune
#        Nc = 250_000 * (h + 1e-10)
#        K  = 10     # a^-1
#        m  = 4.0
#
#        excess = jnp.maximum(N_eff / Nc - 1.0, 0.0)
#
#        plt.imshow(excess)
#        plt.colorbar()
#        plt.show()
#
#        source = K * excess**m
#
#        source_term = source * delta_t
#
#        return (
#            D
#            + (source_term - flux_term/(dy*dx))
#            / (h + 1e-10)
#            * (h > 0).astype(float)
#        )
#
#    return advection_step



def fake_lax_while_loop(conditional, update, initial_state):
    state = initial_state
    while conditional(state):
        state = update(state)
    return state


def make_pic_velocity_solver_function_expl_advection_gpusafe(ny, nx, dy, dx,
                                              b, ice_mask,
                                              n_iterations,
                                              mucoef_0, C_0,
                                              n_timesteps,
                                              sliding="linear",
                                              res_tol=1e-10,
                                              periodic=False,
                                              B_field=None,
                                              temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    nc_velocity_gradient                       = nc_velocity_gradient_function(dy, dx,
                                                                               add_uv_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    
    fc_viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    
    nc_viscosity_fct = node_centred_viscosity_function(ny, nx, dy, dx,
                                                   add_scalar_ghost_cells,
                                                   nc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)


    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
   


    advection_step = make_advection_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
                                            add_uv_ghost_cells, add_scalar_ghost_cells)



    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    basis_vectors = jnp.stack(basis_vectors).astype(jnp.float64)
    #sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
    #                                                    basis_vectors,\
    #                                                    2,
    #                                                    active_indices=(0,1)
    #                                                   )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   
    
    #assemble_jacobian = assemble_sparse_2x2_block_jacobian_function_general(basis_vectors, ny*nx, mask,
    assemble_jacobian = assemble_sparse_2x2_block_jacobian_function(basis_vectors, ny*nx, mask,
                                                                    get_uv_residuals_linear_ssa)


    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    #cg_solver = make_sparse_dpcg_solver_jsp_comp_fori(coords, extract_inverse_diagonal, ny*nx*2,
    #                                        iterations=200)
    cg_solver = make_sparse_dpcg_solver_jsp_comp(coords, extract_inverse_diagonal, ny*nx*2,
                                            iterations=200)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    omega=1


    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=True,
    #                                                          #ksp_type='gmres',
    #                                                          #ksp_type='bcgs',
    #                                                          ksp_type='preonly',
    #                                                          preconditioner="jacobi", #might just about be workable
    #                                                          #preconditioner="sor", #better than jacobi
    #                                                          #preconditioner="sor",
    #                                                          #monitor_ksp=True,
    #                                                          ksp_max_iter=20)

    def conditional(state):
        i, res, init_res, _,_,_,_,_,_,_,_ = state
        return jnp.logical_and(i<n_iterations, jnp.abs(res)>res_tol)


    def momentum_solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask


        mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
        mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
        
        _, r0 = assemble_jacobian(
            u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta
        )
        
        duv = jnp.zeros((nx*ny*2,))

        initial_state = (0, res_fct(r0), res_fct(r0), u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv) 

        def update(state):
            i, res, init_res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = state

            #jax.debug.print("Pic res: {x}", x=res)

            #t0 = time.perf_counter()
            nz_jac_values, rhs = assemble_jacobian(
                u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta
            )
            #nz_jac_values.block_until_ready()
            #t_jac = time.perf_counter() - t0

            #jax.debug.print("NZ jac values: {x}", x=nz_jac_values)
            
            #t0 = time.perf_counter()
            duv = cg_solver(nz_jac_values, rhs, jnp.zeros_like(duv))
            #duv = la_solver(nz_jac_values, rhs)

            #duv.block_until_ready()
            #t_cg = time.perf_counter() - t0


            #t0 = time.perf_counter()
            u_1d = (u_1d + omega*duv[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*duv[(ny*nx):]) * ice_mask
            #u_1d.block_until_ready()
            #t_update = time.perf_counter() - t0

            #t0 = time.perf_counter()
            mu_ew, mu_ns = fc_viscosity_fct(q, u_1d, v_1d)
            mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            #mu_ew.block_until_ready()
            #t_visc = time.perf_counter() - t0

            #print(
            #    f"PIC iter {i:3d} | "
            #    f"Jac: {t_jac:7.4f}s | "
            #    f"CG: {t_cg:7.4f}s | "
            #    f"Upd: {t_update:7.4f}s | "
            #    f"Visc: {t_visc:7.4f}s"
            #)

            return (i+1, res_fct(rhs), init_res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv)


        i, res, init_res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = jax.lax.while_loop(conditional, update, initial_state)
        #i, res, init_res, u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta, duv = fake_lax_while_loop(conditional, update, initial_state)

        #jax.debug.print("Pic res: {x}", x=res)
        #jax.debug.print("Pic res reduction factor: {x}", x=init_res/res)

        return u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx))


    def prognostic_condition(state):
        timestep, _,_,_ = state
        return (timestep<n_timesteps)

    def run_model_forward(q, p, u_trial, v_trial, h_init):
        h = h_init
        u, v = u_trial, v_trial

        initial_state = 0, u, v, h

        def update(state):
            ts, u, v, h = state
            u, v = momentum_solver(q, p, u, v, h)
            h = advection_step(u, v, h.reshape(-1))

            return ts+1, u, v, h

        ts_end, u, v, h = jax.lax.while_loop(prognostic_condition, update, initial_state)
        #ts_end, u, v, h = fake_lax_while_loop(prognostic_condition, update, initial_state)

        return u.reshape((ny, nx)), v.reshape((ny, nx)), h.reshape((ny, nx))

    #def run_model_forward(q, p, u_trial, v_trial, h_init):
    #    for i in range(n_timesteps):
    #        u, v = momentum_solver(q, p, u, v, h)
    #        h = advection_step(u, v, h.reshape(-1))

    #    return u.reshape((ny, nx)), v.reshape((ny, nx)), h.reshape((ny, nx))

    return run_model_forward


def make_pic_velocity_solver_function_acrobatic(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_iterations,
                                                 mucoef_0, C_0, sliding="linear",
                                                 periodic=False, B_field=None,
                                                 temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    #Note: interp_cc_to_nc is not made in a function factory
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    nc_velocity_gradient                       = nc_velocity_gradient_function(dy, dx,
                                                                               add_uv_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    
    viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    
    nc_viscosity_fct = node_centred_viscosity_function(ny, nx, dy, dx,
                                                   add_scalar_ghost_cells,
                                                   nc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)


    beta_fct = beta_function(b, sliding)

    #get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new_noextrap(ny, nx, dy, dx, b,
    #                                                   interp_cc_to_fc,
    #                                                   fc_velocity_gradient,
    #                                                   cc_gradient,
    #                                                   add_uv_ghost_cells,
    #                                                   add_scalar_ghost_cells)
    
    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_acrobatic(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    #j_coord_ar = jnp.arange(ny*nx)
    #pattern = jnp.zeros((nx*ny, nx*ny))*jnp.nan
    #for _, i_coord_ar in zip(basis_vectors, i_coordinate_sets):
    #    mask = ~jnp.isnan(i_coord_ar)

    #    pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
    #                         j_coord_ar[mask].astype(jnp.int32)].set(1)

    #plt.imshow(np.array(pattern[:, 26].reshape((ny,nx))))
    #plt.show()
    #raise




    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
                                                        basis_vectors,\
                                                        2,
                                                        active_indices=(0,1)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=True,
                                                              monitor_ksp=True)
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=True,
    #                                                          #ksp_type='gmres',
    #                                                          #ksp_type='bcgs',
    #                                                          ksp_type='cg',
    #                                                          preconditioner="jacobi", #might just about be workable
    #                                                          #preconditioner="sor", #better than jacobi
    #                                                          monitor_ksp=True,
    #                                                          ksp_max_iter=400)
    #Basically can only be used for newton-krylov type stuff where you're not expecting the LA problem to actually be solved
    #until quite near the end...

    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    #cg_solver = make_sparse_dpgc_solver_comp(sparse_matvec, extract_inverse_diagonal,
    #                                         iterations=200)
    cg_solver = make_sparse_dpcg_solver_jsp_comp(coords, extract_inverse_diagonal, ny*nx*2,
                                            iterations=200)
    #j_solver = make_sparse_damped_jacobi_solver(sparse_matvec, extract_inverse_diagonal, iterations=12)
    #preconditioner = make_point_sor_preconditioner(coords, (ny*nx*2, ny*nx*2))
    #bcgs_solver = make_sparse_bicgstab_solver(sparse_matvec, iterations=200)
    #relax_solver = make_multicoloured_relaxation(sparse_matvec, extract_inverse_diagonal, basis_vectors, ny, nx)
    
    #bcgs_solver = make_sparse_gs_precond_bicgstab_solver(sparse_matvec, extract_inverse_diagonal, 
    #                                                     basis_vectors, ny, nx, iterations=200)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    #res_fct = lambda x: jnp.mean(jnp.abs(x))

    
    omega=1
    def solver(q, p, u_trial, v_trial, h):
        #plt.imshow(q, vmin=-2, vmax=0.5, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        #plt.imshow(p, vmin=-4, vmax=4, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        du = jnp.zeros((nx*ny*2,))
        for i in range(n_iterations):
            #NOTE: making this twice as large makes PIG look a little better...
            #mu_ew = 2*mu_ew
            #mu_ns = 2*mu_ns

            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(jnp.log10(mu_ew[:,1:].reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()
            #plt.imshow(jnp.log10(beta.reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()

            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            print("constructing LA problem")
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])



            #full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #
            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            #
            #plt.imshow(jnp.log(jnp.abs(full_jac-jnp.transpose(full_jac))).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            

            ##print(full_jac[:(ny*nx),:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[(ny*nx):,:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[:(ny*nx),(ny*nx):])
            #print("--------------------------------")
            ##print(full_jac[(ny*nx):,(ny*nx):])
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
    


            #jacrev = jax.jacrev(get_uv_residuals_linear_ssa, argnums=(0,1))
            #dense_dJ_du,  dense_dJ_dv = jacrev(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
            #dense_dJu_du, dense_dJu_dv = dense_dJ_du
            #dense_dJv_du, dense_dJv_dv = dense_dJ_dv

            #dense_full_jac = jnp.block([[dense_dJu_du, dense_dJu_dv],
            #                            [dense_dJv_du, dense_dJv_dv]])



            ##plt.imshow(jnp.log(jnp.abs(dense_full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow((full_jac-dense_full_jac).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx), (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):,:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):, (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),26])).reshape((ny,nx)))
            ##plt.show()
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),(ny*nx)+26])).reshape((ny,nx)))
            ##plt.show()

            ##nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            ##jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta))

            print("solving LA problem")
            #du = la_solver(nz_jac_values, rhs)
            du = cg_solver(nz_jac_values, rhs, du)
            #du = j_solver(nz_jac_values, rhs, du)
            #du = relax_solver(nz_jac_values, rhs, du)
            #du = bcgs_solver(nz_jac_values, rhs, du)


            print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(h>0, cmap="Grays_r")
            #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=0.7, vmin=0)
            #plt.colorbar()
            #plt.show()

            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            mu_nc = nc_viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
            
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, mu_nc, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        plt.colorbar()
        plt.show()

        
        final_residual_pic = res_fct(rhs_new)

        print("Final Picard residual: {}".format(final_residual_pic))
        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver


def make_picnewton_velocity_solver_function_no_cf_extrap_expl_advection(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_pic_iterations, n_newt_iterations, n_timesteps,
                                                 mucoef_0, C_0, sliding="linear",
                                                 periodic=False, B_field=None,
                                                 temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    
    viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new_noextrap(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field)

    
    advection_step = make_advection_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
                                            add_uv_ghost_cells, add_scalar_ghost_cells)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    #j_coord_ar = jnp.arange(ny*nx)
    #pattern = jnp.zeros((nx*ny, nx*ny))*jnp.nan
    #for _, i_coord_ar in zip(basis_vectors, i_coordinate_sets):
    #    mask = ~jnp.isnan(i_coord_ar)

    #    pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
    #                         j_coord_ar[mask].astype(jnp.int32)].set(1)

    #plt.imshow(np.array(pattern[:, 26].reshape((ny,nx))))
    #plt.show()
    #raise




    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
                                                        basis_vectors,\
                                                        2,
                                                        active_indices=(0,1)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=True,
                                                              ksp_type="gmres",
                                                              preconditioner="hypre",
                                                              monitor_ksp=False,
                                                              ksp_max_iter=20)
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=False,
    #                                                          monitor_ksp=False)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    
    omega=1
    

    def momentum_solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        du = jnp.zeros((nx*ny*2,))
        for i in range(n_pic_iterations):
            #print("constructing LA problem")
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           


            full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            
            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            #print("solving LA problem")
            du = la_solver(nz_jac_values, rhs)
            #print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            #print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
        
        final_residual_pic = res_fct(rhs_new)

        #print("Final Picard residual: {}".format(final_residual_pic))
        #print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):
            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, p, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            du = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            #print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        #print("Final Newton residual: {}".format(final_residual))
        #print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")
        
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    def run_model_forward(q, p, u_trial, v_trial, h_init):
        h = h_init
        u, v = u_trial, v_trial

        for ts in range(n_timesteps):
            u, v = momentum_solver(q, p, u, v, h)
            h = advection_step(u, v, h.reshape(-1))

        return u.reshape((ny, nx)), v.reshape((ny, nx)), h.reshape((ny, nx))

    return run_model_forward

def make_picnewton_vel_expl_dam_solver_function_noextrap(ny, nx, dy, dx,
                                                b, ice_mask,
                                                n_pic_iterations, n_newt_iterations, n_timesteps,
                                                mucoef_0, C_0, sliding="linear",
                                                periodic=False, B_field=None,
                                                temperature_field=None,
                                                plt_dir=None):

    if temperature_field is None:
    
        temperature_field = (jnp.zeros((ny,nx))+258.15)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    
    viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new_noextrap(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field)


    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
                                                        basis_vectors,\
                                                        2,
                                                        active_indices=(0,1)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)

    res_fct = lambda x: jnp.max(jnp.abs(x))

    
    omega=1

    #prs_fct                                    = principal_resistive_stress_function(
    #                                                ny, nx, dy, dx,
    #                                                #extrp_over_cf,
    #                                                add_uv_ghost_cells,
    #                                                add_scalar_ghost_cells,
    #                                                cc_gradient, mucoef_0,
    #                                                temperature_field)
    
    rst_dst_fct                                = cc_resistive_and_deviatoric_stress_tensors(
                                                    ny, nx, dy, dx,
                                                    extrapolate_over_cf,
                                                    add_uv_ghost_cells,
                                                    add_scalar_ghost_cells,
                                                    cc_gradient, mucoef_0,
                                                    temperature_field)

    
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024**3, "GB")

    #dam_adv_src_step = make_advsrc_damage_stepper(nx, ny, dx, dy,
    #dam_adv_src_step = make_advsrc_effective_damthk_stepper_threshold_ppmish(nx, ny, dx, dy,
    #                                              interp_cc_to_fc, 
    #                                              add_uv_ghost_cells,
    #                                              add_scalar_ghost_cells,
    #                                              mucoef_0,
    #                                              prs_fct)
    dam_adv_src_step = make_advsrc_effective_damthk_stepper_vmthres_ppmish_DNOTDH(nx, ny, dx, dy,
                                                  interp_cc_to_fc, 
                                                  add_uv_ghost_cells,
                                                  add_scalar_ghost_cells,
                                                  mucoef_0,
                                                  rst_dst_fct,
                                                  plt_dir=plt_dir,
                                                  advtype="FOU")
    
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss / 1024**3, "GB")

    def momentum_solver(q, p, u_trial, v_trial, h):
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        du = jnp.zeros((nx*ny*2,))
    
        process = psutil.Process(os.getpid())
        print(process.memory_info().rss / 1024**3, "GB")
    
        for i in range(n_pic_iterations):
            #print("constructing LA problem")
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )
        
            process = psutil.Process(os.getpid())
            print(process.memory_info().rss / 1024**3, "GB")
    

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            #plt.imshow(rhs[:(ny*nx)].reshape((ny, nx)))
            #plt.colorbar()
            ##plt.imshow(jnp.where((h>0), 1, jnp.nan), cmap="Grays_r", alpha=1)
            #plt.show()

            #print("solving LA problem")
            du = la_solver(nz_jac_values, rhs, x_ig=du)
            #print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
        
        final_residual_pic = res_fct(rhs_new)

        print("Final Picard residual: {}".format(final_residual_pic))
        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):
            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, p, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            #plt.imshow(rhs[:(ny*nx)].reshape((ny, nx)))
            #plt.colorbar()
            ##plt.imshow(jnp.where((h>0), 1, jnp.nan), cmap="Grays_r", alpha=1)
            #plt.show()
            
            du = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        print("Final Newton residual: {}".format(final_residual))
        print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")
        
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    def run_model_forward(q, p, u_trial, v_trial, h_init, D_init):
        h = h_init
        u, v = u_trial, v_trial
        D = D_init

        delta_t = 0

        #os.system(f"mkdir -p {nm_home}/bits_of_data/ss_damage_cook/11/")
        #os.system(f"rm -f {nm_home}/bits_of_data/ss_damage_cook/12/*.png")
        t_cum = 2024

        for ts in range(n_timesteps):
            plt.imshow(D, vmin=0, vmax=1, cmap="cubehelix_r")
            plt.colorbar()
            plt.title(f"year: {t_cum+delta_t:.4f}")
            plt.savefig(f"{plt_dir}/{ts}.png", dpi=150)
            plt.close()


            #q = jnp.log((1-D)/(mucoef_0+1e-10))
            q = jnp.zeros_like(D)
            
            #u, v = momentum_solver(q, p, u, v, h)    
            if ts==0:
                u = jnp.zeros_like(q) + 500
                v = jnp.zeros_like(q)

                #u, v = momentum_solver(jnp.zeros_like(q), p, u, v, h)
            
                plt.imshow(jnp.sqrt(u**2 + v**2).reshape((ny,nx)),
                           vmin=0, vmax=800, cmap="RdYlBu_r")
                plt.colorbar()
                plt.title(f"year: {t_cum+delta_t:.4f}")
                plt.savefig(f"{plt_dir}/speed_{ts}.png", dpi=150)
                plt.close()
            #plt.imshow(jnp.sqrt(u**2 + v**2).reshape((ny,nx)),
            #           vmin=0, vmax=1200, cmap="RdYlBu_r")
            #plt.colorbar()
            #plt.title(f"year: {t_cum+delta_t:.4f}")
            #plt.savefig(f"{plt_dir}/speed_{ts}.png", dpi=150)
            #plt.close()

            delta_t = 0.45*(dx/jnp.max(jnp.sqrt(u**2+v**2)))
            delta_t = jnp.maximum(delta_t, 0.06)

            t_cum += delta_t

            #delta_t = 0.01
            

            floating = jnp.where((h + b) >= (h*(1-c.RHO_I/c.RHO_W)),
                                 0, 1)
            #damage_mask = floating
            damage_mask = jnp.ones_like(floating)

            D = dam_adv_src_step(u, v, h.reshape(-1), D.reshape(-1),
                                 delta_t, ts, damage_mask)
            D = jnp.clip(D, 0, 0.9)
            
            
            #h = jnp.where(D>0.95, 0, h)
            
            ##NEED EVERYTHING TO HAVE A DYNAMIC ICE MASK!!!!!!!
            h = jnp.where(jnp.sqrt(u**2 + v**2)<10_000, h, 0)
            h = jnp.where(dangling_cells(h), 0, h)
            
            bulk_ = bulk_ice(h>0)

            D = jnp.where(bulk_, D, 0)

        return u.reshape((ny, nx)), v.reshape((ny, nx)), D.reshape((ny, nx))

    return run_model_forward

#def make_picnewton_vel_expl_dam_solver_function(ny, nx, dy, dx,
#                                                b, ice_mask,
#                                                n_pic_iterations, n_newt_iterations, n_timesteps,
#                                                mucoef_0, C_0, sliding="linear",
#                                                periodic=False, B_field=None,
#                                                temperature_field=None):
#
#    if temperature_field is None:
#    
#        temperature_field = (jnp.zeros((ny,nx))+258.15)
#
#
#    #functions for various things:
#    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
#    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
#    cc_gradient                                = cc_gradient_function(dy, dx)
#    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
#    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
#    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
#    
#    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
#    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
#    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
#    
#    viscosity_fct = fc_viscosity_function_new_givenT(ny, nx, dy, dx, 
#                                                   extrapolate_over_cf,
#                                                   add_uv_ghost_cells,
#                                                   add_scalar_ghost_cells,
#                                                   interp_cc_to_fc,
#                                                   ew_gradient, ns_gradient,
#                                                   ice_mask, mucoef_0,
#                                                   temperature_field)
#    beta_fct = beta_function(b, sliding)
#
#    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
#                                                       interp_cc_to_fc,
#                                                       ew_gradient, ns_gradient,
#                                                       cc_gradient,
#                                                       add_uv_ghost_cells,
#                                                       add_scalar_ghost_cells,
#                                                       extrapolate_over_cf)
#    
#    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT(
#                                                       ny, nx, dy, dx, b,
#                                                       beta_fct, ice_mask,
#                                                       interp_cc_to_fc,
#                                                       ew_gradient, ns_gradient,
#                                                       cc_gradient,
#                                                       add_uv_ghost_cells,
#                                                       add_scalar_ghost_cells,
#                                                       extrapolate_over_cf,
#                                                       mucoef_0, C_0,
#                                                       temperature_field)
#
#    
#
#    #############
#    #setting up bvs and coords for a single block of the jacobian
#    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 2,
#                                                                                  periodic_x=periodic)
#    #j_coord_ar = jnp.arange(ny*nx)
#    #pattern = jnp.zeros((nx*ny, nx*ny))*jnp.nan
#    #for _, i_coord_ar in zip(basis_vectors, i_coordinate_sets):
#    #    mask = ~jnp.isnan(i_coord_ar)
#
#    #    pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
#    #                         j_coord_ar[mask].astype(jnp.int32)].set(1)
#
#    #plt.imshow(np.array(pattern[:, 26].reshape((ny,nx))))
#    #plt.show()
#    #raise
#
#
#
#
#    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
#    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))
#    mask = (i_coordinate_sets>=0)
#
#
#    sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
#                                                        basis_vectors,\
#                                                        i_coordinate_sets,\
#                                                        j_coordinate_sets,\
#                                                        mask,\
#                                                        2,
#                                                        active_indices=(0,1)
#                                                       )
#    #sparse_jacrev = jax.jit(sparse_jacrev)
#
#
#    i_coordinate_sets = i_coordinate_sets[mask]
#    j_coordinate_sets = j_coordinate_sets[mask]
#    #############
#
#    coords = jnp.stack([
#                    jnp.concatenate(
#                                [i_coordinate_sets,         i_coordinate_sets,\
#                                 i_coordinate_sets+(ny*nx), i_coordinate_sets+(ny*nx)]
#                                   ),\
#                    jnp.concatenate(
#                                [j_coordinate_sets, j_coordinate_sets+(ny*nx),\
#                                 j_coordinate_sets, j_coordinate_sets+(ny*nx)]
#                                   )
#                       ])
#
#   
#    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
#    #                                                          coords,
#    #                                                          (ny*nx*2, ny*nx*2),
#    #                                                          indirect=True,
#    #                                                          monitor_ksp=False)
#    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
#                                                              coords,
#                                                              (ny*nx*2, ny*nx*2),
#                                                              indirect=False,
#                                                              monitor_ksp=False)
#
#    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords,(ny*nx*2, ny*nx*2))
#
#    
#    res_fct = lambda x: jnp.max(jnp.abs(x))
#    #res_fct = lambda x: jnp.mean(jnp.abs(x))
#
#    
#    omega=1
#
#    prs_fct                                    = principal_resistive_stress_function(
#                                                    ny, nx, dy, dx,
#                                                    #extrp_over_cf,
#                                                    add_uv_ghost_cells,
#                                                    add_scalar_ghost_cells,
#                                                    cc_gradient, mucoef_0,
#                                                    temperature_field)
#
#    
#    process = psutil.Process(os.getpid())
#    print(process.memory_info().rss / 1024**3, "GB")
#
#    #dam_adv_src_step = make_advsrc_damage_stepper(nx, ny, dx, dy,
#    dam_adv_src_step = make_advsrc_effective_damthk_stepper_threshold_ppmish(nx, ny, dx, dy,
#                                                  interp_cc_to_fc, 
#                                                  add_uv_ghost_cells,
#                                                  add_scalar_ghost_cells,
#                                                  mucoef_0,
#                                                  prs_fct)
#    
#    process = psutil.Process(os.getpid())
#    print(process.memory_info().rss / 1024**3, "GB")
#
#    def momentum_solver(q, p, u_trial, v_trial, h):
#        
#        u_trial = jnp.where(h>1e-10, u_trial, 0)
#        v_trial = jnp.where(h>1e-10, v_trial, 0)
#
#        u_1d = u_trial.copy().reshape(-1)
#        v_1d = v_trial.copy().reshape(-1)
#        h_1d = h.copy().reshape(-1)
#
#        ice_mask = jnp.where(h>0,1,0).reshape(-1)
#            
#        u_1d = u_1d * ice_mask
#        v_1d = v_1d * ice_mask
#
#        residual = jnp.inf
#        init_res = 0
#
#        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
#        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
#
#        du = jnp.zeros((nx*ny*2,))
#    
#        process = psutil.Process(os.getpid())
#        print(process.memory_info().rss / 1024**3, "GB")
#    
#        for i in range(n_pic_iterations):
#            #print("constructing LA problem")
#            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
#                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
#                                                          )
#        
#            process = psutil.Process(os.getpid())
#            print(process.memory_info().rss / 1024**3, "GB")
#    
#
#            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
#                                             dJv_du[mask], dJv_dv[mask]])
#
#           
#            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
#
#            #print("solving LA problem")
#            du = la_solver(nz_jac_values, rhs, x_ig=du)
#            #print("du norm: {}".format(jnp.max(jnp.abs(du))))
#
#            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
#            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
#            
#            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
#            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
#            
#            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
#            
#            if i==0:
#                initial_residual = jnp.max(rhs)
#            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
#        
#        final_residual_pic = res_fct(rhs_new)
#
#        print("Final Picard residual: {}".format(final_residual_pic))
#        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))
#
#
#        for i in range(n_newt_iterations):
#            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
#            #h = h_1d.reshape((ny,nx))
#
#            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
#                                                             (u_1d, v_1d, q, p, h_1d)
#                                                          )
#
#            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
#                                             dJv_du[mask], dJv_dv[mask]])
#
#           
#            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
#            
#            du = la_solver(nz_jac_values, rhs)
#
#            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
#            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
#            
#            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
#            
#            print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
#
#        final_residual = res_fct(rhs_new)
#
#        print("Final Newton residual: {}".format(final_residual))
#        print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
#        
#        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))
#
#        print("===========================================")
#        
#        
#        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))
#
#    def run_model_forward(q, p, u_trial, v_trial, h_init, D_init):
#        h = h_init
#        u, v = u_trial, v_trial
#        D = D_init
#
#        for ts in range(n_timesteps):
#            plt.imshow(D, vmin=0, vmax=1, cmap="cubehelix_r")
#            plt.colorbar()
#            plt.savefig(f"{nm_home}/bits_of_data/ss_damage_cook/1/{ts}.png", dpi=150)
#            plt.close()
#
#
#            q = jnp.log((1-D)/(mucoef_0+1e-10))
#            
#            u, v = momentum_solver(q, p, u, v, h)    
#            #if ts==0:
#            #    u, v = momentum_solver(jnp.zeros_like(q), p, u, v, h)
#            
#            plt.imshow(jnp.sqrt(u**2 + v**2).reshape((ny,nx)),
#                       vmin=0, cmap="RdYlBu_r")
#            plt.colorbar()
#            plt.savefig(f"{nm_home}/bits_of_data/ss_damage_cook/1/speed_{ts}.png", dpi=150)
#            plt.close()
#
#            #plt.imshow(jnp.sqrt(u**2 + v**2).reshape((ny,nx)),
#            #           vmin=0, vmax=10_000, cmap="RdYlBu_r")
#            #plt.colorbar()
#            #plt.savefig(f"{nm_home}/bits_of_data/damage_gub/speed_{ts}.png", dpi=150)
#            #plt.close()
#
#            delta_t = jnp.minimum(
#                                  (0.5*(dx/jnp.max(jnp.abs(u)))),
#                                  (0.5*(dy/jnp.max(jnp.abs(v))))
#                                 )
#            #delta_t = 0.01
#            
#
#            floating = jnp.where((h + b) >= (h*(1-c.RHO_I/c.RHO_W)),
#                                 0, 1)
#
#            D = dam_adv_src_step(u, v, h.reshape(-1), D.reshape(-1),
#                                 delta_t, ts, floating)
#            D = jnp.clip(D, 0, 0.9)
#            
#
#            h = jnp.where(jnp.sqrt(u**2 + v**2)<20_000, h, 0)
#            bulk_ = bulk_ice(h>0)
#
#            D = jnp.where(bulk_, D, 0)
#
#        return u.reshape((ny, nx)), v.reshape((ny, nx)), D.reshape((ny, nx))
#
#    return run_model_forward


def make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_pic_iterations, n_newt_iterations,
                                                 mucoef_0, C_0, sliding="linear",
                                                 periodic=False, B_field=None,
                                                 temperature_field=None):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    
    fc_velocity_gradient                       = fc_velocity_gradient_function_cf_safe(dy, dx, ny, nx,
                                                                               ice_mask, add_uv_ghost_cells,
                                                                               add_scalar_ghost_cells)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    
    viscosity_fct = fc_viscosity_function_new_givenT_noextrap(ny, nx, dy, dx, 
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new_noextrap(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT_noextrap(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field)

    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    #j_coord_ar = jnp.arange(ny*nx)
    #pattern = jnp.zeros((nx*ny, nx*ny))*jnp.nan
    #for _, i_coord_ar in zip(basis_vectors, i_coordinate_sets):
    #    mask = ~jnp.isnan(i_coord_ar)

    #    pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
    #                         j_coord_ar[mask].astype(jnp.int32)].set(1)

    #plt.imshow(np.array(pattern[:, 26].reshape((ny,nx))))
    #plt.show()
    #raise




    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
                                                        basis_vectors,\
                                                        2,
                                                        active_indices=(0,1)
                                                       )
    #sparse_jacrev = jax.jit(sparse_jacrev)

    mask = (i_coordinate_sets>=0)

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
   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=True,
    #                                                          ksp_type="bcgs",
    #                                                          preconditioner="jacobi",
    #                                                          monitor_ksp=False,
    #                                                          ksp_max_iter=20)
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=True,
    #                                                          #ksp_type='gmres',
    #                                                          #ksp_type='bcgs',
    #                                                          ksp_type='cg',
    #                                                          #preconditioner="jacobi", #might just about be workable
    #                                                          preconditioner="sor", #better than jacobi
    #                                                          monitor_ksp=True,
    #                                                          ksp_max_iter=200)
    #Basically can only be used for newton-krylov type stuff where you're not expecting the LA problem to actually be solved
    #until quite near the end...

    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords,(ny*nx*2, ny*nx*2))


    sparse_matvec, _, extract_inverse_diagonal = make_sparse_matvec(ny*nx*2, coords)  
    #cg_solver = make_sparse_dpgc_solver_comp(sparse_matvec, extract_inverse_diagonal,
    #                                         iterations=500)
    j_solver = make_sparse_damped_jacobi_solver(sparse_matvec, extract_inverse_diagonal, iterations=10000)
    #preconditioner = make_point_sor_preconditioner(coords, (ny*nx*2, ny*nx*2))
    bcgs_solver = make_sparse_bicgstab_solver(sparse_matvec, iterations=200)
    #relax_solver = make_multicoloured_relaxation(sparse_matvec, extract_inverse_diagonal, basis_vectors, ny, nx)
    
    #bcgs_solver = make_sparse_gs_precond_bicgstab_solver(sparse_matvec, extract_inverse_diagonal, 
    #                                                     basis_vectors, ny, nx, iterations=200)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    #res_fct = lambda x: jnp.mean(jnp.abs(x))

    
    omega=1
    

    @custom_vjp
    def solver(q, p, u_trial, v_trial, h):
        #plt.imshow(q, vmin=-2, vmax=0.5, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        #plt.imshow(p, vmin=-4, vmax=4, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        du = jnp.zeros((nx*ny*2,))
        for i in range(n_pic_iterations):
            #NOTE: making this twice as large makes PIG look a little better...
            #mu_ew = 2*mu_ew
            #mu_ns = 2*mu_ns

            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(jnp.log10(mu_ew[:,1:].reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()
            #plt.imshow(jnp.log10(beta.reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()

            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            print("constructing LA problem")
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           


            full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            
            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            #
            #plt.imshow(jnp.log(jnp.abs(full_jac-jnp.transpose(full_jac))).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            

            #print(full_jac[:(ny*nx),:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[(ny*nx):,:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[:(ny*nx),(ny*nx):])
            #print("--------------------------------")
            #print(full_jac[(ny*nx):,(ny*nx):])
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
    


            #jacrev = jax.jacrev(get_uv_residuals_linear_ssa, argnums=(0,1))
            #dense_dJ_du,  dense_dJ_dv = jacrev(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
            #dense_dJu_du, dense_dJu_dv = dense_dJ_du
            #dense_dJv_du, dense_dJv_dv = dense_dJ_dv

            #dense_full_jac = jnp.block([[dense_dJu_du, dense_dJu_dv],
            #                            [dense_dJv_du, dense_dJv_dv]])



            ##plt.imshow(jnp.log(jnp.abs(dense_full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow((full_jac-dense_full_jac).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx), (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):,:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):, (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),26])).reshape((ny,nx)))
            ##plt.show()
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),(ny*nx)+26])).reshape((ny,nx)))
            ##plt.show()

            ##nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            ##jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            print("solving LA problem")
            du = la_solver(nz_jac_values, rhs)
            #du = cg_solver(nz_jac_values, rhs, du)
            #du = j_solver(nz_jac_values, rhs, du)
            #du = relax_solver(nz_jac_values, rhs, du)
            #du = bcgs_solver(nz_jac_values, rhs, du)


            print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(h>0, cmap="Grays_r")
            #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=0.7, vmin=0)
            #plt.colorbar()
            #plt.show()

            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
            
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.colorbar()
        #plt.show()

        
        final_residual_pic = res_fct(rhs_new)

        print("Final Picard residual: {}".format(final_residual_pic))
        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):
            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, p, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            du = la_solver(nz_jac_values, rhs)
            #du = cg_solver(nz_jac_values, rhs, du)
            #du = j_solver(nz_jac_values, rhs, du)
            #du = relax_solver(nz_jac_values, rhs, du)
            #du = bcgs_solver(nz_jac_values, rhs, du)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        print("Final Newton residual: {}".format(final_residual))
        print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")
        
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        ##plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.colorbar()
        #plt.show()
        ##plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        ###plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        ##plt.colorbar()
        ##plt.show()


        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, p, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, p, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa, \
                              (u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, p, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, p, h = res
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
        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1)
                                      )
        _, _, q_bar, p_bar, _ = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        #return (q_bar.reshape((ny, nx)), p_bar.reshape((ny,nx)), None, None, None)
        return (q_bar.reshape((ny, nx)), p_bar.reshape((ny,nx)), None, None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver

def make_picnewton_velocity_solver_function_full_cvjp(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_pic_iterations, n_newt_iterations,
                                                 mucoef_0, C_0, sliding="linear",
                                                 periodic=False, B_field=None,
                                                 temperature_field=None):

    if temperature_field is None:
    
        temperature_field = (jnp.zeros((ny,nx))+258.15)


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    
    #extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function_cornersafe(ice_mask)
    #extrapolate_over_cf                        = mean_linear_extrapolate_over_cf_function(ice_mask)
    
    viscosity_fct = fc_viscosity_function_new_givenT(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0,
                                                   temperature_field)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_pnotC_givenT(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf,
                                                       mucoef_0, C_0,
                                                       temperature_field)

    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 2,
                                                                                  periodic_x=periodic)
    #j_coord_ar = jnp.arange(ny*nx)
    #pattern = jnp.zeros((nx*ny, nx*ny))*jnp.nan
    #for _, i_coord_ar in zip(basis_vectors, i_coordinate_sets):
    #    mask = ~jnp.isnan(i_coord_ar)

    #    pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
    #                         j_coord_ar[mask].astype(jnp.int32)].set(1)

    #plt.imshow(np.array(pattern[:, 26].reshape((ny,nx))))
    #plt.show()
    #raise




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

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
    #                                                          coords,
    #                                                          (ny*nx*2, ny*nx*2),
    #                                                          indirect=True,
    #                                                          monitor_ksp=False)
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)

    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords,(ny*nx*2, ny*nx*2))

    
    res_fct = lambda x: jnp.max(jnp.abs(x))
    #res_fct = lambda x: jnp.mean(jnp.abs(x))

    
    omega=1
    

    @custom_vjp
    def solver(q, p, u_trial, v_trial, h):
        #plt.imshow(q, vmin=-2, vmax=0.5, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        #plt.imshow(p, vmin=-4, vmax=4, cmap="RdBu")
        #plt.colorbar()
        #plt.show()
        
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        for i in range(n_pic_iterations):
            #NOTE: making this twice as large makes PIG look a little better...
            #mu_ew = 2*mu_ew
            #mu_ns = 2*mu_ns

            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(jnp.log10(mu_ew[:,1:].reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()
            #plt.imshow(jnp.log10(beta.reshape((ny,nx))[40:-5, 20:-30]))
            #plt.colorbar()
            #plt.show()

            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            #print("constructing LA problem")
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           


            #full_jac = jnp.zeros((ny*nx*2, ny*nx*2))
            #full_jac = full_jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #
            #plt.imshow(jnp.log(jnp.abs(full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            #plt.colorbar()
            #plt.show()
            

            #print(full_jac[:(ny*nx),:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[(ny*nx):,:(ny*nx)])
            #print("--------------------------------")
            #print(full_jac[:(ny*nx),(ny*nx):])
            #print("--------------------------------")
            #print(full_jac[(ny*nx):,(ny*nx):])
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
            #print("--------------------------------")
    


            #jacrev = jax.jacrev(get_uv_residuals_linear_ssa, argnums=(0,1))
            #dense_dJ_du,  dense_dJ_dv = jacrev(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
            #dense_dJu_du, dense_dJu_dv = dense_dJ_du
            #dense_dJv_du, dense_dJv_dv = dense_dJ_dv

            #dense_full_jac = jnp.block([[dense_dJu_du, dense_dJu_dv],
            #                            [dense_dJv_du, dense_dJv_dv]])



            ##plt.imshow(jnp.log(jnp.abs(dense_full_jac[:,:])).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow((full_jac-dense_full_jac).reshape((ny*nx*2,2*nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx), (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()
            #
            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):,:(ny*nx)])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[(ny*nx):, (ny*nx):])).reshape((ny*nx,nx*ny)))
            ##plt.colorbar()
            ##plt.show()

            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),26])).reshape((ny,nx)))
            ##plt.show()
            ##plt.imshow(jnp.log(jnp.abs(full_jac[:(ny*nx),(ny*nx)+26])).reshape((ny,nx)))
            ##plt.show()

            ##nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            ##jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            #print("solving LA problem")
            du = la_solver(nz_jac_values, rhs)

            #print("du norm: {}".format(jnp.max(jnp.abs(du))))

            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            #plt.imshow(jnp.sqrt(u_1d**2 + v_1d**2 + 1).reshape((ny,nx)))
            #plt.colorbar()
            #plt.show()

            #plt.imshow(h>0, cmap="Grays_r")
            #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=0.7, vmin=0)
            #plt.colorbar()
            #plt.show()

            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            
            if i==0:
                initial_residual = jnp.max(rhs)
            #    print(f"INIT RES: {initial_residual}")
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")
            
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.colorbar()
        #plt.show()

        if n_pic_iterations>0:     
            final_residual_pic = res_fct(rhs_new)

            print("Final Picard residual: {}".format(final_residual_pic))
            print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):
            #h_1d = jnp.where(jnp.sqrt(u_1d**2 + v_1d**2 + 1)<3e4, h_1d, 0)
            #h = h_1d.reshape((ny,nx))

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, p, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))
            
            du = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))

            #plt.imshow(jnp.log10(jnp.abs(rhs_new[:(nx*ny)])).reshape((ny,nx)), alpha=1, vmin=0)
            #plt.colorbar()
            #plt.show()
            
            print(f"nonlinear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        print("Final Newton residual: {}".format(final_residual))
        
        if n_pic_iterations>0:
           print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")
        
        #plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        ##plt.imshow(jnp.log10(jnp.abs(-jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))[(ny*nx):])).reshape((ny,nx)), alpha=1, vmin=0)
        #plt.colorbar()
        #plt.show()


        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, p, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, p, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa, \
                              (u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, p, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, p, h = res
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
        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1)
                                      )
        _, _, q_bar, p_bar, _ = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        #return (q_bar.reshape((ny, nx)), p_bar.reshape((ny,nx)), None, None, None)
        return (q_bar.reshape((ny, nx)), p_bar.reshape((ny,nx)), None, None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver


def make_picnewton_velocity_solver_function_cvjp(ny, nx, dy, dx,
                                                 b, ice_mask,
                                                 n_pic_iterations, n_newt_iterations,
                                                 mucoef_0, sliding="linear",
                                                 periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf, mucoef_0)

    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)


    
    res_fct = lambda x: jnp.max(jnp.abs(x))
    #res_fct = lambda x: jnp.mean(jnp.abs(x))

    omega = 1

    @custom_vjp
    def solver(q, C, u_trial, v_trial, h):
        plt.imshow(q)
        plt.colorbar()
        plt.show()

        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask = jnp.where(h>0,1,0).reshape(-1)
            
        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
        beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        for i in range(n_pic_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            #nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            
            du = la_solver(nz_jac_values, rhs)

            #u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            #v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask
            
            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        #plt.imshow(rhs_new[:(ny*nx)].reshape((ny,nx)))
        #plt.colorbar()
        #plt.show()
        #plt.imshow(rhs_new[(ny*nx):].reshape((ny,nx)))
        #plt.colorbar()
        #plt.show()
        #raise

        final_residual_pic = res_fct(rhs_new)

        print("Final Picard residual: {}".format(final_residual_pic))
        print("Picard residual reduction factor: {}".format(initial_residual/final_residual_pic))


        for i in range(n_newt_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                                             (u_1d, v_1d, q, C, h_1d)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, C, h_1d))
            
            du = la_solver(nz_jac_values, rhs)

            u_1d = (u_1d + du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + du[(ny*nx):]) * ice_mask
            
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, C, h_1d))
            
            print(f"linear residual reduction factor: {res_fct(rhs)/res_fct(rhs_new)}")

        final_residual = res_fct(rhs_new)

        print("Final Newton residual: {}".format(final_residual))
        print("Newton residual reduction factor: {}".format(final_residual_pic/final_residual))
        
        print("TOTAL residual reduction factor: {}".format(initial_residual/final_residual))

        print("===========================================")


        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, C, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa, \
                                        (u.reshape(-1), v.reshape(-1), q, C, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, C, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, C, h = res
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
        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, C, h.reshape(-1)
                                      )
        _, _, q_bar, _, _ = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        return (q_bar.reshape((ny, nx)), None, None, None, None)


    solver.defvjp(solver_fwd, solver_bwd)

    return solver

def make_picard_velocity_solver_function_full_cvjp(ny, nx, dy, dx,
                                                   b, ice_mask, n_iterations,
                                                   mucoef_0, C_0, sliding="linear",
                                                   periodic=False):


    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(ice_mask)
    

    viscosity_fct = fc_viscosity_function_new(ny, nx, dy, dx, 
                                                   extrapolate_over_cf,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   ew_gradient, ns_gradient,
                                                   ice_mask, mucoef_0)
    beta_fct = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_new(ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf)
    
    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct, ice_mask,
                                                       interp_cc_to_fc,
                                                       ew_gradient, ns_gradient,
                                                       cc_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       extrapolate_over_cf, mucoef_0)

    

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),
    #                                                          ksp_type="gmres",
    #                                                          preconditioner="hypre",
    #                                                          precondition_only=False,
    #                                                          ksp_max_iter=60,
    #                                                          monitor_ksp=False)
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)



    @custom_vjp
    def solver(q, p, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                                 (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta)
                                                          )

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

           
            #nz_jac_values = jnp.where(jnp.abs(nz_jac_values) < 1e-10, 0.0, nz_jac_values)
            #jax.debug.print("{x}", x=nz_jac_values)

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))

            
            du = la_solver(nz_jac_values, rhs)

            u_1d = u_1d + du[:(ny*nx)]
            v_1d = v_1d + du[(ny*nx):]
            
            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta))
            if i==0:
                initial_residual = jnp.max(rhs)
            print(f"linear residual reduction factor: {jnp.max(jnp.abs(rhs))/jnp.max(jnp.abs(rhs_new))}")

        final_residual = jnp.max(jnp.abs(rhs_new))

        print("----------")
        print("Final residual: {}".format(final_residual))
        print("Total residual reduction factor: {}".format(initial_residual/final_residual))
        print("----------")
        
        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))



    def solver_fwd(q, p, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        u, v = solver(q, p, u_trial, v_trial, h)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa, \
                                        (u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1))
                                                      )
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

        #dJ_dvel_nz_values = jnp.where(jnp.abs(dJ_dvel_nz_values) < 1e-10, 0.0, dJ_dvel_nz_values)

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, p, h)
        #fwd_residuals = (u, v, q)

        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        
        u, v, dJ_dvel_nz_values, q, p, h = res
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
        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1)
                                      )
        _, _, q_bar, p_bar, _ = pullback_function((lambda_u, lambda_v))
        
#        #bwd has to return a tuple of cotangents for each primal input
#        #of solver, so have to return this 1-tuple:
#        return (mu_bar.reshape((ny, nx)), )

        #I wonder if I can get away with just returning None for u_trial_bar and v_trial_bar...
        return (q_bar.reshape((ny, nx)), p_bar.reshape((ny, nx)), None, None, None)


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

def forward_adjoint_and_second_order_adjoint_solvers_picard(ny, nx, dy, dx, h, b,\
                                                     C, n_iterations, mucoef_0,\
                                                     periodic=False):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)
    
    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
        
    add_uv_ghost_cells, add_sc_ghost_cells     = add_ghost_cells_fcts(ny, nx, periodic=periodic)

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
                                                                                     periodic_x=periodic)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient, mucoef_0)
    fc_viscosity = fc_viscosity_function(ny, nx, dy, dx, extrapolate_over_cf, add_uv_ghost_cells,
                                         add_sc_ghost_cells,
                                         interp_cc_to_fc, ew_gradient, ns_gradient, h_1d, mucoef_0)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       b,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf, mucoef_0)
    
    linear_ssa_residuals = compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_no_rhs = compute_linear_ssa_residuals_function_fc_visc_no_rhs(
                                                       ny, nx, dy, dx,\
                                                       h_1d, b, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   

    #Note the insane number of ksp iterations!!!!!! Ill conditioned matrices in SOA cals.
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
                                                              ksp_type="gmres",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,
                                                              monitor_ksp=False,\
                                                              ksp_max_iter=400)


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

def forward_adjoint_and_second_order_adjoint_solvers_nl_fudge(ny, nx, dy, dx, h, b,\
                                                     C, n_iterations, mucoef_0,\
                                                     periodic=False):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)
    
    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    add_uv_ghost_cells, add_sc_ghost_cells     = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #if periodic:
    #    add_uv_ghost_cells, add_sc_ghost_cells = add_ghost_cells_fcts_funny_periodic_stream_case(ny, nx)
    #else:
    #    add_uv_ghost_cells, add_sc_ghost_cells = add_ghost_cells_fcts(ny, nx)

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
                                                                                     periodic_x=periodic)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient, mucoef_0)
    fc_viscosity = fc_viscosity_function(ny, nx, dy, dx, extrapolate_over_cf, add_uv_ghost_cells,
                                         add_sc_ghost_cells,
                                         interp_cc_to_fc, ew_gradient, ns_gradient, h_1d, mucoef_0)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       b,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf, mucoef_0)
    
    linear_ssa_residuals = compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_nl_fudge = compute_linear_ssa_residuals_function_fc_visc_nl_fudge(
                                                       ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_no_rhs = compute_linear_ssa_residuals_function_fc_visc_no_rhs(
                                                       ny, nx, dy, dx,\
                                                       h_1d, b, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   

    #Note the insane number of ksp iterations!!!!!! Ill conditioned matrices in SOA cals.
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
    #                                                          ksp_type="gmres",\
    #                                                          preconditioner="hypre",\
    #                                                          precondition_only=False,
    #                                                          monitor_ksp=False,\
    #                                                          ksp_max_iter=400)

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)

    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)

    #picard_solver = make_picard_solver(ny, nx, sparse_jacrev, mask, la_solver, get_u_v_residuals, fc_visc)

    def solve_fwd_problem(q, u_trial, v_trial):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        #u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,), coords)
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,))

        u = jnp.where(h>1e-10, u.reshape((ny,nx)), 0)
        v = jnp.where(h>1e-10, v.reshape((ny,nx)), 0)

        return u, v


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
       
            # just making sure
            u = jnp.where(h_1d>1e-10, u_new, 0)
            v = jnp.where(h_1d>1e-10, v_new, 0)


            #u = 0.9*u + 0.1*u_new
            #v = 0.9*v + 0.1*v_new
            #u = u_new
            #v = v_new
            
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
                               linear_ssa_residuals_nl_fudge, 1, (mu_bar_ew, mu_bar_ns, rhs))
        lx = jnp.where(h>1e-10, lx, 0)
        ly = jnp.where(h>1e-10, ly, 0)


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
        mu_x, mu_y = newton_solver(x_trial, y_trial, linear_ssa_residuals_nl_fudge,
                                   1, (mu_bar_ew, mu_bar_ns, rhs))

        mu_x = jnp.where(h>1e-10, mu_x, 0)
        mu_y = jnp.where(h>1e-10, mu_y, 0)

        #solve second equation for beta
        #NOTE: make functional essentially a function just of u,v to avoid the
        #difficulties with what to do with q gradients
        functional_fixed_q = lambda u, v: functional(u, v, q)
        gradient_j = jax.grad(functional_fixed_q, argnums=(0,1))
        direct_hvp_x, direct_hvp_y = jax.jvp(gradient_j,
                                             (u.reshape(-1), v.reshape(-1)),
                                             (mu_x.reshape(-1), mu_y.reshape(-1)))[1]
        rhs_1_x, rhs_1_y = div_tensor_field((h * mu_bar * perturbation_direction)[...,None,None] *\
                                        membrane_strain_rate(lx.reshape(-1), ly.reshape(-1)) *\
                                        (1/c.GLEN_N)
                                           )
        ##NOTE: Was trying ways of getting a fudgy version of du^2Gl for the beta rhs, but
        ##it's really hard - at least for me.
        #H_mu_x, H_mu_y = div_tensor_field((h * mu_bar)[...,None,None] * membrane_strain_rate(mu_x.reshape(-1), mu_y.reshape(-1)))
        #rhs_1_x = rhs_1_x - (1/c.GLEN_N - 1) * lx * H_mu_x
        #rhs_1_y = rhs_1_y - (1/c.GLEN_N - 1) * ly * H_mu_y

        rhs = - jnp.concatenate([(rhs_1_x.reshape(-1) + direct_hvp_x),
                                 (rhs_1_y.reshape(-1) + direct_hvp_y)])

        print("solving for beta")
        beta_x, beta_y = newton_solver(x_trial, y_trial, linear_ssa_residuals_nl_fudge,
                                       1, (mu_bar_ew, mu_bar_ns, rhs))

        beta_x = jnp.where(h>1e-10, beta_x, 0)
        beta_y = jnp.where(h>1e-10, beta_y, 0)

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
                                    )*(1/c.GLEN_N) +\
              double_dot_contraction(
                            cc_vector_field_gradient(beta_x.reshape(-1), beta_y.reshape(-1)),
                            membrane_strain_rate(u.reshape(-1), v.reshape(-1))
                                    )
                                              )

        return hvp

    return solve_fwd_problem, solve_adjoint_problem, solve_soa_problem

def forward_adjoint_and_second_order_adjoint_solvers(ny, nx, dy, dx, h, b,\
                                                     C, n_iterations, mucoef_0,\
                                                     periodic=False):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)
    
    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    
    add_uv_ghost_cells, add_sc_ghost_cells     = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #if periodic:
    #    add_uv_ghost_cells, add_sc_ghost_cells = add_ghost_cells_fcts_funny_periodic_stream_case(ny, nx)
    #else:
    #    add_uv_ghost_cells, add_sc_ghost_cells = add_ghost_cells_fcts(ny, nx)

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
                                                                                     periodic_x=periodic)

    #calculate cell-centred viscosity based on velocity and q
    cc_viscosity = cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient, mucoef_0)
    fc_viscosity = fc_viscosity_function(ny, nx, dy, dx, extrapolate_over_cf, add_uv_ghost_cells,
                                         add_sc_ghost_cells,
                                         interp_cc_to_fc, ew_gradient, ns_gradient, h_1d, mucoef_0)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       b,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf, mucoef_0)
    
    linear_ssa_residuals = compute_linear_ssa_residuals_function_fc_visc(ny, nx, dy, dx,\
                                                       h_1d, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    linear_ssa_residuals_no_rhs = compute_linear_ssa_residuals_function_fc_visc_no_rhs(
                                                       ny, nx, dy, dx,\
                                                       h_1d, b, beta_eff,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_uv_ghost_cells,\
                                                       add_sc_ghost_cells,\
                                                       extrapolate_over_cf)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)

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

   

    #Note the insane number of ksp iterations!!!!!! Ill conditioned matrices in SOA cals.
    #la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
    #                                                          ksp_type="gmres",\
    #                                                          preconditioner="hypre",\
    #                                                          precondition_only=False,
    #                                                          monitor_ksp=False,\
    #                                                          ksp_max_iter=400)

    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords,
                                                              (ny*nx*2, ny*nx*2),
                                                              indirect=False,
                                                              monitor_ksp=False)

    newton_solver = generic_newton_solver_no_cjvp(ny, nx, sparse_jacrev, mask, la_solver)

    #picard_solver = make_picard_solver(ny, nx, sparse_jacrev, mask, la_solver, get_u_v_residuals, fc_visc)

    def solve_fwd_problem(q, u_trial, v_trial):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)
        
        #u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,), coords)
        u, v = newton_solver(u_trial, v_trial, get_u_v_residuals, n_iterations, (q,))

        u = jnp.where(h>1e-10, u.reshape((ny,nx)), 0)
        v = jnp.where(h>1e-10, v.reshape((ny,nx)), 0)

        return u, v


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
       
            # just making sure
            u = jnp.where(h_1d>1e-10, u_new, 0)
            v = jnp.where(h_1d>1e-10, v_new, 0)


            #u = 0.9*u + 0.1*u_new
            #v = 0.9*v + 0.1*v_new
            #u = u_new
            #v = v_new
            
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
        lx = jnp.where(h>1e-10, lx, 0)
        ly = jnp.where(h>1e-10, ly, 0)


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

        mu_x = jnp.where(h>1e-10, mu_x, 0)
        mu_y = jnp.where(h>1e-10, mu_y, 0)

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

        beta_x = jnp.where(h>1e-10, beta_x, 0)
        beta_y = jnp.where(h>1e-10, beta_y, 0)

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





def quasi_newton_coupled_comp_1d(C, B_int, iterations, dt, bmr, acc=None):
    
    if acc is None:
        acc=accumulation.copy()

    mom_res = make_linear_momentum_residual_osd_at_gl()
    adv_res = make_adv_residual(dt, acc)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B_int * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta

    def continue_condition(state):
        _,_,_, i,res,resrat = state
        return i<iterations


    def step(state):
        u, h, h_init, i, prev_res, prev_resrat = state

        beta = new_beta(u, h)
        mu = new_mu(u)

        jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
        jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

        full_jacobian = jnp.block(
                                  [ [jac_mom_res[0], jac_mom_res[1]],
                                    [jac_adv_res[0], jac_adv_res[1]] ]
                                  )
    
        rhs = jnp.concatenate((-mom_res(u, h, mu, beta),
                               -adv_res(u, h, h_init, bmr)))
    
        dvar = lalg.solve(full_jacobian, rhs)
    
        u = u.at[:].set(u+dvar[:n])
        h = h.at[:].set(h+dvar[n:])

        #TODO: add one for the adv residual too...
        res = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 

        return u, h, h_init, i+1, res, prev_res/res


    def iterator(u_init, h_init):    

        resrat = np.inf
        res = np.inf

        initial_state = u_init, h_init, h_init, 0, res, resrat

        u, h, h_init, itn, res, resrat = jax.lax.while_loop(continue_condition, step, initial_state)

        return u, h, res
       
    return iterator




