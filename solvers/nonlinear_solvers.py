#1st Party
import sys

#3rd Party
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants_years as c
from grid import *

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers/")
import residuals as rdl
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp
from residuals import *



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

def make_newton_velocity_solver_function_custom_vjp(ny, nx, dy, dx,\
                                                    h, b, C,\
                                                    n_iterations,\
                                                    mucoef_0,\
                                                    periodic=False):

    beta_eff = C.copy()
    h_1d = h.reshape(-1)

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    #add_uv_ghost_cells, add_cont_ghost_cells   = add_ghost_cells_fcts(ny, nx)
    #add_scalar_ghost_cells                     = add_ghost_cells_periodic_continuation_function(ny, nx) if periodic else add_cont_ghost_cells
    extrapolate_over_cf                        = linear_extrapolate_over_cf_function(h)

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

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),\
                                                              ksp_type="gmres",\
                                                              preconditioner="hypre",\
                                                              precondition_only=False,\
                                                              ksp_max_iter=40)



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

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]


        res_final = jnp.max(jnp.abs(jnp.concatenate(
                                    get_u_v_residuals(u_1d, v_1d, q)
                                                   )
                                   )
                           )
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


def make_couple_quasi_newton_solver_function(ny, nx, dy, dx,
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


def make_diva_velocity_solver_function(ny, nx, dy, dx,
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

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),
                                                              ksp_type="gmres",
                                                              preconditioner="hypre",
                                                              precondition_only=False,
                                                              ksp_max_iter=60,
                                                              monitor_ksp=False)



    def solver(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

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

    return solver


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

   
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp(coords, (ny*nx*2, ny*nx*2),
                                                              ksp_type="gmres",
                                                              preconditioner="hypre",
                                                              precondition_only=False,
                                                              ksp_max_iter=60,
                                                              monitor_ksp=False)



    @custom_vjp
    def solver(q, C, u_trial, v_trial, h):
        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        residual = jnp.inf
        init_res = 0

        for i in range(n_iterations):

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d)
            beta = beta_fct(C, u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

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




