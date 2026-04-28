#1st party
import os
import sys
import time

#3rd party
import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO as jax_bcoo

#local apps

nm_home = os.environ['NM_HOME']

sys.path.insert(1, os.path.join(nm_home, 'utils'))
from sparsity_utils import jax_coo_to_csr


def make_sparse_matvec(matrix_width, coords):
    #j-coordinates typically look like jnp.arange(nr*nc)
    #but I think it's best to make this generic
    
    i_coords = coords[0,:]
    j_coords = coords[1,:]
    diag_mask = (i_coords == j_coords)

    @jax.jit
    def sparse_matvec(vals, x):
        aij_xj = vals * x[j_coords]
    
        sumj_aij_xj = jnp.zeros(matrix_width).at[i_coords].add(aij_xj)
    
        #Need to test whether the following might be faster!
        #sumj_aij_xj = jax.ops.segment_sum(aij_xj, i_coords, num_segments=matrix_width)
    
        return sumj_aij_xj

    @jax.jit
    def A_inner_product(vals, vec):
        return jnp.dot(vec, sparse_matvec(vals, vec))

    @jax.jit
    def extract_inverse_diagonal(vals):
        diag_vals = vals[diag_mask]
        diag_indices = i_coords[diag_mask]
        
        return 1/(jnp.zeros(matrix_width).at[diag_indices].add(diag_vals) + 1e-15)


    return sparse_matvec, A_inner_product, extract_inverse_diagonal


def sparse_dpcg_solver(sp_matvec, inverse_diag_fct, vals, b, x0, iterations=10):

    M_inv = inverse_diag_fct(vals)

    r = b - sp_matvec(vals, x0)
    x = x0.copy()

    d = M_inv * r

    rs_old = jnp.dot(r, M_inv * r)

    for i in range(iterations):
        #print(jnp.max(jnp.abs(r)))
        #print(jnp.abs(rs_old))

        Ad = sp_matvec(vals, d)

        alpha = rs_old / jnp.dot(Ad, d)
        
        x = x + alpha * d
        
        r = r - alpha * Ad
        
        rs_new = jnp.dot(r, M_inv * r)

        beta = rs_new/rs_old

        d = M_inv * r + beta * d

        rs_old = rs_new

    print(jnp.abs(rs_old))
    
    return x


def make_sparse_dpgc_solver_comp(sp_matvec, inverse_diag_fct, iterations=10, tol=1e-5):
        
    def conditional(state):
        i, _, r, _, rs = state
        return jnp.logical_and(i<iterations, jnp.abs(rs)>tol)

    def solver(vals, b, x0):
    
        M_inv = inverse_diag_fct(vals)
    
        r0 = b - sp_matvec(vals, x0)
    
        d0 = M_inv * r0
    
        rs0 = jnp.dot(r0, M_inv * r0)
    
        initial_state = (0, x0, r0, d0, rs0)
    
        def update(state):
            i, xi, r, d, rs = state
    
            Ad = sp_matvec(vals, d)
    
            alpha = rs / jnp.dot(Ad, d)
    
            xi = xi + alpha * d
    
            r = r - alpha * Ad
    
            rs_new = jnp.dot(r, M_inv * r)
    
            beta = rs_new/rs
    
            d = M_inv * r + beta * d
    
            rs = rs_new

            return (i+1, xi, r, d, rs)
    
        i, xi, r, d, rs = jax.lax.while_loop(conditional, update, initial_state)
        jax.debug.print("LA final residual {x}", x=jnp.abs(rs))
        return xi

    return jax.jit(solver)

def fake_lax_while_loop(conditional, update, initial_state):
    state = initial_state
    while conditional(state):
        state = update(state)
    return state

def make_sparse_dpcg_solver_jsp_comp(coords, inverse_diag_fct, jac_width, iterations=10, tol=1e-20):

    def conditional(state):
        i, _, r, _, rs = state
        return jnp.logical_and(i<iterations, jnp.abs(rs)>tol)
    
    def solver(vals, b, x0):

        A_sp = jax_bcoo((vals, coords.T), shape=(jac_width, jac_width))
    
        M_inv = inverse_diag_fct(vals)
  
        #jax.debug.print("M_inv: {x}", x=M_inv)

        r0 = b - A_sp @ x0

        d0 = M_inv * r0
    
        rs0 = jnp.dot(r0, M_inv * r0)
    
        initial_state = (0, x0, r0, d0, rs0)
            
        #jax.debug.print("IS Max: {x}",x=(jnp.max(x0), jnp.max(r0), jnp.max(d0), jnp.max(rs0)))
    
        def update(state):
            i, xi, r, d, rs = state
   
            #t0 = time.perf_counter()
            Ad = A_sp @ d
            #Ad.block_until_ready()
            #t_matvec = time.perf_counter()-t0



            #t_0 = time.perf_counter()

            alpha = rs / jnp.dot(Ad, d)
    
            xi = xi + alpha * d
    
            r = r - alpha * Ad
    
            rs_new = jnp.dot(r, M_inv * r)
    
            beta = rs_new/rs
    
            d = M_inv * r + beta * d
    
            rs = rs_new
            
            #Ad.block_until_ready()
            #t_rest = time.perf_counter()-t0
        
            #print(
            #    f"CG iter {i:3d} | "
            #    f"Matvec: {t_matvec:7.4f}s | "
            #    f"Rest:   {t_rest:7.4f}s | "
            #)


            return (i+1, xi, r, d, rs)
    
        i, xi, r, d, rs = jax.lax.while_loop(conditional, update, initial_state)
        #i, xi, r, d, rs = fake_lax_while_loop(conditional, update, initial_state)
        
        #jax.debug.print("LA final residual {x}", x=jnp.abs(rs))
    
        return xi
    
    return jax.jit(solver)
    #return solver




def make_sparse_dpcg_solver_jsp_comp_fori(
    coords,
    inverse_diag_fct,
    jac_width,
    iterations=10,
    tol=1e-20,
):

    def solver(vals, b, x0):

        # Sparse matrix (unchanged)
        A_sp = jax_bcoo((vals, coords.T),
                        shape=(jac_width, jac_width))

        M_inv = inverse_diag_fct(vals)

        # Initial CG state (same math as before)
        r0  = b - A_sp @ x0
        d0  = M_inv * r0
        rs0 = jnp.dot(r0, M_inv * r0)

        # State = (x, r, d, rs)
        init_state = (x0, r0, d0, rs0)

        def body(i, state):
            x, r, d, rs = state

            # Convergence test (scalar, but used only for masking)
            converged = rs < tol

            Ad = A_sp @ d

            alpha = rs / jnp.dot(Ad, d)

            x_new = x + alpha * d
            r_new = r - alpha * Ad

            rs_new = jnp.dot(r_new, M_inv * r_new)
            beta   = rs_new / rs
            d_new  = M_inv * r_new + beta * d

            # Masked updates after convergence
            x  = jnp.where(converged, x,  x_new)
            r  = jnp.where(converged, r,  r_new)
            d  = jnp.where(converged, d,  d_new)
            rs = jnp.where(converged, rs, rs_new)

            return (x, r, d, rs)

        x, r, d, rs = jax.lax.fori_loop(
            0, iterations, body, init_state
        )

        return x

    return jax.jit(solver)


def make_point_sor_preconditioner(coordinates, jac_shape, omega=1.0):
    """
    Returns a function M(r, vals) that applies ONE forward SOR sweep:
        z = M^{-1} r
    using the CSR sparsity pattern (iptr, indices).

    Arguments:
        omega  : relaxation factor (1.0 = GS, <1 underrelax, >1 overrelax)

    The returned function expects:
        M(r, vals)
    where vals is the CSR data *for this evaluation*.
    """

    nnz = coordinates[0].size

    iptr, indices, _, order = jax_coo_to_csr(jnp.zeros(nnz, dtype=jnp.float64),
                                      coordinates, jac_shape)
    iptr = jnp.array(iptr)
    indices = jnp.array(indices)


    @jax.jit
    def sor_apply(vals, r):
        """
        Apply one forward SOR sweep to approximate z = M^{-1} r.
        vals: CSR values (nnz,), same ordering as iptr/indices.
        r   : right hand side vector
        """
        N = r.shape[0]

        # Initial guess z = 0 for preconditioning
        z0 = jnp.zeros_like(r)

        def body(i, z):
            start = iptr[i]
            end   = iptr[i+1]

            js   = indices[start:end]
            aij  = vals[start:end]

            # Identify diagonal entry
            diag_mask = (js == i)
            a_ii = aij[diag_mask][0]   # guaranteed exactly one diagonal per row

            # Lower part: uses updated z[j]
            lower_mask = (js < i)
            sum_lower  = jnp.sum(aij[lower_mask] * z[js[lower_mask]])

            # Upper part: uses OLD z0[j] == 0 (since z0 is zero)
            upper_mask = (js > i)
            sum_upper  = jnp.sum(aij[upper_mask] * z0[js[upper_mask]])

            # Gauss–Seidel update
            new_val = (r[i] - sum_lower - sum_upper) / a_ii
            z_new_i = (1 - omega) * z[i] + omega * new_val

            return z.at[i].set(z_new_i)

        # Sequential row sweep
        z_final = lax.fori_loop(0, N, body, z0)
        return z_final

    return sor_apply




def make_multicoloured_relaxation(sparse_matvec, inv_diag_fct, basis_vectors, 
                                 ny, nx, ncolours=9, omega=1.4, iterations=1):
    
    N = ny * nx
    colour_sets_scalar = [ jnp.where(basis_vectors[c] == 1)[0] for c in range(ncolours)]
    
    colour_sets = [
        jnp.concatenate([rows, rows + N])
        for rows in colour_sets_scalar
    ]

    #@jax.jit
    def relax(vals, b, x0):
        x = x0
        inv_diag = inv_diag_fct(vals)

        for it in range(iterations):
            for rows in colour_sets:               # rows = indices for this colour
                r = b - sparse_matvec(vals, x)     # recompute residual using latest x
                delta = omega * inv_diag[rows] * r[rows]
                x = x.at[rows].add(delta[rows])
            
            for rows in colour_sets[::-1]:         # rows = indices for this colour
                r = b - sparse_matvec(vals, x)     # recompute residual using latest x
                delta = omega * inv_diag[rows] * r[rows]
                x = x.at[rows].add(delta[rows])

            #jax.debug.print("GS residual norm: {x}", x=jnp.dot(r, r))

        return x
    
    return relax



def make_sparse_gs_precond_bicgstab_solver(sp_matvec,
                                           inv_diag_fct,
                                           basis_vectors,
                                           ny, nx,
                                           iterations=200,
                                           tol=1e-6):


    relax_solver =  make_multicoloured_relaxation(sp_matvec, inv_diag_fct, basis_vectors,
                                                  ny, nx, omega=1, iterations=1)


    @jax.jit
    def solver(vals, b, x0):
        preconditioner = lambda r: relax_solver(vals, r, jnp.zeros_like(r))

        # Initial residual
        r0 = b - sp_matvec(vals, x0)
        r_hat = r0  # shadow residual (fixed)

        rho_old = jnp.array(1.0)
        alpha   = jnp.array(1.0)
        omega   = jnp.array(1.0)

        v = jnp.zeros_like(b)
        p = jnp.zeros_like(b)

        # Store residual norm
        rs = jnp.dot(r0, r0)

        state = (0, x0, r0, r_hat, p, v, rho_old, alpha, omega, rs)

        def cond_fn(state):
            i, x, r, r_hat, p, v, rho_old, alpha, omega, rs = state
            return jnp.logical_and(i < iterations, jnp.sqrt(rs) > tol)

        def body_fn(state):
            i, x, r, r_hat, p, v, rho_old, alpha, omega, rs = state
        
            jax.debug.print("BiCGStab res {val}", val=jnp.sqrt(rs))

            rho_new = jnp.dot(r_hat, r)

            # Breakdown protection
            rho_new = jnp.where(rho_new == 0.0, 1e-30, rho_new)

            beta = (rho_new / rho_old) * (alpha / omega)

            # p_k = r + beta*(p - omega*v)
            p_new = r + beta * (p - omega * v)

            # Apply preconditioner
            y = preconditioner(p_new)

            # v = A y
            v_new = sp_matvec(vals, y)

            alpha_new = rho_new / jnp.dot(r_hat, v_new + 1e-30)

            # s = r - alpha*v
            s = r - alpha_new * v_new

            # Early exit check
            s_norm = jnp.sqrt(jnp.dot(s, s))

            def update_from_s(state_s):
                # Apply preconditioner to s
                z = preconditioner(s)
                t = sp_matvec(vals, z)

                omega_new = jnp.dot(t, s) / jnp.dot(t, t + 1e-30)

                x_new = x + alpha_new * y + omega_new * z
                r_new = s - omega_new * t

                rs_new = jnp.dot(r_new, r_new)

                return (i+1, x_new, r_new, r_hat, p_new, v_new,
                        rho_new, alpha_new, omega_new, rs_new)

            def update_skip(state_s):
                # If s is small enough, update x directly
                x_new = x + alpha_new * y
                r_new = s
                rs_new = s_norm**2
                return (i+1, x_new, r_new, r_hat, p_new, v_new,
                        rho_new, alpha_new, omega, rs_new)

            # If s is very small → skip omega step
            return jax.lax.cond(s_norm < 1e-14,
                            update_skip,
                            update_from_s,
                            operand=None)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        i, x_final, r_final, r_hat, p, v, rho_old, alpha, omega, rs_final = final_state

        jax.debug.print("BiCGStab final residual {r}", r=jnp.sqrt(rs_final))
        return x_final

    return solver

def make_sparse_bicgstab_solver(sp_matvec,
                                precond=None,   # function r -> M^{-1} r (e.g. point SOR sweep)
                                iterations=200,
                                tol=1e-6):

    """
    Returns a matrix-free BiCGStab solver:
        x = solver(vals, b, x0)

    sp_matvec(vals, x): required
    precond(r): optional (apply M^{-1} r)
    """

    if precond is None:
        # Identity preconditioner if none supplied
        precond = lambda r: r

    @jax.jit
    def solver(vals, b, x0):

        # Initial residual
        r0 = b - sp_matvec(vals, x0)
        r_hat = r0  # shadow residual (fixed)

        rho_old = jnp.array(1.0)
        alpha   = jnp.array(1.0)
        omega   = jnp.array(1.0)

        v = jnp.zeros_like(b)
        p = jnp.zeros_like(b)

        # Store residual norm
        rs = jnp.dot(r0, r0)

        state = (0, x0, r0, r_hat, p, v, rho_old, alpha, omega, rs)

        def cond_fn(state):
            i, x, r, r_hat, p, v, rho_old, alpha, omega, rs = state
            return jnp.logical_and(i < iterations, jnp.sqrt(rs) > tol)

        def body_fn(state):
            i, x, r, r_hat, p, v, rho_old, alpha, omega, rs = state
        
            jax.debug.print("BiCGStab res {val}", val=jnp.sqrt(rs))

            rho_new = jnp.dot(r_hat, r)

            # Breakdown protection
            rho_new = jnp.where(rho_new == 0.0, 1e-30, rho_new)

            beta = (rho_new / rho_old) * (alpha / omega)

            # p_k = r + beta*(p - omega*v)
            p_new = r + beta * (p - omega * v)

            # Apply preconditioner
            y = precond(p_new)

            # v = A y
            v_new = sp_matvec(vals, y)

            alpha_new = rho_new / jnp.dot(r_hat, v_new + 1e-30)

            # s = r - alpha*v
            s = r - alpha_new * v_new

            # Early exit check
            s_norm = jnp.sqrt(jnp.dot(s, s))

            def update_from_s(state_s):
                # Apply preconditioner to s
                z = precond(s)
                t = sp_matvec(vals, z)

                omega_new = jnp.dot(t, s) / jnp.dot(t, t + 1e-30)

                x_new = x + alpha_new * y + omega_new * z
                r_new = s - omega_new * t

                rs_new = jnp.dot(r_new, r_new)

                return (i+1, x_new, r_new, r_hat, p_new, v_new,
                        rho_new, alpha_new, omega_new, rs_new)

            def update_skip(state_s):
                # If s is small enough, update x directly
                x_new = x + alpha_new * y
                r_new = s
                rs_new = s_norm**2
                return (i+1, x_new, r_new, r_hat, p_new, v_new,
                        rho_new, alpha_new, omega, rs_new)

            # If s is very small → skip omega step
            return jax.lax.cond(s_norm < 1e-14,
                            update_skip,
                            update_from_s,
                            operand=None)

        final_state = jax.lax.while_loop(cond_fn, body_fn, state)
        i, x_final, r_final, r_hat, p, v, rho_old, alpha, omega, rs_final = final_state

        jax.debug.print("BiCGStab final residual {r}", r=jnp.sqrt(rs_final))
        return x_final

    return solver

def make_sparse_damped_jacobi_solver(sp_matvec, inverse_diag_fct,
                                     iterations=10, tol=1e-5, omega=0.75):

    def solver(vals, b, x0):

        M_inv = inverse_diag_fct(vals)

        # Initial residual
        r0 = b - sp_matvec(vals, x0)
        rs0 = jnp.dot(r0, r0)   # plain L2 residual is fine for Jacobi

        # State: (iter, x, r, rs)
        initial_state = (0, x0, r0, rs0)

        def cond_fun(state):
            i, x, r, rs = state
            return jnp.logical_and(i < iterations, jnp.sqrt(rs) > tol)

        def body_fun(state):
            i, x, r, rs = state

            # Jacobi update: x_{k+1} = x_k + ω M^{-1} r_k
            x_new = x + omega * (M_inv * r)

            # New residual
            r_new = b - sp_matvec(vals, x_new)
            
            rs_new = jnp.dot(r_new, r_new)
            #jax.debug.print("Damped Jacobi res: {res}", res=jnp.sqrt(rs_new))

            return (i + 1, x_new, r_new, rs_new)

        i, x_final, r_final, rs_final = jax.lax.while_loop(
            cond_fun, body_fun, initial_state
        )

        jax.debug.print("Damped Jacobi final residual: {res}", res=jnp.sqrt(rs_final))
        return x_final

    return jax.jit(solver)

def sparse_cg_solver(sp_matvec, vals, b, x0, iterations=10):
    r = b - sp_matvec(vals, x0)
    d = r.copy()
    x = x0.copy()

    rs_old = jnp.dot(r, r)

    for i in range(iterations):
        Ad = sp_matvec(vals, d)

        alpha = rs_old / jnp.dot(Ad, d)
        
        x = x + alpha * d
        
        r = r - alpha * Ad
        rs_new = jnp.dot(r, r)

        beta = rs_new/rs_old

        d = r + beta * d

        rs_old = rs_new

    return x


