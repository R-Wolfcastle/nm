
#3rd party
import jax
import jax.numpy as jnp



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
        
        #If there are zeros in this, we're screwed anyway
        return 1/jnp.zeros(matrix_width).at[diag_indices].add(diag_vals)


    return sparse_matvec, A_inner_product, extract_inverse_diagonal


def sparse_dpcg_solver(sp_matvec, inverse_diag_fct, vals, b, x0, iterations=10):

    M_inv = inverse_diag_fct(vals)

    r = b - sp_matvec(vals, x0)
    x = x0.copy()

    d = M_inv * r

    rs_old = jnp.dot(r, M_inv * r)

    for i in range(iterations):
        #print(jnp.max(jnp.abs(r)))
        print(rs_old)

        Ad = sp_matvec(vals, d)

        alpha = rs_old / jnp.dot(Ad, d)
        
        x = x + alpha * d
        
        r = r - alpha * Ad
        
        rs_new = jnp.dot(r, M_inv * r)

        beta = rs_new/rs_old

        d = M_inv * r + beta * d

        rs_old = rs_new

    return x



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


