"""
NOTE: This docstring was made by MS Copilot

Geometric multigrid (GMG) solver for 2D cell-centred, two-field systems stored
in local stencil form.

The intended use case is the solution of linear systems arising from the
linearisation of a residual function

    R(u, v) = 0

where each residual depends only on a 3x3 neighbourhood of cells. Rather than
forming a sparse matrix explicitly, the Jacobian is stored as a stencil

    A.shape = (2, 2, 3, 3, ny, nx)

whose dimensions correspond to

    (residual component,
     degree of freedom component,
     stencil y-offset,
     stencil x-offset,
     cell y,
     cell x)

i.e. the four Jacobian blocks

    [[dRu/du, dRu/dv],
     [dRv/du, dRv/dv]]

are stored as 3x3 stencils at every cell centre.

The file provides:

- Matrix-free stencil operations (matvec, transpose, restriction of inactive
  cells, etc.).
- Construction of coarse-grid operators using the Galerkin product

      Ac = R Af P

  without ever assembling RAP explicitly.
- Cell-centred geometric restriction and prolongation operators.
- Four-colour (red/black/blue/pink) block Gauss-Seidel and block Jacobi
  smoothers.
- Recursive V-cycles.
- A pure multigrid solver.
- A GMRES solver preconditioned by multigrid.
- Utilities for extracting stencil Jacobians from linear maps using coloured
  probing.

Multigrid hierarchy
-------------------

Unknowns are cell-centred and the grid is repeatedly coarsened by factors of
two. A coarse cell represents a 2x2 block of fine cells.

Restriction:
    Residuals are restricted by summing over 2x2 children.

Prolongation:
    Corrections are prolongated using mask-aware bilinear interpolation between
    coarse- and fine-cell centres. The usual 9:3:3:1 bilinear weights are
    renormalised near inactive cells and domain boundaries.

Inactive cells:
    Cells with mask == 0 are treated as homogeneous Dirichlet cells (u = v = 0)
    by removing all couplings and inserting a 2x2 identity block on the
    diagonal.

Coarsest level:
    The coarsest operator is converted to a dense matrix and solved directly
    using LU factorisation.

Implementation notes
--------------------

This code is designed primarily for use under jax.jit. Many small Python loops
(e.g. over stencil offsets or colours) are intentionally written in a form that
XLA will unroll during compilation. The resulting compiled kernels are typically
much more efficient than treating these operations as dynamic loops.

The multigrid hierarchy is stored as a plain dictionary of JAX arrays, allowing
it to be passed freely through jitted functions, lax loops and recursive
V-cycles.
"""


import math

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl


nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'utils'))
from stencil_matrix_operations import *

jax.config.update("jax_enable_x64", True)




###############################################################################
#Important Grid-transfer gubbins!!
###############################################################################

subpositions = ((0, 0), (0, 1), (1, 0), (1, 1))


def shift(a, oy, ox):
    """
    (oy, ox) = (offset in y, offset in x)
    a[..., i+oy, j+ox], zero outside the domain."""
    ny, nx = a.shape[-2:]
    pad = [(0,0), (0,0), (0,0), (0,0), (1,1), (1,1)]
    ap = jnp.pad(a, pad)
    return ap[..., (1+oy):(1+oy+ny), (1+ox):(1+ox+nx)]


def restrict(r):
    """
    Restrict the residual of a linear equation r:
    Simple sum. Note, no divide by 4 - no need to do that when restricting the residual.
    IF the error equation is RAP e = R r, then a factor of 1/4 on both sides makes no difference.
    Each coarse cell contains 4 fine cells. reshape so the 4 children are in dimensions 2
    and 4 and then sum over them.
    """
    _, nyf, nxf = r.shape
    return r.reshape(2, nyf // 2, 2, nxf // 2, 2).sum(axis=(2, 4))


def coarsen_mask(m):
    """
    Same as restriction of residual, but if _any_ children are ice-filled, then the
    coarsened cell is considered ice-filled.
    """
    nyf, nxf = m.shape
    return m.reshape(nyf // 2, 2, nxf // 2, 2).max(axis=(1, 3))


def prolongation_weights(mc, mf):
    """
    weights for bilinear interpolation (https://en.wikipedia.org/wiki/Bilinear_interpolation)
    of fine cell centres from surrounding coarse cell-centres.
    9 for closest coarse cc, 3 for equidistant cccs and 1 for furthers ccc. Divided by 16.
    In this case, however, not dividing by 16 all the time - dividing by the sum of the weights,
    as all weights that are associated with ice-free cells are set to zero!
    
    mc: coarse mask
    mf: fine mask
    Cells outside the domain are treated as having mask value zero through the use of shift(),
    which pads with zeros.
    
    Weights have shape (4 subpositions, 4 neighbours, nyc, nxc). They correspond to the following:
    for each coarse cell centrei P, there are 4 fine cell centres 0.25 away from it. For each of these
    subpositions, what are the weights associated with the neighbour coarse cell centres to P.

    For each of the four fine-cell centres created when a coarse cell is refined (NW, NE, SW, SE), 
    the weights describe how that fine-cell value is reconstructed from the surrounding
    coarse-cell-centre values.
    """
    W = []
    for si, sj in subpositions:
        #offsets of coarse ccs given subposition of the fine cc
        oy = -1 if si == 0 else 1
        ox = -1 if sj == 0 else 1
        #weight corresponding to: P - the closest to the fine point 
        #                         N or S, depending on the subposition
        #                         W or E, depending on the subposition
        #                         The corner neighbour cc, depending on the subposition
        w = jnp.stack([9.0 * mc,
                       3.0 * shift(mc, oy, 0),
                       3.0 * shift(mc, 0, ox),
                       1.0 * shift(mc, oy, ox)])
        s = w.sum(axis=0)
        w = w / jnp.where(s > 0, s, 1.0)
        W.append(w * mf[si::2, sj::2][None])
    return jnp.stack(W)


def prolong(e, W):
    """
    Prolong error e (shpae [2, nyc, nxc]) to shape [2, 2*nyc, 2*nxc] with precomputed
    set of weights W given by prolongation_weights defined above.
    
    Weights have shape (4 subpositions, 4 neighbours, nyc, nxc). They correspond to the following:
    for each coarse cell centrei P, there are 4 fine cell centres 0.25 away from it. For each of these
    subpositions, what are the weights associated with the neighbour coarse cell centres to P.
    """
    _, nyc, nxc = e.shape
    out = jnp.zeros((2, 2 * nyc, 2 * nxc), e.dtype)

    #loop over the subpositions of the fine cells surrounding a certain coarse cell centre
    #subpositions are indexed by: nw, ne, sw, se
    for k, (si, sj) in enumerate(subpositions):
        oy = -1 if si == 0 else 1
        ox = -1 if sj == 0 else 1
        #bilinear interpolation (if the weights are right!)
        val = (W[k, 0] * e + W[k, 1] * shift(e, oy, 0)
               + W[k, 2] * shift(e, 0, ox) + W[k, 3] * shift(e, oy, ox))
        #do all nw's at the same time across the domain etc.
        #si,sj = (0,0): NW children
        #si,sj = (0,1): NE children
        #si,sj = (1,0): SW children
        #si,sj = (1,1): SE children
        out = out.at[:, si::2, sj::2].set(val)
    return out


def galerkin_coarse_stencil(Af, W, nyc, nxc, probe_batch_size=2):
    """
    Af: fine-grid operator
    W: weights of the bilinear interpolation that forms the prolongation funciton

    Produces a stencil version of the coarse-grid operator by Galerkin magic!
    Uses the 18 coloured probes again.

    #Defines the Galerkin coarse operator
    #Ac v = R Af P v
    #without ever assembling explicit sparse matrices.
    """
    #imagining v is an error on coarse grid, of shape [2, nyc, nxc]
    lin = lambda v: restrict(stencil_matvec(Af, prolong(v, W)))
    return stencil_from_linear_map(lin, nyc, nxc, probe_batch_size)


###############################################################################
#Smoothers and coarse solve
###############################################################################

def block_diag_inverse(A):
    """
    Inverse of the block matrix [[dRx/du, dRx/dv],[dRy/du, dRy/dv]] where everything is at
    cell centres.
    """
    a, b = A[0, 0, 1, 1], A[0, 1, 1, 1]
    c, d = A[1, 0, 1, 1], A[1, 1, 1, 1]
    det = a * d - b * c
    #inverse of determinant (where it existst)
    idet = jnp.where(det != 0, 1.0 / jnp.where(det != 0, det, 1.0), 0.0)
    return jnp.stack([jnp.stack([d * idet, -b * idet]),
                      jnp.stack([-c * idet, a * idet])])


def apply_block(Dinv, r):
    """
    Does what it says on the tin. Apply the inverse of the inverse of the 2x2 block matrix
    to a cell-centred residual.
    """
    return jnp.stack([Dinv[0, 0] * r[0] + Dinv[0, 1] * r[1],
                      Dinv[1, 0] * r[0] + Dinv[1, 1] * r[1]])


def four_colours(ny, nx, dtype=jnp.float64):
    """Masks of the four (i mod 2, j mod 2) colours that separate dependencies of cell-centred
    residuals. USed to order the Gauss-Seidel method.
    output is list of nyxny binary arrays
    """
    i, j = cell_index(ny, nx)
    return [((i % 2 == si) & (j % 2 == sj)).astype(dtype) for si, sj in subpositions]


def dense_index(ny, nx):
    """
    For an nyxnx grid and 2 fields of dofs, and a 3x3 stencil, what are the ij indices
    of the nonzero entries of a big sparse matrix that map onto the stencil-level
    representations in utils/stencil_matrix_operations
    """
    a, b, di, dj, i, j = np.meshgrid(jnp.arange(2), jnp.arange(2),  jnp.arange(3),
                                     jnp.arange(3), jnp.arange(ny), jnp.arange(nx),
                                     indexing="ij")
    ni, nj = i + di - 1, j + dj - 1
    valid = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
    rows = a * ny * nx + i * nx + j
    cols = b * ny * nx + jnp.clip(ni, 0, ny - 1) * nx + jnp.clip(nj, 0, nx - 1)
    return rows.ravel(), cols.ravel(), valid.ravel().astype(np.float64)


###############################################################################
#The GMG solver factory
###############################################################################

def transpose_for_adjoint(A, mask):
    """Stencil for the adjoint solve: eliminate inactive cells first, then transpose."""
    return stencil_transpose(eliminate_inactive(A, mask))


def make_stencil_gmg_solver(ny, nx, max_coarse_cells=1024, nu1=2, nu2=2,
                            smoother="gs4", jacobi_omega=0.7, restart=20, max_restarts=30,
                            galerkin_batch_size=2):
    """
    Build the multigrid functions for one grid size (all shapes are fixed here).

        gmg_setup, gmg_solve, gmg_solve_mg = make_stencil_gmg_solver(ny, nx)

        H = gmg_setup(A, mask)                    # once per linearisation
        x, its, rel_res = gmg_solve(H, b, rtol)   # b, x flat, length 2*ny*nx, [u; v]
        x, cycles, rel_res = gmg_solve_mg(H, b, fixed_cycles=8)   # pure multigrid

    For the adjoint, set up from transpose_for_adjoint(A, mask) instead of A.
    H is a plain dict of arrays, so it can be passed through jit and loops freely.
    """
    
    if smoother not in ("gs4", "jacobi"):
        raise ValueError("smoother must be 'gs4' or 'jacobi'")


    #Caluclate the number of levels: halve until the coarsest grid is small enough.
    #Defined by max_course_cells. Default is 1024 (e.g. 32x32). By default do a direct bottom solve
    #using the JAX version of Scipy's version of the LAPAK LU solve.
    n_levels = 1
    while True:
        f = 2 ** (n_levels - 1)
        cells = math.ceil(ny / f) * math.ceil(nx / f)
        if cells <= max_coarse_cells or min(math.ceil(ny / (2 * f)), math.ceil(nx / (2 * f))) < 2:
            break
        n_levels += 1

    #number of cells in target fine grid
    f = 2 ** (n_levels - 1)

    #pad the existing grid so it is a multiple of 2^(n_levels - 1) so we can coarsen to there
    nyp, nxp = math.ceil(ny / f) * f, math.ceil(nx / f) * f
    shapes = [(nyp // 2 ** l, nxp // 2 ** l) for l in range(n_levels)]
    
    #Dense indices for the coarsest grid to hand to thedirect bottom solver
    rows, cols, valid = dense_index(*shapes[-1])
    n_coarse = 2 * shapes[-1][0] * shapes[-1][1]
    shape = (2, nyp, nxp)

    #function for padding a variable x (operator or vector) so that it has the size compatible
    #with coarsening. Pads only the spatial dimensions.
    def pad(x):
        return jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, nyp - ny), (0, nxp - nx)])

    def unpad(x):
        return x[..., :ny, :nx]

    #SETUP!

    #Create a dictionary for the multigrid solver!
    #This is done on each Newton iterate - A is the Jacobian of either the linear or nonlinear
    #residual equation, and it is fixed for the internal linear solve by GMG or GMG-GMRES.
    #Looks like:
    #
    # H = {
    #     "levels": [
    #         {
    #             "A":    fine-grid stencil operator
    #                     shape (2,2,3,3,ny_l,nx_l),
    #
    #             "Dinv": inverse of the 2x2 point-block diagonal of A
    #                     shape (2,2,ny_l,nx_l),
    #
    #             "W":    prolongation weights from level l+1 -> l
    #                     shape (4,4,ny_{l+1},nx_{l+1}),
    #
    #             "m":    active-cell mask on level l
    #                     shape (ny_l,nx_l)
    #         },
    #         ...
    #     ],
    #
    #     "coarse_lu": LU factorisation fucntion for the bottom solve,
    #
    #     "m0":        finest-grid mask,
    #
    #     "A0":        finest-grid stencil operator
    # }
    #
    # The hierarchy stores levels 0...(n_levels-2); the final coarsest
    # level is represented by the dense LU factorisation in "coarse_lu".
    def gmg_setup(A, mask):
        m = pad(mask.astype(A.dtype))
        A = eliminate_inactive(pad(A), m)

        levels = []
        for l in range(n_levels - 1):
            nyc, nxc = shapes[l + 1]
            mc = coarsen_mask(m)
            W = prolongation_weights(mc, m)
            levels.append(dict(A=A, Dinv=block_diag_inverse(A), W=W, m=m))
            #probes evaluated at once when forming R A P are set by galerkin_batch_size;
            #each holds several fine-grid fields, so it sets setup's peak memory
            A = eliminate_inactive(galerkin_coarse_stencil(A, W, nyc, nxc, galerkin_batch_size), mc)
            m = mc
        M = jnp.zeros((n_coarse, n_coarse), A.dtype).at[rows, cols].add(A.ravel() * valid)
        return dict(levels=levels, coarse_lu=jsl.lu_factor(M),
                    m0=levels[0]["m"] if levels else m,
                    A0=levels[0]["A"] if levels else A)


    #SMOOTHER!
    #at the moment only Jacobi or Gauss-Seidel (4-colour, i.e. red/black/blue/pink).
    def smooth(lev, colours, x, b, sweeps, reverse):
        A, Dinv = lev["A"], lev["Dinv"]
        #Dinv: block diagonal inverse of A
        for _ in range(sweeps):
            if smoother == "gs4":
                #Gauss-Seidel red/black (/blue/pink)
                for cm in (colours[::-1] if reverse else colours):
                    #add correction multiplied by colour masks each time
                    x = x + cm * apply_block(Dinv, b - stencil_matvec(A, x))
            else:
                #Jacobi iteration! Update all cells simultaneously
                #https://en.wikipedia.org/wiki/Jacobi_method.
                x = x + jacobi_omega * apply_block(Dinv, b - stencil_matvec(A, x))
        return x


    #V-CYCLE!
    def vcycle(H, l, b):
        """
        H: the dictionary defined in gmg_setup.
        l: what level we're on: 0 finest, n_levels-1 coarsest.
        b: rhs of linear equation - typically a restricted version of the residual one level up.

        Function is defined recursively.
        Provides an update for the solution to the linear equation (not the solution itself).
        So it provides a solution to the error equation.
        """
        #If finest level, do a dense solve with lu decomposition.
        #This routine seems to work pretty well on the GPU
        if l == n_levels - 1:
            return jsl.lu_solve(H["coarse_lu"], b.reshape(-1)).reshape(b.shape)

        lev = H["levels"][l]
        #recalculate the colour masks for level
        colours = four_colours(*shapes[l], b.dtype)
        #Do a smooth
        x = smooth(lev, colours, jnp.zeros_like(b), b, nu1, False)
        #calculate residual
        r = b - stencil_matvec(lev["A"], x)
        #recursive update of x by additional v-cycle
        #https://en.wikipedia.org/wiki/Multigrid_method
        x = x + prolong(vcycle(H, l + 1, restrict(r)), lev["W"])

        #Smooth again
        return smooth(lev, colours, x, b, nu2, True)


    #GMG METHOD IN FULL!
    def gmg_solve_mg(H, b_flat, rtol=1e-8, max_cycles=50, fixed_cycles=None):
        """
        This applies pure geometric multigrid as a linear solver.
        There is a version below written by an LLM that uses this GMG as a preconditioner
        to GMRES.

        GMG: x = x + V(b - Ax) where V is the v-cycle function. Initial guess is x=0

        Options for fixed number of v-cycles vs iterating until convergence:

        fixed_cycles=None: iterate until ||b - A x|| <= rtol*||b|| or max_cycles
                           (a data-dependent while_loop).
        fixed_cycles=k:    exactly k V-cycles in a fori_loop, with no stopping test,
                           so nothing is read back to the host.
        Returns (x, cycles, rel_res).
        """
        
        #Put b on the same grid as A0 and mask it by ice-filled cells, in case that's not
        #done in the Newotn loop.
        b = pad(b_flat.reshape(2, ny, nx)) * H["m0"]
        A0 = H["A0"]
        bnorm = jnp.linalg.norm(b)
        safe = jnp.where(bnorm > 0, bnorm, 1.0)

        #Function for appling a v-cycle
        def cycle(x):
            return x + vcycle(H, 0, b - stencil_matvec(A0, x))

        x0 = jnp.zeros_like(b)
        #Either a lax fori or a lax while loop depending on whether fixed number of v-cycles
        #or a condition-based assessment of convergence.
        if fixed_cycles is not None:
            x = jax.lax.fori_loop(0, fixed_cycles, lambda i, x: cycle(x), x0)
            k = fixed_cycles
        else:
            def cond(s):
                _, k, res = s
                return (res > rtol * bnorm) & (k < max_cycles)

            def body(s):
                x, k, _ = s
                x = cycle(x)
                return x, k + 1, jnp.linalg.norm(b - stencil_matvec(A0, x))

            x, k, _ = jax.lax.while_loop(cond, body, (x0, 0, bnorm))

        #Relative norm of residual compared to start
        rel = jnp.linalg.norm(b - stencil_matvec(A0, x)) / safe
        return unpad(x).reshape(-1), k, rel
   

    #GMG-PRECONDITIONED GMRES!
    def gmg_solve(H, b_flat, rtol=1e-8, atol=0.0):
        b = pad(b_flat.reshape(2, ny, nx)) * H["m0"]
        A0 = H["A0"]
        matvec = lambda x: stencil_matvec(A0, x.reshape(shape)).reshape(-1)
        psolve = lambda r: vcycle(H, 0, r.reshape(shape)).reshape(-1)
        x, its, rel = gmres(matvec, psolve, b.reshape(-1), rtol, atol, restart, max_restarts)
        return unpad(x.reshape(shape)).reshape(-1), its, rel

    return gmg_setup, gmg_solve, gmg_solve_mg



###############################################################################
# GMRES (restarted, right preconditioned) - NOTE: THis is written by Claude!
###############################################################################

def gmres(matvec, psolve, b, rtol, atol, restart, max_restarts):
    """Solve A x = b; `psolve` must be a fixed linear operator. Returns (x, its, rel_res)."""
    m = restart
    dtype = b.dtype
    bnorm = jnp.linalg.norm(b)
    target = jnp.maximum(rtol * bnorm, atol)

    def cycle(x):
        r = b - matvec(x)
        beta = jnp.linalg.norm(r)
        V = jnp.zeros((m + 1, b.shape[0]), dtype).at[0].set(r / jnp.where(beta > 0, beta, 1.0))
        H = jnp.zeros((m + 1, m), dtype)
        cs = jnp.zeros((m,), dtype)
        sn = jnp.zeros((m,), dtype)
        g = jnp.zeros((m + 1,), dtype).at[0].set(beta)

        def cond(s):
            k, _, _, _, _, g, stop = s
            return (k < m) & (~stop) & (jnp.abs(g[k]) > target)

        def body(s):
            k, V, H, cs, sn, g, _ = s
            w = matvec(psolve(V[k]))
            active = (jnp.arange(m + 1) <= k).astype(dtype)
            h = (V @ w) * active                    # classical Gram-Schmidt, twice
            w = w - V.T @ h
            h2 = (V @ w) * active
            w = w - V.T @ h2
            h = h + h2
            hn = jnp.linalg.norm(w)
            V = V.at[k + 1].set(w / jnp.where(hn > 0, hn, 1.0))
            h = h.at[k + 1].set(hn)

            def rotate(i, h):
                hi, hj = h[i], h[i + 1]
                rotated = h.at[i].set(cs[i] * hi + sn[i] * hj).at[i + 1].set(-sn[i] * hi + cs[i] * hj)
                return jnp.where(i < k, rotated, h)

            h = jax.lax.fori_loop(0, m, rotate, h)
            hk, hk1 = h[k], h[k + 1]
            t = jnp.hypot(hk, hk1)
            ts = jnp.where(t > 0, t, 1.0)
            c = jnp.where(t > 0, hk / ts, 1.0)
            s_ = jnp.where(t > 0, hk1 / ts, 0.0)
            h = h.at[k].set(c * hk + s_ * hk1).at[k + 1].set(0.0)
            cs = cs.at[k].set(c)
            sn = sn.at[k].set(s_)
            gk = g[k]
            g = g.at[k].set(c * gk).at[k + 1].set(-s_ * gk)
            H = H.at[:, k].set(h)
            return (k + 1, V, H, cs, sn, g, hn == 0)

        k, V, H, cs, sn, g, _ = jax.lax.while_loop(
            cond, body, (0, V, H, cs, sn, g, jnp.array(False)))

        act = jnp.arange(m) < k
        R = jnp.where(act[:, None] & act[None, :], H[:m, :m], jnp.eye(m, dtype=dtype))
        y = jsl.solve_triangular(R, jnp.where(act, g[:m], 0.0), lower=False)
        return x + psolve(V[:m].T @ y), jnp.abs(g[k]), k

    def outer_cond(s):
        _, res, _, n = s
        return (res > target) & (n < max_restarts)

    def outer_body(s):
        x, _, its, n = s
        x, res, k = cycle(x)
        return x, res, its + k, n + 1

    x, res, its, _ = jax.lax.while_loop(
        outer_cond, outer_body, (jnp.zeros_like(b), bnorm, 0, 0))
    return x, its, res / jnp.where(bnorm > 0, bnorm, 1.0)


