"""
A load of things for working with operators/matrices in stencil form.
Moving away from sparse-matrix stuff the CPU code did.

Operator storage: 
        A[a, b, di, dj, i, j]
is basically the coefficient coupling equation a (0: x-momentum, 1: y-momentum) at cell
(i, j) to unknown dof  b (0: u, 1: v) at cell (i + di - 1, j + dj - 1). Entries
that point outside the domain are set to zero. 
BCs are already folded into the in-domain coefficients by AD through the ghost-cell stuff.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def stencil_matvec(A, x):
    """
    y = A x if A is shape (2, 2, 3, 3, ny, nx) and x is shape (2, ny, nx).
    
    Not to be used outside of jit as those for-loops get unrolled. That goes for lots of stuff
    written in this file.
    """
    #This kind of thing is confusing to me. Remember that the jacobian is a map and
    #the stencil is the derivative of the residual wrt all the neighbours, so each
    #element should just be multiplied by the x in the corresponding location and then
    #the sum just taken over the stencil. The result is the total perturbation to the 
    #residual at that cell.

    ny, nx = x.shape[-2:]
    #pad in the y and x directions:
    xp = jnp.pad(x, ((0, 0), (1, 1), (1, 1)))
    y = jnp.zeros_like(x)
    for di in range(3):
        for dj in range(3):
            #The parts of x that produce non-zero additions to the sum.
            #e.g. if di and dj were 0, then this would be the northwest enighbour
            #of each cell. Or if (di, dj) were (1, 2) it would be the central-east.
            xs = xp[:, di:(di+ny), dj:(dj+nx)]

            #in the stencil, multiply the northwest component of the jacobian with
            #all the northwest parts of x if you know what I mean. Add at each cell.
            y = y + A[:, 0, di, dj] * xs[0] + A[:, 1, di, dj] * xs[1]
    return y


def stencil_transpose(A):
    """
    Given a matrix A defined in the stencil way, what is the equivalent transpose?
    Stencil of A^T: AT[a,b,di,dj,i,j] = A[b,a,2-di,2-dj,i+di-1,j+dj-1].
    """
    #The stencil shoudl get reflected through its centre. Each point is sensitivity
    #of cell-centred residual to pertubation in neighbour cells. 
    #So the opposite should say what perturbation corresponds to what sensitivity. So
    #the dirctional dependency should be reversed.

    ny, nx = A.shape[-2:]
    #only pad the final 2 dims
    Ap = jnp.pad(A, ((0,0),(0,0),(0,0),(0,0),(1,1),(1,1)))


    #example. di=0, dj=1 is the north cell centre. 2-di, 2-dj = 2,1 i.e. south centre.
    #Think about all the north coefficients A[..., 0, 1, :, :]. Those are the sensitiviteis
    #of the cell centred residuals to the north values. 
    #Maybe think about a single one... A[..., 0, 1, 20, 40]. Sensitivity of (20, 40) cell centred
    #x or y res to north u or v. The transposed version should be sensitivity of pertn (19, 40) to
    #a change in the residual to the south.

    #Bloody hell. How to do this...
    #Think A[..., di, dj, i, j] means sensitivity of (i,j) to (i+di-1,j+dj-1).
    #So AT should referse that.

    rows = []
    for di in range(3):
        row = []
        for dj in range(3):
            #Reflecting the stencil through the centre and swapping the 
            blk = Ap[:, :, 2-di, 2-dj, di:(di+ny), dj:(dj+nx)]

            #The first two dimensions should just swap places and add that
            row.append(jnp.swapaxes(blk, 0, 1))
        rows.append(jnp.stack(row, axis=2))
    return jnp.stack(rows, axis=2)


#NOTE on constants: nothing here precomputes per-cell arrays with NumPy. A NumPy
#array used inside a jitted function is embedded in the compiled program as a
#constant, and XLA then constant-folds operations on it into further constants.
#At large grid sizes that bloats the GPU binary by gigabytes (CUDA_ERROR_OUT_OF_MEMORY
#when it loads). Instead, cell indices come from lax.broadcasted_iota, which XLA
#generates on the fly, and anything depending on a probe number goes through an
#optimization_barrier so it can't be folded either.

def cell_index(ny, nx):
    """(i, j) index arrays, generated on the device. Pretty neat.
    e.g. cell_index(2,3) = (
                            [[0,0,0],
                             [1,1,1]],

                            [[0,1,2]
                             [0,1,2]]
                           )
    corresponding to (I, J)
    I hope...
    """
    return (jax.lax.broadcasted_iota(jnp.int32, (ny, nx), 0),
            jax.lax.broadcasted_iota(jnp.int32, (ny, nx), 1))


def colour(i, j):
    """Colour of the cell centre in a radius-3 stencil.
    Every 3x3 neighbourhood holds each colour once.
    [[0,1,2],
     [3,4,5],
     [6,7,8]]
    """
    return (i % 3) * 3 + (j % 3)


def stencil_from_linear_map(lin, ny, nx, probe_batch_size=None, dtype=jnp.float64):
    """
    Read the 9-point 2x2-block stencil ((dRudu, dRudv),(dRvdu, dRvdv)) of a linear map
    called lin: (2,ny,nx)->(2,ny,nx)
    (e.g. linearised versions of the residuals functions)
    There are 18 different coloured probes (if the map really has a 3x3 stencil).

    Sometimes used the notation a and b to mean: a: the number of outputs of the residuals function and
    b: the number of different input fields. So the Jacobian is axb block structured. Should really be the same...
    E.g. the momentum equation has a=2, b=2 for the x, y parts of the equation and the u, v degrees of freedom.

    It's quite memory intensive, so the probe_batch_size limits the number of basis
    vectors that can be probed at any one time suggested by the chatbot. If it's set to None, then it'll do 18.

    Output is the Jacobian, with nz values stored at each cell centre. Has shape (2,2,3,3,ny,nx) meaning, for each grid cell
    2 by 2 block ((dRudu, dRudv),(dRvdu, dRvdv)) and, for each, a 3 by 3 block of residual at cell wrt all neighbouring dofs.
    """
    #the different numbers that correspond to the colours in the coloured domain.
    #these will be used by probe_response to construct, for each number, the basis
    #vector with ones at the locations of those numbers.
    #optimization_barrier stops constant folding at comp time. The size of the graph
    #was blowing up, so putting that here ensures it has to be instantiated and various
    #calculations downstream have to be done and runtime, rather than constant values saved.
    ks = jax.lax.optimization_barrier(jnp.arange(18, dtype=jnp.int32))


    #This contains a load of recalculation of stuff that looks like it might be best to calculate
    #before and store. However, it's quite a lot more memory efficient to do it this way.
    #I think Steph mentioned recalculating rather than storing as a philosophy and it seems
    #to be the right way to go about things... Having said that, I might just create i, j right
    #at the start and have that be a big, all pervasive state variable.
    def probe_response(k):
        i, j = cell_index(ny, nx)
        #m is one where colour and zero elsewhere. I've called these "basis vectors" in the
        #petsc-cpu set up.
        m = (colour(i, j) == k % 9).astype(dtype)
        #b is 0 if k<9, 1 if 9<k<18, etc. 
        b = k // 9

        #jnp.stack([m * (b == 0), m * (b == 1)]) basically makes an array of shape [2, ny, nx]
        #with numbers 0,1 representing the basis vectors, but split into [u,v].
        #k-0,...,8 represent u degrees of freedom and 9,...,17 v degrees of freedom
        return lin(jnp.stack([m * (b == 0), m * (b == 1)]))

    #map probe_response over the different numners [0...17] to generate jvp for each basis vector
    #comes out with shape [n_colours*2, a, nx, ny] = [18, 2, nx, ny]
    T = jax.lax.map(probe_response, ks, batch_size=probe_batch_size or 18)
    
    #shape: [b, n_colours, a, ny, nx]
    T = T.reshape(2, 9, 2, ny, nx)
    #shape: [a, b, n_colours, ny, nx] = [2, 2, 9, ny, nx] - 
    #looking more like the right thing for the Jacobian now!
    T = jnp.transpose(T, (2, 0, 1, 3, 4))


    #Final thing to do is make this [a, b, 3, 3, ny, nx]
    #slot (di, dj) of cell (i, j) lives in the probe for the colour of its
    #neighbour (i+di-1, j+dj-1) and slots pointing outside the domain are zero
    i, j = cell_index(ny, nx)
    slots = []
    for di in range(3):
        for dj in range(3):
            ni, nj = i + di - 1, j + dj - 1
            #+3 just there to keep it non-negative
            nb_colour = colour(ni + 3, nj + 3)
            inside = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
            val = jnp.take_along_axis(T, jnp.broadcast_to(nb_colour, (2, 2, 1, ny, nx)), axis=2)[:, :, 0]
            slots.append(jnp.where(inside, val, 0.0))
    return jnp.stack(slots, axis=2).reshape(2, 2, 3, 3, ny, nx)


def stencil_from_residual(residual_uv, u_1d, v_1d, ny, nx, probe_batch_size=None):
    """
    Jacobian stencil of residual_uv(u_1d, v_1d) -> (Ru_1d, Rv_1d) at (u_1d, v_1d),
    via one linearisation and 18 vmapped JVPs. Basically an aaplication of stencil_from_linear_map
    using linearised residual_uv as the linear map.
    """
    _, f_lin = jax.linearize(residual_uv, u_1d, v_1d)

    def lin(x):
        ru, rv = f_lin(x[0].reshape(-1), x[1].reshape(-1))
        return jnp.stack([ru.reshape(ny, nx), rv.reshape(ny, nx)])

    #Returned stencil has shape [2, 2, 3, 3, ny, nx] corresponding to 
    #[n_residual_components, n_dof_types, stencil_radius, stencil_radiue, ny, nx]
    return stencil_from_linear_map(lin, ny, nx, probe_batch_size)


def eliminate_inactive(A, mask):
    """
    Treat cells with mask==0 as Dirichlet zeros: identity rows, no couplings into them.
    In the sparse-matrix version of things, this is dealt with just by setting beta to 1
    in ice-free cells, however, it is necessary to do this as the grid is padded up to a
    multuple of 2^(n_levels - 1) so that it can be repeatedly halved. Most importantly, the
    coarse operator RAP is built from the fine operator, not from the residual, so beta=1 in
    the residual function doesn't carry through.
    """
    ny, nx = mask.shape
    m = mask.astype(A.dtype)
    #pad the mask with a ring of zeros
    mp = jnp.pad(m, 1)
    #create 3x3 stencil of masks with shape [3,3,ny,nx]. "neighbour activity"
    #So each location is one or zero depending on whether there is ice there or not. 
    #E.g. [0,0,ny,nx]=1 if there is ice in the northwest neighbour or 0 if there isn't.
    nb = jnp.stack([jnp.stack([mp[di:(di+ny), dj:(dj+nx)] for dj in range(3)])
                    for di in range(3)])
    
    #(nb*m_[None, None]: zero out all ice-free neighbour stencils (and expand so has same
    #number of dims as A.
    #zero out all stencil coefficients corresponding to connections to ice-free cells
    A = A * ((nb * m)[None, None])

    #Add 1s in stencil centres in ice-free cells on the DIAGONAL of the 2x2 jacobian
    #so no dependence of Rx on v or Ry on u.
    return A.at[:, :, 1, 1].add(jnp.eye(2, dtype=A.dtype)[:, :, None, None] * (1 - m))


def check_stencil(residual_uv, u_1d, v_1d, A, ny, nx, key=None):
    """Relative mismatch between the stencil matvec and a JVP of the residual (debug)."""
    key = jax.random.PRNGKey(0) if key is None else key
    x = jax.random.normal(key, (2, ny, nx))
    _, jv = jax.jvp(residual_uv, (u_1d, v_1d), (x[0].reshape(-1), x[1].reshape(-1)))
    ref = jnp.stack([jv[0].reshape(ny, nx), jv[1].reshape(ny, nx)])
    return jnp.linalg.norm(stencil_matvec(A, x) - ref) / jnp.linalg.norm(ref)


