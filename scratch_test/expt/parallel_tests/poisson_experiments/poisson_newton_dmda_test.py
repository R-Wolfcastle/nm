from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


np.set_printoptions(precision=3, suppress=False, linewidth=np.inf, threshold=np.inf)


def spherical_wave(nr, nc, amplitude=1, frequency=10):
    y = jnp.linspace(0, 1, nr)
    x = jnp.linspace(0, 1, nc)
    yy, xx = jnp.meshgrid(y, x, indexing='ij')

    cy, cx = (0.5, 0.5)
    r = jnp.sqrt((yy - cy)**2 + (xx - cx)**2)

    wave = amplitude * (1 + jnp.sin(2 * jnp.pi * frequency * r))

    return wave


def create_domain(nr, nc, dy, dx):
    C = spherical_wave(nr, nc, frequency=10, amplitude=500)
    
    #plt.imshow(C)
    #plt.colorbar()
    #plt.show()
    
    F = jnp.ones((nr, nc))*2000
    return C, F

nr = int(2**7)
nc = int(2**7)
dy = 1/nr
dx = 1/nc

C, F = create_domain(nr, nc, dy, dx)

def create_dmda_object(nr, nc):

    dm = PETSc.DMDA().create(
        dim=2,
        dof=1, #DOFs per node
        sizes=(nr, nc),
        stencil_width=1,
        stencil_type=PETSc.DMDA.StencilType.STAR,
        #Still not sure exactly how to make use of this...
        boundary_type=(
            PETSc.DM.BoundaryType.GHOSTED,
            PETSc.DM.BoundaryType.GHOSTED
        )
    )

    return dm


def residual(stencil_u,
             cp,
             fp,
             west_bc,
             east_bc,
             north_bc,
             south_bc):

    up = stencil_u[0]
    ue = stencil_u[1]
    uw = stencil_u[2]
    un = stencil_u[3]
    us = stencil_u[4]

    # Ghost-cell Dirichlet BCs
    uw = jnp.where(west_bc,  -up, uw)
    ue = jnp.where(east_bc,  -up, ue)
    un = jnp.where(north_bc, -up, un)
    us = jnp.where(south_bc, -up, us)


    x_fluxes = (ue - 2*up + uw)
    y_fluxes = (un - 2*up + us)
    volume_term = - (cp*up + fp)*dx*dy

    return 1 * (x_fluxes + y_fluxes ) + volume_term



batched_res_fct = jax.jit(jax.vmap(
                        residual,
                        in_axes=(0, 0, 0, 0, 0, 0, 0)
                          )
                  )

#Jacobian function! will produce:
#(dRp/dup, dRp/due, dRp/duw, dRp/dun, dRp/dus)
jac_fun = jax.grad(residual, argnums=0)

#Batch this baby, so it vmaps over the first dimension of stencil_us, C and f.
#They're all flattened (well, stencil_us is shape (N, 5)
batched_jac_fun = jax.jit(
    jax.vmap(
        jac_fun,
        in_axes=(0, 0, 0, 0, 0, 0, 0)
    )
)


def assemble_jacobian(matrix, y0, y1, x0, x1, jacobian_values):
    #To keep track of which jac_values we're looking at.
    k = 0
    #I guess you can't get away from having to loop over the entire mesh, but the hard vmap part
    #of things is already done, so it's just populating something that's the issue now.
    for y in range(y0, y1):
        for x in range(x0, x1):
    
            
            #Mat.Stencil() doesn't seem to really mean anything about the stencil,
            #it just seems to be the matrix row corresponding to the physical grid cell (y, x)
            row = PETSc.Mat.Stencil()
            
            row.index = (y, x)
            #I think you're just telling PETSc that you're assembling the particular row
            #associated with point (y,x).
   
            coeffs = jacobian_values[k]

            col = PETSc.Mat.Stencil()
            
            col.index = (y, x)
            matrix.setValueStencil(row, col, coeffs[0])
            
            col.index = (y, x + 1)
            matrix.setValueStencil(row, col, coeffs[1])
            
            col.index = (y, x - 1)
            matrix.setValueStencil(row, col, coeffs[2])
            
            col.index = (y - 1, x)
            matrix.setValueStencil(row, col, coeffs[3])
            
            col.index = (y + 1, x)
            matrix.setValueStencil(row, col, coeffs[4])
            
            k += 1

def create_direct_solver(A):

    ksp = PETSc.KSP().create()
    
    ksp.setOperators(A)
    
    ksp.setType("preonly")
    
    pc = ksp.getPC()
    pc.setType("lu")
    
    ksp.setUp()

    return ksp, None


def create_ksp(A,
               ksp_type="gmres",
               ksp_max_iter=20,
               monitor_ksp=False,
               preconditioner='hypre'):
    
    opts = PETSc.Options()
    opts['ksp_max_it'] = ksp_max_iter

    opts['ksp_norm_type'] = "unpreconditioned"

    if monitor_ksp:
        opts['ksp_monitor'] = None
        opts['ksp_converged_reason'] = None
        opts['ksp_view'] = None
    
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
    

    ksp.setUp()

    return ksp





max_iters = 4


#CREATE DMDA OBJECT THING
dm = create_dmda_object(nr, nc)

#create "global" and "local" vectors. These are PETSc objects whose construction
#is informed by dm. One is the part of the actual global data owned by this rank,
#the local one has got halo cells and things, I think.
u_global = dm.createGlobalVec()
u_local  = dm.createLocalVec()

#Create the matrix! It should do some pre-populating and stuff, I think.
A = dm.createMatrix()

rank = MPI.COMM_WORLD.rank

#The ranges in x and y of the part of the mesh that this rank owns.
(y0, y1), (x0, x1) = dm.getRanges()

west_bc  = np.zeros((y1-y0, x1-x0), dtype=bool)
east_bc  = np.zeros((y1-y0, x1-x0), dtype=bool)
north_bc = np.zeros((y1-y0, x1-x0), dtype=bool)
south_bc = np.zeros((y1-y0, x1-x0), dtype=bool)

if x0 == 0:
    west_bc[:,0] = True
if x1 == nc:
    east_bc[:,-1] = True
if y0 == 0:
    north_bc[0,:] = True
if y1 == nr:
    south_bc[-1,:] = True
   


#create a PETSc global vector object for the RHS
b = dm.createGlobalVec()

#Create one for update
du = dm.createGlobalVec()



#Create the C and f fields in the right shape.
#God it's so long-winded!!
c = dm.createGlobalVec()
c_local = dm.createLocalVec()
c_array = dm.getVecArray(c)[:,:]
c_array[:,:] = C[y0:y1, x0:x1]

f = dm.createGlobalVec()
f_local = dm.createLocalVec()
f_array = dm.getVecArray(f)[:,:]
f_array[:,:] = F[y0:y1, x0:x1]

dm.globalToLocal(c, c_local)
dm.globalToLocal(f, f_local)

cp = dm.getVecArray(c_local)[:,:][1:-1, 1:-1]
fp = dm.getVecArray(f_local)[:,:][1:-1, 1:-1]



#initial guess
u = dm.getVecArray(u_global)
u[:,:] = 0.0

for newton_iter in range(max_iters):

    #halo exchange stuff
    dm.globalToLocal(u_global, u_local)



    #print(f"RANK: {rank}")
    #print("global patch:")
    #print(dm.getVecArray(u_global)[:,:])
    #print("local patch:")
    #print(dm.getVecArray(u_local)[:,:])



    ### Create local stencil points

    ul = dm.getVecArray(u_local)[:,:]

    up = ul[1:-1, 1:-1]
    
    ue = ul[1:-1, 2:]
    uw = ul[1:-1, :-2]
    
    un = ul[:-2, 1:-1]
    us = ul[2:, 1:-1]
   
    
    #create the (5, n) vector of values
    local_stencils = jnp.stack(
        [up, ue, uw, un, us],
        axis=-1
    )
    
    #what I called "n" just above
    n_local = local_stencils.shape[0] * local_stencils.shape[1]
    #reshape it to (n, 5)
    local_stencils = local_stencils.reshape((n_local, 5))
   
    west_bc  = west_bc.reshape(n_local)
    east_bc  = east_bc.reshape(n_local)
    north_bc = north_bc.reshape(n_local)
    south_bc = south_bc.reshape(n_local)


    ### Jacobian coefficients
    jac_values = np.asarray(batched_jac_fun(
                        local_stencils,
                        cp.reshape(n_local),
                        fp.reshape(n_local),
                        west_bc,
                        east_bc,
                        north_bc,
                        south_bc
                            )
                  )

    #print(jac_values.reshape((nr, nc, 5))[2,:,:])

    ### build RHS
    bg = dm.getVecArray(b)[:,:]
    
    flat_residual = - np.asarray(batched_res_fct(
                        local_stencils,
                        cp.reshape(n_local),
                        fp.reshape(n_local),
                        west_bc,
                        east_bc,
                        north_bc,
                        south_bc
                                )
                      )

    print(f"RANK: {rank}")
    print(np.max(np.abs(flat_residual)))
    
    #Remember, Trys, these vectors are 2D!
    bg[:,:] = flat_residual.reshape((y1-y0, x1-x0))
   

    # assemble Jacobian
    A.zeroEntries()

    assemble_jacobian(
        A,
        y0, y1, x0, x1,
        jac_values,
    )

    A.assemble()

    ### solve J du = -R
    ksp = create_ksp(A)
    du.set(0.0)
    ksp.solve(b, du)

    ### update
    u_global.axpy(1.0, du)



    ### convergence
    du_norm = du.norm()
    #print(du_norm)

    if du_norm < 1e-4:
        break


#Thanks MS Copilot
def plot_dmda_solution(dm, u, nr, nc):

    rank = MPI.COMM_WORLD.rank
    comm = MPI.COMM_WORLD

    #
    # local owned portion (not ghosted)
    #
    ulocal = dm.getVecArray(u)[:,:]

    (y0, y1), (x0, x1) = dm.getRanges()

    payload = (
        y0,
        y1,
        x0,
        x1,
        np.array(ulocal, copy=True)
    )

    gathered = comm.gather(payload, root=0)

    if rank == 0:

        U = np.zeros((nr, nc))

        for y0, y1, x0, x1, block in gathered:

            U[y0:y1, x0:x1] = block

        plt.figure()
        plt.imshow(U, origin='upper', cmap="gnuplot2")
        plt.colorbar()
        plt.show()

        return U


#def plot_dmda_solution_scattered(dm, u, nr, nc):
#
#    rank = MPI.COMM_WORLD.rank
#    comm = MPI.COMM_WORLD
#
#    #
#    # local owned portion (not ghosted)
#    #
#    ulocal = dm.getVecArray(u)[:,:]
#
#    (y0, y1), (x0, x1) = dm.getRanges()
#
#    payload = (
#        y0,
#        y1,
#        x0,
#        x1,
#        np.array(ulocal, copy=True)
#    )
#
#    gathered = comm.gather(payload, root=0)
#
#    if rank == 0:
#
#        U = np.zeros((nr, nc))
#
#        plot, subplots = plt.figure(figsize=, subplots=)
#
#        for y0, y1, x0, x1, block in gathered:
#
#            U[y0:y1, x0:x1] = block
#
#            plt.figure()
#            plt.imshow(U, origin='upper', cmap="gnuplot2")
#            plt.colorbar()
#            plt.show()
#
#        return U
def plot_dmda_solution_scattered(dm, u, nr, nc, pad=2):

    rank = MPI.COMM_WORLD.rank
    comm = MPI.COMM_WORLD

    ulocal = dm.getVecArray(u)[:, :]

    (y0, y1), (x0, x1) = dm.getRanges()

    payload = (
        y0,
        y1,
        x0,
        x1,
        np.array(ulocal, copy=True)
    )

    gathered = comm.gather(payload, root=0)

    if rank == 0:

        px, py = dm.getProcSizes()

        # Determine unique processor-grid boundaries
        xs = sorted(set(x0 for y0, y1, x0, x1, block in gathered))
        ys = sorted(set(y0 for y0, y1, x0, x1, block in gathered))

        blocks = [[None for _ in range(px)] for _ in range(py)]

        for y0, y1, x0, x1, block in gathered:
            i = xs.index(x0)
            j = ys.index(y0)
            blocks[j][i] = block

        row_heights = [
            max(blocks[j][i].shape[0] for i in range(px))
            for j in range(py)
        ]

        col_widths = [
            max(blocks[j][i].shape[1] for j in range(py))
            for i in range(px)
        ]

        H = sum(row_heights) + pad * (py - 1)
        W = sum(col_widths) + pad * (px - 1)

        canvas = np.full((H, W), np.nan)

        yoff = 0
        for j in range(py):

            xoff = 0
            for i in range(px):

                block = blocks[j][i]
                h, w = block.shape

                canvas[yoff:yoff+h, xoff:xoff+w] = block

                xoff += col_widths[i] + pad

            yoff += row_heights[j] + pad

        plt.figure(figsize=(8, 8))
        plt.imshow(canvas, origin="upper", cmap="gnuplot2")
        plt.colorbar()
        plt.axis("off")
        plt.show()

        return canvas

def plot_dmda_solution_subplots(dm, u):

    rank = MPI.COMM_WORLD.rank
    comm = MPI.COMM_WORLD

    ulocal = dm.getVecArray(u)[:, :]

    (y0, y1), (x0, x1) = dm.getRanges()

    payload = (
        y0,
        y1,
        x0,
        x1,
        np.array(ulocal, copy=True)
    )

    gathered = comm.gather(payload, root=0)

    if rank != 0:
        return None

    px, py = dm.getProcSizes()

    xs = sorted(set(x0 for y0, y1, x0, x1, block in gathered))
    ys = sorted(set(y0 for y0, y1, x0, x1, block in gathered))

    blocks = [[None for _ in range(px)] for _ in range(py)]
    metadata = [[None for _ in range(px)] for _ in range(py)]

    for y0, y1, x0, x1, block in gathered:

        i = xs.index(x0)
        j = ys.index(y0)

        blocks[j][i] = block
        metadata[j][i] = (x0, x1, y0, y1)

    # Shared colour scale across all subdomains
    vmin = min(
        np.nanmin(block)
        for row in blocks
        for block in row
    )

    vmax = max(
        np.nanmax(block)
        for row in blocks
        for block in row
    )

    fig, axes = plt.subplots(
        py,
        px,
        figsize=(4 * px, 4 * py),
        squeeze=False,
        constrained_layout=True
    )

    for j in range(py):
        for i in range(px):

            ax = axes[j, i]

            block = blocks[j][i]
            x0, x1, y0, y1 = metadata[j][i]

            im = ax.imshow(
                block,
                origin="upper",
                cmap="gnuplot2",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(
                f"x:[{x0},{x1})  y:[{y0},{y1})"
            )

            ax.set_xlabel("local x")
            ax.set_ylabel("local y")

    ## One shared colourbar
    #fig.colorbar(
    #    im,
    #    ax=axes.ravel().tolist(),
    #    shrink=0.9,
    #    pad=0.02
    #)
    fig.colorbar(
        im,
        ax=axes,
        location="right",
        shrink=0.9
    )

    #plt.tight_layout()
    plt.show()

    return blocks

def plot_dmda_solution_subplots(dm, u):

    rank = MPI.COMM_WORLD.rank
    comm = MPI.COMM_WORLD

    ulocal = dm.getVecArray(u)[:, :]

    (y0, y1), (x0, x1) = dm.getRanges()

    payload = (
        y0,
        y1,
        x0,
        x1,
        np.array(ulocal, copy=True)
    )

    gathered = comm.gather(payload, root=0)

    if rank != 0:
        return None

    # Determine processor-grid structure from the actual decomposition
    xs = sorted(set(x0 for y0, y1, x0, x1, block in gathered))
    ys = sorted(set(y0 for y0, y1, x0, x1, block in gathered))

    px = len(xs)
    py = len(ys)

    blocks = [[None for _ in range(px)] for _ in range(py)]
    metadata = [[None for _ in range(px)] for _ in range(py)]

    for y0, y1, x0, x1, block in gathered:

        i = xs.index(x0)
        j = ys.index(y0)

        blocks[j][i] = block
        metadata[j][i] = (x0, x1, y0, y1)

    # Shared colour scale
    vmin = min(
        np.nanmin(block)
        for row in blocks
        for block in row
        if block is not None
    )

    vmax = max(
        np.nanmax(block)
        for row in blocks
        for block in row
        if block is not None
    )

    fig, axes = plt.subplots(
        py,
        px,
        figsize=(4 * px, 4 * py),
        squeeze=False,
        constrained_layout=True
    )

    im = None

    for j in range(py):
        for i in range(px):

            ax = axes[j, i]

            block = blocks[j][i]

            if block is None:
                ax.axis("off")
                continue

            x0, x1, y0, y1 = metadata[j][i]

            im = ax.imshow(
                block,
                origin="upper",
                cmap="gnuplot2",
                vmin=vmin,
                vmax=vmax
            )

            ax.set_title(
                f"x:[{x0},{x1})  y:[{y0},{y1})"
            )

            ax.set_xlabel("local x")
            ax.set_ylabel("local y")

    if im is not None:
        fig.colorbar(
            im,
            ax=axes,
            location="right",
            shrink=0.9
        )

    plt.show()

    return blocks

#plot_dmda_solution(dm, u_global, nr, nc)
#plot_dmda_solution_scattered(dm, u_global, nr, nc)
plot_dmda_solution_subplots(dm, u_global)

#############gunk:



#def residual_wo_boundary_stuff(stencil_u, cp, fp):
#    #give the stencil values as a vector so JAX can grad it
#
#    up = stencil_u[0]
#    ue = stencil_u[1]
#    uw = stencil_u[2]
#    un = stencil_u[3]
#    us = stencil_u[4]
#
#    #FV discretisation (happens to be same as FD)
#    return (
#        ue
#        + uw
#        + un
#        + us
#        - 4.0 * up
#        - cp * up
#        - fp
#    )
#
#
#def residual_simple(stencil_u,
#             cp,
#             fp,
#             west_bc,
#             east_bc,
#             north_bc,
#             south_bc):
#
#    up = stencil_u[0]
#    ue = stencil_u[1]
#    uw = stencil_u[2]
#    un = stencil_u[3]
#    us = stencil_u[4]
#
#    # Ghost-cell Dirichlet BCs
#    uw = jnp.where(west_bc,  -up, uw)
#    ue = jnp.where(east_bc,  -up, ue)
#    un = jnp.where(north_bc, -up, un)
#    us = jnp.where(south_bc, -up, us)
#
#    return (
#        ue
#        + uw
#        + un
#        + us
#        - 4.0*up
#        - cp*up
#        - fp
#    )



#def assemble_jacobian_dfct(matrix, y0, y1, x0, x1, jacobian_values):
#    #To keep track of which jac_values we're looking at.
#    k = 0
#    #I guess you can't get away from having to loop over the entire mesh, but the hard vmap part
#    #of things is already done, so it's just populating something that's the issue now.
#    for y in range(y0, y1):
#        for x in range(x0, x1):
#    
#            
#            #Mat.Stencil() doesn't seem to really mean anything about the stencil,
#            #it just seems to be the matrix row corresponding to the physical grid cell (y, x)
#            row = PETSc.Mat.Stencil()
#            
#            row.index = (y, x)
#            #I think you're just telling PETSc that you're assembling the particular row
#            #associated with point (y,x).
#    
#            boundary = (
#                x == 0 or
#                x == nc - 1 or
#                y == 0 or
#                y == nr - 1
#            )
#    
#            if boundary:
#    
#                #Tell PETSc that the column index is the one associated with gridpoint (y,x)
#                col = PETSc.Mat.Stencil()
#                col.index = (y, x)
#    
#                #I mean, this is pretty hacky, to be honest. I don't want this dirichlet stuff.
#                #But, this should just set everything to zero on the boundary
#                matrix.setValueStencil(
#                    row,
#                    col,
#                    1.0
#                )
#    
#            else:
#    
#                coeffs = jacobian_values[k]
#    
#                col = PETSc.Mat.Stencil()
#    
#                col.index = (y, x)
#                matrix.setValueStencil(row, col, coeffs[0])
#    
#                col.index = (y, x + 1)
#                matrix.setValueStencil(row, col, coeffs[1])
#    
#                col.index = (y, x - 1)
#                matrix.setValueStencil(row, col, coeffs[2])
#    
#                col.index = (y - 1, x)
#                matrix.setValueStencil(row, col, coeffs[3])
#    
#                col.index = (y + 1, x)
#                matrix.setValueStencil(row, col, coeffs[4])
#    
#                k += 1

