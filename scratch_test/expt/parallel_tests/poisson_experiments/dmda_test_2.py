
from petsc4py import PETSc
from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp


nr = 1028
nc = 1028


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


def residual(stencil_u, cp, fp):
    #give the stencil values as a vector so JAX can grad it

    up = stencil_u[0]
    ue = stencil_u[1]
    uw = stencil_u[2]
    un = stencil_u[3]
    us = stencil_u[4]

    #FV discretisation (happens to be same as FD)
    return (
        ue
        + uw
        + un
        + us
        - 4.0 * up
        - cp * up
        - fp
    )


batched_res_fct = jax.jit(jax.vmap(
                        residual,
                        in_axes=(0, 0, 0)
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
        in_axes=(0, 0, 0)
    )
)



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






#Create something to give JAX
ug = dm.getVecArray(u_global)
ug[:,:] = 1

#Populate local stuff with global gubbins. Work out halo cells and whatnot
dm.globalToLocal(u_global, u_local)

#Get a numpy array for the local patch
ul = dm.getVecArray(u_local)[:,:]






up = ul[1:-1, 1:-1]

ue = ul[1:-1, 2:]
uw = ul[1:-1, :-2]

un = ul[:-2, 1:-1]
us = ul[2:, 1:-1]





if x0 == 0:
    uw[:, 0] = -up[:, 0]

if x1 == nc:
    ue[:, -1] = -up[:, -1]

if y0 == 0:
    un[0, :] = -up[0, :]

if y1 == nr:
    us[-1, :] = -up[-1, :]






#create the (5, n) vector of values
local_stencils = jnp.stack(
    [up, ue, uw, un, us],
    axis=-1
)

#what I called "n" just above
n_local = local_stencils.shape[0] * local_stencils.shape[1]
#reshape it to (n, 5)
local_stencils = local_stencils.reshape((n_local, 5))




#setting these babies up!
local_cp = np.zeros(n_local)
local_fp = np.ones(n_local)





#vmap over the local stencil values, C and f coeffs, and cast it all
#as a numpy array for use with PETSc.
jac_values = np.asarray(
    batched_jac_fun(
        local_stencils,
        local_cp,
        local_fp
    )
)

#should look like this at each point:
# jac_values[k] =
#
# [dRp_dup,
#  dRp_due,
#  dRp_duw,
#  dRp_dun,
#  dRp_dus]






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

        boundary = (
            x == 0 or
            x == nc - 1 or
            y == 0 or
            y == nr - 1
        )

        if boundary:

            #Tell PETSc that the column index is the one associated with gridpoint (y,x)
            col = PETSc.Mat.Stencil()
            col.index = (y, x)

            #I mean, this is pretty hacky, to be honest. I don't want this dirichlet stuff.
            #But, this should just set everything to zero on the boundary
            A.setValueStencil(
                row,
                col,
                1.0
            )

        else:

            coeffs = jac_values[k]

            col = PETSc.Mat.Stencil()

            col.index = (y, x)
            A.setValueStencil(row, col, coeffs[0])

            col.index = (y, x + 1)
            A.setValueStencil(row, col, coeffs[1])

            col.index = (y, x - 1)
            A.setValueStencil(row, col, coeffs[2])

            col.index = (y - 1, x)
            A.setValueStencil(row, col, coeffs[3])

            col.index = (y + 1, x)
            A.setValueStencil(row, col, coeffs[4])

        k += 1

#Still not entirely sure what this means.
#Basically, sort it all out. Do your comms, build the sparse representation, etc.
A.assemble()




#create a PETSc global vector object for the forcing on the RHS
b = dm.createGlobalVec()

bg = dm.getVecArray(b)[:,:]



flat_residual = - np.asarray(batched_res_fct(
                    local_stencils,
                    local_cp,
                    local_fp
                            )
                  )
#Remember, Trys, these vectors are 2D!
bg[:,:] = flat_residual.reshape((y1-y0, x1-x0))




du = dm.createGlobalVec()

ksp = PETSc.KSP().create()

ksp.setOperators(A)

ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")

ksp.setUp()

ksp.solve(b, du)


u_sol = u_global.copy()

u_sol_vals = dm.getVecArray(u_sol)
u_sol_vals[:,:] += dm.getVecArray(du)[:,:]



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
        plt.imshow(U, origin='upper')
        plt.colorbar()
        plt.show()

        return U

plot_dmda_solution(dm, u_sol, nr, nc)






