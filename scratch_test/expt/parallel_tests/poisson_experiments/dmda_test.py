#1st party
from pathlib import Path
import sys
import time


#3rd party
from petsc4py import PETSc
from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=3, suppress=False, linewidth=np.inf, threshold=np.inf)



def make_stencil_residual_function(coarsest_delta,\
                                   stencil_case=1):
    #case 1: uniform stencil
    #case 2: coarse/fine interface to the right
    #case 3: fine/coarse interface to the right
    #etc
    match stencil_case:
        case 1:
            def stencil_residual(stencil_us, cp, fp):
                
                up = stencil_us[0]
                ue = stencil_us[1]
                uw = stencil_us[2]
                un = stencil_us[3]
                us = stencil_us[4]

                x_fluxes = ue - 2*up + uw
                y_fluxes = un - 2*up + us

                volume_term = - (cp * up + fp) * coarsest_delta**2

                return x_fluxes + y_fluxes + volume_term


        case _:
            raise ValueError(f"Unknown stencil case: {stencil_case}")


    return stencil_residual

def nonzero_jac_values_for_row(stencil_us, cp, fp,
                               stencil_residual_function):
    #returns tuple of (dRp/dup, dRp/due, dRp/duw, dRp/dun, dRp/dus)
    return jax.grad(stencil_residual_function, argnums=0)(stencil_us, cp, fp)

def nonzero_jac_values(local_stencils, C, f, stencil_residual_function):
    local_jac_fct = jax.grad(stencil_residual_function, argnums=0)

    mapped_local_jac_fct = jax.vmap(local_jac_fct(stencil_us, cp, fp),
                                    in_axes = (0, 0, 0))

    nz_jacobian_values = mapped_local_jac_fct(local_stencils, C, f)
                                        
    return nz_jacobian_values

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
    #C = 199
    
    
    #plt.imshow(C)
    #plt.colorbar()
    #plt.show()
    #raise
    
    f = 1
    #f = jnp.zeros((nr, nc))
    #f = f.at[int(nr/4):int(nr/2), int(nc/4):int(nc/2)].set(1)

    return C, f



def create_dmda_object(nr, nc):
    dm = PETSc.DMDA()
    dm.create(dim=2,
              dof=1,
              sizes=(nr, nc),
              proc_sizes=None, #Think this means PETSc can just choose?
              stencil_type=PETSc.DMDA.StencilType.STAR,
              stencil_width=1,
              boundary_type=(PETSc.DM.BoundaryType.GHOSTED,
                             PETSc.DM.BoundaryType.GHOSTED) #one for each dim
              ) 

    return dm



def residual(stencil_u, cp, fp):

    up = stencil_u[0]
    ue = stencil_u[1]
    uw = stencil_u[2]
    un = stencil_u[3]
    us = stencil_u[4]

    return (
        ue + uw + un + us
        - 4*up
        - cp*up
        - fp
    )


jac_fun = jax.grad(residual, argnums=0)



def global_index(y, x):
    return y*nc + x



nr = 128
nc = 128


dm = create_dmda_object(nr, nc)


A = dm.createMatrix()


rank = MPI.COMM_WORLD.rank

print(
    rank,
    dm.getRanges()
)

(y0, y1), (x0, x1) = dm.getRanges()

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
            x == nc-1 or
            y == 0 or
            y == nr-1
        )

        if boundary:

            #Tell PETSc that the column index is the one associated with gridpoint (y,x)
            col = PETSc.Mat.Stencil()
            col.index = (y, x)

            A.setValueStencil(row, col, 1.0)

        else:

            col = PETSc.Mat.Stencil()

            col.index = (y, x)
            A.setValueStencil(row, col, -4.0)

            col.index = (y, x+1)
            A.setValueStencil(row, col, 1.0)

            col.index = (y, x-1)
            A.setValueStencil(row, col, 1.0)

            col.index = (y-1, x)
            A.setValueStencil(row, col, 1.0)

            col.index = (y+1, x)
            A.setValueStencil(row, col, 1.0)

A.assemble()

b = dm.createGlobalVec()

bg = dm.getVecArray(b)

for y in range(y0, y1):
    for x in range(x0, x1):

        ly = y - y0
        lx = x - x0

        if (
            x == 0 or
            x == nc-1 or
            y == 0 or
            y == nr-1
        ):
            bg[ly, lx] = 0.0

        else:
            bg[ly, lx] = 1.0


u = dm.createGlobalVec()

ksp = PETSc.KSP().create()

ksp.setOperators(A)

ksp.setType("preonly")

pc = ksp.getPC()
pc.setType("lu")

ksp.setUp()

ksp.solve(b, u)



#
# Gather solution to rank 0 for inspection
#
#u_global = u.getArray()
#
#print(
#    f"rank {MPI.COMM_WORLD.rank} "
#    f"local min={u_global.min():.6e} "
#    f"local max={u_global.max():.6e}"
#)
#
#
#
#u_local = dm.createLocalVec()
#
#dm.globalToLocal(u, u_local)
#
#ul = dm.getVecArray(u_local)[:,:]
#
#print(f"rank {rank}")
#print(ul[1:-1,1:-1])
#
#raise



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

plot_dmda_solution(dm, u, nr, nc)


raise



#A = dm.createMatrix()

(y0, y1), (x0, x1) = dm.getRanges()

for y in range(y0, y1):
    for x in range(x0, x1):

        row = (y, x)

        boundary = (
            x == 0 or
            x == nc-1 or
            y == 0 or
            y == nr-1
        )

        if boundary:

            A.setValuesStencil(
                row,
                row,
                1.0
            )

        else:

            A.setValueStencil(row, (y, x),   -4.0)
            A.setValueStencil(row, (y, x-1),  1.0)
            A.setValueStencil(row, (y, x+1),  1.0)
            A.setValueStencil(row, (y-1, x),  1.0)
            A.setValueStencil(row, (y+1, x),  1.0)

A.assemble()

b = dm.createGlobalVec()

bg = dm.getVecArray(b)

for y in range(y0, y1):
    for x in range(x0, x1):

        if (
            x == 0 or
            x == nc-1 or
            y == 0 or
            y == nr-1
        ):
            bg[y-y0, x-x0] = 0.0
        else:
            bg[y-y0, x-x0] = 1.0


u = dm.createGlobalVec()

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setType("cg")
ksp.getPC().setType("jacobi")

ksp.solve(b, u)


ulocal = dm.createLocalVec()
dm.globalToLocal(u, ulocal)

u_np = dm.getVecArray(ulocal)[:,:]

print("rank =", MPI.COMM_WORLD.rank)
print(u_np)











raise



















nr = int(2**5)
nc = int(2**5)
dy = 1/nr
dx = 1/nc
C, f = create_domain(nr, nc, dy, dx)


dm = create_dmda_object(nr, nc)

u = dm.createGlobalVec()
u_local = dm.createLocalVec()

rank = MPI.COMM_WORLD.rank

ug = dm.getVecArray(u)

for i in range(ug.shape[0]):
    for j in range(ug.shape[1]):
        ug[i,j] = 100*i + j

dm.globalToLocal(u, u_local)

ul = dm.getVecArray(u_local)[:,:]

##I have no fucking idea how the indexing goes when using ul itself. It's absolutely barmy.
##But you seem to be able to get out a version of the array as a numpy array that makes some sense
##if you do:
#ul_np = ul[:,:]


print(f"RANK: {rank}")



up = ul[1:-1,1:-1]

uw = ul[1:-1,0:-2]
ue = ul[1:-1,2:]

un = ul[0:-2,1:-1]
us = ul[2:,1:-1]

print(up.shape)
print(uw.shape)
print(ue.shape)
print(un.shape)
print(us.shape)


raise




print(type(ul_np))
print(ul_np)
print(ul_np.shape)

print(ul_np[0,0])
print(ul_np[1,1])

raise
print(ul[:,:])

print(type(ul))
print(ul.shape)

print(ul[0,0])
print(ul[1,1])

print(type(ul[1:3,1:3]))

print(ul[1:3,1:3])

raise


# neighbours
ue = ul[1:-1,2:]
uw = ul[1:-1,:-2]

un = ul[:-2,1:-1]
us = ul[2:,1:-1]

(y_first, y_last), (x_first, x_last) = dm.getRanges()

# homogeneous Dirichlet
if x_first == 0:
    uw[:,0] = -up[:,0]

if x_last == nc:
    ue[:,-1] = -up[:,-1]

if y_first == 0:
    un[0,:] = -up[0,:]

if y_last == nr:
    us[-1,:] = -up[-1,:]

local_stencils = np.stack(
    [up, ue, uw, un, us],
    axis=-1
)

print(rank)
print(local_stencils.shape)
print(local_stencils[0,0,:])





























raise

dm = create_dmda_object(nr, nc)

#Create vector compatible with the DMDA object, which stores all the
#mesh dimensions, ownership ranges, ghost cell stuff, stencil info, MPI gubbins, ...
u = dm.createGlobalVec()

#guess like a temporary working array with cells owned by this rank, and ghost cells.
#for use in the actual calculations.
u_local = dm.createLocalVec()


#You access the "data" in the arrays by using dm.getVecArray. These are just views.
#The section of the global vector owned by each rank
ug = dm.getVecArray(u)
#Basically ghosted version of ug as far as I can tell...
ul = dm.getVecArray(u_local)


##https://petsc.org/release/petsc4py/reference/petsc4py.PETSc.Vec.html#petsc4py.PETSc.Vec
#print(type(ug))
#print(ug.shape)
#print(ug.sizes)
#
#print(type(ul))
#print(ul.shape)
#print(ug.shape)
#print(ul.sizes)


rank = MPI.COMM_WORLD.rank

#You have to slice it, as that's operating on the view of the global vector u.
#If you did ug = (rank+1)**2, it would reassign ug as an int. It's basic stuff, but it's
#taking a long time for me to get my head around it...
ug[:,:] = (rank+1)**2

#Basically says to take the field u and fill u_local with the owned values
#plus whatever neighbour values are needed in the ghost cells.
#I guess it must do some message passing between ranks to get it sorted.
dm.globalToLocal(u, u_local)
#It throws away whatever is currentlu in u_local, and fills it with the stuff in u.

#ul = dm.getVecArray(u_local)

#print(f"rank {rank}")
#print(ul[:,:])



def apply_dirichlet_ghost_cells(dm, ul):
    # Bounds for the local part of the grid this process owns
    (y_first, y_last), (x_first, x_last) = dm.getRanges()

    # top boundary
    if y_last == nr:
        ul[0, :] = -ul[1 ,:]
    ## bottom boundary
    #if y_first == 0:
    #    ul[-1,:] = -ul[-2,:]
    ## right boundary
    #if x_last == nc:
    #    ul[:,-1] = -ul[:,-2]
    ## left boundary
    #if x_first == 0:
    #    ul[:, 0] = -ul[:, 1]



#dm.globalToLocal(u, u_local)


#It seems that this returns a vewi where the indexing corresponds to the owned region
#and the halo data is living in extra storage somewhere you can't reach with numpy indexing!
#Soooooo weird!
ul = dm.getVecArray(u_local)
#E.g. print(ul[33,:]) gives: IndexError: index 34 is out of bounds for axis 0 with size 34


print("before")
print(ul[:3,:])

tmp = np.array(ul[1,:], copy=True)

ul[0,:] = -tmp

print("after")
print(ul[:3,:])


raise

apply_dirichlet_ghost_cells(dm, ul)

print(f"rank {rank}")
print(ul[:,:])




raise

print(ug.shape)
print(ug)
print(ug.size)
raise

dm.globalToLocal(u, u_local)
ul = dm.getVecArray(u_local)

print("global shape =", ug.shape)
print("local shape  =", ul.shape)


#plt.imshow(C)
#plt.colorbar()
#plt.show()

raise


u_init = jnp.ones((nr, nc)).reshape(-1)

#1 should be enough as the problem is linear but seemingly benefits from another
n_iterations = 1
#solver = make_newton_solver(C, f, n_iterations)
solver = make_newton_solver_sparse_jac(C, f, n_iterations)

t0 = time.time()
u_final = solver(u_init)
t1 = time.time()
print("Solver time with nr={}: {}s".format(nr, t1-t0))

plt.imshow(u_final.reshape((nr,nc)), cmap="gnuplot2", vmin=0)
plt.colorbar()
plt.show()

#plt.figure(figsize=(10, 4))
#plt.plot(u_final.reshape((nr,nc))[1250,:])
#plt.show()
raise
