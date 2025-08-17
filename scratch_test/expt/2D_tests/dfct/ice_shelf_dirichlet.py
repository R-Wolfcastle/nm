#1st party
from pathlib import Path
import sys
import time


#local apps
sys.path.insert(1, "../../../utils/")
from sparsity_utils import scipy_coo_to_csr,\
                           basis_vectors_and_coords_2d_square_stencil,\
                           make_sparse_jacrev_fct_new,\
                           make_sparse_jacrev_fct_shared_basis
import constants as c
from plotting_stuff import show_vel_field


#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.set_printoptions(precision=1, suppress=False, linewidth=np.inf, threshold=np.inf)



def solve_petsc_sparse(values, coordinates, jac_shape,\
                       b, ksp_type='gmres', preconditioner='hypre',\
                       precondition_only=False):
    
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)

    #rows_local = int(jac_shape[0] / size)

    #A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), bsize=[rows_local, jac_shape], comm=comm)
    A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr.astype(np.int32), j.astype(np.int32), values), comm=comm)
    
    b = PETSc.Vec().createWithArray(b, comm=comm)
    
    x = b.duplicate()
    
    #set ksp iterations
    opts = PETSc.Options()
    opts['ksp_max_it'] = 20
    opts['ksp_monitor'] = None
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

        #pc.apply(b, x)
        #print((A*x - b).norm())
        #raise


    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()
    

    x_jnp = jnp.array(x.getArray())

    return x_jnp


def interp_cc_to_fc_function(ny, nx):

    def interp_cc_to_fc(var):
        
        var_ew = jnp.zeros((ny, nx+1))
        var_ew = var_ew.at[:, 1:-1].set(0.5*(var[:, 1:]+var[:, -1:]))
        var_ew = var_ew.at[:, 0].set(var[:, 0])
        var_ew = var_ew.at[:, -1].set(var[:, -1])

        var_ns = jnp.zeros((ny+1, nx))
        var_ns = var_ns.at[1:-1, :].set(0.5*(var[:-1, :]+var[1:, :]))
        var_ns = var_ns.at[0, :].set(var[0, :])
        var_ns = var_ns.at[-1, :].set(var[-1, :])

        return var_ew, var_ns

    return interp_cc_to_fc


def cc_vel_gradient_function(ny, nx, dy, dx):

    def cc_vel_gradient(var, bcs="dirichlet"):
        dvar_dx = jnp.zeros((ny, nx))
        dvar_dy = jnp.zeros((ny, nx))
        
        dvar_dx = dvar_dx.at[:, 1:-1].set((0.5/dx) * (var[:,2:] - var[:,:-2]))
        dvar_dx = dvar_dx.at[:, 0].set((0.5/dx) * (var[:, 1] + var[:, 0])) #using dirichlet bc
        dvar_dx = dvar_dx.at[:,-1].set((0.5/dx) * (var[:,-1] - var[:,-2])) #extrapolating u over cf
        
        dvar_dy = dvar_dy.at[1:-1, :].set((0.5/dy) * (var[:-2,:] - var[2:,:]))
        dvar_dy = dvar_dy.at[0, :].set((0.5/dy) * (-var[0, :] - var[1, :]))
        dvar_dy = dvar_dy.at[-1,:].set((0.5/dy) * (var[-2, :] + var[-1,:]))

        return dvar_dx, dvar_dy

    return cc_vel_gradient

def cc_s_gradient_function(ny, nx, dy, dx):

    def cc_s_gradient(var, bcs="neuman"):
        dvar_dx = jnp.zeros((ny, nx))
        dvar_dy = jnp.zeros((ny, nx))
        
        dvar_dx = dvar_dx.at[:, 1:-1].set((0.5/dx) * (var[:,2:] - var[:,:-2]))
        dvar_dx = dvar_dx.at[:, 0].set((0.5/dx) * (var[:, 1] - var[:, 0])) #using neuman bc
        dvar_dx = dvar_dx.at[:,-1].set((0.5/dx) * (-var[:,-2]))
        
        dvar_dy = dvar_dy.at[1:-1, :].set((0.5/dy) * (var[:-2,:] - var[2:,:]))
        dvar_dy = dvar_dy.at[0, :].set((0.5/dy) * (var[0, :] - var[1, :])) #using neuman bc
        dvar_dy = dvar_dy.at[-1,:].set((0.5/dy) * (var[-2,:] - var[-1,:])) #using neuman bc

        return dvar_dx, dvar_dy

    return cc_s_gradient


def fc_gradient_functions(ny, nx, dy, dx):

    def ew_face_gradient(var):
        dvar_dx_ew = jnp.zeros((ny, nx+1))
        dvar_dy_ew = jnp.zeros((ny, nx+1))


        dvar_dx_ew = dvar_dx_ew.at[:, 1:-1].set((var[:,1:]-var[:,:-1])/dx)
        dvar_dx_ew = dvar_dx_ew.at[:, 0].set( 2*var[:, 0]/dx)
        dvar_dx_ew = dvar_dx_ew.at[:,-1].set(-2*var[:,-1]/dx)

        #internals
        dvar_dy_ew = dvar_dy_ew.at[1:-1, 1:-1].set((var[:-2, 1:]  +\
                                                    var[:-2, :-1] -\
                                                    var[2:, :-1]  -\
                                                    var[2:, 1:]
                                                   )/(4*dy))
        #upper and lower boundaries
        dvar_dy_ew = dvar_dy_ew.at[0, 1:-1].set(  -(var[0, 1:]  +\
                                                    var[0, :-1] +\
                                                    var[1, 1:]  +\
                                                    var[1, :-1]
                                                   )/(4*dy))
        dvar_dy_ew = dvar_dy_ew.at[-1, 1:-1].set(  (var[-2, 1:]  +\
                                                    var[-2, :-1] +\
                                                    var[-1, 1:]  +\
                                                    var[-1, :-1]
                                                   )/(4*dy))
        #corner points
        dvar_dy_ew = dvar_dy_ew.at[0,  0].set(-2*var[0,  0]/(4*dy))
        dvar_dy_ew = dvar_dy_ew.at[0, -1].set(-2*var[0, -1]/(4*dy))
        dvar_dy_ew = dvar_dy_ew.at[-1, 0].set( 2*var[-1, 0]/(4*dy))
        dvar_dy_ew = dvar_dy_ew.at[-1,-1].set( 2*var[-1,-1]/(4*dy))
    
        #due to dirichlet bcs, dvar_dy_ew is 0 on left and right boundaries

        return dvar_dx_ew, dvar_dy_ew


    def ns_face_gradient(var):
        dvar_dx_ns = jnp.zeros((ny+1, nx))
        dvar_dy_ns = jnp.zeros((ny+1, nx))

        dvar_dy_ns = dvar_dy_ns.at[1:-1,:].set((var[:-1,:]-var[1:,:])/dy)
        dvar_dy_ns = dvar_dy_ns.at[0, :].set(-2*var[0, :]/dy)
        dvar_dy_ns = dvar_dy_ns.at[-1,:].set( 2*var[-1,:]/dy)

        #internals
        dvar_dx_ns = dvar_dx_ns.at[1:-1, 1:-1].set((var[:-1, 2:] +\
                                                    var[1:,  2:] -\
                                                    var[:-1,:-2] -\
                                                    var[1:, :-2]
                                                   )/(4*dx))
        #left and right boundaries
        dvar_dx_ns = dvar_dx_ns.at[1:-1, 0].set(   (var[:-1, 1]  +\
                                                    var[1:,  1]  +\
                                                    var[:-1, 0]  +\
                                                    var[1:,  0]
                                                   )/(4*dx))
        dvar_dx_ns = dvar_dx_ns.at[1:-1, -1].set( -(var[:-1, -1] +\
                                                    var[1:,  -1] +\
                                                    var[:-1, -2] +\
                                                    var[1:,  -2]
                                                   )/(4*dx))
        #dvar_dx_ns = dvar_dx_ns.at[1:-1, -1].set(  (var[:-1, -1] +\
        #                                            var[1:,  -1] -\
        #                                            var[:-1, -2] -\
        #                                            var[1:,  -2]
        #                                           )/(4*dx))
        #corner points
        dvar_dx_ns = dvar_dx_ns.at[0,  0].set(( 2*var[0,  0])/(4*dx))
        dvar_dx_ns = dvar_dx_ns.at[-1, 0].set(( 2*var[-1, 0])/(4*dx))
        #dvar_dx_ns = dvar_dx_ns.at[0, -1].set((-2*var[0, -1])/(4*dx))
        #dvar_dx_ns = dvar_dx_ns.at[-1,-1].set((-2*var[-1,-1])/(4*dx))
        ##dvar_dx_ns = dvar_dx_ns.at[0, -1].set((0)/(4*dx))
        ##dvar_dx_ns = dvar_dx_ns.at[-1,-1].set((0)/(4*dx)) #I'm not sure how it could be any different!
        dvar_dx_ns = dvar_dx_ns.at[0, -1].set((var[0,-2]-var[0,-1])/(4*dx))
        dvar_dx_ns = dvar_dx_ns.at[-1, -1].set(-(var[-1,-2]-var[-1,-1])/(4*dx))

        #due to dbcs, ddx_ns is 0 on upper and lower boundaries

        return dvar_dx_ns, dvar_dy_ns

    
    return ew_face_gradient, ns_face_gradient


def compute_u_v_residuals_function(ny, nx, dy, dx):

    interp_cc_to_fc = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient = fc_gradient_functions(ny, nx, dy, dx)
    cc_s_gradient = cc_s_gradient_function(ny, nx, dy, dx)


    def compute_u_v_residuals(u_1d, v_1d, h_1d, mu_bar):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)


        #volume_term
        dsdx, dsdy = cc_s_gradient(s)
        volume_x = -c.RHO_I * c.g * h * dsdx
        volume_y = -c.RHO_I * c.g * h * dsdy


        #momentum_term
        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        ##cc_derivatives, e.g. for viscosity calculation
        #dudx_cc, dudy_cc = cc_gradient(u)
        #dvdx_cc, dvdy_cc = cc_gradient(v)

        #interpolate things onto face-cenres
        mu_ew, mu_ns = interp_cc_to_fc(mu_bar)
        #the only reason for doing the above rather than having mu on cell centres is
        #that it makes DIVA a little easier when we come to it. Should only incur a
        #second-order error.
        h_ew, h_ns = interp_cc_to_fc(h)


        #to account for calving front boundary condition:
        #NOTE: this is after driving term has been calculated
        mu_ew = mu_ew.at[:, -1].set(0)

        
        visc_x = mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx


        visc_y = mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx


        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return compute_u_v_residuals



def qn_velocity_solver_function(ny, nx, dy, dx, mucoef, n_iterations):
    cc_vel_gradient = cc_vel_gradient_function(ny, nx, dy, dx)
    
    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx)


    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1)

    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(nr*nc), len(basis_vectors))
    mask = (i_coordinate_sets>=0)


    sparse_jacrev = make_sparse_jacrev_fct_shared_basis(
                                                        basis_vectors,\
                                                        i_coordinate_sets,\
                                                        j_coordinate_sets,\
                                                        mask,\
                                                        2,
                                                        active_indices=(0,1)
                                                          )


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

    
    def new_viscosity(u, v):
        #all these things are 2d and returns a 2d array
        
        dudx, dudy = cc_vel_gradient(u.reshape((ny, nx)))
        dvdx, dvdy = cc_vel_gradient(v.reshape((ny, nx)))

        return B * mucoef * (dudx**2 + dvdy**2 + dudx*dvdy +\
                    0.25*(dudy+dvdx)**2 + c.EPSILON_VISC)**(0.5*(1/c.GLEN_N - 1))


    def solver(u_trial, v_trial, h):
        u_1d = u_trial.copy().flatten()
        v_1d = v_trial.copy().flatten()
        h_1d = h.flatten()

        residual = jnp.inf

        for i in range(n_iterations):

            mu = new_viscosity(u_1d, v_1d)

            dJu_du, dJu_dv, dJv_du, dJv_dv = sparse_jacrev(get_u_v_residuals, (u_1d, v_1d, h_1d, mu))

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            jac = jnp.zeros((2*ny*nx, 2*ny*nx))
            jac = jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #t = 2*nr*nc
            #print(jac[:int(t/2), :int(t/2)])
            #print(jac[int(t/2+1):t :int(t/2)])
            #print(jac[:int(t/2), int(t/2+1):int(t)])
            #print(jac[int(t/2+1):int(t), int(t/2+1):int(t)])
            #raise
            #print(nz_jac_values.shape)
            #print(jac)
            #raise

            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, h_1d, mu))
      
            old_residual = residual
            residual = jnp.max(jnp.abs(-rhs))
            
            print(residual)
            print(old_residual/residual)

            du = solve_petsc_sparse(nz_jac_values,\
                                    coords,\
                                    (nr*nc*2, nr*nc*2),\
                                    rhs,\
                                    ksp_type="bcgs",\
                                    preconditioner="hypre",\
                                    precondition_only=False)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver












A = 5e-25
B = 2 * (A**(-1/3))

#epsilon_visc = 1e-5/(3.15e7)
epsilon_visc = 3e-13

lx = 100_000
ly = 150_000

#nr, nc = 64, 64
nr, nc = 90, 60


x = jnp.linspace(0, lx, nc)
y = jnp.linspace(0, ly, nr)

delta_x = x[1]-x[0]
delta_y = y[1]-y[0]


thk = jnp.zeros((nr, nc))+500

b = jnp.zeros_like(thk)-600

mucoef = jnp.ones_like(thk)

u_init = jnp.zeros_like(thk)
v_init = jnp.zeros_like(thk)

n_iterations = 10

solver = qn_velocity_solver_function(nr, nc, delta_y, delta_x, mucoef, n_iterations)

u_out, v_out = solver(u_init, v_init, thk)

show_vel_field(u_out*c.S_PER_YEAR, v_out*c.S_PER_YEAR)

plt.imshow(v_out*c.S_PER_YEAR)
plt.colorbar()
plt.show()

plt.imshow(u_out*c.S_PER_YEAR)
plt.colorbar()
plt.show()

plt.plot((u_out*c.S_PER_YEAR)[:,40])
plt.show()



