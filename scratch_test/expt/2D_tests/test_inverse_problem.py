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

np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)


def solve_petsc_sparse(values, coordinates, jac_shape,\
                       b, ksp_type='gmres', preconditioner='hypre',\
                       precondition_only=False, monitor_ksp=False):
    
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
    if monitor_ksp:
        opts['ksp_monitor'] = None
    opts['ksp_rtol'] = 1e-20
    
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


#NOTE: finish thinking about this!
def create_la_solver(coordinates, jac_shape,\
                     ksp_type='gmres', preconditioner='hypre',\
                     precondition_only=False, monitor_ksp=False):
    

    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    def solve_petsc_sparse(values, b):

        iptr, j, values = scipy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)



        #rows_local = int(jac_shape[0] / size)

        #A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), bsize=[rows_local, jac_shape], comm=comm)
        A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr.astype(np.int32), j.astype(np.int32), values), comm=comm)
    
        b = PETSc.Vec().createWithArray(b, comm=comm)
    
        x = b.duplicate()
    
    #set ksp iterations
    opts = PETSc.Options()
    opts['ksp_max_it'] = 20
    if monitor_ksp:
        opts['ksp_monitor'] = None
    opts['ksp_rtol'] = 1e-20
    
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
        var_ew = var_ew.at[:, 1:-1].set(0.5*(var[:, 1:]+var[:, :-1]))
        var_ew = var_ew.at[:, 0].set(var[:, 0])
        var_ew = var_ew.at[:, -1].set(var[:, -1])

        var_ns = jnp.zeros((ny+1, nx))
        var_ns = var_ns.at[1:-1, :].set(0.5*(var[:-1, :]+var[1:, :]))
        var_ns = var_ns.at[0, :].set(var[0, :])
        var_ns = var_ns.at[-1, :].set(var[-1, :])

        return var_ew, var_ns

    return interp_cc_to_fc


def cc_gradient_function(dy, dx):

    def cc_gradient(var):

        dvar_dx = (0.5/dx) * (var[1:-1, 2:] - var[1:-1,:-2])
        dvar_dy = (0.5/dy) * (var[:-2,1:-1] - var[2:, 1:-1])

        return dvar_dx, dvar_dy

    return cc_gradient


def fc_gradient_functions(dy, dx):

    def ew_face_gradient(var):
        
        dvar_dx_ew = (var[1:-1, 1:] - var[1:-1, :-1])/dx

        dvar_dy_ew = (var[:-2, 1:] + var[:-2, :-1] - var[2:, 1:] - var[2:, :-1])/(4*dy)
        
        return dvar_dx_ew, dvar_dy_ew
    
    def ns_face_gradient(var):
        
        dvar_dy_ns = (var[:-1, 1:-1]-var[1:, 1:-1])/dy

        dvar_dx_ns = (var[:-1, 2:] + var[1:, 2:] - var[:-1, :-2] - var[1:, :-2])/(4*dx)
        
        return dvar_dx_ns, dvar_dy_ns
    
    return ew_face_gradient, ns_face_gradient


def add_ghost_cells_fcts(ny, nx):

    def add_reflection_ghost_cells(u_int, v_int):

        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #edges
        u = u.at[0, 1:-1].set( u[1, 1:-1])
        u = u.at[-1,1:-1].set( u[-2,1:-1])
        u = u.at[1:-1, 0].set(-u[1:-1, 1])
        u = u.at[1:-1,-1].set(-u[1:-1,-2])
        #corner points
        u = u.at[0,0].set(-u[1,1])
        u = u.at[-1,-1].set(-u[-2,-2])
        u = u.at[-1,0].set(-u[-2,1])
        u = u.at[0,-1].set(-u[1,-2])

        
        v = jnp.zeros((ny+2, nx+2))
        v = v.at[1:-1,1:-1].set(v_int)
        #edges
        v = v.at[0, 1:-1].set(-v[1, 1:-1])
        v = v.at[-1,1:-1].set(-v[-2,1:-1])
        v = v.at[1:-1, 0].set( v[1:-1, 1])
        v = v.at[1:-1,-1].set( v[1:-1,-2])
        #corner points
        v = v.at[0,0].set(-v[1,1])
        v = v.at[-1,-1].set(-v[-2,-2])
        v = v.at[-1,0].set(-v[-2,1])
        v = v.at[0,-1].set(-v[1,-2])

        return u, v

    def add_continuation_ghost_cells(h_int):

        h = jnp.zeros((ny+2, nx+2))
        h = h.at[1:-1,1:-1].set(h_int)
        #edges
        h = h.at[0, 1:-1].set(h[1, 1:-1])
        h = h.at[-1,1:-1].set(h[-2,1:-1])
        h = h.at[1:-1, 0].set(h[1:-1, 1])
        h = h.at[1:-1,-1].set(h[1:-1,-2])
        #corner points
        h = h.at[0,0].set(h[1,1])
        h = h.at[-1,0].set(h[-2,1])
        h = h.at[0,-1].set(h[1,-2])
        h = h.at[-1,-1].set(h[-2,-2])

        return h

    return add_reflection_ghost_cells, add_continuation_ghost_cells

def extrapolate_over_cf_function():

    def extrapolate_over_cf():

        return u, v

    return add_caving_front_ghosts


def compute_u_v_residuals_function(ny, nx, dy, dx, \
                                   interp_cc_to_fc,\
                                   ew_gradient,\
                                   ns_gradient,\
                                   cc_gradient,\
                                   add_rflc_ghost_cells,\
                                   add_cont_ghost_cells):

    
    def compute_u_v_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta):

        u, v = add_rflc_ghost_cells(u_1d.reshape((ny,nx)),\
                                    v_1d.reshape((ny,nx)))

        u_alive = u[1:-1, 1:-1]
        v_alive = v[1:-1, 1:-1]
        h = h_1d.reshape((ny, nx))

        s_gnd = h + b #b is globally defined
        s_flt = h * (1-c.RHO_I/c.RHO_W)
        s = jnp.maximum(s_gnd, s_flt)

        s = add_cont_ghost_cells(s)

        #volume_term
        dsdx, dsdy = cc_gradient(s)

        volume_x = - (beta * u_alive + c.RHO_I * c.g * h * dsdx) * dx * dy
        volume_y = - (beta * v_alive + c.RHO_I * c.g * h * dsdy) * dy * dx

        #TODO: set bespoke conditions at the front for dvel_dx! otherwise
        #over-estimating the viscosity on the faces near the front!
        #u, v = extrapolate_over_cf(u, v)

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
        h_ew, h_ns = interp_cc_to_fc(h)


        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))


        #mu_ns = mu_ns.at[:, -2].set(0) #screws everything up for some reason!
        #mu_ns = mu_ns*0

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(dudy_ns[1:, :] + dvdx_ns[1:, :])*0.5*dx


        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx
        
        ##removing the thickness makes speeds look better!
        #visc_x = mu_ew[:, 1:]*(2*dudx_ew[:, 1:] + dvdy_ew[:, 1:])*dy   -\
        #         mu_ew[:,:-1]*(2*dudx_ew[:,:-1] + dvdy_ew[:,:-1])*dy   +\
        #         mu_ns[:-1,:]*(dudy_ns[:-1,:] + dvdx_ns[:-1,:])*0.5*dx -\
        #         mu_ns[1:, :]*(dudy_ns[1:,:] + dvdx_ns[1:,:])*0.5*dx


        #visc_y = mu_ew[:, 1:]*(dudy_ew[:, 1:] + dvdx_ew[:, 1:])*0.5*dy -\
        #         mu_ew[:,:-1]*(dudy_ew[:,:-1] + dvdx_ew[:,:-1])*0.5*dy +\
        #         mu_ns[:-1,:]*(2*dvdy_ns[:-1,:] + dudx_ns[:-1,:])*dx   -\
        #         mu_ns[1:, :]*(2*dvdy_ns[1:, :] + dudx_ns[1:, :])*dx



        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y


        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_u_v_residuals)



def qn_velocity_solver_function(ny, nx, dy, dx, mucoef, C, n_iterations):

    #functions for various things:
    interp_cc_to_fc                            = interp_cc_to_fc_function(ny, nx)
    ew_gradient, ns_gradient                   = fc_gradient_functions(dy, dx)
    cc_gradient                                = cc_gradient_function(dy, dx)
    add_rflc_ghost_cells, add_cont_ghost_cells = add_ghost_cells_fcts(ny, nx)

    get_u_v_residuals = compute_u_v_residuals_function(ny, nx, dy, dx,\
                                                       interp_cc_to_fc,\
                                                       ew_gradient, ns_gradient,\
                                                       cc_gradient,\
                                                       add_rflc_ghost_cells,\
                                                       add_cont_ghost_cells)

    mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)

    #############
    #setting up bvs and coords for a single block of the jacobian
    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1)

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

    
    def new_viscosity_fc(u, v):
        u, v = add_rflc_ghost_cells(u.reshape((ny, nx)), v.reshape((ny, nx)))

        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)

        #This is right I think, but I don't get convergence if i square epsilon_visc...
        return B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1)),\
               B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        #return B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
        #            0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC)**(0.5*(1/c.GLEN_N - 1)),\
        #       B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
        #            0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC)**(0.5*(1/c.GLEN_N - 1))

        #return jnp.zeros_like(mucoef_ew) + 1e12, jnp.zeros_like(mucoef_ns) + 1e12

    @jax.jit
    def new_viscosity(u, v):
        dudx, dudy = cc_gradient(u)
        dvdx, dvdy = cc_gradient(v)

        return B * mucoef * (dudx**2 + dvdy**2 + dudx*dvdy +\
                    0.25*(dudy+dvdx)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

    @jax.jit
    def new_beta(u, v):
        return C.copy()


    def solver(u_trial, v_trial, h):
        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.reshape(-1)

        residual = jnp.inf

        for i in range(n_iterations):

            mu_ew, mu_ns = new_viscosity_fc(u_1d, v_1d)

            ## --- DEBUG: Check magnitudes in SSA balance ---
            ## Compute strain rates in cell centers
            #dudx_cc, dudy_cc = cc_gradient(u_1d.reshape((ny, nx)))
            #dvdx_cc, dvdy_cc = cc_gradient(v_1d.reshape((ny, nx)))
            #eps_xx = dudx_cc
            #eps_yy = dvdy_cc
            #eps_xy = 0.5 * (dudy_cc + dvdx_cc)
            #
            ## Effective strain rate squared
            #eps_sq = eps_xx**2 + eps_yy**2 + 2*eps_xy**2
            #
            #print("max strain rate [1/s]:", float(jnp.max(jnp.sqrt(eps_sq))))
            #print("min strain rate [1/s]:", float(jnp.min(jnp.sqrt(eps_sq))))
            #
            ## Check viscosities (mu_ew and mu_ns from new_viscosity_fc)
            #print("max mu_ew [Pa s]:", float(jnp.max(mu_ew)))
            #print("min mu_ew [Pa s]:", float(jnp.min(mu_ew)))
            #print("max mu_ns [Pa s]:", float(jnp.max(mu_ns)))
            #print("min mu_ns [Pa s]:", float(jnp.min(mu_ns)))
            

            beta_eff = new_beta(u_1d, v_1d)

            #dJu_du, dJu_dv, dJv_du, dJv_dv = sparse_jacrev(get_u_v_residuals, \
            #                                (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta_eff))
            ##pooosssibly had these the wrong way round and the below is correct...
            #TODO: CHECK!!!!!!!!!
            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_u_v_residuals, \
                                            (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta_eff))

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],\
                                             dJv_du[mask], dJv_dv[mask]])

            
            ###FOR DEBUGGING
            #jac = jnp.zeros((2*ny*nx, 2*ny*nx))
            #jac = jac.at[coords[0,:], coords[1,:]].set(nz_jac_values)
            #t = 2*nr*nc
            #np.set_printoptions(suppress=True)
            #print("dJ_u/du")
            #print(np.array(jac[:int(t/2), :int(t/2)]))
            #print("-----")
            #print("dJ_u/dv")
            #print(np.array(jac[int(t/2+1):, :int(t/2)]))
            #print("-----")
            #print("dJ_v/du")
            #print(np.array(jac[:int(t/2), int(t/2+1):int(t)]))
            #print("-----")
            #print("dJ_v/dv")
            #print(np.array(jac[int(t/2+1):int(t), int(t/2+1):int(t)]))
            #print("-----")
            #print("-----")
            #raise
            ###print(nz_jac_values.shape)
            ###print(jac)
            ##################

            rhs = -jnp.concatenate(get_u_v_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta_eff))
      
            old_residual = residual
            residual = jnp.max(jnp.abs(-rhs))
            
            if i==0:
                print("Initial residual: {}".format(residual))
            else:
                print("residual: {}".format(residual))
                print("Residual reduction factor: {}".format(old_residual/residual))
            print("------")

            du = solve_petsc_sparse(nz_jac_values,\
                                    coords,\
                                    (ny*nx*2, ny*nx*2),\
                                    rhs,\
                                    ksp_type="bcgs",\
                                    preconditioner="hypre",\
                                    precondition_only=False)

            u_1d = u_1d+du[:(ny*nx)]
            v_1d = v_1d+du[(ny*nx):]

        res_final = jnp.max(jnp.abs(jnp.concatenate(get_u_v_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta_eff))))
        print("----------")
        print("Final residual: {}".format(res_final))

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))

    return solver



def make_misfit(u_obs, reg_param, solve_function):

    cc_gradient = cc_gradient_function(dy, dx)

    def misfit(mucoef_guess):
        u_mod = solve(mucoef_guess)


        misfit_term = jnp.sum((u_mod - u_obs)**2)
        
        #dmu_dx, dmu_dy = cc_gradient(mucoef_guess)
        #regularisation = reg_param * jnp.sum(jnp.vdot(dmu_dx, dmu_dx) + jnp.vdot(dmu_dy, dmu_dy))

        return misfit_term + regularisation

    return misfit


def gradient_descent_function(misfit_function, iterations=400, step_size=0.01):
    def gradient_descent(initial_guess):
        get_grad = jax.jacrev(misfit_function)
        ctrl_i = initial_guess
        for i in range(iterations):
            print(i)
            grads = get_grad(ctrl_i)
            #print(grads)
            ctrl_i = ctrl_i.at[:].set(ctrl_i - step_size*grads)
        return ctrl_i
    return gradient_descent








#A = 5e-25
A = c.A_COLD
B = 0.5 * (A**(-1/c.GLEN_N))


lx = 150_000
ly = 200_000

resolution = 2000 #m

nr = int(ly/resolution)
nc = int(lx/resolution)

lx = nr*resolution
ly = nc*resolution


#nr, nc = 64, 64
#nr, nc = 96*2, 64*2


x = jnp.linspace(0, lx, nc)
y = jnp.linspace(0, ly, nr)

delta_x = x[1]-x[0]
delta_y = y[1]-y[0]

thk_profile = 500# - 300*x/lx
thk = jnp.zeros((nr, nc))+thk_profile
thk = thk.at[:, -1:].set(0)

b = jnp.zeros_like(thk)-600


mucoef = jnp.ones_like(thk)
mucoef = mucoef.at[20:-20,-15:-13].set(0.25)

#plt.imshow(mucoef, vmin=0, vmax=1, cmap="cubehelix")
#plt.colorbar()
#plt.show()

C = jnp.zeros_like(thk)
C = C.at[:1, :].set(1e16)
C = C.at[:, :1].set(1e16)
C = C.at[-1:,:].set(1e16)
C = jnp.where(thk==0, 1, C)


#plt.imshow(jnp.log10(C))
#plt.show()


u_init = jnp.zeros_like(thk)
v_init = jnp.zeros_like(thk)

n_iterations = 20

solver = qn_velocity_solver_function(nr, nc, delta_y, delta_x, mucoef, C, n_iterations)

u_out, v_out = solver(u_init, v_init, thk)

show_vel_field(u_out*c.S_PER_YEAR, v_out*c.S_PER_YEAR)

plt.imshow(v_out*c.S_PER_YEAR)
plt.colorbar()
plt.show()

plt.imshow(u_out*c.S_PER_YEAR)
plt.colorbar()
plt.show()

#plt.plot((u_out*c.S_PER_YEAR)[:,40])
#plt.show()



