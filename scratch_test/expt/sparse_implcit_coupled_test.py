#1st party
import sys

#local apps
sys.path.insert(1, '../')
from sparsity_utils import basis_vectors_etc,\
                           make_sparse_jacrev_fct,\
                           dodgy_coo_to_csr,\
                           make_sparse_jacrev_fct_multiprime,\
                           basis_vectors_etc_nonsquare,\
                           make_sparse_jacrev_fct_multiprime_no_densify

#3rd party
from petsc4py import PETSc
from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp
from jax.experimental.sparse import BCOO

import scipy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm





# u lives on cell centres
#mu lives on face centres
#(as does h, s, phi, ...)

def make_vto(mu):

    def vto(u, h):

        s_gnd = h + b
        s_flt = h*(1-0.917/1.027)
        s = jnp.maximum(s_gnd, s_flt)
        s = s.at[-1].set(0)


        p_W = 1.027 * jnp.maximum(0, h-s)
        p_I = 0.917 * h
        phi = 1 - (p_W / p_I)
        C = 300 * phi
        C = jnp.where(s_gnd>s_flt, C, 0)



        mu_longer = jnp.zeros((n+1,))
        mu_longer = mu_longer.at[:n].set(mu)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)

        mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()


        sliding = 0.5 * (C[1:(n+1)] + C[:n]) * u[:n] * dx


        flux = h * mu_nl * dudx

        h_grad_s = 0.917 * 0.5 * (h[1:(n+1)] + h[:n]) * (s[1:(n+1)] - s[:n])
        # h_grad_s = 0.5 * (h[1:(n+1)] + h[:n]) * (s[1:(n+1)] - s[:n]) / dx

        # plt.plot(h_grad_s)
        # plt.show()

        # sgrad = jnp.zeros((n,))
        # sgrad = sgrad.at[-1].set(-s)

        return flux[1:(n+1)] - flux[:n] - h_grad_s - sliding

    return vto


def make_advo_linear_differencing(dt):

    # @jax.jit
    def advo(u, h, h_old):

        h_faces = jnp.zeros((n+2,)) #1 longer than h (which is n+1 long). h considered at cell centres for advo and u at cell centres for vto.
        h_faces = h_faces.at[2:n+1].set(h[1:n] + (h[1:n] - h[:n-1])/2) #does this do anything to the accuracy?
        ##h_faces = h_faces.at[2:n+1].set(h[1:n] + (h[2:n+1] - h[:n-1])/4) #this central difference form of the derivative is unstable

        #kooky option: expand around i-3/2
        #h_faces = h_faces.at[2:n+1].set(0.5*(h[:n-1]+h[1:n]) + 1.5*(h[1:n] - h[:n-1])) #does this do anything to the accuracy?
        #also seems to do fine. Seems a little less stable than the version above as we reduce delta_x
        #but it's a bit more stable than FOU when we reduce delta_x
        #not much to choose between them on stability when we increase delta_t. Seems like the above option might be best.
        

        thk_flux   = jnp.zeros((n+2,))
        thk_flux   = thk_flux.at[2:n+1].set(h_faces[2:n+1] * u[1:])
        thk_flux   = thk_flux.at[0].set(-u[0] * h[0]) #gets in reflection bc for u and that dhdx=0 (so h0 = h1 = h-1)
        thk_flux   = thk_flux.at[1].set(u[0] * h[0])

        #thickness flux has a discontinuous first derivatve at the grounding
        #line. Is that alright? Maybe can be helped by taking us and hs from
        #a wider region around the gl?

        thk_flux   = thk_flux.at[-1].set(thk_flux[-2]) #no thickness change



        accm_term  = accumulation * dx
        dhdt       = (h - h_old) * dx / dt


        # advection_eq = jnp.zeros((n+1,))
        # #icy points:
        # advection_eq = advection_eq.at[:n].set(
        #     dhdt[:n] + thk_flux[:n] - thk_flux[1:n+1] - accm_term[:n]
        # )
        # advection_eq = advection_eq.at[-1].set(h[-1]) # so that jac is 1 in LR...
        # return advection_eq


        # return dhdt + thk_flux[:n+1] - thk_flux[1:n+2] - accm_term. Nope. Think about basic integration.
        return dhdt - thk_flux[:n+1] + thk_flux[1:n+2] - accm_term

    return advo


def make_advo_first_order_upwind(dt):

    # @jax.jit
    def advo(u, h, h_old):

        # thk_flux   = jnp.zeros((n+2,)) #ghost cell at either end
        # thk_flux   = thk_flux.at[1:n+1].set(0.5 * (h[:n] + h[1:n+1]) * u) #no upwinding
        # thk_flux   = thk_flux.at[1:n+1].set(h[:n] * u) #first-order upwinding
        # thk_flux   = thk_flux.at[0].set(thk_flux[1]) #no divergence of uh at ID
        # thk_flux   = thk_flux.at[-1].set(thk_flux[-2]) #no thickness change


        thk_flux   = jnp.zeros((n+2,))
        # thk_flux   = thk_flux.at[1:n+1].set(0.5 * (h[:n] + h[1:n+1]) * u) #no upwinding
        thk_flux   = thk_flux.at[1:n+1].set(h[:n] * u) #first order upwinding
        thk_flux   = thk_flux.at[0].set(-u[0] * h[0]) #gets in reflection bc for u and that dhdx=0 (so h0 = h1 = h-1)
        thk_flux   = thk_flux.at[1].set(u[0] * h[0])

        #thickness flux has a discontinuous first derivatve at the grounding
        #line. This isn't great! Maybe can be helped by taking us and hs from
        #a wider region around the gl?

        thk_flux   = thk_flux.at[-1].set(thk_flux[-2]) #no thickness change



        accm_term  = accumulation * dx
        dhdt       = (h - h_old) * dx / dt


        # advection_eq = jnp.zeros((n+1,))
        # #icy points:
        # advection_eq = advection_eq.at[:n].set(
        #     dhdt[:n] + thk_flux[:n] - thk_flux[1:n+1] - accm_term[:n]
        # )
        # advection_eq = advection_eq.at[-1].set(h[-1]) # so that jac is 1 in LR...
        # return advection_eq


        # return dhdt + thk_flux[:n+1] - thk_flux[1:n+2] - accm_term
        return dhdt - thk_flux[:n+1] + thk_flux[1:n+2] - accm_term

    return advo




def plotgeom(thk):

    s_gnd = b + thk
    s_flt = thk*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk

    #plot b, s and base on lhs y axis, and C on rhs y axis
    fig, ax1 = plt.subplots(figsize=(10,5))

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    #legend
    ax1.legend(loc='upper right')

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")

    plt.show()




def plotboth(thk, speed, title=None, savepath=None, axis_limits=None, show_plots=True):
    s_gnd = b + thk
    s_flt = thk*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)

    base = s-thk


    fig, ax1 = plt.subplots(figsize=(10,5))
    ax2 = ax1.twinx()

    ax1.plot(s, label="surface")
    # ax1.plot(base, label="base")
    ax1.plot(base, label="base")
    ax1.plot(b, label="bed")

    ax2.plot(speed, color='k', marker=".", linewidth=0, label="speed")

    #legend
    ax1.legend(loc='lower left')
    #slightly lower
    ax2.legend(loc='center left')
    #stop legends overlapping

    #axis labels
    ax1.set_xlabel("x")
    ax1.set_ylabel("elevation")
    ax2.set_ylabel("speed")

    if axis_limits is not None:
        ax1.set_ylim(axis_limits[0])
        ax2.set_ylim(axis_limits[1])

    if title is not None:
        plt.title(title)
    if savepath is not None:
        plt.savefig(savepath)

    if show_plots:
        plt.show()



def plotu(u):

    fig, ax = plt.subplots(figsize=(10,5))

    # colors = cm.coolwarm(jnp.linspace(0, 1, len(us)))

    # for i, u in enumerate(np.array(us) / 3.5):
    #     plt.plot(x, u, color=colors[i], label=str(i))
    # # change all spines
    # for axis in ['bottom','left']:
    #     plt.gca().spines[axis].set_linewidth(2.5)
    # for axis in ['top','right']:
    #     plt.gca().spines[axis].set_linewidth(0)

    plt.plot(x, u, color='k')

    #increase size of x and y tick labels
    plt.tick_params(axis='both', which='major', labelsize=12)

    #axis label
    plt.gca().set_ylabel("speed")

    plt.show()



#solve:
def make_solver(u_trial, h_trial, dt, num_iterations, num_timesteps, intermediates=False):

    # @jax.jit
    def newton_solve(mu):
        hus = []

        vto = make_vto(mu)

        #print(np.array(vto(u_trial, h_trial)))
        #print(np.array(vto(u_trial, h_trial)).shape)

        #advo = make_advo_first_order_upwind(dt) #more stable under larger dt
        advo = make_advo_linear_differencing(dt) #more stable under larger dx

        vto_jac_fn = jacfwd(vto, argnums=(0,1))
        advo_jac_fn = jacfwd(advo, argnums=(0,1))

        u = u_trial.copy()
        h = h_trial.copy()



        # if intermediates:
        #     us = [u]

        h_old = h_trial.copy()


        for j in range(num_timesteps):
            print(j)
            for i in range(num_iterations):

                vto_jac = vto_jac_fn(u, h)
                advo_jac = advo_jac_fn(u, h, h_old)

                #print(vto_jac[0].shape) #(n,n)
                #print(vto_jac[1].shape) #(n,n+1)
                #print(advo_jac[0].shape) #(n+1,n)
                #print(advo_jac[1].shape) #(n+1,n+1)

                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [advo_jac[0], advo_jac[1]]]
                                )
                #print(full_jacobian.shape) #given the above, will have shape (n+n+1, n+n+1)

                #print(np.array(vto_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(advo_jac[0]))
                #print("-------------------")
                #print("-------------------")
                #print("-------------------")
                #print(np.array(full_jacobian))
                #raise


                # np.set_printoptions(linewidth=200)
                # print(np.array_str(full_jacobian, precision=2, suppress_small=True))
                # np.set_printoptions(linewidth=75)
                # print(full_jacobian.shape)


                rhs = jnp.concatenate((-vto(u, h), -advo(u, h, h_old)))

                dvar = lalg.solve(full_jacobian, rhs)

                u = u.at[:].set(u+dvar[:n])
                h = h.at[:].set(h+dvar[n:])


                # plt.plot(dvar[:n])
                # plt.show()
                # plt.plot(dvar[n:])
                # plt.show()


            hus.append([h, u])

            plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
                    savepath="../misc/full_implicit_tests/{}_{}.png".format(j+1,i),\
                    axis_limits = [[-15, 30],[0, 150]], show_plots=False)

            # plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
            #          savepath=None,\
            #          axis_limits = [[-15, 30],[0, 150]], show_plots=True)

            h_old = h.copy()




            # if intermediates:
              # us.append(u)

        # if intermediates:
        #   return u, us
        # else:
        #   return u

        return u, h, hus

    return newton_solve

#TODO: Refactor this jesus christ. it's also in sparse_jacobian_petsc_test.py... and it's rubbish anyway!
def solve_petsc_sparse(values, coordinates, jac_shape, b, ksp_type='gmres', preconditioner='hypre', precondition_only=False):
    comm = PETSc.COMM_WORLD
    size = comm.Get_size()

    iptr, j, values = dodgy_coo_to_csr(values, coordinates, jac_shape, return_decomposition=True)

    
    A = PETSc.Mat().createAIJ(size=jac_shape, csr=(iptr, j, values), comm=comm)
    
    b = PETSc.Vec().createWithArray(b, comm=comm)
    
    x = b.duplicate()
    
    
    # Create a linear solver
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)

    ksp.setOperators(A)
    #ksp.setFromOptions()
    
    
    if preconditioner == 'hypre':
        pc = ksp.getPC()
        pc.setType('hypre')
        pc.setHYPREType('boomeramg')
    else:
        pc = ksp.getPC()
        pc.setType(preconditioner)

    if precondition_only:
        pc.apply(b, x)
    else:
        ksp.solve(b, x)
    
    # Print the solution
    #x.view()

    x_jnp = jnp.array(x.getArray())

    return x_jnp


def make_solver_sparse_jvp(u_trial, h_trial, dt, num_iterations, num_timesteps, intermediates=False):

    def newton_solve(mu):
        hus = []


        vto = make_vto(mu)
        #advo = make_advo_first_order_upwind(dt) #more stable under larger dt
        advo = make_advo_linear_differencing(dt) #more stable under larger dx


        u = u_trial.copy()
        h = h_trial.copy()

        h_old = h_trial.copy()

       
        #it seems like maybe jax vjp can't handle just single arguments, so
        #we might have to aggregate teh stencils by operator (if ykwim..?)
    
        bvs_vt, i_cs_dvt_du, j_cs_dvt_du = basis_vectors_etc(n, 1)
        _, i_cs_dvt_dh, j_cs_dvt_dh = basis_vectors_etc_nonsquare(n, n+1, 1)

        bvs_ad, i_cs_dadv_du, j_cs_dadv_du = basis_vectors_etc_nonsquare(n+1, n, 3)
        _, i_cs_dadv_dh, j_cs_dadv_dh = basis_vectors_etc(n+1, 5)

        #bvs_gh, i_cs_gh, j_cs_gh = basis_vectors_etc(n+1, 5)
        sparse_jacrev_vt,_  = make_sparse_jacrev_fct_multiprime_no_densify(bvs_vt)
        sparse_jacrev_adv,_ = make_sparse_jacrev_fct_multiprime_no_densify(bvs_ad)

        
        is_dvt_du  = i_cs_gu.copy()
        js_dvt_du  = j_cs_gu.copy()
        is_dvt_dh  = i_cs_gu.copy() + n
        js_dvt_dh  = j_cs_gu.copy()
        is_dadv_du = i_cs_gh.copy() + n
        js_dadv_du = j_cs_gh.copy()
        is_dadv_dh = i_cs_gh.copy() + n
        js_dadv_dh = j_cs_gh.copy() + n

        print(is_dvt_du.shape)
        print(js_dvt_du.shape)
        print(is_dvt_dh.shape)
        print(js_dvt_dh.shape)
        print(is_dadv_du.shape)
        print(js_dadv_du.shape)
        print(is_dadv_dh.shape)
        print(js_dadv_dh.shape)
        raise

        for j in range(num_timesteps):
            print(j)
            for i in range(num_iterations):
                
                #TODO: work out wtf is going on here and everywhere...

                dvt_du_values, dvt_dh_values = sparse_jacrev_gu(vto, (u, h))
                dadv_du_values, dadv_dh_values, dadv_dhold_values = sparse_jacrev_gh(advo, (u, h, h_old))

                print(dvt_du_values.shape)
                print(dvt_dh_values.shape)
                print(dadv_du_values.shape)
                print(dadv_dh_values.shape)
                print(dadv_dhold_values.shape)
                raise

                all_values = jnp.concatenate((dvt_du_values, dvt_dh_values, dadv_du_values, dadv_dh_values))
                all_is = jnp.concatenate((is_dvt_du, is_dvt_dh, is_dadv_du, is_dadv_dh))
                all_js = jnp.concatenate((js_dvt_du, js_dvt_dh, js_dadv_du, js_dadv_dh))
                all_coords = jnp.column_stack((all_is, all_js))

                print(all_values.shape)
                print(all_coords.shape)
                raise

                rhs = jnp.concatenate((-vto(u, h), -advo(u, h, h_old)))

                dvar = solve_petsc_sparse(all_values, all_coords, (2*n + 1, 2*n + 1), rhs)

                u = u.at[:].set(u+dvar[:n])
                h = h.at[:].set(h+dvar[n:])


                # plt.plot(dvar[:n])
                # plt.show()
                # plt.plot(dvar[n:])
                # plt.show()


            hus.append([h, u])

            plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
                    savepath="../misc/full_implicit_tests/{}_{}.png".format(j+1,i),\
                    axis_limits = [[-15, 30],[0, 150]], show_plots=False)

            # plotboth(h, u, title="Timestep {}, iteration {}".format(j+1, i),\
            #          savepath=None,\
            #          axis_limits = [[-15, 30],[0, 150]], show_plots=True)

            h_old = h.copy()




            # if intermediates:
              # us.append(u)

        # if intermediates:
        #   return u, us
        # else:
        #   return u

        return u, h, hus

    return newton_solve





lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)


mu_base = 0.1
mu = jnp.zeros((n,)) + 1
mu = mu.at[:].set(mu*mu_base)



accumulation = jnp.zeros((n+1,))
# accumulation = accumulation.at[:n].set(100*(1-(2*(x-0.3))**2))
accumulation = accumulation.at[:n].set(500)





#OVERDEEPENED BED


h = jnp.zeros((n+1,))
# h = h.at[:n].set(20*jnp.exp(-2*x*x*x*x))
h = h.at[:n].set(20*jnp.exp(-0.5*x*x*x*x))
h = h.at[-1].set(0)


b_intermediate = jnp.zeros((n+1,))-12
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-2)**2))

s_gnd = b_intermediate + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)

b = jnp.zeros((n+1,))-12
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-2)**2))
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-3)**2))
b = b.at[:n].set((x**0.5)*(b[:n] - 5*jnp.exp(-(5*x-3)**2)))

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)


# #linear sliding, constant C:
# C = jnp.where(s_gnd>s_flt, 1, 0)

# #linear sliding, ramped C:
p_W = 1.027 * jnp.maximum(0, h-s)
p_I = 0.917 * h
phi = 1 - (p_W / p_I)
C = 300 * phi
C = jnp.where(s_gnd>s_flt, C, 0)
# C = C.at[0].set(100)

#linear slding, artificial ramp:
# C = jnp.where(s_gnd>s_flt, jnp.maximum(0, 1-2.2*jnp.linspace(0,1,n+1)), 0)

base = s - h

effective_base = s.copy()
effective_base = effective_base.at[:n].set(s[:n] - h[:n]*mu/mu_base)

epsilon = 1e-10





##plot b, s and base on lhs y axis, and C on rhs y axis
#fig, ax1 = plt.subplots(figsize=(10,5))
#ax2 = ax1.twinx()
#
#ax1.plot(s, label="surface")
## ax1.plot(base, label="base")
#ax1.plot(effective_base, label="base")
#ax1.plot(b, label="bed")
#
#ax2.plot(C, color='k', marker=".", linewidth=0, label="sliding coefficient")
#
##legend
#ax1.legend(loc='upper right')
##slightly lower
#ax2.legend(loc='center')
##stop legends overlapping
#
##axis labels
#ax1.set_xlabel("x")
#ax1.set_ylabel("elevation")
#ax2.set_ylabel("sliding coefficient")
#
#plt.show()







u_trial = jnp.exp(x)-1
h_trial = h.copy()


#newton_solve = make_solver(u_trial, h_trial, 2e-4, 10, 20, intermediates=False)
##newton_solve = make_solver(u_trial, h_trial, 2e-3, 10, 20, intermediates=False)

newton_solve = make_solver_sparse_jvp(u_trial, h_trial, 2e-4, 10, 20, intermediates=False)



u_end, h_end, hus = newton_solve(mu)








