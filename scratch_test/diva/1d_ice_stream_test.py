import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)


#Note, for making the sparsity stuff a bit easier, we're co-locating everything.
#Bite me Suhas V.


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

        h_faces = jnp.zeros((n+2,))
        h_faces = h_faces.at[2:n+1].set(h[1:n] + (h[1:n] - h[:n-1])/2) #does this do anything to the accuracy?
        ##h_faces = h_faces.at[2:n+1].set(h[1:n] + (h[2:n+1] - h[:n-1])/4) #this central difference form of the derivative is unstable

        #kooky option: expand around i-3/2
        #h_faces = h_faces.at[2:n+1].set(0.5*(h[:n-1]+h[1:n]) + 1.5*(h[1:n] - h[:n-1])) #does this do anything to the accuracy?
        #also seems to do fine. Seems a little less stable than the version above as we reduce delta_x
        #but it's a bit more stable than FOU when we reduce delta_x
        #not much to choose between them on stability when we increase delta_t. Seems like the above option might be best.
        

        thk_flux   = jnp.zeros((n+2,))
        thk_flux   = thk_flux.at[2:n+1].set(h_faces[2:n+1] * u[1:]) #first order upwinding
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


        # return dhdt + thk_flux[:n+1] - thk_flux[1:n+2] - accm_term
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

        print(np.array(vto(u_trial, h_trial)))
        print(np.array(vto(u_trial, h_trial)).shape)

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


                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [advo_jac[0], advo_jac[1]]]
                                )
                print(full_jacobian)
                raise

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





lx = 1
n = 8
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
newton_solve = make_solver(u_trial, h_trial, 2e-3, 10, 20, intermediates=False)




u_end, h_end, hus = newton_solve(mu)








