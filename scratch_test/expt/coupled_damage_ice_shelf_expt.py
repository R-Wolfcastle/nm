import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=2, suppress=False, linewidth=np.inf)


# u lives on cell centres
#mu lives on face centres
#(as does h, s, phi, ...)

def make_vto(h):
    #NOTE:
    h = h.at[-1].set(0)

    s_gnd = h + b
    s_flt = h*(1-0.917/1.027)
    s = jnp.maximum(s_gnd, s_flt)
    s = s.at[-1].set(0)


    p_W = 1027 * jnp.maximum(0, h-s)
    p_I = 917 * h
    #phi = 1 - (p_W / p_I)
    phi = 1
    C = C_scale * phi
    C = jnp.where(s_gnd>s_flt, C, 0)

    C = jnp.where(h==0, 1, C)

    h_face = jnp.zeros((n+1,))
    h_face = h_face.at[1:n-1].set(0.5*(h[:n-2] + h[1:n-1]))
    h_face = h_face.at[0].set(h_face[1])
        

    h_grad_s = jnp.zeros((n,))
    h_grad_s = h_grad_s.at[1:n-1].set(917 * 9.81 * h[1:n-1] * 0.5 * (s[2:] - s[:n-2]))
    h_grad_s = h_grad_s.at[0].set(917 * 9.81 * h[0] * (s[1] - s[0]))


    def vto(u, d):
        mu = 1-d

        #face-centred:
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n-1].set((u[1:n-1] - u[:n-2])/dx) #remember, the final point has no thickness so we shouldn't treat it the same
        dudx = dudx.at[-2].set(dudx[-3].copy())
        dudx = dudx.at[0].set(dudx[1]) #Not sure what kind of boundary condition this is but does mean u can't change...
        #dudx = dudx.at[0].set(-dudx[1]) #Gets in the idea of reflection

        mu_face = jnp.zeros((n+1,))
        mu_face = mu_face.at[1:n-1].set(0.5*(mu[:n-2] + mu[1:n-1]))
        mu_face = mu_face.at[-2].set(mu_face[-3].copy())
        mu_face = mu_face.at[0].set(mu_face[1])


#        # calculate mu_nl on cell centres then interpolate to face centres:
#        dudx_centres = jnp.zeros((n,))
#        dudx_centres = dudx_centres.at[1:n-1].set((u[2:n] - u[:n-2])/(2*dx))
#        dudx_centres = dudx_centres.at[0].set((u[1] - u[0])/dx)
#        dudx_centres = dudx_centres.at[-1].set((u[-1] - u[-2])/dx)
#
#        mu_nl_centres = mu * (jnp.abs(dudx_centres)+epsilon)**(-2/3)
#
#        mu_nl_faces = jnp.zeros((n+1,))
#        mu_nl_faces = mu_nl_faces.at[1:n].set(0.5*(mu_nl_centres[:n-1] + mu_nl_centres[1:n]))
#        mu_nl_faces = mu_nl_faces.at[0].set(mu_nl_faces[1])
#        mu_nl_faces = mu_nl_faces.at[-1].set(mu_nl_faces[-2])

        #alternative is to calculate based on values already at face centres
        #mu_nl_faces = mu_face * (jnp.abs(dudx)+epsilon)**(-2/3)
        mu_nl_faces = mu_face
        #neither work at the moment

        sliding = C * u * dx

        flux = h_face * mu_nl_faces * dudx
        flux_diff = flux[1:] - flux[:-1]
        #flux_diff = flux_diff.at[-1].set(0)

#        #print(h_face)
#        #print(mu_nl_faces)
#        #print(dudx)
#        print(flux_diff)
#        print(h_grad_s)
#        print(sliding)
#        raise

        #flux = h_face * mu_face * dudx #This works fine.
        #print(flux)
        #raise
        
        return flux_diff - h_grad_s - sliding
    
    return vto



def make_solver_u_only(u_trial, intermediates=False):

    def newton_solve(d):
        vto = make_vto(h)

        #for plotting, printing and debugging:
#        vto(u_trial, d)
#        raise

        vto_jac = jacfwd(vto, argnums=0)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(9):
            jac = vto_jac(u, d)
            print(jac)

            rhs = -vto(u, d)
            
            du = lalg.solve(jac, rhs)
            #du = solve_petsc_dense(jac, rhs, preconditioner='hypre')
            # du = linear_solve(jac, rhs) #making this change alone makes no difference to the hessian computation

            u = u.at[:].set(u+du)
            if intermediates:
                us.append(u)

        if intermediates:
            return u, us
        else:
            return u

    return newton_solve


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
            for i in range(num_iterations):

                vto_jac = vto_jac_fn(u, h)
                advo_jac = advo_jac_fn(u, h, h_old)


                full_jacobian = jnp.block(
                                          [[vto_jac[0], vto_jac[1]],
                                          [advo_jac[0], advo_jac[1]]]
                                )
            #    print(full_jacobian)
            #    raise

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





lx = 100_000
n = 1000
dx = lx/n
x = jnp.linspace(0,lx,n)

#mu_base = 0.1
mu = jnp.zeros((n,)) + 1e15
#mu = mu.at[:].set(mu*mu_base)



#accumulation = jnp.zeros((n+1,))
#accumulation = accumulation.at[:n].set(500)



#OVERDEEPENED BED


h = 2000*jnp.exp(-3*((x/lx)**4))
h = h.at[-1].set(0)

#print(h)
#raise


b = jnp.zeros((n,))-400
# b = b.at[:n].set(b[:n] - 4*jnp.exp(-(5*x-2)**2))

s_gnd = b + h
s_flt = h*(1-0.917/1.027)
s = jnp.maximum(s_gnd, s_flt)
s = s.at[-1].set(0)

#b = jnp.zeros((n,))-400
#b = b.at[:].set(((x/lx)**0.5)*(b - 200*jnp.exp(-(5*(x/lx)-3)**2)))

h = jnp.minimum(s-b, s/(1-0.917/1.027))
h = h.at[-1].set(0)



# #linear sliding, constant C:
C_scale = 6e9
# #linear sliding, ramped C:
p_W = 1027 * jnp.maximum(0, h-s)
p_I = 917 * h
phi = 1
#phi = 1 - (p_W / p_I)
C = C_scale * phi
C = jnp.where(s_gnd>s_flt, C, 0)
# C = C.at[0].set(100)

#linear slding, artificial ramp:
# C = jnp.where(s_gnd>s_flt, jnp.maximum(0, 1-2.2*jnp.linspace(0,1,n+1)), 0)

base = s - h


epsilon = 1e-10


#plotgeom(h)
#raise


u_trial = (jnp.exp(x/lx)-1)*1e-4
h_trial = h.copy()


newton_speed_solve = make_solver_u_only(u_trial, intermediates=True)

u_end, u_ints = newton_speed_solve(1-mu)


h_nonzero_mask = np.where(jnp.abs(h)>0+1e-5, 1, np.nan)

plt.figure()
for u in u_ints:
    plt.plot(np.array(u*h_nonzero_mask*(3.14e7)))
plt.show()








