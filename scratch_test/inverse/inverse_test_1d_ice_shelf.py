""" 1D Ice shelf tests """

#external
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#internal
import os

#local apps
# - 





@custom_vjp
@jax.jit
def linear_solve(A, b):
    # the reason for writing it out like this, despite the fact that lalg.solve
    #is already differentiable and probably has a primitive is that we want to
    #be able to easily replace the solver.
    solution = lalg.solve(A, b)
    return solution

def linear_solve_fwd(A, b):
    solution = linear_solve(A, b)
    return solution, (A, b, solution)

def linear_solve_bwd(res, c_bar):
    A, b, x = res

    lambda_ = linear_solve(jnp.transpose(A), c_bar)
    # calling this here means that higher-order derivatives can be calculated without further definition.

    b_bar = lambda_.copy()
    A_bar = - jnp.outer(lambda_, x)

    return A_bar, b_bar

linear_solve.defvjp(linear_solve_fwd, linear_solve_bwd)





def make_solver_custom_vjp(u_trial, intermediates=False):

    @custom_vjp
    @jax.jit
    def newton_solve(mu):
        vto = make_vto(s)
        vto_jac = jacfwd(vto, argnums=0)
        # vto_jac = jacrev(vto)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            jac = vto_jac(u, mu)
            rhs = -vto(u, mu)
            # du = lalg.solve(jac, rhs)
            du = linear_solve(jac, rhs)
            u = u.at[:].set(u+du)
            if intermediates:
              us.append(u)

        if intermediates:
          return u, us
        else:
          return u


    def newton_solve_fwd(mu):
      #what residuals do we need eh? (i.e. gubbins we need to keep track of for the backward pass)
      vto = make_vto(s)

      u = newton_solve(mu) #REALLY NEED TO THINK ABOUT WHAT TO DO IF INTERMEDIATES!!!


      #use AD enjine to calculate the jacobians (so we're not going completely oldschool)
      # vto_jac_u_func = jacfwd(vto, argnums=0)
      # vto_jac_mu_func = jacfwd(vto, argnums=1)
      # vto_jac_u = vto_jac_u_func(u, mu)
      # vto_jac_mu = vto_jac_mu_func(u, mu)
      # AKA (just keeping stuff around until everyone is very happy with JAX):
      vto_jac_u = jacfwd(vto, argnums=0)(u, mu) #can retrieve this from fwd computation if do properly I think.
      vto_jac_mu = jacfwd(vto, argnums=1)(u, mu)

      #What's kind of nuts is that using u_trial, instead of the current u makes this waayyyyy faster for the
      #inverse problem! THIS DESERVES SOME SERIOUS THOUGHT!!!!!!!!

      #this is a little weird... It seems like we ought to be able to take some
      #of the stuff from the /actual/ forward run, rather than running some of
      #it twice...

      #Returns primal output and residuals to be used in backward pass by f_bwd.
      return u, (vto_jac_u, vto_jac_mu)


    def newton_solve_bwd(res, u_bar):
      vto_jac_u, vto_jac_mu = res # Gets residuals computed in f_fwd

      vto_jac_u_transpose = jnp.transpose(vto_jac_u)
      vto_jac_mu_transpose = jnp.transpose(vto_jac_mu)

      #call to solve adjoint problem:
      # lambda_ = lalg.solve(vto_jac_u_transpose, -u_bar)
      lambda_ = linear_solve(vto_jac_u_transpose, -u_bar)

      mu_bar = jnp.matmul(vto_jac_mu_transpose, lambda_)

      return (mu_bar,)

    newton_solve.defvjp(newton_solve_fwd, newton_solve_bwd)

    return newton_solve


def make_solver(u_trial, intermediates=False):

    def newton_solve(mu):
        vto = make_vto(s)

        vto_jac = jacfwd(vto)

        # vto_jac = jacrev(vto)

        u = u_trial.copy()

        if intermediates:
            us = [u]

        for i in range(10):
            jac = vto_jac(u, mu)
            rhs = -vto(u, mu)
            
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



def make_vto(s):

    def vto(u, mu):
        mu_longer = jnp.zeros((n+1,))
        mu_longer = mu_longer.at[:n].set(mu)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        dudx = dudx.at[0].set(2*u[0]/dx)

        mu_nl = mu_longer * (jnp.abs(dudx)+epsilon)**(-2/3)
        # mu_nl = mu_longer.copy()

        flux = mu_nl * dudx

        sgrad = jnp.zeros((n,))
        sgrad = sgrad.at[-1].set(-s)

        return flux[1:n+1] - flux[:n] - sgrad

    return vto



def make_misfit(u_obs, reg_param):

    #@jax.jit
    def misfit(mu_internal):
        u_mod = newton_solve(mu_internal)

        misfit = jnp.sum((u_obs-u_mod)**2)
        regularisation = reg_param * jnp.sum(((mu_internal[1:]-mu_internal[:-1])/dx)**2)
        # print(regularisation)

        return misfit + regularisation

    return misfit


def make_misfit_custom_vjp(u_obs, reg_param):

    #@jax.jit
    def misfit(mu_internal):
        u_mod = newton_solve_custom_vjp(mu_internal)

        misfit = jnp.sum((u_obs-u_mod)**2)
        regularisation = reg_param * jnp.sum(((mu_internal[1:]-mu_internal[:-1])/dx)**2)
        # print(regularisation)

        return misfit + regularisation

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




lx = 1
n = 100
dx = lx/n
x = jnp.linspace(0,lx,n)
s = 1

epsilon = 1e-10


u_data = jnp.array([0.005,0.015,0.02515305,0.0354623,0.04593097,0.05656237,0.06735988,0.07832697,0.08946723,0.10078432,0.11228199,0.12396412,0.13583466,0.14789769,0.1601574,0.17261808,0.18528415,0.19816016,0.21125075,0.22456075,0.23809506,0.25185877,0.26585707,0.28009537,0.29457912,0.30931404,0.32430595,0.33956087,0.355085,0.3708847,0.3869665,0.4033372,0.42000377,0.43697336,0.45425335,0.47185138,0.48977527,0.50803316,0.5266334,0.5455845,0.56489545,0.5845754,0.6046338,0.62508047,0.6459254,0.667179,0.68885213,0.7109558,0.7335015,0.7565012,0.77996707,0.8039118,0.8283485,0.85329086,0.8787528,0.9047489,0.9312942,0.95840424,0.9860952,1.0143838,1.0432874,1.0728238,1.1030117,1.1338705,1.1654202,1.1976814,1.8376814,1.8714314,1.9059602,1.9412919,1.9774518,2.0144658,2.052361,2.0911655,2.1309092,2.1716218,2.2133358,2.2560837,2.2999003,2.3448217,2.3908849,2.4381292,2.4865952,2.5363257,2.5873647,2.6397586,2.6935558,2.748807,2.8055649,2.863885,2.9238248,2.9854453,3.0488095,3.1139839,3.1810384,3.2500458,3.3210826,3.3942294,3.4695704,3.5471945])

key = jax.random.PRNGKey(42)
u_data_noisy = u_data + 0.02*jax.random.normal(key, (len(u_data),))

u_trial = jnp.exp(x)-1
newton_solve = make_solver(u_trial)

mu_initial_guess = 1-x/2

misfit_function_noisy = make_misfit(u_data_noisy, 1e-4)
#misfit_function_noisy_custom_vjp = make_misfit_custom_vjp(u_data_noisy, 1e-4)


gradient_descent_func = gradient_descent_function(misfit_function_noisy)
mu_final = gradient_descent_func(mu_initial_guess)

u_mod = newton_solve(mu_final)

plt.scatter(x,u_data_noisy)
plt.plot(x,u_mod)
plt.savefig("../misc/ice_shelf_ip_misc.png")














