from pathlib import Path
import sys
import os

nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'utils'))
from plotting_stuff import *



from PIL import Image

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=10, suppress=False, linewidth=np.inf, threshold=np.inf)

def interp_mu_onto_faces(mu_centre):
    mu_face = jnp.zeros((n+1,))
    mu_face = mu_face.at[1:-1].set(0.5 * (mu_centre[:-1] + mu_centre[1:]))
    mu_face = mu_face.at[-1].set(mu_centre[-1]) #doesn't actually do anything to mb as
                                                #dudx defined to be zero there, but
                                                #useful for translating back!
    mu_face = mu_face.at[0].set(mu_centre[1])
    return mu_face

def interp_mu_onto_centres(mu_face):
    mu_centre = jnp.zeros((n,))
    mu_centre = mu_centre.at[1:-1].set(0.5 * (mu_face[1:-2] + mu_face[2:-1]))
    mu_centre = mu_centre.at[0].set(mu_face[0])
    mu_centre = mu_centre.at[-1].set(mu_face[-1])
    return mu_centre


def make_linear_momentum_residual_mu_face():
    #one-sided differences at the grounding line!

    def mom_res(u, h, mu_face, beta):

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)
        
        is_floating = s_flt > s_gnd
        ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)


        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)


        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #one-sided differences at gl
        h_grad_s = h_grad_s.at[ffci].set(h[ffci] * (s[ffci+1] - s[ffci]))
        h_grad_s = h_grad_s.at[ffci-1].set(h[ffci-1] * (s[ffci-1] - s[ffci-2]))
     
        #scale
        h_grad_s = rho * g * h_grad_s

        #print(flux)
        #print(sliding)
        #print(h_grad_s)

        #plt.plot(-h_grad_s)
        #plt.plot(sliding)
        #plt.plot(flux)
        #plt.show()
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res

def make_linear_momentum_residual():
    #one-sided differences at the grounding line!

    def mom_res(u, h, mu_centre, beta):

        mu_face = interp_mu_onto_faces(mu_centre)

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)
        
        is_floating = s_flt > s_gnd
        ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)


        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #one-sided differences at gl
        h_grad_s = h_grad_s.at[ffci].set(h[ffci] * (s[ffci+1] - s[ffci]))
        h_grad_s = h_grad_s.at[ffci-1].set(h[ffci-1] * (s[ffci-1] - s[ffci-2]))
     
        #scale
        h_grad_s = rho * g * h_grad_s

        #print(flux)
        #print(sliding)
        #print(h_grad_s)

        #plt.plot(-h_grad_s)
        #plt.plot(sliding)
        #plt.plot(flux)
        #plt.show()
        
        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res

def make_linear_momentum_residual_all_afloat():
    #one-sided differences at the grounding line!

    def mom_res(u, h, mu_centre, beta):

        mu_face = interp_mu_onto_faces(mu_centre)

        s_gnd = h + b #b is globally defined
        s_flt = h*(1-rho/rho_w)
        s = jnp.maximum(s_gnd, s_flt)

        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:n].set((u[1:n] - u[:n-1])/dx)
        #dudx = dudx.at[-2].set(dudx[-3])
        dudx = dudx.at[-1].set(0)
        ##set (or 'use' I guess) reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)


        sliding = beta * u * dx
        #making sure the Jacobian is full rank!
        sliding = sliding.at[:].set(jnp.where(h>0, jnp.where(s_gnd>s_flt, sliding, 0), u * dx))

        
        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n].set(0.5 * (h[1:n] + h[:n-1]))
        h_face = h_face.at[-1].set(0)
        h_face = h_face.at[0].set(h[0])


        flux = h_face * mu_face * dudx


        h_grad_s = jnp.zeros((n,))
        h_grad_s = h_grad_s.at[1:n-1].set(h[1:n-1] * 0.5 * (s[2:n] - s[:n-2]))
        h_grad_s = h_grad_s.at[-1].set(-h[-1] * 0.5 * s[-2])
        #h_grad_s = h_grad_s.at[-2].set(-0.1)
        h_grad_s = h_grad_s.at[0].set(h[0] * 0.5 * (s[1] - s[0]))
      
        #scale
        h_grad_s = rho * g * h_grad_s

        return flux[1:] - flux[:-1] - h_grad_s - sliding
        #return - h_grad_s - sliding
        #return flux[1:] - flux[:-1] - sliding

    return mom_res


def make_adv_residual(dt, accumulation):
    
    def adv_res(u, h, h_old, basal_melt_rate):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)

        acc = jnp.where(s_gnd<s_flt, accumulation-basal_melt_rate, accumulation)
        acc = acc.at[:].set(jnp.where(h>0, acc, 0))

        h_face = jnp.zeros((n+1,))
        h_face = h_face.at[1:n+1].set(h[:n]) #upwind values
        h_face = h_face.at[0].set(h[0].copy())

        u_face = jnp.zeros((n+1,))
        u_face = u_face.at[1:n].set(0.5*(u[1:n]+u[:n-1]))
        u_face = u_face.at[-1].set(2*u[-1] - u[-2]) #extrapolating u (linear)

        h_flux = h_face * u_face
        #the two lines below were supposed to stop everything piling up in the last
        #cell but they had the opposite effect for some reason...
        #h_flux = h_flux.at[-2].set(h_flux[-3]) #stop everythin piling up at the end.
        #NOTE: This changes things a lot:
        #h_flux = h_flux.at[-1].set(h_flux[-2].copy())
        h_flux = h_flux.at[0].set(0)


        #return  (h - h_old)/dt - ( h_flux[1:(n+1)] - h_flux[:n] )/dx - acc
        #I think the above is a sign error in the flux!
        return  (h - h_old)*dx + dt*( h_flux[1:(n+1)] - h_flux[:n] ) - dt*dx*acc

    return adv_res



def define_z_coordinates(n_levels, thk):
    s_gnd = b + thk
    s_flt = thk*(1-rho/rho_w)

    surface = jnp.maximum(s_gnd, s_flt)
    base = surface-thk

    base = jnp.maximum(base, b) #just to make sure

    #Choosing quadratic spacing. However, the vertical profiles I have seen look more
    #like the maximum curvature is quite high in the ice column, so maybe uniform
    #spacing would make more sense. But it's based on equation 6. If you assume viscosity
    #is vertically uniform, then dudz is linear so u is quadratic in z.
    v_coords_1d = jnp.linspace(0,1,n_levels)**2
    
    v_coords_expanded = v_coords_1d[None, :]
   
    base_expanded = base[:, None]
    thk_expanded = thk[:, None]

    z_coords_2d = base_expanded + thk_expanded*v_coords_expanded
    #v_coords_2d = base_expanded*0 + vdd_coords_expanded
    #return v_coords_2d, z_coords_2d
    return z_coords_2d


#unfortunately, it seems that np and jnp in-built trapz only take scalar dz.
def vertically_integrate(field, z_coords, preserve_structure=False):
    #last dimension in field should be vertical
    #trapezium rule

    dzs = z_coords[..., 1:]-z_coords[..., :-1]

    au_curve_segments = 0.5 * dzs * (field[..., 1:]+field[..., :-1])

    integrated_field = jnp.cumsum(au_curve_segments, axis=-1)
    integrated_field = jnp.concatenate([jnp.zeros_like(dzs[...,-1])[...,None]\
                                        , integrated_field], axis=-1)

    if preserve_structure:
        return integrated_field
    else:
        return integrated_field[..., -1]


def vertically_average(field, z_coords):
    hs = z_coords[...,-1]-z_coords[...,0]

    v_int = vertically_integrate(field, z_coords)

    v_avg = v_int/(hs+1e-10)

    ##jax.debug.print("{}",((v_avg-field[...,-5])/v_avg))
    #jax.debug.print("field: {}",field)
    #jax.debug.print("vertical average: {}",v_avg)

    return v_avg



def make_mom_solver_diva(iterations, rheology_n=3, mode="DIVA", compile_=False):

    mom_res = make_linear_momentum_residual()
    #mom_res = make_linear_momentum_residual_all_afloat()

    jac_mom_res_fn = jacfwd(mom_res, argnums=0)


    #equation 2b
    def new_viscosity(u_va, dudz, zs):
        #If face-centred:
   #     dudx = jnp.zeros((n+1,))
   #     dudx = dudx.at[1:-1].set((u_va[1:] - u_va[:-1])/dx)
   #     dudx = dudx.at[-1].set(dudx[-2])
   #     #set reflection boundary condition
   #     dudx = dudx.at[0].set(2*u_va[0]/dx)

        #If cell-centred
        dudx = jnp.zeros_like(u_va)
        dudx = dudx.at[1:-1].set(0.5 * (u_va[2:] - u_va[:-2]) / dx)
        dudx = dudx.at[0].set(0.5 * (u_va[1]+u_va[0]) / dx) #reflection bc, remember
        dudx = dudx.at[-1].set(dudx[-2])
    
        #mu_vv = 0.5 * B * (jnp.abs(dudx)[...,None]**2 + 0.25*dudz**2 + epsilon_visc)**(0.5*(1/rheology_n - 1))
        #not sure about the factor of 1/2...
        #mu_vv = B * (jnp.abs(dudx)[...,None]**2 + 0.25*dudz**2 + epsilon_visc**2)**(0.5*(1/rheology_n - 1))
        mu_vv = B * (jnp.abs(dudx)[...,None]**2 + 0.25*dudz**2 + epsilon_visc**2)**(0.5*(1/rheology_n - 1))
        
        mu_va = vertically_average(mu_vv, zs)

        return mu_vv, mu_va


    #NOTE: This function is (and should be) unused.
    def new_viscosity_ssa(u):
        
        #dudx = jnp.zeros((n+1,))
        #dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        #dudx = dudx.at[-1].set(dudx[-2])
        ##set reflection boundary condition
        #dudx = dudx.at[0].set(2*u[0]/dx)
    
        #mu_va = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #If cell-centred
        dudx = jnp.zeros_like(u)
        dudx = dudx.at[1:-1].set(0.5 * (u[2:] - u[:-2]) / dx)
        dudx = dudx.at[0].set(0.5 * (u[1]+u[0]) / dx) #reflection bc, remember
        dudx = dudx.at[-1].set(dudx[-2])
    
        mu_va = B * (jnp.abs(dudx)**2 + epsilon_visc**2)**(-1/3)

        return mu_va

    def arthern_function(mu_vv, zs, m=1, only_return_surface=True):

        ss_expanded = zs[...,-1][...,None]
        hs_expanded = (zs[...,-1]-zs[...,0])[...,None]
        vv_field = (1/mu_vv) * ((ss_expanded-zs)/hs_expanded)**m
        
        fm = vertically_integrate(vv_field, zs, preserve_structure=True)

        if only_return_surface:
            return fm[...,-1]
        else:
            return fm


    def new_beta(u_base, zs):
        h = zs[...,-1] - zs[...,0]
        
        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u_base)**(2/3) + (1e-8)**(2/3))) * grounded_mask

        return beta


    #f2 = arthern_function(mu_vv, zs, 2)
    #equation 5
    def new_beta_eff(u_base, f2, zs):
        
        beta = new_beta(u_base, zs)
        
        #jax.debug.print("f2: {}", f2)
        #jax.debug.print("beta factor: {}", (1 / (1 + beta*f2)))

        beta_eff = beta / (1 + beta*f2)

        #jax.debug.print("beta: {}", beta)
        #jax.debug.print("beta_eff: {}", beta_eff)
        
        return beta, beta_eff


    #equation 6
    def new_dudz(mu_vv, u_va, beta_eff, zs):

        s = zs[...,-1][...,None]
        h = (zs[...,-1] - zs[...,0])[...,None]
       
        dudz = (1/mu_vv) *\
               beta_eff[...,None] *\
               u_va[...,None] *\
               (s - zs) *\
               (1/h)

        return dudz

 
    #equation 4 (combined with 5 a bit).
    def new_u_vv(dudz, u_va, beta, beta_eff, zs, mu_vv, f2):

        f1_vv = arthern_function(mu_vv, zs, m=1, only_return_surface=False)

        f2 = arthern_function(mu_vv, zs, m=2)

    #    jax.debug.print("beta: {}", beta)
    #    jax.debug.print("f2: {}", f2)
    #    jax.debug.print("u_va factor: {}", 1/(1+beta*f2))

        pre_add = (u_va / (1 + beta*f2))[...,None] 
        prefactor_exp = (beta_eff * u_va)[...,None]

        u_vv = pre_add + prefactor_exp * f1_vv

        return u_vv


    #equation 3
    def setup_and_solve_linear_prob(u_va, mu_va, beta_eff, zs):
        h = zs[...,-1]-zs[...,0]

        jac_mom_res = jac_mom_res_fn(u_va, h, mu_va, beta_eff)

        delta_u = lalg.solve(jac_mom_res, -mom_res(u_va, h, mu_va, beta_eff))

        u_va = u_va + delta_u

        residual = jnp.max(jnp.abs(mom_res(u_va, h, mu_va, beta_eff)))

        return u_va, residual


    def step_ssa(state):
        u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i, res, resrat = state

        dudz = jnp.zeros_like(dudz)

        #update viscosity
        mu_vv, mu_va = new_viscosity(u_va, dudz, zs)
        #mu_va = new_viscosity_ssa(u_va)
        
    #    jax.debug.print("{}", mu_va)

        #update beta_eff
        beta_eff = new_beta(u_vv[...,0], zs)
        #Note, you can't get away with calculating beta_eff from new_beta_eff unfortunatel

        #solve linear problem
        u_va, residual = setup_and_solve_linear_prob(u_va, mu_va, beta_eff, zs)
        resrat = res/residual

        #update dudz
        #dudz = new_dudz(mu_vv, u_va, beta_eff, zs)

        #update u_vv
        #u_vv = new_u_vv(dudz, u_va, beta, beta_eff, zs, mu_vv, f2)
        u_vv = jnp.zeros_like(u_vv) + u_va[...,None]

        #jax.debug.print("res: {}", residual)
        #jax.debug.print("resrat: {}", resrat)
        
#        jax.debug.print("u_va: {}", u_va)
#        ##jax.debug.print("u_vv: {}", u_vv)
#        jax.debug.print("u_va from vi of u_vv: {}", vertically_average(u_vv, zs))

        return u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i+1, residual, resrat


    def step(state):
        
        u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i, res, resrat = state

        #update viscosity
        mu_vv, mu_va = new_viscosity(u_va, dudz, zs)

        #update beta_eff
        #f2 = jax.lax.cond(i>0,
        #                lambda _: arthern_function(mu_vv, zs, m=2),
        #                lambda _: jnp.zeros_like(u_va),
        #                operand=None)
        f2 = arthern_function(mu_vv, zs, m=2)
        beta, beta_eff = new_beta_eff(u_vv[...,0], f2, zs)

        #solve linear problem
        u_va, residual = setup_and_solve_linear_prob(u_va, mu_va, beta_eff, zs)
        resrat = res/residual

        #update dudz
        dudz = new_dudz(mu_vv, u_va, beta_eff, zs)
        #jax.debug.print("dudz: {}", dudz[...,1])

        #update u_vv
        u_vv = new_u_vv(dudz, u_va, beta, beta_eff, zs, mu_vv, f2)
        #NOTE: Something's going quite wrong as u_vv inconsistent with u_va (i.e.
        #avg of u_vv not the same as u_va...)

        #jax.debug.print("res: {}", residual)
        #jax.debug.print("resrat: {}", resrat)
        
        #jax.debug.print("u_va: {}", u_va)
        ##jax.debug.print("u_vv: {}", u_vv)
        #jax.debug.print("u_va from vi of u_vv: {}", vertically_average(u_vv, zs))

        #jax.debug.print("u_va error: {}", (u_va-vertically_average(u_vv, zs))/u_va)

        return u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i+1, residual, resrat


    def continue_condition(state):
        _,_,_,_,_,_,_,_, i ,_,_ = state
        return i<iterations


    def iterator(u_va_init, dudz_init, zs):
        
        residual_ratio = jnp.inf
        residual = 1


        dummy_small = jnp.zeros_like(u_va_init)
        dummy_large = jnp.zeros_like(dudz_init)
        
        beta_init = dummy_small
        u_vv_init = dummy_large + u_va_init[...,None]

        initial_state = u_va_init, u_vv_init, dummy_small, dummy_large,\
                        dudz_init, beta_init, dummy_small, zs, 0, residual, residual_ratio

        #u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, itns, residual, residual_ratio

        if mode=="DIVA":
            internal_step = step
        elif mode=="SSA":
            internal_step = step_ssa
        else:
            print("Invalid mode given, defaulting to DIVA")
            internal_step = step


        out_state = jax.lax.while_loop(continue_condition, internal_step, initial_state)
        
        return out_state

    return iterator



def make_mom_solver_ssa(iterations, rheology_n=3, compile_=False):


    #mom_res = make_linear_momentum_residual_mu_face()
    mom_res = make_linear_momentum_residual()
    
    jac_mom_res_fn = jacfwd(mom_res, argnums=0)


    def new_viscosity(u):
        
        #dudx = jnp.zeros((n+1,))
        #dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        #dudx = dudx.at[-1].set(dudx[-2])
        ##set reflection boundary condition
        #dudx = dudx.at[0].set(2*u[0]/dx)
    
        #mu_va = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #If cell-centred
        dudx = jnp.zeros_like(u)
        dudx = dudx.at[1:-1].set(0.5 * (u[2:] - u[:-2]) / dx)
        dudx = dudx.at[0].set(0.5 * (u[1]+u[0]) / dx) #reflection bc, remember
        dudx = dudx.at[-1].set(dudx[-2])
    
        mu_va = B * (jnp.abs(dudx)**2 + epsilon_visc**2)**(-1/3)

        return mu_va


    #just keeping this here to make it look more like the diva solver
    def new_beta(u, h):
        
        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask

        return beta


    def setup_and_solve_linear_prob(u_va, mu_va, beta_eff, h):

        jac_mom_res = jac_mom_res_fn(u_va, h, mu_va, beta_eff)

        delta_u = lalg.solve(jac_mom_res, -mom_res(u_va, h, mu_va, beta_eff))

        u_va = u_va + delta_u

        residual = jnp.max(jnp.abs(mom_res(u_va, h, mu_va, beta_eff)))

        return u_va, residual


    def step(state):
        
        u_va, mu_va, beta, h, i, res, resrat = state

        #update viscosity
        mu_va = new_viscosity(u_va)
        
        jax.debug.print("{}",mu_va)

        #update beta_eff
        beta = new_beta(u_va, h)

        #solve linear problem
        u_va, residual = setup_and_solve_linear_prob(u_va, mu_va, beta, h)
        resrat = res/residual

        jax.debug.print("res: {}", residual)
        jax.debug.print("resrat: {}", resrat)

        return u_va, mu_va, beta, h, i+1, residual, resrat


    def continue_condition(state):
        _,_,_,_, i ,_,_ = state
        return i<iterations


    def iterator(u_va_init, h):
        
        residual_ratio = jnp.inf
        residual = 1
        
        dummy_small  = jnp.zeros((n,))
        dummy_medium = jnp.zeros((n+1,))

        initial_state = u_va_init, dummy_small, dummy_small, h, 0, residual, residual_ratio

        out_state = jax.lax.while_loop(continue_condition, step, initial_state)

        return out_state

    return iterator







n = 101
l = 1_800_000
x = jnp.linspace(0, l, n)
dx = x[1]-x[0]

rho = 900
rho_w = 1000

siy = 3.15e7

g = 9.8

accumulation = jnp.zeros_like(x)+0.3/(3.15e7)


C = 7.624e6

#A = 4.6146e-24
A = 5e-25
#A = 5e-24 #This works, but I have to change the timestep from 1e10 to 5e8 which is a bit of a bummer.

B = 2 * (A**(-1/3))

#epsilon_visc = 1e-5/(3.15e7)
epsilon_visc = 3e-13


#b = 720 - 778.5*x/750_000
b = 729 - 2184.8*(x/750_000)**2 + 1031.72*(x/750_000)**4 - 151.72*(x/750_000)**6
#b = 729 - 2184.8*(x/750_000)**2 + 1031.72*(x/750_000)**4 - 151.72*(x/750_000)**6

x_s = x/l
#h_init = jnp.zeros_like(x)+100
h_init = 4000*jnp.exp(-2*((x_s)**15))
#h_init = 4000 - 3500*x_s*x_s
#h_init = 500 + 4000*jnp.exp(-2*((x_s+0.35)**15))


h_init = jnp.load("./possible_starting_thk.npy")

u_trial = jnp.zeros_like(x)
h_trial = h_init.copy()


#
##NOTE: different mom res for this required bc stuff to do w/
##osd at gl goes wrong if no grounded ice. easy fix will do another time
#b = jnp.zeros_like(b)-1000
#h_trial = jnp.zeros_like(b)+800




n_levels = 31

z_coordinates = define_z_coordinates(n_levels, h_trial)




#plt.figure()
#for i in range(z_coordinates.shape[-1]):
#    plt.plot(z_coordinates[...,i])
#plt.show()
##plotgeoms([z_coordinates[...,i] for i in range(z_coordinates.shape[-1])], b, z_coordinates.shape[-1])
#raise





u_va_init = u_trial
dudz_init = jnp.zeros((n,n_levels))



#DIVA SSA:
n_iterations = 15

mom_solver = make_mom_solver_diva(n_iterations, mode="SSA")

u_va_divassa, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, itns, res, resrat = mom_solver(u_va_init, dudz_init, z_coordinates)


#DIVA DIVA:
n_iterations = 15

mom_solver = make_mom_solver_diva(n_iterations, mode="DIVA")

u_va_divadiva, u_vv_divadiva, mu_va, mu_vv, dudz, beta, beta_eff, zs, itns, res, resrat = mom_solver(u_va_init, dudz_init, z_coordinates)


plotboth(h_trial, b, u_va_divadiva)

##SSA:
#n_iterations = 15
#mom_solver = make_mom_solver_ssa(n_iterations)
#
#u_va_ssa, mu_end, beta_end, h_end, its, res_end, resrat_end = mom_solver(u_trial, h_trial)




#plt.plot(u_va_ssa*siy, label="ssa")
#plt.plot(u_va_divassa*siy, label="diva-ssa")
#plt.plot(u_va_divadiva*siy, label="diva-diva")
#plt.legend()
#plt.show()
#
#raise





#u_vv_avg = vertically_average(u_vv_divadiva, zs)

#plt.plot((u_vv_avg-u_va_divadiva)*siy)
#plt.show()
#raise




###########PLOTTING STUFF:

#mom_solver_ssa = make_mom_solver_diva(n_iterations, mode="SSA")
#u_va_ssa,_,_,_,_,_,_,_,_,_,_ = mom_solver_ssa(u_va_init, dudz_init, z_coordinates)

#percentage_diff_from_vert_mean = (100/u_va[...,None])*(u_vv-u_va[...,None])
#va_alt = vertically_average(u_vv, z_coordinates)
#percentage_diff_from_vert_mean = (100/va_alt[...,None])*(u_vv-va_alt[...,None])

#print(np.array2string(np.array(100*(va_alt-u_va)/u_va), formatter={'float_kind': lambda x: f"{x:.2f}"}))
#raise




diff_from_va = u_vv_divadiva - u_va_divassa[...,None]
#diff_from_va = u_vv_divadiva - u_va_divadiva[...,None]



X, Z = np.meshgrid(x, np.arange(n_levels), indexing='ij')  # (n, 11)

# Replace Z with the actual z_coords from your data
Z = z_coordinates  # shape (n, 11)


#percentage_diff_from_ssa = (100/u_va_ssa[...,None])*(u_vv-u_va_ssa[...,None])


#plt.figure(figsize=(8, 4))
#contour = plt.contourf(X, Z, u_vv*3.15e7, levels=10001, cmap='RdYlBu_r', vmin=0, vmax=5e2)
#plt.colorbar(contour, label='Speed (m/y)')
#plt.ylabel('Elevation (m)')
#plt.show()


# Plot difference in vert profile from mean
plt.figure(figsize=(8, 4))
#contour = plt.contourf(X, Z, percentage_diff_from_vert_mean, levels=101, cmap='RdBu_r', vmin=-25, vmax=25)
contour = plt.contourf(X, Z, diff_from_va*siy, levels=101, cmap='RdBu_r', vmin=-200, vmax=200)
#plt.colorbar(contour, label='Speed difference from vertical average (m/a)')
plt.colorbar(contour, label='Speed difference from SSA (m/a)')
plt.ylabel('Elevation (m)')
for i in range(z_coordinates.shape[-1]):
    plt.plot(x, z_coordinates[...,i], c="k", alpha=0.1)
plt.show()




h = z_coordinates[...,-1]-z_coordinates[...,0]
s_gnd = h + b #b is globally defined
s_flt = h*(1-rho/rho_w)
s = jnp.maximum(s_gnd, s_flt)

is_floating = s_flt > s_gnd
ffci = jnp.where(jnp.any(is_floating), jnp.argmax(is_floating), n)

plt.plot(u_vv_divadiva[ffci-1,:]*3.15e7, z_coordinates[ffci-1,:])
plt.ylabel("Elevation (m)")
plt.xlabel("Speed at GL (m/a)")
plt.show()


raise


plotboth(h_trial, b, u_va)

raise

plt.plot(u_va)
plt.show()
raise

##test:
#
#base = jnp.zeros((1,))
#surface = jnp.ones_like(base)
#n_levels = 51
#
#v_coords, z_coords = define_z_coordinates(n_levels, base, surface)
#
#u_test = 1 + 10*(((z_coords-base)/(surface-base))**3)
#
##plt.plot((z_coords[0,:]-base)/(surface-base), u_test[0,:])
##plt.show()
##raise
#
##plt.figure(figsize=(5,5))
##for i in range(u_test.shape[-1]):
##    plt.plot(u_test[:,i])
##plt.show()
##raise
#
#u_vi = vertically_integrate(u_test, z_coords, preserve_structure=False)
#
#u_vi_true = jnp.zeros_like(base) + 3.5
#
#
#print(u_vi)
#print(u_vi_true)
#print(u_vi - u_vi_true)
#raise





