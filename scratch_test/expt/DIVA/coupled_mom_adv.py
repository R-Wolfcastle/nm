from pathlib import Path
import sys


sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
from plotting_stuff import *



from PIL import Image

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import vmap


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


def make_adv_residual(dt, accumulation):
    
    def adv_res(u, h, h_old):
        s_gnd = h + b
        s_flt = h*(1-rho/rho_w)

        acc = jnp.where(s_gnd<s_flt, accumulation, accumulation)
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

    v_coords_1d = jnp.linspace(0,1,n_levels)**2
    #v_coords_3d = jnp.broadcast_to(v_coords_1d, (base.shape[0], base.shape[1], n_levels))
    
    v_coords_expanded = v_coords_1d[None, :] 
    #The ellipses are moot because this wouldn't work for 3d - it would have to be [None, None, :]
   
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


def interp_field_onto_new_zs(field, zs, zs_new):
    #Note: the in_axes at the bottom makes this specific to 2d arrays
    #to make it more general, let's just flatten the arrays and then un-flatten
    #them again (rather than nesting vmaps... which gets confusing).

    def interp_single_point(field_i, zs_i, zs_i_new):
        indices_hi = jnp.searchsorted(zs_i, zs_i_new, side="right")
        indices_lo = jnp.clip(indices_hi - 1, 0, zs_i.shape[0] - 2)

        zs_0 = zs_i[indices_lo]
        zs_1 = zs_i[indices_lo + 1]
        ys_0 = field_i[indices_lo]
        ys_1 = field_i[indices_lo + 1]

        #linear_iterp:
        ws = (zs_i_new - zs_0) / (zs_1 - zs_0 + 1e-12)
    
        return ys_0 + ws * (ys_1 - ys_0)

    # Vectorize over spatial dimension(s)
    interp_fn = vmap(interp_single_point, in_axes=(0, 0, 0))
    return interp_fn(field, zs, zs_new)


def interp_fields_onto_new_zs(fields, z, z_new):
    #Same as the above, but for multiple fields.
    #we want to not have to keep doing the searchsort and stuff for 
    #each field, but I'll fix that at a later date..

    def interp_single_point(field_i, zs_i, zs_i_new):
        indices_hi = jnp.searchsorted(zs_i, zs_i_new, side="right")
        indices_lo = jnp.clip(indices_hi - 1, 0, zs_i.shape[0] - 2)

        zs_0 = zs_i[indices_lo]
        zs_1 = zs_i[indices_lo + 1]
        ys_0 = field_i[indices_lo]
        ys_1 = field_i[indices_lo + 1]

        #linear_iterp:
        ws = (zs_i_new - zs_0) / (zs_1 - zs_0 + 1e-12)
    
        return ys_0 + ws * (ys_1 - ys_0)

    #Vectorise over spatial dimension(s)
    interp_fn = vmap(interp_single_point, in_axes=(1, 0, 0))
    #Vectorise over different fields
    interp_fn_whole = vmap(interp_fn, in_axes=(0,None,None))

    return interp_fn_whole(fields, zs, zs_new)


def make_full_solver(iterations, timestep, acc=0, rheology_n=3, mode="DIVA", compile_=False):

    mom_res = make_linear_momentum_residual()
    adv_res = make_adv_residual(timestep, acc)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))


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
        mu_vv = B * (jnp.abs(dudx)[...,None]**2 + 0.25*dudz**2 + epsilon_visc**2)**(0.5*(1/rheology_n - 1))
        
        mu_va = vertically_average(mu_vv, zs)

        return mu_vv, mu_va


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


        beta_eff = beta / (1 + beta*f2)
        
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

        pre_add = (u_va / (1 + beta*f2))[...,None] 
        prefactor_exp = (beta_eff * u_va)[...,None]

        u_vv = pre_add + prefactor_exp * f1_vv

        return u_vv

        
    #equation 3
    def setup_and_solve_full_linear_prob(u_va, mu_va, beta_eff, zs, h_old):
        h = zs[...,-1]-zs[...,0]

        jac_mom_res = jac_mom_res_fn(u_va, h, mu_va, beta_eff)
        jac_adv_res = jac_adv_res_fn(u_va, h, h_old)

        full_jacobian = jnp.block(
                                  [ [jac_mom_res[0], jac_mom_res[1]],
                                    [jac_adv_res[0], jac_adv_res[1]] ]
                                 )
    
        rhs = jnp.concatenate((-mom_res(u_va, h, mu_va, beta_eff), -adv_res(u_va, h, h_old)))
    
        dvar = lalg.solve(full_jacobian, rhs)
    
        u_va = u_va.at[:].set(u_va+dvar[:n])
        h = h.at[:].set(h+dvar[n:])

        momres = jnp.max(jnp.abs(mom_res(u_va, h, mu_va, beta_eff)))

        return u_va, h, momres


    def setup_and_solve_linear_prob(u_va, mu_va, beta_eff, zs):
        h = zs[...,-1]-zs[...,0]

        jac_mom_res = jac_mom_res_fn(u_va, h, mu_va, beta_eff)

        delta_u = lalg.solve(jac_mom_res, -mom_res(u_va, h, mu_va, beta_eff))

        u_va = u_va + delta_u

        residual = jnp.max(jnp.abs(mom_res(u_va, h, mu_va, beta_eff)))

        return u_va, residual


    def step_ssa(state):
        h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i, res, resrat = state

        dudz = jnp.zeros_like(dudz)

        #update viscosity
        mu_vv, mu_va = new_viscosity(u_va, dudz, zs)
        
        jax.debug.print("{}",mu_va)

        #update beta_eff
        beta_eff = new_beta(u_vv[...,0], zs)

        #solve linear problem
        u_va, h_new, residual = setup_and_solve_full_linear_prob(u_va, mu_va, beta_eff, zs, h_old)
        resrat = res/residual


        zs = define_z_coordinates(n_levels, h_new)


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

        return h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i+1, residual, resrat


    def step(state):
        
        h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, i, res, resrat = state

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
        u_va, h_new, residual = setup_and_solve_full_linear_prob(u_va, mu_va, beta_eff, zs, h_old)
        resrat = res/residual

        #define new z coords and interpolate vv things onto them.
        zs_new = define_z_coordinates(n_levels, h_new)
        u_vv  = interp_field_onto_new_zs(u_vv, zs, zs_new)
        mu_vv = interp_field_onto_new_zs(mu_vv, zs, zs_new)

        #update dudz
        dudz = new_dudz(mu_vv, u_va, beta_eff, zs_new)
        #jax.debug.print("dudz: {}", dudz[...,1])

        #update u_vv
        u_vv = new_u_vv(dudz, u_va, beta, beta_eff, zs_new, mu_vv, f2)
        #NOTE: Something's going quite wrong as u_vv inconsistent with u_va (i.e.
        #avg of u_vv not the same as u_va...)

        #jax.debug.print("res: {}", residual)
        #jax.debug.print("resrat: {}", resrat)
        
        #jax.debug.print("u_va: {}", u_va)
        ##jax.debug.print("u_vv: {}", u_vv)
        #jax.debug.print("u_va from vi of u_vv: {}", vertically_average(u_vv, zs))

        #jax.debug.print("u_va error: {}", (u_va-vertically_average(u_vv, zs))/u_va)


        return h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs_new, i+1, residual, resrat


    def continue_condition(state):
        _,_,_,_,_,_,_,_,_, i ,_,_ = state
        return i<iterations


    def iterator(u_va_init, dudz_init, zs):
        h_old = zs[...,-1]-zs[...,0]
        
        residual_ratio = jnp.inf
        residual = 1


        dummy_small = jnp.zeros_like(u_va_init)
        dummy_large = jnp.zeros_like(dudz_init)
        
        beta_init = dummy_small
        u_vv_init = dummy_large + u_va_init[...,None]

        initial_state = h_old, u_va_init, u_vv_init, dummy_small, dummy_large,\
                        dudz_init, beta_init, dummy_small, zs, 0, residual, residual_ratio

        #h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, itns, residual, residual_ratio

        if mode=="DIVA":
            internal_step = step
        elif mode=="SSA":
            internal_step = step_ssa
        else:
            print("Invalid mode given, defaulting to DIVA")
            internal_step = step


        out_state = jax.lax.while_loop(continue_condition, step, initial_state)
        #out_state = jax.lax.while_loop(continue_condition, step_ssa, initial_state)
        
        return out_state

    return iterator


def make_mom_solver_diva(iterations, rheology_n=3, mode="DIVA", compile_=False):

    mom_res = make_linear_momentum_residual()

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
        mu_vv = B * (jnp.abs(dudx)[...,None]**2 + 0.25*dudz**2 + epsilon_visc**2)**(0.5*(1/rheology_n - 1))
        
        mu_va = vertically_average(mu_vv, zs)

        return mu_vv, mu_va


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

        beta_eff = beta / (1 + beta*f2)
        
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
        
        jax.debug.print("{}",mu_va)

        #update beta_eff
        beta_eff = new_beta(u_vv[...,0], zs)

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
        jax.debug.print("dudz: {}", dudz[...,1])

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


        out_state = jax.lax.while_loop(continue_condition, step, initial_state)
        #out_state = jax.lax.while_loop(continue_condition, step_ssa, initial_state)
        
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
    
        mu_va = B * (jnp.abs(dudx) + epsilon_visc)**(-2/3)

        return mu_va


    #just keeping this here to make it look more like the diva solver
    def new_beta_eff(u, h):
        
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
        beta = new_beta_eff(u_va, h)

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


def make_joint_qn_solver(C, B, iterations, dt, acc=0, compile_=False):
    
    mom_res = make_linear_momentum_residual()
    adv_res = make_adv_residual(dt, acc)

    jac_mom_res_fn = jacfwd(mom_res, argnums=(0,1))
    jac_adv_res_fn = jacfwd(adv_res, argnums=(0,1))
    
    def new_mu(u):
    
        dudx = jnp.zeros((n+1,))
        dudx = dudx.at[1:-1].set((u[1:] - u[:-1])/dx)
        dudx = dudx.at[-1].set(dudx[-2])
        #set reflection boundary condition
        dudx = dudx.at[0].set(2*u[0]/dx)
    
        mu_nl = B * (jnp.abs(dudx)+epsilon_visc)**(-2/3)

        #mu_nl = B * (epsilon_visc**(-2/3))

        return mu_nl

    def new_beta(u, h):

        grounded_mask = jnp.where((b+h)<(h*(1-rho/rho_w)), 0, 1)

        #beta = C * ((jnp.abs(u))**(-2/3)) * grounded_mask
        beta = C * (1/(jnp.abs(u)**(2/3) + (1e-8)**(2/3))) * grounded_mask
        #beta = C * (1e-6)**(-2/3) * grounded_mask

        return beta

    def continue_condition(state):
        _,_,_, i,res,resrat = state
        return i<iterations


    def step(state):
        u, h, h_init, i, prev_res, prev_resrat = state

        beta = new_beta(u, h)
        mu = new_mu(u)

        jac_mom_res = jac_mom_res_fn(u, h, mu, beta)
        jac_adv_res = jac_adv_res_fn(u, h, h_init, bmr)

        full_jacobian = jnp.block(
                                  [ [jac_mom_res[0], jac_mom_res[1]],
                                    [jac_adv_res[0], jac_adv_res[1]] ]
                                  )
    
        rhs = jnp.concatenate((-mom_res(u, h, mu, beta), -adv_res(u, h, h_init, bmr)))
    
        dvar = lalg.solve(full_jacobian, rhs)
    
        u = u.at[:].set(u+dvar[:n])
        h = h.at[:].set(h+dvar[n:])

        #TODO: add one for the adv residual too...
        res = jnp.max(jnp.abs(mom_res(u, h, mu, beta))) 

        return u, h, h_init, i+1, res, prev_res/res


    def iterator(u_init, h_init):    

        resrat = np.inf
        res = np.inf

        initial_state = u_init, h_init, h_init, 0, res, resrat

        u, h, h_init, itn, res, resrat = jax.lax.while_loop(continue_condition, step, initial_state)

        return u, h, res


    if compile_:
        return jax.jit(iterator)
    else:
        return iterator










n = 101
l = 1_800_000
x = jnp.linspace(0, l, n)
dx = x[1]-x[0]

rho = 900
rho_w = 1000

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



n_levels = 51

z_coordinates = define_z_coordinates(n_levels, h_trial)


#plt.figure()
#for i in range(z_coordinates.shape[-1]):
#    plt.plot(z_coordinates[...,i])
#plt.show()
##plotgeoms([z_coordinates[...,i] for i in range(z_coordinates.shape[-1])], b, z_coordinates.shape[-1])
#raise




##SSA:
#n_iterations = 5
#mom_solver = make_mom_solver_ssa(n_iterations)
#
#u_va_end, mu_end, beta_end, h_end, its, res_end, resrat_end = mom_solver(u_trial, h_trial)
#
#plotboth(h_trial, b, u_va_end)
#
#raise





#DIVA:
n_iterations = 15
timestep = 1e9
#accumulation = 2/(3.15e7) #2 m/yr
accumulation = 0
solver = make_full_solver(n_iterations, timestep, acc=accumulation)


u_va_init = u_trial
dudz_init = jnp.zeros((n,n_levels))


zs = z_coordinates.copy()
u_va = u_va_init.copy()
dudz = dudz_init.copy()
hs = []
us = []
for i in range(250):
    print("Year: {}".format(int((i+1)*timestep/(3.15e7))))
    h_old, u_va, u_vv, mu_va, mu_vv, dudz, beta, beta_eff, zs, itns, res, resrat = solver(u_va, dudz, zs)
    
    hs.append(zs[...,-1]-zs[...,0])
    us.append(u_va)
#plotboths(hs, b, us, len(hs))


siy=3.15e7

X, Z = np.meshgrid(x, np.arange(n_levels), indexing='ij')  # (n, 11)

# Replace Z with the actual z_coords from your data
Z = zs

plt.figure(figsize=(8, 4))
#contour = plt.contourf(X, Z, u_vv*siy, levels=101, cmap='gnuplot2', vmin=0, vmax=10000)
contour = plt.contourf(X, Z, (u_vv-u_vv[...,0][...,None])*siy, levels=101, cmap='gnuplot2', vmin=0, vmax=1)
plt.colorbar(contour, label='Speed (m/a)')
plt.ylabel('Elevation (m)')
for i in range(z_coordinates.shape[-1]):
    plt.plot(x, zs[...,i], c="white", alpha=0.15)
plt.show()


#jnp.save("./possible_starting_thk.npy", hs[-1])


raise

