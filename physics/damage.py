#1st party
import sys
import os

#3rd party
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import vertically_average
from grid import principal_resistive_stress_function,\
                 cc_resistive_and_deviatoric_stress_tensors,\
                 cc_deviatoric_stress_tensor_cf_safe

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_layered_advection_stepper

#Huth and Duddu epdf/10.1029/2020MS002292

@jax.jit
def macaulay_bracket(field):
    return jnp.maximum(field, 0)


@jax.jit
def overburden_pressure(z_coords):
    #NOTE: z_coords[..., -1] should be the surface and z_coords[..., 0] the base.
    return c.RHO_I * c.g * (z_coords[..., -1][..., None] - z_coords)


@jax.jit
def two_component_cauchy_stress(rst, z_coordinates):
    #rst: 2x2, vertically averaged resistive stress tensor
    return rst[..., None] - overburden_pressure(z_coordinates)

@jax.jit
def water_pressure(z_coords):
    return c.RHO_W * c.g * jnp.maximum(-z_coords, 0)

@jax.jit
def vertically_average_damage(damage, z_coords):
    return jnp.minimum(vertically_average(damage, z_coords), c.dmg.vaD_max)


#@jax.jit
#def hayhurst_stress_fct(dst, damage,
#                    z_coords,
#                    alpha=0.21, 
#                    beta=0.63,
#                    lambda_=0.16):
#    """
#    dst = deviatoric_stress_tensor
#    """
#    #NOTE NOTE NOTE: Need to ensure basal water pressure only included over
#    #floating ice.
#
#    #Ignoring water pressure for now. do - water_pressure(z_coords) -\
#    p_eff = overburden_pressure(z_coords) - water_pressure(z_coords) -\
#            ((dst[:,:,0,0] + dst[:,:,1,1])[..., None])#/(1-damage)
#
#    pds = 0.5 * (dst[:,:,0,0] + dst[:,:,1,1] + jnp.sqrt(
#                           (dst[:,:,0,0] + dst[:,:,1,1])**2 -\
#                           4*(dst[:,:,0,0]*dst[:,:,1,1] - dst[:,:,0,1]**2)
#                                             )
#                 )[..., None]/(1-damage)
#
#    term1 = alpha * (pds - p_eff)
#
#    term2 = beta * jnp.sqrt(1.5 * (dst[:,:,0,0]**2 +\
#                                   2*dst[:,:,1,0]**2 +\
#                                   dst[:,:,1,1]**2 +\
#                                   (dst[:,:,0,0] + dst[:,:,1,1])**2)[..., None]
#                           )
#
#    term3 = -3 * lambda_ * p_eff
#
#    return term1 + term2 + term3

@jax.jit
def hayhurst_stress_fct(dst, damage,
                    z_coords,
                    alpha=0.21, 
                    beta=0.63,
                    lambda_=0.16):
    """
    dst = deviatoric_stress_tensor
    """
    #NOTE NOTE NOTE: Need to ensure basal water pressure only included over
    #floating ice.

    #Ignoring water pressure for now. do - water_pressure(z_coords) -\
    p_eff = overburden_pressure(z_coords) - water_pressure(z_coords) -\
            ((dst[:,:,0,0] + dst[:,:,1,1])[..., None])#/(1-damage)

    pds = 0.5 * (dst[:,:,0,0] + dst[:,:,1,1] + jnp.sqrt(
                           (dst[:,:,0,0] + dst[:,:,1,1])**2 -\
                           4*(dst[:,:,0,0]*dst[:,:,1,1] - dst[:,:,0,1]**2)
                                             )
                 )[..., None]#/(1-damage)

    term1 = alpha * (pds - p_eff)

    term2 = beta * jnp.sqrt(1.5 * (dst[:,:,0,0]**2 +\
                                   2*dst[:,:,1,0]**2 +\
                                   dst[:,:,1,1]**2 +\
                                   (dst[:,:,0,0] + dst[:,:,1,1])**2)[..., None]
                           )

    term3 = -3 * lambda_ * p_eff

    return term1 + term2 + term3


@jax.jit
def vertically_averaged_damage_nye(principal_rs, z_coordinates):
    overburden = overburden_pressure(z_coordinates)
    vv_principal_cauchy_stress = principal_rs[..., None] - overburden

    #print(z_coordinates.shape)
    #print(principal_rs.shape)
    #print(overburden.shape)
    #print(vv_principal_cauchy_stress.shape)
    #raise

    indicator_nye = jnp.where(vv_principal_cauchy_stress>0, 1, 0)
    
    #plt.imshow(np.transpose(np.array(indicator_nye[75,:,:]))[::-1, :])
    #plt.colorbar()
    #plt.show()

    #plt.imshow(np.transpose(np.array(vv_principal_cauchy_stress[75,:,:]))[::-1, :])
    #plt.colorbar()
    #plt.show()
    #raise


    return vertically_average(indicator_nye, z_coordinates)


def nye_vertical_damage_function(ny, nx, dy, dx,
                                 add_uv_ghost_cells,
                                 add_s_ghost_cells,
                                 cc_gradient,
                                 mucoef_0, temp_cc=None):

    prs_function = principal_resistive_stress_function(ny, nx, dy, dx,
                                                       add_uv_ghost_cells,
                                                       add_s_ghost_cells,
                                                       cc_gradient,
                                                       mucoef_0, temp_cc=temp_cc)
    def va_damage_nye(q, u, v, z_coords):

        prs = prs_function(q, u, v, z_coords[..., -1] - z_coords[..., 0])

        #plt.imshow(prs)
        #plt.colorbar()
        #plt.show()

        return jnp.minimum(vertically_averaged_damage_nye(prs, z_coords), 0.9)
    
    return jax.jit(va_damage_nye)
    #return va_damage_nye

@jax.jit
def vertically_averaged_damage_nye_surfandbase(principal_rs, z_coordinates):
    overburden = overburden_pressure(z_coordinates)
    wp = water_pressure(z_coordinates)

    vv_principal_cauchy_stress_surf = principal_rs[..., None] - overburden
    vv_principal_cauchy_stress_base = principal_rs[..., None] - overburden + wp

    indicator_nye = jnp.where(
        (vv_principal_cauchy_stress_surf>0) | (vv_principal_cauchy_stress_base>0), 1, 0
    )

    return vertically_average(indicator_nye, z_coordinates)


def nye_vertical_damage_surfandbase_function(ny, nx, dy, dx,
                                 add_uv_ghost_cells,
                                 add_s_ghost_cells,
                                 cc_gradient,
                                 mucoef_0, temp_cc=None):

    prs_function = principal_resistive_stress_function(ny, nx, dy, dx,
                                                       add_uv_ghost_cells,
                                                       add_s_ghost_cells,
                                                       cc_gradient,
                                                       mucoef_0, temp_cc=temp_cc)
    def va_damage_nye(q, u, v, z_coords):

        prs = prs_function(q, u, v, z_coords[..., -1] - z_coords[..., 0])

        return vertically_averaged_damage_nye_surfandbase(prs, z_coords)

    return jax.jit(va_damage_nye)
    #return va_damage_nye


def isotropic_creep_damage_source_function(ny, nx, dy, dx,
                                       add_uv_ghost_cells,
                                       add_s_ghost_cells,
                                       cc_gradient,
                                       mucoef_0, temp_cc=None):

    dst_function = cc_deviatoric_stress_tensor_cf_safe(ny, nx, dy, dx,
                                                       add_uv_ghost_cells,
                                                       add_s_ghost_cells,
                                                       cc_gradient,
                                                       mucoef_0, temp_cc=temp_cc)
    def source_term(q, u, v, damage, z_coords):

        dst = dst_function(q*0, u, v, z_coords[..., -1] - z_coords[..., 0])

        #3D field
        hayhurst_stress = hayhurst_stress_fct(dst, damage, z_coords)

        #plt.imshow(vertically_average(hayhurst_stress, z_coords))
        #plt.colorbar()
        #plt.show()
        #plt.close()

        #jax.debug.print("{x}",x=hayhurst_stress[:,:,-1])

        return c.dmg.B_STAR  * (1/(1-damage))**c.dmg.k_STAR *\
               macaulay_bracket(hayhurst_stress - c.dmg.sigma_th)**c.dmg.r
    
    return jax.jit(source_term)
    #return source_term


def make_isotropic_creep_damage_stepper(nx, ny, dx, dy,
                                        interp_cc_to_fc, 
                                        add_uv_ghost_cells,
                                        add_s_ghost_cells,
                                        cc_gradient,
                                        mucoef_0,
                                        method="PPM",
                                        temp_cc=None,
                                        max_n_shrinks=6):

    layered_advection_stepper = make_layered_advection_stepper(
        nx, ny, dx, dy, interp_cc_to_fc,
        add_uv_ghost_cells, add_s_ghost_cells,
        method=method, conservative=False)
    #single_layer_advection_stepper = make_advection_stepper(nx, ny, dx, dy, interp_cc_to_fc, 
    #                                               add_uv_ghost_cells, add_s_ghost_cells,
    #                                               method=method, conservative=False)

    source_fct = isotropic_creep_damage_source_function(ny, nx, dy, dx,
                                       add_uv_ghost_cells,
                                       add_s_ghost_cells,
                                       cc_gradient,
                                       mucoef_0, temp_cc=temp_cc)

    def icd_stepper(q, u, v, damage, z_coords, delta_t):
        source = source_fct(q, u, v, damage, z_coords)
        
        #plt.imshow(vertically_average(source, z_coords))
        #plt.colorbar()
        #plt.show()
        #plt.close()

        ice_mask = jnp.where((z_coords[..., -1] - z_coords[..., 0])>1e-3, 1, 0)[..., None]
        
        damage_init = damage
        va_damage_init = vertically_average_damage(damage, z_coords)
        
        def update_with_delta_t(delta_t):
            damage_trial = layered_advection_stepper(u.reshape(-1),
                                v.reshape(-1),
                                (z_coords[..., -1] - z_coords[..., 0]).reshape(-1),
                                damage, source, delta_t=delta_t)

            return jnp.minimum(c.dmg.D_max, damage_trial)*ice_mask
        
        def dD(damage_trial):
            va_damage = vertically_average(damage_trial, z_coords)
            return jnp.max(jnp.abs(va_damage - va_damage_init))

        def conditional_(state):
            _, _, n_shrinks, dD_val = state
            #jax.debug.print("dD: {x}", x=dD_val)
            return (dD_val >= c.dmg.dD_max) & (n_shrinks<max_n_shrinks)

        def loop_body(state):
            _, dt, n_shrinks, _ = state
            
            dt_new = dt / c.dmg.dt_shrink_factor

            damage_trial = update_with_delta_t(dt_new)
            return (damage_trial, dt_new, n_shrinks + 1, dD(damage_trial))


        #plt.imshow(source[:,:,-1])
        #plt.colorbar()
        #plt.show()
        #plt.close()

        damage_trial0 = update_with_delta_t(delta_t)
        init_state = (damage_trial0, delta_t, 0, dD(damage_trial0))
 
        damage_final, delta_t_used, n_shrinks, dD_final = jax.lax.while_loop(
            conditional_, loop_body, init_state)


        #Rupture
        damage_final = jnp.where(damage_final>c.dmg.D_cr, c.dmg.D_max, damage_final)

        jax.debug.print("max damage layer: {x}", x=jnp.max(damage_final))

        #damage = step_with_delta_t(u, v, z_coords, damage, delta_t, source)

        va_damage_final = vertically_average_damage(damage_final, z_coords)
        va_damage_final = jnp.where(va_damage_final>c.dmg.vaD_cr, c.dmg.vaD_max, va_damage_final)
        
        jax.debug.print("max vertically averaged damage: {x}", x=jnp.max(va_damage_final))
        jax.debug.print("n shrinks used: {x}", x=n_shrinks)

        #plt.imshow(damage[:,:,-1])
        #plt.colorbar()
        #plt.show()
        #plt.close()

        return damage_final, va_damage_final, delta_t_used

    return jax.jit(icd_stepper)
    #return icd_stepper

#def make_isotropic_creep_damage_stepper(nx, ny, dx, dy,
#                                        interp_cc_to_fc,
#                                        add_uv_ghost_cells,
#                                        add_s_ghost_cells,
#                                        cc_gradient,
#                                        mucoef_0,
#                                        method="PPM"):
#    layered_advection_stepper = make_layered_advection_stepper(
#        nx, ny, dx, dy, interp_cc_to_fc,
#        add_uv_ghost_cells, add_s_ghost_cells,
#        method=method, conservative=False)
#
#    source_fct = isotropic_creep_damage_source_function(
#        ny, nx, dy, dx,
#        add_uv_ghost_cells, add_s_ghost_cells,
#        cc_gradient, mucoef_0, temp_cc=None)
#
#    def icd_stepper(q, u, v, damage, h, z_coords, delta_t):
#        source = source_fct(q, u, v, damage, z_coords)
#        damage = layered_advection_stepper(
#            u.reshape(-1), v.reshape(-1), h.reshape(-1),
#            damage, source, delta_t=delta_t)
#        return jnp.minimum(damage, 0.9)
#    return jax.jit(icd_stepper)


