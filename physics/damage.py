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
from grid import principal_resistive_stress_function


@jax.jit
def overburden_pressure(z_coords):
    #NOTE: z_coords[..., -1] should be the surface and z_coords[..., 0] the base.
    return c.RHO_I * c.g * (z_coords[..., -1][..., None] - z_coords)


@jax.jit
def two_component_cauchy_stress(rst, z_coordinates):
    #rst: 2x2, vertically averaged resistive stress tensor
    return rst[..., None] - overburden_pressure(z_coordinates)


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
def water_pressure(z_coords):
    return c.RHO_W * c.g * jnp.maximum(-z_coords, 0)


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

    #return jax.jit(va_damage_nye)
    return va_damage_nye
