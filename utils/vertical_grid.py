#1st party
import sys

#3rd party
import jax
import jax.numpy as jnp

#local apps
sys.path.insert(0,"/Users/eartsu/new_model/testing/nm/utils/")
import constants_years as c



def define_z_coordinates(bed, thk, n_levels):
    s_gnd = bed + thk
    s_flt = thk*(1-rho/rho_w)

    surface = jnp.maximum(s_gnd, s_flt)
    base = surface-thk

    base = jnp.maximum(base, bed) #just to make sure

    #quadratic spacing:
    #v_coords_1d = jnp.linspace(0,1,n_levels)**2
    #linear spacing:
    v_coords_1d = jnp.linspace(0,1,n_levels)
    
    v_coords_expanded = v_coords_1d[None, None, :] 
   
    base_expanded = base[:, :, None]
    thk_expanded = thk[:, :, None]

    z_coords = base_expanded + thk_expanded*v_coords_expanded
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



