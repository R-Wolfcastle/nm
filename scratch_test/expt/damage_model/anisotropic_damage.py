#1st party
import os
import sys

#3rd party
import jax
from jax import custom_vjp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from jax.scipy.signal import convolve2d


nm_home = os.environ['NM_HOME']

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import define_z_coordinates, vertically_average, interp_field_onto_new_zs
from grid import (add_ghost_cells_fcts, cc_gradient_function, cc_vel_gradient_function,
                  interp_cc_with_ghosts_to_fc_function, fc_velocity_gradient_function_noextrap,
                  fc_viscosity_function_new_givenT_noextrap_dt, gl_unaware_driving_stress_function,
                  beta_function)
from sparsity_utils import basis_vectors_and_coords_2d_square_stencil, \
                           make_sparse_jacrev_fct_shared_basis_new
from thermo import B_from_T
from standard_domains import mismip_domain_symm
from plotting_stuff import make_plot_mismip_field_function

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_layered_advection_stepper, make_advection_stepper, \
                              print_residual_things
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp_given_csr

sys.path.insert(1, os.path.join(nm_home, 'physics'))
from damage import macaulay_bracket, overburden_pressure, water_pressure, \
                   vertically_average_damage



@jax.jit
def symmetric_2x2_inverse(A11, A22, A12):
    """(I - D)^-1 for D given as (D11, D22, D12). inverse of
    [[1-D11, -D12], [-D12, 1-D22]]."""
    m11, m22, m12 = 1 - A11, 1 - A22, -A12
    det = m11 * m22 - m12 * m12
    det = jnp.where(jnp.abs(det) < 1e-12, jnp.sign(det) * 1e-12 + 1e-12, det)
    inv11 = m22 / det
    inv22 = m11 / det
    inv12 = -m12 / det
    return inv11, inv22, inv12


@jax.jit
def symmetric_2x2_product_symmetric_part(A11, A22, A12, S11, S22, S12):
    """0.5*(A@S + S@A) for two symmetric 2x2 tensors A=(A11,A22,A12),
    S=(S11,S22,S12). Used for both the stress transform (Eq. 6/18, with
    A = (I-D)^-1) and the strain-rate transform (Eq. 7, with A = I-D)."""
    #Note that A and S both have to be symmetric for this to work out.
    out11 = A11 * S11 + A12 * S12
    out22 = A22 * S22 + A12 * S12
    out12 = 0.5 * (A11 * S12 + A12 * S22 + S11 * A12 + S12 * A22)
    return out11, out22, out12


@jax.jit
def evals_and_evecs_of_symm_2x2_tensor(S11, S22, S12, eps=1e-12):
    """Evals and evec (components) for a symmetric 2x2 tensor.
    Returns (s1, s2, cos_theta, sin_theta), where s1 >= s2."""

    trace = S11 + S22
    det = S11 * S22 - S12 * S12
    #discriminant is 1/4 * (b^2-4*a*c)
    disc = jnp.maximum(0.25 * trace * trace - det, 0.0)
    #root is 1/2 * sqrt(b^2-4*a*c)
    root = jnp.sqrt(disc)
    #quadratic formula to find eigenvalues
    s1 = 0.5 * trace + root
    s2 = 0.5 * trace - root

    #principal eigenvector is (cos(theta), sin(theta))
    #and second evec is (-sin(theta), cos(theta))
    angle = 0.5 * jnp.arctan2(2 * S12, S11 - S22 + eps)
    cos_t = jnp.cos(angle)
    sin_t = jnp.sin(angle)
    return s1, s2, cos_t, sin_t


@jax.jit
def project_damage_tensor_to_bounds(D11, D22, D12, D_min, D_max):
    """clip the actual eigenvalues of the (vertically-averated) 2x2
    in-plane damage tensor to [D_min, D_max], then reconstruct D11, D22,
    D12 from the clipped eigenvalues and the (unchanged) eigenvector
    directions. Clipping D11 and D22 independently while leaving D12
    unconstrained does NOT keep the tensor physical. D12 can still push
    the true principal damage <D1> arbitrarily far past D_max even when
    D11, D22 are each individually below it, and can drive (I-D) toward
    singularity, causing runaway blowup in the (I-D)^-1-based damage rate
    and in the momentum coupling. Such a bloody nuisance."""
    s1, s2, cos_t, sin_t = evals_and_evecs_of_symm_2x2_tensor(D11, D22, D12)
    s1_c = jnp.clip(s1, D_min, D_max)
    s2_c = jnp.clip(s2, D_min, D_max)

    D11_c = s1_c * cos_t**2 + s2_c * sin_t**2
    D22_c = s1_c * sin_t**2 + s2_c * cos_t**2
    D12_c = (s1_c - s2_c) * cos_t * sin_t

    return D11_c, D22_c, D12_c


@jax.jit
def hayhurst_stress_tensorial(eff11, eff22, eff12, p_eff,
                              alpha=c.dmg.alpha, beta=c.dmg.beta,
                              lambda_=c.dmg.lambda_):
    """Tensorial generalization of the existing (isotropic)
    hayhurst_stress_fct. Returns (chi, s1, cos_t, sin_t)."""
    s1, s2, cos_t, sin_t = evals_and_evecs_of_symm_2x2_tensor(eff11, eff22, eff12)

    term1 = alpha * (s1 - p_eff)
    term2 = beta * jnp.sqrt(1.5 * (eff11**2 + 2 * eff12**2 + eff22**2 +
                                   (eff11 + eff22)**2))
    term3 = -3 * lambda_ * p_eff

    chi = term1 + term2 + term3
    return chi, s1, cos_t, sin_t


def jaumann_corotation_function(dy, dx, add_uv_ghost_cells):
    """Returns rotation terms of source fct 
    the Huth Eq. 8 spin-correction WD - DW. D33 is unaffected bc no vertical shear
    """
    cc_gradient_vel = cc_vel_gradient_function(dy, dx, add_uv_ghost_cells)

    def corotation_terms(u, v, D11, D22, D12):
        du_dx, du_dy, dv_dx, dv_dy = cc_gradient_vel(u, v)
        w = 0.5 * (dv_dx - du_dy)
        w = w[..., None]

        #w = W21, so these are the components of W@D - D@W.
        rot11 = -2 * w * D12
        rot22 =  2 * w * D12
        rot12 =  w * (D11 - D22)
        return rot11, rot22, rot12

    return jax.jit(corotation_terms)



#Stuff usef for the "nonlocal" smoothing of damage rate: 

def _gaussian_kernel_2d(dx, dy, l_c, cutoff=3.0):
    """Compact Gaussian kernel for a fixed-grid approximation to the
    normalised material-point integral in the Huth paper. I'm not
    100% certain about this, but I hope it works out."""
    rx = max(1, int(np.ceil(cutoff * float(l_c) / float(dx))))
    ry = max(1, int(np.ceil(cutoff * float(l_c) / float(dy))))
    xs = jnp.arange(-rx, rx + 1, dtype=jnp.float64) * dx
    ys = jnp.arange(-ry, ry + 1, dtype=jnp.float64) * dy
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    kernel = jnp.exp(-0.5 * ((xx/l_c)**2 + (yy/l_c)**2))
    return kernel / jnp.sum(kernel)


def nonlocal_regularisation_function(ny, nx, dy, dx, l_c=c.dmg.l_c):
    """Layerwise masked Gaussian averaging, normalised by available ice.

    This prevents ice-free cells and the rectangular box boundary from
    diluting or spreading the margin source. It regularises only the local
    irreversible production tensor, never advection, rotation or rupture.
    """
    kernel = _gaussian_kernel_2d(dx, dy, l_c)
    
    @jax.jit
    def regularise(field_3d, ice_mask_2d):
        mask = ice_mask_2d.astype(field_3d.dtype)
        den = convolve2d(mask, kernel, mode="same", boundary="fill")
        den = jnp.maximum(den, 1.0e-12)
        def one_layer(field_2d):
            num = convolve2d(field_2d * mask, kernel,
                             mode="same", boundary="fill")
            return jnp.where(mask > 0, num / den, 0.0)
        return jax.vmap(one_layer, in_axes=2, out_axes=2)(field_3d)
    return regularise






@jax.jit
def effective_inplane_strain_rate(D11, D22, D12, D33, e11, e22, e12):
    """Eq. 7 in Alex's paper to compute the "undamaged" strain rate.
    You know... the strain rate as if there were no damage...
    """

    A11, A22, A12 = 1 - D11, 1 - D22, -D12
    temp11, temp22, temp12 = symmetric_2x2_product_symmetric_part(
                                       A11, A22, A12, e11, e22, e12)

    #The strain rate tensor has to have zero trace, so that has to be accounted for here
    #in this kind of roundabout way...

    e33 = -(e11 + e22)
    temp33 = (1 - D33) * e33

    trace = temp11 + temp22 + temp33
    e_t11 = temp11 - trace / 3
    e_t22 = temp22 - trace / 3
    e_t12 = temp12

    return e_t11, e_t22, e_t12



##############################################################################
##############################################################################
##############################################################################





def anisotropic_creep_damage_source_function(ny, nx, dy, dx,
                                             add_uv_ghost_cells,
                                             add_s_ghost_cells,
                                             cc_gradient,
                                             mucoef_0, temp_cc=None,
                                             gamma=c.dmg.gamma):

    if temp_cc is None:
        temp_cc = jnp.zeros((ny, nx)) + 263.15

    cc_gradient_vel = cc_vel_gradient_function(dy, dx, add_uv_ghost_cells)
    B_cc = B_from_T(temp_cc)   #(ny, nx). NOTE this is z-independent! NEed to
                               #fix at some point

    corotation_fct = jaumann_corotation_function(dy, dx, add_uv_ghost_cells)

    #As per, the equations and sections referenced here are from Huth et al., 2021
    #DOI: 10.1029/2020MS002292

    def source_term(q, u, v, D11, D22, D12, D33, z_coords):
        #build the "undamaged" deviatoric stress (Eq. 14) from the:
            #the raw strain-rate in the effective viscosity
            #the transformed "undamaged" strain rate
        #I'm not sure quite why it's like this - need to ask Alex/Ravi

        du_dx, du_dy, dv_dx, dv_dy = cc_gradient_vel(u, v)
        e11 = du_dx
        e22 = dv_dy
        e12 = 0.5 * (du_dy + dv_dx)

        #Raw (undamaged) viscosity, Eq. 15
        eps_e_sq = e11**2 + e22**2 + e11*e22 + e12**2 + c.EPSILON_VISC**2
        eta_raw = B_cc * mucoef_0 * eps_e_sq**(0.5*(1/c.GLEN_N - 1))

        e11_b, e22_b, e12_b = e11[..., None], e22[..., None], e12[..., None]
        eta_b = eta_raw[..., None]

        #Eq. 7: damage-transformed ("undamaged") strain rate, per layer.
        e_t11, e_t22, e_t12 = effective_inplane_strain_rate(
                                       D11, D22, D12, D33, e11_b, e22_b, e12_b)

        #Eq. 14: the "undamaged" deviatoric stress (ish). You know what I mean.
        sigma_p11 = 2 * eta_b * e_t11
        sigma_p22 = 2 * eta_b * e_t22
        sigma_p12 = 2 * eta_b * e_t12

        p_eff = overburden_pressure(z_coords) - water_pressure(z_coords) -\
                (sigma_p11 + sigma_p22)

        #Eq. 18: the "damaged" deviatoric stress (the (I-D)^-1 amplification of
        #the "undamaged" stress given by eq 14.
        inv11, inv22, inv12 = symmetric_2x2_inverse(D11, D22, D12)
        eff11, eff22, eff12 = symmetric_2x2_product_symmetric_part(
                                     inv11, inv22, inv12,
                                     sigma_p11, sigma_p22, sigma_p12)


        #Eq. 19: Computation of the Hayhurst stress! As well as the orienttions of the
        #principal stress vector (xi1 in Eq 9).
        chi, s1, cos_t, sin_t = hayhurst_stress_tensorial(eff11, eff22, eff12, p_eff)

        #Trace term in equation 9.
        tr_term = (inv11 * cos_t**2 + 2 * inv12 * cos_t * sin_t + inv22 * sin_t**2)

        #Eq. 9: Evaluation of subcritical creep production.
        #The separate brittle threshold is applied after transport,
        #following Section 3.5.
        rate_mag = c.dmg.B_STAR * macaulay_bracket(chi - c.dmg.sigma_th)**c.dmg.r *\
                   tr_term**c.dmg.k_STAR

        #Once a layer has reached the brittle threshold, its state is handled
        #by the rupture rule rather than continued creep accumulation.
        D1_current = evals_and_evecs_of_symm_2x2_tensor(D11, D22, D12)[0]
        rate_mag = jnp.where(D1_current < c.dmg.D_cr, rate_mag, 0.0)

        dD11_f = rate_mag * ((1 - gamma) + gamma * cos_t**2)
        dD22_f = rate_mag * ((1 - gamma) + gamma * sin_t**2)
        dD12_f = rate_mag * gamma * cos_t * sin_t
        dD33   = rate_mag * (1 - gamma)

        rot11, rot22, rot12 = corotation_fct(u, v, D11, D22, D12)
        #Return irreversible production separately. The nonlocal operator
        #acts on production only; objective rotation is local and added once.
        return dD11_f, dD22_f, dD12_f, dD33, rot11, rot22, rot12

    return jax.jit(source_term)



def make_anisotropic_creep_damage_stepper(nx, ny, dx, dy,
                                          interp_cc_to_fc,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          cc_gradient,
                                          mucoef_0,
                                          method="PPM",
                                          temp_cc=None,
                                          gamma=c.dmg.gamma,
                                          l_c=c.dmg.l_c,
                                          max_n_shrinks=6):

    #Need to make sure dont_allow_negative is set to False as it turns out that
    #D12 can be negative! That's a bit weird but of course actually fine as
    #it's the invariants of a tensor (like principal damages, here) that have 
    #physical meaning.
    layered_advection_stepper = make_layered_advection_stepper(
        nx, ny, dx, dy, interp_cc_to_fc,
        add_uv_ghost_cells, add_s_ghost_cells,
        method=method, conservative=False,
        dont_allow_negative=False)

    cc_gradient_vel = cc_vel_gradient_function(dy, dx, add_uv_ghost_cells)

    source_fct = anisotropic_creep_damage_source_function(ny, nx, dy, dx,
                                       add_uv_ghost_cells,
                                       add_s_ghost_cells,
                                       cc_gradient,
                                       mucoef_0, temp_cc=temp_cc,
                                       gamma=gamma)

    regularise = nonlocal_regularisation_function(ny, nx, dy, dx, l_c=l_c)
    
    #Max vals that the damage components take after rupture
    DMAX1, DMAX2, DMAX3 = 0.99, 0.98, 0.98

    def _d1_bar(D11, D22, D12, z_coords):
        D11_bar = vertically_average(D11, z_coords)
        D22_bar = vertically_average(D22, z_coords)
        D12_bar = vertically_average(D12, z_coords)
        s1, _, _, _ = evals_and_evecs_of_symm_2x2_tensor(D11_bar, D22_bar, D12_bar)
        return s1

    def acd_stepper(q, u, v, D11, D22, D12, D33,
                   D11_bar_in, D22_bar_in, D12_bar_in, D33_bar_in,
                   z_coords, delta_t):

        #Compute the source term!
        (prod11, prod22, prod12, prod33,
         rot11, rot22, rot12) = source_fct(q, u, v, D11, D22,
                                           D12, D33, z_coords)

        h = z_coords[..., -1] - z_coords[..., 0]
        ice_mask_2d = jnp.where(h > 1e-3, 1.0, 0.0)
        ice_mask = ice_mask_2d[..., None]

        #Fully ruptured layers don't drive the creep law or the
        #adaptive time step. Otherwise (I-D)^-1 reaches about 100 at D1=0.99,
        #and the fourth-power factor creates a bloody massive rate multiplier
        #giving us tiny timesteps.
        D1_old = evals_and_evecs_of_symm_2x2_tensor(D11, D22, D12)[0]
        active_layer = ((D1_old < c.dmg.D_cr) & (ice_mask > 0)).astype(D11.dtype)

        #Huth's adaptive measure uses vertically averaged LOCAL damage. Keep
        #the pre-nonlocal production, but only on still-subcritical layers.
        #P
        local_prod11 = prod11 * active_layer
        local_prod22 = prod22 * active_layer
        local_prod12 = prod12 * active_layer
        local_prod33 = prod33 * active_layer

        #Equation 24: Huth-style "nonlocality" bit: regularise the local production 
        #increment _within_ each layer individually, with local normalisation by where
        #thre is ice. Basically smooth with a gaussian using JAX SciPy.
        #It's not that clear what the size of the fracture
        #process zone is, but apparently ought to be large enough to remove grid-
        #dependence. That's maybe a couple of km? Set in c.dmg.l_c
        prod11 = regularise(local_prod11, ice_mask_2d)
        prod22 = regularise(local_prod22, ice_mask_2d)
        prod12 = regularise(local_prod12, ice_mask_2d)
        prod33 = regularise(local_prod33, ice_mask_2d)

        #Add the rotation/spin terms. Note that the other additional bit of the
        #source term coming from the non-conservativism is added during the
        #advection step already.
        dD11, dD22 = prod11 + rot11, prod22 + rot22
        dD12, dD33 = prod12 + rot12, prod33

        #Eq. 21: control the step using the maximum principal
        #component of the VERTICALLY AVERAGED LOCAL damage increment. I screwed that up
        #before... Using the maximum single-layer rate is too restrictive when damage is
        #concentrated near the surface or base.
        local_prod11_va = vertically_average(local_prod11, z_coords)
        local_prod22_va = vertically_average(local_prod22, z_coords)
        local_prod12_va = vertically_average(local_prod12, z_coords)
        local_prod1_va_rate = evals_and_evecs_of_symm_2x2_tensor(
            local_prod11_va, local_prod22_va, local_prod12_va)[0]
        max_local_dav_rate = jnp.max(
            jnp.maximum(local_prod1_va_rate, 0.0) * ice_mask_2d)

        def update_with_delta_t(delta_t):
            u_flat = u.reshape(-1)
            v_flat = v.reshape(-1)
            h_flat = h.reshape(-1)

            #Advect with a source
            D11_t = layered_advection_stepper(u_flat, v_flat, h_flat, D11, dD11, delta_t=delta_t)
            D22_t = layered_advection_stepper(u_flat, v_flat, h_flat, D22, dD22, delta_t=delta_t)
            D33_t = layered_advection_stepper(u_flat, v_flat, h_flat, D33, dD33, delta_t=delta_t)
            D12_t = layered_advection_stepper(u_flat, v_flat, h_flat, D12, dD12, delta_t=delta_t)


            #Adjust things based on rupture criteria!

            #Layerwise brittle rupture. Alex's setup seems to use
            #Dcrit=0.6 and principal maxima 0.99, 0.98, 0.98. But he also mentions
            #possible values of 0.45 in Section 3.5
            D1_trial, D2_trial, cos_t, sin_t = evals_and_evecs_of_symm_2x2_tensor(
                D11_t, D22_t, D12_t)

            failed = D1_trial >= c.dmg.D_cr

            D1_new = jnp.where(failed, DMAX1, D1_trial)
            
            D2_target = (1.0 - gamma) * DMAX1
            D3_target = (1.0 - gamma) * DMAX1
            D2_new = jnp.where(failed, jnp.maximum(D2_trial, D2_target), D2_trial)
            D3_new = jnp.where(failed, jnp.maximum(D33_t, D3_target), D33_t)


            ##Where D1_trial is greater than critical val (0.6 ish), how far is D1_trial from
            ##the max val allowed (0.99 ish)
            #jump = jnp.where(failed,
            #                 jnp.maximum(DMAX1 - D1_trial, 0.0), 0.0)

            #D1_new = jnp.where(failed, DMAX1, D1_trial)
            #D2_new = jnp.where(failed,
            #    jnp.maximum(D2_trial, D2_trial + (1.0-gamma)*jump),
            #    D2_trial)
            #D3_new = jnp.where(failed,
            #    jnp.maximum(D33_t, D33_t + (1.0-gamma)*jump),
            #    D33_t)

            #Make extra sure!
            D1_new = jnp.clip(D1_new, 0.0, DMAX1)
            D2_new = jnp.clip(D2_new, 0.0, DMAX2)
            D3_new = jnp.clip(D3_new, 0.0, DMAX3)

            #Reproject into current coordinate system
            D11_t = (D1_new*cos_t**2 + D2_new*sin_t**2) * ice_mask
            D22_t = (D1_new*sin_t**2 + D2_new*cos_t**2) * ice_mask
            D12_t = ((D1_new-D2_new)*cos_t*sin_t) * ice_mask
            D33_t = D3_new * ice_mask
            return D11_t, D22_t, D12_t, D33_t

        #The target timestep 
        HUTH_DD_TARGET = 0.05

        #An alternative to Equation 23. Trial for the timestep.
        dt_damage = jnp.where(max_local_dav_rate > 0.0, 
                              HUTH_DD_TARGET / max_local_dav_rate, 
                              delta_t)

        #default to the CFL timestep if everything larger
        delta_t_used = jnp.minimum(delta_t, dt_damage)

        D11_final, D22_final, D12_final, D33_final = update_with_delta_t(delta_t_used)

        #n_shrinks = jnp.asarray(0)

        dD_final = delta_t_used * max_local_dav_rate

        jax.debug.print("max vertically averaged ACTIVE local dD: {x}", x=dD_final)
        jax.debug.print("accepted dt (days): {x}", x=365.0 * delta_t_used)
        jax.debug.print("active layer fraction: {x}", x=jnp.mean(active_layer))
        jax.debug.print("max local D1 layer: {x}",
                        x=jnp.max(evals_and_evecs_of_symm_2x2_tensor(D11_final, D22_final, D12_final)[0]))
        #jax.debug.print("n shrinks used: {x}", x=n_shrinks)

        #Derive the momentum-facing tensor from the same updated 3-D
        #nonlocal state. This avoids separate local and 2-D damage histories.
        D11_bar = vertically_average(D11_final, z_coords)
        D22_bar = vertically_average(D22_final, z_coords)
        D12_bar = vertically_average(D12_final, z_coords)
        D33_bar = vertically_average(D33_final, z_coords)
        ice_mask_2d = jnp.where(h > 0, 1, 0)
        D11_bar *= ice_mask_2d
        D22_bar *= ice_mask_2d
        D12_bar *= ice_mask_2d
        D33_bar *= ice_mask_2d
        D11_bar, D22_bar, D12_bar = project_damage_tensor_to_bounds(
            D11_bar, D22_bar, D12_bar, 0.0, c.dmg.vaD_max)
        D33_bar = jnp.clip(D33_bar, 0.0, c.dmg.vaD_max)

        #Full-thickness rupture: this is the stage at which all components
        #become maximally damaged for the SSA momentum solve.
        D1_bar_check = evals_and_evecs_of_symm_2x2_tensor(D11_bar, D22_bar, D12_bar)[0]
        failed_bar = D1_bar_check >= c.dmg.vaD_cr
        D11_bar = jnp.where(failed_bar, c.dmg.vaD_max, D11_bar)
        D22_bar = jnp.where(failed_bar, c.dmg.vaD_max, D22_bar)
        D12_bar = jnp.where(failed_bar, 0.0, D12_bar)
        D33_bar = jnp.where(failed_bar, c.dmg.vaD_max, D33_bar)

        d1_bar_final = evals_and_evecs_of_symm_2x2_tensor(D11_bar, D22_bar, D12_bar)[0]
        jax.debug.print("max vertically averaged D1: {x}", x=jnp.max(d1_bar_final))

        return (D11_final, D22_final, D12_final, D33_final,
               D11_bar, D22_bar, D12_bar, D33_bar,
               delta_t_used)

    return jax.jit(acd_stepper)





##############################################################################
##############################################################################
##############################################################################



#These are the big two functions!
#The recommendation in Huth, for reasons I'm not completely sure I understand but am
#willing to trust him on, is to compute the viscosities using the "raw" strain rates
#computed just from gradients of the velocities. This is visible in equation 14.

#In these functions, you can see how if the damage were isotropic, it would end up that
#the damage behaved like the enhancement factor (i.e just multiplying effective viscosity
#by a factor of (1-D).



def compute_linear_ssa_residuals_function_fc_visc_anisotropic_damage_dt(
                                          ny, nx, dy, dx, b,
                                          interp_cc_to_fc,
                                          fc_vel_gradient,
                                          add_uv_ghost_cells,
                                          add_s_ghost_cells,
                                          hgrads_fct):

    def compute_linear_ssa_residuals(u_1d, v_1d, h_1d, mu_ew, mu_ns, beta, ice_mask,
                                     D11_bar, D22_bar, D12_bar, D33_bar):

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        hdsdx, hdsdy = hgrads_fct(h, b)

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx

        h_g = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_g)

        dudx_ew, dudy_ew, dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns, dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)

        #Need to interpolate each component of the tensor onto face centres
        D11_g = add_s_ghost_cells(D11_bar); D11_ew, D11_ns = interp_cc_to_fc(D11_g)
        D22_g = add_s_ghost_cells(D22_bar); D22_ew, D22_ns = interp_cc_to_fc(D22_g)
        D12_g = add_s_ghost_cells(D12_bar); D12_ew, D12_ns = interp_cc_to_fc(D12_g)
        D33_g = add_s_ghost_cells(D33_bar); D33_ew, D33_ns = interp_cc_to_fc(D33_g)


        e11_ew, e22_ew, e12_ew = dudx_ew, dvdy_ew, 0.5*(dudy_ew + dvdx_ew)
        e11_ns, e22_ns, e12_ns = dudx_ns, dvdy_ns, 0.5*(dudy_ns + dvdx_ns)

        
        #Compute the strain rate you would have if there were no damage (not quite, but
        #you know what I mean - the strain rates multiplied by (1-D)
        et11_ew, et22_ew, et12_ew = effective_inplane_strain_rate(
            D11_ew, D22_ew, D12_ew, D33_ew, e11_ew, e22_ew, e12_ew)
        et11_ns, et22_ns, et12_ns = effective_inplane_strain_rate(
            D11_ns, D22_ns, D12_ns, D33_ns, e11_ns, e22_ns, e12_ns)


        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*et11_ew[:, 1:] + et22_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*et11_ew[:,:-1] + et22_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*et12_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*et12_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*et12_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*et12_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*et22_ns[:-1,:] + et11_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*et22_ns[1:, :] + et11_ns[1:, :])*dx

        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_linear_ssa_residuals)


def compute_ssa_uv_residuals_function_anisotropic_damage_dt(ny, nx, dy, dx, b,
                                   beta_fct,
                                   interp_cc_to_fc,
                                   fc_vel_gradient,
                                   add_uv_ghost_cells,
                                   add_s_ghost_cells,
                                   mucoef_0,
                                   C_0, temp_cc,
                                   hgrads_fct):

    temp_cc_g = add_s_ghost_cells(temp_cc)
    B_cc = B_from_T(temp_cc_g)
    B_ew, B_ns = interp_cc_to_fc(B_cc)

    def compute_uv_residuals(u_1d, v_1d, q, p, h_1d, ice_mask,
                             D11_bar, D22_bar, D12_bar, D33_bar):

        mucoef = mucoef_0 * jnp.exp(q)
        C = C_0 * jnp.exp(p)

        u = u_1d.reshape((ny, nx))
        v = v_1d.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        hdsdx, hdsdy = hgrads_fct(h, b)
        beta = beta_fct(C, u, v, h)

        volume_x = - (beta * u + c.RHO_I * c.g * hdsdx) * dx * dy
        volume_y = - (beta * v + c.RHO_I * c.g * hdsdy) * dy * dx

        dudx_ew, dudy_ew, dvdx_ew, dvdy_ew,\
        dudx_ns, dudy_ns, dvdx_ns, dvdy_ns = fc_vel_gradient(u, v)

        h_g = add_s_ghost_cells(h)
        h_ew, h_ns = interp_cc_to_fc(h_g)

        mucoef_g = add_s_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef_g)

        #viscosity: unchanged from the isotropic residual, from raw
        #(undamaged) strain rate. see the note at top of this section.
        mu_ew = B_ew * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))
        mu_ns = B_ns * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        mu_ew = mu_ew.at[:, 1:].set(jnp.where(ice_mask==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(ice_mask==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(ice_mask==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(ice_mask==0, 0, mu_ns[:-1,:]))

        D11_g = add_s_ghost_cells(D11_bar); D11_ew, D11_ns = interp_cc_to_fc(D11_g)
        D22_g = add_s_ghost_cells(D22_bar); D22_ew, D22_ns = interp_cc_to_fc(D22_g)
        D12_g = add_s_ghost_cells(D12_bar); D12_ew, D12_ns = interp_cc_to_fc(D12_g)
        D33_g = add_s_ghost_cells(D33_bar); D33_ew, D33_ns = interp_cc_to_fc(D33_g)

        e11_ew, e22_ew, e12_ew = dudx_ew, dvdy_ew, 0.5*(dudy_ew + dvdx_ew)
        e11_ns, e22_ns, e12_ns = dudx_ns, dvdy_ns, 0.5*(dudy_ns + dvdx_ns)

        et11_ew, et22_ew, et12_ew = effective_inplane_strain_rate(
            D11_ew, D22_ew, D12_ew, D33_ew, e11_ew, e22_ew, e12_ew)
        et11_ns, et22_ns, et12_ns = effective_inplane_strain_rate(
            D11_ns, D22_ns, D12_ns, D33_ns, e11_ns, e22_ns, e12_ns)

        visc_x = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*et11_ew[:, 1:] + et22_ew[:, 1:])*dy   -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*et11_ew[:,:-1] + et22_ew[:,:-1])*dy   +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*et12_ns[:-1,:])*0.5*dx -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*et12_ns[1:, :])*0.5*dx

        visc_y = 2 * mu_ew[:, 1:]*h_ew[:, 1:]*(2*et12_ew[:, 1:])*0.5*dy -\
                 2 * mu_ew[:,:-1]*h_ew[:,:-1]*(2*et12_ew[:,:-1])*0.5*dy +\
                 2 * mu_ns[:-1,:]*h_ns[:-1,:]*(2*et22_ns[:-1,:] + et11_ns[:-1,:])*dx   -\
                 2 * mu_ns[1:, :]*h_ns[1:, :]*(2*et22_ns[1:, :] + et11_ns[1:, :])*dx

        x_mom_residual = visc_x + volume_x
        y_mom_residual = visc_y + volume_y

        return x_mom_residual.reshape(-1), y_mom_residual.reshape(-1)

    return jax.jit(compute_uv_residuals)



def make_picnewton_velocity_solver_function_anisotropic_damage_dt(
                                                 ny, nx, dy, dx, b,
                                                 max_n_pic_iterations,
                                                 max_n_newt_iterations,
                                                 mucoef_0, C_0,
                                                 sliding="linear",
                                                 pic_reduction_tol=1e-3,
                                                 newton_tol=1e0,
                                                 periodic=False,
                                                 temperature_field=None,
                                                 newton_max_backtracks=32):

    if temperature_field is None:
        temperature_field = (jnp.zeros((ny,nx))+263.15)

    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(ny, nx)
    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(ny, nx, periodic=periodic)
    fc_velocity_gradient                       = fc_velocity_gradient_function_noextrap(
                                                                                dy, dx, ny, nx,
                                                                                add_uv_ghost_cells)
    viscosity_fct = fc_viscosity_function_new_givenT_noextrap_dt(ny, nx, dy, dx,
                                                   add_uv_ghost_cells,
                                                   add_scalar_ghost_cells,
                                                   interp_cc_to_fc,
                                                   fc_velocity_gradient,
                                                   mucoef_0,
                                                   temperature_field)

    hgrads_fct = gl_unaware_driving_stress_function(dy, dx)
    beta_fct   = beta_function(b, sliding)

    get_uv_residuals_linear_ssa = compute_linear_ssa_residuals_function_fc_visc_anisotropic_damage_dt(
                                                       ny, nx, dy, dx, b,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       hgrads_fct)

    get_uv_residuals_nonlinear_ssa = compute_ssa_uv_residuals_function_anisotropic_damage_dt(
                                                       ny, nx, dy, dx, b,
                                                       beta_fct,
                                                       interp_cc_to_fc,
                                                       fc_velocity_gradient,
                                                       add_uv_ghost_cells,
                                                       add_scalar_ghost_cells,
                                                       mucoef_0, C_0,
                                                       temperature_field,
                                                       hgrads_fct)

    basis_vectors, i_coordinate_sets = basis_vectors_and_coords_2d_square_stencil(ny, nx, 1,
                                                                                  periodic_x=periodic)
    i_coordinate_sets = jnp.concatenate(i_coordinate_sets)
    j_coordinate_sets = jnp.tile(jnp.arange(ny*nx), len(basis_vectors))

    sparse_jacrev = make_sparse_jacrev_fct_shared_basis_new(
                                                        basis_vectors, 2, active_indices=(0,1))
    mask = (i_coordinate_sets>=0)
    i_coordinate_sets = i_coordinate_sets[mask]
    j_coordinate_sets = j_coordinate_sets[mask]

    coords = jnp.stack([
                    jnp.concatenate([i_coordinate_sets, i_coordinate_sets,
                                     i_coordinate_sets+(ny*nx), i_coordinate_sets+(ny*nx)]),
                    jnp.concatenate([j_coordinate_sets, j_coordinate_sets+(ny*nx),
                                     j_coordinate_sets, j_coordinate_sets+(ny*nx)])
                       ])

    #Using a direct solver. Necessary I think, but I'm not too sure...
    la_solver = create_sparse_petsc_la_solver_with_custom_vjp_given_csr(
                                                              coords, (ny*nx*2, ny*nx*2),
                                                              indirect=False, monitor_ksp=False)

    res_fct = lambda x: jnp.max(jnp.abs(x))
    omega = 1

    @custom_vjp
    def solver(q, p, u_trial, v_trial, h, D11_bar, D22_bar, D12_bar, D33_bar):

        u_trial = jnp.where(h>1e-10, u_trial, 0)
        v_trial = jnp.where(h>1e-10, v_trial, 0)

        u_1d = u_trial.copy().reshape(-1)
        v_1d = v_trial.copy().reshape(-1)
        h_1d = h.copy().reshape(-1)

        ice_mask_2d = jnp.where(h>0,1,0)
        ice_mask = ice_mask_2d.reshape(-1)

        u_1d = u_1d * ice_mask
        v_1d = v_1d * ice_mask

        residual = jnp.inf
        init_res = 0
        initial_residual = jnp.inf

        mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d, ice_mask_2d)
        beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

        rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns,
                            beta, ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))

        for i in range(max_n_pic_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_linear_ssa,
                                (u_1d, v_1d, h_1d, mu_ew, mu_ns, beta, ice_mask_2d,
                                D11_bar, D22_bar, D12_bar, D33_bar))

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns,
                                beta, ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))

            old_residual, residual, init_res = print_residual_things(
                                                  residual, rhs, init_res, i, print_=True)
            if i==0:
                initial_residual = jnp.max(jnp.abs(rhs))
            if ((residual/init_res) < pic_reduction_tol) or ((i > 0) and (residual < newton_tol)):
                break

            du = la_solver(nz_jac_values, rhs)
            u_1d = (u_1d + omega*du[:(ny*nx)]) * ice_mask
            v_1d = (v_1d + omega*du[(ny*nx):]) * ice_mask

            mu_ew, mu_ns = viscosity_fct(q, u_1d, v_1d, ice_mask_2d)
            beta = beta_fct(C_0*jnp.exp(p), u_1d.reshape((ny,nx)), v_1d.reshape((ny,nx)), h)

            rhs_new = -jnp.concatenate(get_uv_residuals_linear_ssa(u_1d, v_1d, h_1d, mu_ew, mu_ns,
                                beta, ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))

        final_residual_pic = res_fct(rhs_new)
        print("Picard final residual:", final_residual_pic)

        for i in range(max_n_newt_iterations):

            dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                                (u_1d, v_1d, q, p, h_1d, ice_mask_2d,
                                D11_bar, D22_bar, D12_bar, D33_bar))

            nz_jac_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],
                                             dJv_du[mask], dJv_dv[mask]])

            rhs = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d,
                                ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))

            old_residual, residual, init_res = print_residual_things(
                                                  residual, rhs, init_res, i, print_=True)
            if (i > 0) and (residual < newton_tol):
                break

            du = la_solver(nz_jac_values, rhs)

            step_scale = 1.0
            u_new = (u_1d+step_scale*du[:(ny*nx)]) * ice_mask
            v_new = (v_1d+step_scale*du[(ny*nx):]) * ice_mask
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_new, v_new, q, p, h_1d,
                                ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))
            res_new = res_fct(rhs_new)

            n_backtracks = 0
            while res_new > residual and n_backtracks < newton_max_backtracks:
                step_scale *= 0.5
                u_new = (u_1d+step_scale*du[:(ny*nx)]) * ice_mask
                v_new = (v_1d+step_scale*du[(ny*nx):]) * ice_mask
                rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_new, v_new, q, p, h_1d,
                                ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))
                res_new = res_fct(rhs_new)
                n_backtracks += 1

            if n_backtracks > 0:
                print(f"  backtracked {n_backtracks} times, step_scale={step_scale}")

            u_1d, v_1d = u_new, v_new
            rhs_new = -jnp.concatenate(get_uv_residuals_nonlinear_ssa(u_1d, v_1d, q, p, h_1d,
                                ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))

        return u_1d.reshape((ny, nx)), v_1d.reshape((ny, nx))


    def solver_fwd(q, p, u_trial, v_trial, h, D11_bar, D22_bar, D12_bar, D33_bar):
        u, v = solver(q, p, u_trial, v_trial, h, D11_bar, D22_bar, D12_bar, D33_bar)

        ice_mask_2d = jnp.where(h>0,1,0)

        dJu_du, dJv_du, dJu_dv, dJv_dv = sparse_jacrev(get_uv_residuals_nonlinear_ssa,
                            (u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1),
                            ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar))
        dJ_dvel_nz_values = jnp.concatenate([dJu_du[mask], dJu_dv[mask],
                                             dJv_du[mask], dJv_dv[mask]])

        fwd_residuals = (u, v, dJ_dvel_nz_values, q, p, h, ice_mask_2d,
                         D11_bar, D22_bar, D12_bar, D33_bar)
        return (u, v), fwd_residuals


    def solver_bwd(res, cotangent):
        u, v, dJ_dvel_nz_values, q, p, h, ice_mask_2d,\
            D11_bar, D22_bar, D12_bar, D33_bar = res
        u_bar, v_bar = cotangent

        lambda_ = la_solver(dJ_dvel_nz_values, -jnp.concatenate([u_bar.reshape(-1), v_bar.reshape(-1)]),
                            transpose=True)
        lambda_u = lambda_[:(ny*nx)]
        lambda_v = lambda_[(ny*nx):]

        _, pullback_function = jax.vjp(get_uv_residuals_nonlinear_ssa,
                                u.reshape(-1), v.reshape(-1), q, p, h.reshape(-1),
                                ice_mask_2d, D11_bar, D22_bar, D12_bar, D33_bar)
        _, _, q_bar, p_bar, _, _, _, _, _, _ = pullback_function((lambda_u, lambda_v))

        return (q_bar.reshape((ny, nx)), p_bar.reshape((ny,nx)), None, None, None,
               None, None, None, None)

    solver.defvjp(solver_fwd, solver_bwd)
    return solver


def regrid_mismip_experiment(new_resolution, x_old, y_old, thk_old, half=True):
    """
    Rebuild the MISMIP+ half-domain at `new_resolution` metres, and
    bilinearly interpolate `thk_old` (defined on the (y_old, x_old) grid)
    onto it. Returns the same tuple ordering as mismip_domain_symm, with
    the interpolated thickness field in place of the discarded initial-
    guess placeholder mismip_domain_symm itself returns there.

    x_old, y_old: the 1-D coordinate arrays thk_old was defined on
                  (as returned by an earlier mismip_domain_symm call).
    thk_old: thickness field, shape (len(y_old), len(x_old)).
    """
    (lx, ly, nr, nc, x, y, delta_x, delta_y, _, b,
     C_0, mucoef_0, q, ice_mask, surface, grounded) = mismip_domain_symm(
                                          resolution=new_resolution, half=half)

    interpolator = RegularGridInterpolator(
        (np.asarray(y_old), np.asarray(x_old)),
        np.asarray(thk_old),
        method="linear",
        bounds_error=False,
        fill_value=None    #extrapolate (nearest) rather than silently
                           #zeroing edge cells. the new grid spans
                           #exactly the same domain, so this only ever
                           #guards against floating-point edge mismatches,
                           #never a real out-of-domain query.
    )

    yy_new, xx_new = np.meshgrid(np.asarray(y), np.asarray(x), indexing="ij")
    query_pts = np.stack([yy_new.ravel(), xx_new.ravel()], axis=-1)

    thk_new = interpolator(query_pts).reshape((nr, nc))
    thk_new = jnp.clip(jnp.asarray(thk_new), 0.0, None)   #no negative
                                                          #thickness from
                                                          #interpolation
                                                          #overshoot near
                                                          #the calving front

    return (lx, ly, nr, nc, x, y, delta_x, delta_y, thk_new, b,
           C_0, mucoef_0, q, ice_mask, surface, grounded)

if __name__ == "__main__":


    resolution = 1000
    n_levels = 41
    n_pic_iterations = 50
    n_newt_iterations = 50

    (
        lx, ly, nr, nc,
        x, y, delta_x,
        delta_y, _, b,
        C_0, mucoef_0, q,
        ice_mask, surface,
        grounded
    ) = mismip_domain_symm(resolution=resolution, half=True)

    thk = jnp.load(
        f"{nm_home}/bits_of_data/mismip_plus_experiments/full_attepmt_schoof/ssa/ice0/"
        f"thickness_WmSlidingC1e4_1km_res_HalfDomain_789.7538years.npy"
    )

    #resolution = 500
    #(
    #    lx, ly, nr, nc,
    #    x, y, delta_x,
    #    delta_y, thk, b,
    #    C_0, mucoef_0, q,
    #    ice_mask, surface,
    #    grounded
    #) = regrid_mismip_experiment(resolution, x_r, y_r, thk, half=True)
   
    p = jnp.zeros_like(q)

    temp_field = jnp.zeros_like(thk) + (273.15 - 8.930363929212376)
    

    plot_mismip_field = make_plot_mismip_field_function(b, x, y, y_exaggeration=2)

    add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
    cc_gradient                                = cc_gradient_function(delta_y, delta_x)
    interp_cc_to_fc                            = interp_cc_with_ghosts_to_fc_function(nr, nc)

    vel_solver = make_picnewton_velocity_solver_function_anisotropic_damage_dt(
        nr, nc, delta_y, delta_x, b,
        n_pic_iterations, n_newt_iterations,
        mucoef_0, C_0, sliding="schoof", temperature_field=temp_field)

    adv_stepper = make_advection_stepper(nc, nr, delta_x, delta_y,
                                        interp_cc_to_fc,
                                        add_uv_ghost_cells, add_scalar_ghost_cells,
                                        method="PPM")

    damage_stepper = make_anisotropic_creep_damage_stepper(nc, nr, delta_x, delta_y,
                                                            interp_cc_to_fc,
                                                            add_uv_ghost_cells,
                                                            add_scalar_ghost_cells,
                                                            cc_gradient,
                                                            mucoef_0,
                                                            method="PPM",
                                                            temp_cc=temp_field,
                                                            gamma=c.dmg.gamma,
                                                            l_c=c.dmg.l_c)

    u, v = jnp.zeros_like(thk), jnp.zeros_like(thk)
    h = thk

    #no initial damage!
    D11 = jnp.zeros((nr, nc, n_levels))
    D22 = jnp.zeros((nr, nc, n_levels))
    D12 = jnp.zeros((nr, nc, n_levels))
    D33 = jnp.zeros((nr, nc, n_levels))
    D11_bar = jnp.zeros((nr, nc))
    D22_bar = jnp.zeros((nr, nc))
    D12_bar = jnp.zeros((nr, nc))
    D33_bar = jnp.zeros((nr, nc))
    z_coords = define_z_coordinates(b, h, n_levels)

    outdir = f"{nm_home}/bits_of_data/damage_figures/mismip/22_anisotropic/"
    os.makedirs(outdir, exist_ok=True)

    t_cum = 0
    for i in range(1000):

        d1_bar_for_plot = evals_and_evecs_of_symm_2x2_tensor(D11_bar, D22_bar, D12_bar)[0]

        plot_mismip_field(d1_bar_for_plot, h, vmin=0, vmax=0.9, cmap="coolwarm",
                          cbar_label="Max principal damage <D1>",
                          filepath=f"{outdir}/D1_{i}.png",
                          title=f"Max principal damage {t_cum:.2f} years")
        plot_mismip_field(D12_bar, h, vmin=-0.1, vmax=0.1, cmap="PuOr",
                          cbar_label="D12 (orientation)",
                          filepath=f"{outdir}/D12_{i}.png",
                          title=f"D12 {t_cum:.2f} years")
        plt.close('all')

        u, v = vel_solver(q, p, u, v, h, D11_bar, D22_bar, D12_bar, D33_bar)

        speed = jnp.sqrt(u**2 + v**2)
        delta_t_cfl = 0.95 * delta_x / jnp.maximum(jnp.max(speed), 1.0e-12)
        delta_t = jnp.minimum(delta_t_cfl, 12.0/365.0)
        if i == 0:
            delta_t = jnp.minimum(delta_t, 2.0/365.0)

        (D11, D22, D12, D33,
         D11_bar, D22_bar, D12_bar, D33_bar,
         delta_t_used) = damage_stepper(q, u, v, D11, D22, D12, D33,
                                        D11_bar, D22_bar, D12_bar, D33_bar,
                                        z_coords, delta_t)

        t_cum += delta_t_used

        h = adv_stepper(u.reshape(-1), v.reshape(-1), h.reshape(-1), source=0, delta_t=delta_t_used)

        plot_mismip_field(speed, h, cmap="RdYlBu_r", vmin=0, vmax=2000,
                          cbar_label="Speed (m a^-1)", filepath=f"{outdir}/speed{i}.png",
                          title=f"Speed {t_cum:.2f} years")
        plt.close('all')

        z_coords_new = define_z_coordinates(b, h, n_levels)
        D11 = interp_field_onto_new_zs(D11, z_coords, z_coords_new)
        D22 = interp_field_onto_new_zs(D22, z_coords, z_coords_new)
        D12 = interp_field_onto_new_zs(D12, z_coords, z_coords_new)
        D33 = interp_field_onto_new_zs(D33, z_coords, z_coords_new)
        z_coords = z_coords_new

        if t_cum>1.0:
            break

