#1st party
import os
import sys

#3rd party
import jax
import jax.numpy as jnp

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import vertically_integrate, vertically_average
from thermo import B_from_T

"""
Mostly from: A comparison of the stability and performance of depth-integrated
ice-dynamics solvers from Robinson et al., 2021.
"""


def diva_cc_viscosity_function(dy, dx, cc_vel_gradient, mucoef_0, temp_cc):
    """
    3D effective viscosity at cell centres, mu_vv: (ny, nx, nz), following
    Eq. (13): horizontal (membrane) strain-rate terms come from the
    vertically-averaged velocity (constant with z), the vertical-shear terms
    come from dudz/dvdz (which vary with z).

    Also returns mu_va, the vertically-averaged viscosity (ny, nx), which is
    what actually goes into the SSA momentum residual after
    interpolation to faces.
    """
    
    B_cc = B_from_T(temp_cc)
    
    def diva_viscosity(q, u_va, v_va, dudz, dvdz, zs):
        mucoef = mucoef_0 * jnp.exp(q)

        dudx, dudy, dvdx, dvdy = cc_vel_gradient(u_va, v_va)
        #add another dimenion
        dudx = dudx[..., None]
        dudy = dudy[..., None]
        dvdx = dvdx[..., None]
        dvdy = dvdy[..., None]

        eps_e_sq = dudx**2 + dvdy**2 + dudx*dvdy \
                   + 0.25*(dudy + dvdx)**2 \
                   + 0.25*dudz**2 + 0.25*dvdz**2

        mu_vv = (B_cc * mucoef)[..., None] * \
                (eps_e_sq + c.EPSILON_VISC**2)**(0.5*(1/c.GLEN_N - 1))

        mu_va = vertically_average(mu_vv, zs)

        return mu_vv, mu_va

    return jax.jit(diva_viscosity)


def arthern_function(mu_vv, zs, m=1, only_return_surface=True):
    """
    F_m (Eq. 15) = int_b^s (1/mu) * ((s-z)/H)^m dz
    """
    s_expanded = zs[..., -1][..., None]
    h_expanded = (zs[..., -1] - zs[..., 0])[..., None]

    integrand = (1 / mu_vv) * ((s_expanded - zs) / h_expanded) ** m
    fm = vertically_integrate(integrand, zs, preserve_structure=True)

    if only_return_surface:
        return fm[..., -1]
    else:
        return fm


def diva_beta_eff_function(beta_fct):
    """
    beta_fct is what would be used in the SSA formulation - so is the
    sliding function applied to the basal ice veloctity.
    beta_eff is according to equation 19, or 12 in Arthern 2015.
    """
    def beta_eff(C, u_base, v_base, h, f2):
        beta = beta_fct(C, u_base, v_base, h)
        beta_eff_val = beta / (1 + beta * f2)
        
        beta_eff_val = jnp.where(h>0, beta_eff_val, 1)
        beta = jnp.where(h>0, beta, 1)

        return beta, beta_eff_val

    return jax.jit(beta_eff)


def diva_vertical_shear_function():
    """
    dudz, dvdz at cell centres (Eq. 21, generalized to 2D by treating x and y
    basal shear stress components independently -
    also: Lipscomb et al. (2019) - explanation of CISM's DIVA solver):
    """
    def new_dudz_dvdz(mu_vv, u_va, v_va, beta_eff, zs):
        s = zs[..., -1][..., None]
        h = (zs[..., -1] - zs[..., 0])[..., None]

        tau_bx = (beta_eff * u_va)[..., None]
        tau_by = (beta_eff * v_va)[..., None]

        dudz = tau_bx * (s - zs) / ( mu_vv * h )
        dvdz = tau_by * (s - zs) / ( mu_vv * h )

        return dudz, dvdz

    return jax.jit(new_dudz_dvdz)


def diva_reconstruct_3d_velocity_function():
    """
    Reconstructs the full 3D velocity field (Eq. 16/18) once the outer
    Picard loop on (u_va, v_va) has converged.
    Kind of need it to define u_base
    """
    def reconstruct(u_va, v_va, dudz, dvdz, beta, f2, zs):
        u_base = u_va / (1 + beta * f2)
        v_base = v_va / (1 + beta * f2)

        u_vv = u_base[..., None] + vertically_integrate(dudz, zs, preserve_structure=True)
        v_vv = v_base[..., None] + vertically_integrate(dvdz, zs, preserve_structure=True)

        return u_vv, v_vv

    return jax.jit(reconstruct)
