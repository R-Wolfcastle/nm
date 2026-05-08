import jax
import jax.numpy as jnp

import os
import sys
nm_home = os.environ['NM_HOME']
sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c


import jax
import jax.numpy as jnp


# ============================================================
# CONSISTENT CENTRAL DIFFERENCE (NO EXTRA SLICING INSIDE)
# ============================================================
def grad(u, dx, dy):
    ux = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    uy = (u[2:, :] - u[:-2, :]) / (2 * dy)
    return ux, uy


# ============================================================
# ENERGY FUNCTIONAL (FIXED DOMAIN HANDLING)
# ============================================================
def ssa_energy(u, v, h, b, beta, B, rho, g, dx, dy, s, eps=1e-12):

    # ------------------------------------------------------------
    # define ONE consistent interior field ONCE
    # ------------------------------------------------------------
    u0 = u
    v0 = v
    h0 = h
    beta0 = beta
    s0 = s

    # ------------------------------------------------------------
    # gradients define the reduced domain automatically
    # ------------------------------------------------------------
    ux, uy = grad(u0, dx, dy)
    vx, vy = grad(v0, dx, dy)
    sx, sy = grad(s0, dx, dy)

    # matching interior region (IMPORTANT FIX)
    h_i = h0[1:-1, 1:-1]

    # strain invariant (NOW consistent shapes)
    I = (
        ux**2 +
        vy**2 +
        ux * vy +
        0.25 * (uy + vx)**2
    )

    # Glen dissipation (n=3 → power 2)
    diss = h_i * (2 * B / 4.0) * (I + eps)**2.0

    # basal drag on same interior region
    drag = 0.5 * beta0[1:-1, 1:-1] * (
        u0[1:-1, 1:-1]**2 + v0[1:-1, 1:-1]**2
    )

    # driving stress
    driving = rho * g * h_i * (
        u0[1:-1, 1:-1] * sx +
        v0[1:-1, 1:-1] * sy
    )

    return jnp.sum(diss + drag - driving)


# ============================================================
# RESIDUAL (first variation)
# ============================================================
def residual(u, v, h, b, beta, B, rho, g, dx, dy, s):

    return jax.grad(ssa_energy, argnums=(0, 1))(
        u, v, h, b, beta, B, rho, g, dx, dy, s
    )


# ============================================================
# FULL SYSTEM RESIDUAL (for Jacobian)
# ============================================================
def residual_full(u, v, h, b, beta, B, rho, g, dx, dy, s):

    Ru, Rv = residual(u, v, h, b, beta, B, rho, g, dx, dy, s)

    return jnp.concatenate([Ru.ravel(), Rv.ravel()])


# ============================================================
# JACOBIAN (symmetric Hessian of energy)
# ============================================================
def jacobian(u, v, h, b, beta, B, rho, g, dx, dy, s):

    return jax.jacfwd(residual_full)(u, v, h, b, beta, B, rho, g, dx, dy, s)







def tiny_ice_shelf():
    lx = 1_500
    ly = 1_500
    resolution = 250 #m

    nr = int(ly/resolution)
    nc = int(lx/resolution)

    lx = nr*resolution
    ly = nc*resolution

    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)

    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    thk_profile = 500# - 300*x/lx
    thk = jnp.zeros((nr, nc))+thk_profile
    thk = thk.at[:,  -1:].set(0)
    thk = thk.at[-5:,-4:].set(0)

    b = jnp.zeros_like(thk)-600
    b = b.at[:1, :].set(-440)
    b = b.at[:, :1].set(-440)
    b = b.at[-1:, :].set(-440)

    mucoef = jnp.ones_like(thk)

    C = jnp.zeros_like(thk)
    C = C.at[:2, :].set(1e12)
    C = C.at[:, :2].set(1e12)
    C = C.at[-2:,:].set(1e12)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile

    q = jnp.zeros_like(C)
    
    grounded = jnp.where((b+thk)>thk*(1-0.917/1.027), 1, 0)
    surface = jnp.maximum(thk+b, thk*(1-c.RHO_I/c.RHO_W))

    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b,\
           C, mucoef_0, q, jnp.where(thk>0,1,0), surface, grounded 

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = tiny_ice_shelf()



# example initialization
u = jnp.zeros((nr, nc))
v = jnp.zeros((nr, nc))

s = surface  # from your function



Ru, Rv = residual(u, v, thk, b, C, c.B_COLD, c.RHO_I, c.g, delta_x, delta_y, s)

J = jacobian(u, v, thk, b, C, c.B_COLD, c.RHO_I, c.g, delta_x, delta_y, s)









raise

######################

def grad(u, dx, dy):
    ux = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    uy = (u[2:, :] - u[:-2, :]) / (2 * dy)
    return ux, uy


def ssa_energy(u, v, h, b, beta, B, rho, g, dx, dy, s, eps=1e-12):

    # interior slices
    u0 = u[1:-1, 1:-1]
    v0 = v[1:-1, 1:-1]
    h0 = h[1:-1, 1:-1]

    # gradients (interior-aligned)
    ux, uy = grad(u, dx, dy)
    vx, vy = grad(v, dx, dy)
    dsdx, dsdy = grad(s, dx, dy)

    h0_g = h[1:-1, 1:-1]

    # strain invariant
    I = (
        ux**2 +
        vy**2 +
        ux * vy +
        0.25 * (uy + vx)**2
    )

    # Glen dissipation (n=3 → power 2)
    diss = h0_g * (2 * c.B_COLD / 4.0) * (I + eps)**2.0

    # basal drag
    drag = 0.5 * beta[1:-1, 1:-1] * (u0**2 + v0**2)

    # driving stress
    driving = rho * g * h0_g * (u0 * dsdx + v0 * dsdy)

    return jnp.sum(diss + drag - driving)


def tiny_ice_shelf():
    lx = 1_500
    ly = 1_500
    resolution = 250 #m

    nr = int(ly/resolution)
    nc = int(lx/resolution)

    lx = nr*resolution
    ly = nc*resolution

    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)

    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]

    thk_profile = 500# - 300*x/lx
    thk = jnp.zeros((nr, nc))+thk_profile
    thk = thk.at[:,  -1:].set(0)
    thk = thk.at[-5:,-4:].set(0)

    b = jnp.zeros_like(thk)-600
    b = b.at[:1, :].set(-440)
    b = b.at[:, :1].set(-440)
    b = b.at[-1:, :].set(-440)

    mucoef = jnp.ones_like(thk)

    C = jnp.zeros_like(thk)
    C = C.at[:2, :].set(1e12)
    C = C.at[:, :2].set(1e12)
    C = C.at[-2:,:].set(1e12)
    C = jnp.where(thk==0, 1e8, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile

    q = jnp.zeros_like(C)
    
    grounded = jnp.where((b+thk)>thk*(1-0.917/1.027), 1, 0)
    surface = jnp.maximum(thk+b, thk*(1-c.RHO_I/c.RHO_W))

    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b,\
           C, mucoef_0, q, jnp.where(thk>0,1,0), surface, grounded 

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = tiny_ice_shelf()



R = jax.grad(ssa_energy, argnums=(0, 1))


J = jax.jacfwd(lambda u, v: jax.grad(ssa_energy, argnums=(0,1))(u,v)[0])(u, v)





