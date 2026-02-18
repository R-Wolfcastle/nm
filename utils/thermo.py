import sys
sys.path.insert(1, "/users/eartsu/new_model/testing/nm/utils/")
import constants_years as c

import jax
import jax.numpy as jnp


@jax.jit
def rate_factor(T, P=0):
    #Q = c.Q_LOW if T<263.15 else c.Q_HIGH

    T_standard_recip = 1/(263.15 + c.P_SCALING*P)
    T_h_recip        = 1/(T + c.P_SCALING*P)

    A_cold = c.A_0 * jnp.exp( -( c.Q_LOW/c.R) * (T_h_recip - T_standard_recip) )  
    A_warm = c.A_0 * jnp.exp( -(c.Q_HIGH/c.R) * (T_h_recip - T_standard_recip) )  

    return jnp.minimum(A_cold, A_warm)


@jax.jit
def rate_factor_ctlw(T, LW=0, P=0):
    #Q = c.Q_LOW if T<263.15 else c.Q_HIGH

    T_standard_recip = 1/(263.15 + c.P_SCALING*P)
    T_h_recip        = 1/(T + c.P_SCALING*P)

    A_cold = c.A_0 * jnp.exp( -( c.Q_LOW/c.R) * (T_h_recip - T_standard_recip) )  
    A_warm = c.A_0 * jnp.exp( -(c.Q_HIGH/c.R) * (T_h_recip - T_standard_recip) )

    A_dry = jnp.minimum(A_cold, A_warm)

    #Cutoff mentioned in Aschwanden JGlac 2012 paper
    LW = jnp.maximum(LW, 0.01)

    #LW is zero for non-temperate ice, so this should work out fine, I think.

    return A_dry*(1+181.25*LW)


