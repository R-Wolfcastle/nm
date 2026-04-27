#1st party
import sys
import time

##local apps
sys.path.insert(1, "../../../../solvers/")
from nonlinear_solvers import make_pic_velocity_solver_function_acrobatic,\
                              make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap,\
                              make_pic_velocity_solver_function_gpusafe,\
                              make_pic_velocity_solver_function_expl_advection_gpusafe

sys.path.insert(1, "../../../../utils/")
from plotting_stuff import show_vel_field, make_gif, show_damage_field,\
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,\
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette
from grid import binary_erosion, cc_gradient_function, add_ghost_cells_fcts
import constants_years as c

#3rd party
import numpy as np
import jax
import jax.numpy as jnp
import xarray as xr
import scipy
from scipy.optimize import minimize as scinimize
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)





def mucoef_rifted(x,y,resolution):

    L = len(x)*resolution
    W = len(y)*resolution
    
    # crack parallel to front
    L3 = 32.0e+3 #crack to front
    L2 = 4.0e+3  #crack width
    L1 = L - L3 - L2  #left hand boundary to crack
    
    #crack parallel to shear margin
    W1 = 32.0e+3 # y = 0 to crack
    W2 = 4.0e+3  # crack width
    W3 = W - W1 + W2

    mucoef = jnp.ones((len(y), len(x)))

    #rift parallel to front
    mucoef = jnp.where(((x>L1) & (x < L1 + L2) & (y > W1) & (y < W - W1)), 0.25, mucoef)
    #(higher) damage parallel to shear margin @ y = W1
    mucoef = jnp.where(((y > W1) & (y < W1 + W2)), 0.25, mucoef)
    #(lower) damage parallel to shear margin @y = W - W1
    mucoef = jnp.where(((y > W - W1 - W2) & (y < W - W1)), 0.5, mucoef)

    mucoef = jnp.flipud(mucoef)
            
    return mucoef


def stickiness(x, y, resolution):
    BETA_MAX = 2.0e+3
    BETA_MID = 1.0e+3
    
    L = len(x)*resolution
    W = len(y)*resolution
    
    # crack parallel to front
    L3 = 32.0e+3 #crack to front
    L2 = 4.0e+3  #crack width
    L1 = L - L3 - L2  #left hand boundary to crack
    
    #crack parallel to shear margin
    W1 = 32.0e+3 # y = 0 to crack
    W2 = 4.0e+3  # crack width
    W3 = W - W1 + W2

    beta = jnp.zeros_like(x)+BETA_MAX
    #beta = jnp.where(((y > W1) & (y < W - W1)), 0.01 * BETA_MID * (1.0 + jnp.cos(16.0 * jnp.pi * x/L)), beta)
    beta = jnp.where(((y > W1) & (y < W - W1)), BETA_MID * (1.0 + jnp.cos(16.0 * jnp.pi * x/L)), beta)

    return beta

def wonky_stream_rotated():
    # --- build original domain ---
    lx = 128_000
    ly = 128_000
    resolution = 1000

    nr0 = int(ly/resolution)
    nc0 = int(lx/resolution)

    x0 = jnp.linspace(0, lx, nc0)
    y0 = jnp.linspace(0, ly, nr0)
    xx0, yy0 = jnp.meshgrid(x0, y0)

    delta_x = x0[1]-x0[0]
    delta_y = y0[1]-y0[0]

    thk0 = jnp.zeros((nr0,nc0)) + 512 - 256*x0/lx
    thk0 = thk0.at[:,-2:].set(0)
    #thk0 = thk0.at[70:90, -10:].set(0)
    thk0 = thk0.at[int(70*1000/resolution):int(90*1000/resolution), -int(10*1000/resolution):].set(0)
    b0   = jnp.zeros((nr0,nc0)) - 256 - 256*x0/lx

    C0 = stickiness(xx0, yy0, resolution)
    grounded0 = jnp.where((b0+thk0)>thk0*(1-0.917/1.027), 1, 0)
    C0 = jnp.where((grounded0>0) | (thk0==0), C0, 0)

    # side damping
    C0 = C0.at[:2,:].set(1e12)
    C0 = C0.at[-2:,:].set(1e12)
    C0 = C0.at[:,:2].set(1e12)

    surface0 = jnp.maximum(thk0+b0, thk0*(1-c.RHO_I/c.RHO_W))
    b0 = jnp.where(C0>1e11, 0.01 + surface0 - thk0, b0)

    mucoef0 = mucoef_rifted(xx0, yy0, resolution)
    q0 = jnp.zeros_like(C0)
    ice_mask0 = (thk0>0).astype(int)

    # ---------------------------------------
    #  ROTATION BY -45 degrees (clockwise)
    # ---------------------------------------
    theta = jnp.pi/8
    R = jnp.array([[ jnp.cos(theta), -jnp.sin(theta)],
                   [ jnp.sin(theta),  jnp.cos(theta)]])

    # New padded domain dimensions
    L = lx
    W = ly
    diag = int(jnp.ceil(jnp.sqrt(2)*L/resolution))
    nr = diag
    nc = diag

    # Build new coords
    x = jnp.linspace(-diag*resolution/2, diag*resolution/2, nc)
    y = jnp.linspace(-diag*resolution/2, diag*resolution/2, nr)
    XX, YY = jnp.meshgrid(x, y)

    # Map (XX,YY) back to original coordinates via inverse rotation
    invR = R.T
    XY_old = jnp.stack([
        invR[0,0]*XX + invR[0,1]*YY,
        invR[1,0]*XX + invR[1,1]*YY
    ], axis=0)

    Xold = XY_old[0] + L/2
    Yold = XY_old[1] + W/2

    # nearest neighbour lookup indices
    i = jnp.round(Yold / resolution).astype(int)
    j = jnp.round(Xold / resolution).astype(int)

    valid = (i>=0)&(i<nr0)&(j>=0)&(j<nc0)

    def sample(arr):
        out = jnp.zeros_like(XX)
        out = out.at[valid].set(arr[i[valid], j[valid]])
        return out

    thk  = sample(thk0)
    b    = sample(b0)
    C    = sample(C0)
    mucoef_0 = sample(mucoef0)
    q    = sample(q0)
    surface  = sample(surface0)
    grounded = sample(grounded0)
    ice_mask = sample(ice_mask0)

    return (diag*resolution, diag*resolution,
            nr, nc, x, y, resolution, resolution,
            thk, b, C, mucoef_0, q, ice_mask, surface, grounded)

def wonky_stream():
    lx = 128_000
    ly = 128_000

    resolution = 1000

    nr = int(ly/resolution)
    nc = int(lx/resolution)

    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)
   
    xx, yy = jnp.meshgrid(x,y)

    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]


    thk = jnp.zeros((nr,nc)) + 512 - 256*x/lx
    thk = thk.at[:,-2:].set(0)
    b = jnp.zeros((nr, nc)) - 256 - 256*x/lx

   
    C = stickiness(xx, yy, resolution)
    
    grounded = jnp.where((b+thk)>thk*(1-0.917/1.027), 1, 0)
    C = jnp.where((grounded>0) | (thk==0), C, 0)
    
    C = C.at[:1,:].set(1e12)
    C = C.at[-1:,:].set(1e12)
    C = C.at[:,:1].set(1e12)
    

    surface = jnp.maximum(thk+b, thk * (1-c.RHO_I/c.RHO_W))

    b = jnp.where(C>1e11, 0.01+surface-thk, b)
    
    mucoef_0 = mucoef_rifted(xx, yy, resolution)

    q = jnp.zeros_like(C)

    ice_mask = jnp.where(thk>0, 1, 0)


    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded




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



#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = tiny_ice_shelf()
lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = wonky_stream()
#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = wonky_stream_rotated()



u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 75

#solver = make_pic_velocity_solver_function_gpusafe(nr, nc, delta_y, delta_x,
#                                                   b, ice_mask, n_iterations,
#                                                   mucoef_0, C, sliding="basic_weertman")
#
#
#u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#
#show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=3000)
#
#raise
#
#solver_comp = make_picnewton_velocity_solver_function_full_cvjp_no_cf_extrap(nr, nc,
#                                                         delta_y, delta_x,
#                                                         b, ice_mask,
#                                                         70, 1,
#                                                         mucoef_0, C,
#                                                         sliding="basic_weertman")
#
#u_out_comp, v_out_comp = solver_comp(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#raise
#show_vel_field(u_out_comp, v_out_comp, cmap="RdYlBu_r", vmin=0, vmax=3000)
#
#show_vel_field(u_out-u_out_comp, v_out-v_out_comp, cmap="RdBu_r", vmin=-200, vmax=200)

n_timesteps = 5

prognostic_solver = make_pic_velocity_solver_function_expl_advection_gpusafe(nr, nc, delta_y, delta_x,
                                                   b, ice_mask, n_iterations,
                                                   mucoef_0, C, n_timesteps, sliding="basic_weertman")



## Warm-up
#u, v, h = prognostic_solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#u.block_until_ready()

# Timed run
#t0 = time.time()
u, v, h = prognostic_solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#u.block_until_ready()
#print("Total runtime:", time.time() - t0)

raise
show_vel_field(u, v, cmap="RdYlBu_r", vmin=0, vmax=3000)

plt.imshow(h)
plt.colorbar()
plt.show()

