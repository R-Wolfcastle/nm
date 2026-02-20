
#1st party
import sys
import time

##local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers/")
from nonlinear_solvers import make_newton_coupled_solver_function,\
        make_newton_velocity_solver_function_custom_vjp_dynamic_thk,\
        make_newton_velocity_solver_function_custom_vjp,\
        make_picard_velocity_solver_function_custom_vjp,\
        make_picnewton_velocity_solver_function_cvjp,\
        make_picnewton_velocity_solver_function_full_cvjp

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils/")
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





#def make_misfit_function_speed_basic(u_obs, u_c, reg_param, solver):
#
#    def misfit_function(mucoef_internal):
#
#        u_mod, v_mod = solver(mucoef_internal.reshape(u_obs.shape))
#
#        u_mod = u_mod*c.S_PER_YEAR
#        v_mod = v_mod*c.S_PER_YEAR
#
#        misfit = jnp.sum(uc * (u_obs - jnp.sqrt(u_mod**2 + v_mod**2 + 1e-12))**2)/(u_mod.size)
#
#        regularisation = reg_param * jnp.sum((1-mucoef_internal)**2)/(u_mod.size)
#
#        return misfit + regularisation
#
#    return misfit_function
#
#
#
#def lbfgsb_function(misfit_function, iterations=50):
#    def lbfgsb(initial_guess):
#
#        get_grad = jax.grad(misfit_function)
#
#        print("starting opt")
#        #need the callback to give intermediate vals etc. will sort later.
#        result = scinimize(misfit_function, 
#                           initial_guess, 
#                           jac = lambda x: get_grad(x), 
#                           method="L-BFGS-B", 
#                           bounds=[(0.1, 2)] * initial_guess.size, 
#                           options={"maxiter": iterations} #Note: disp is depricated
#                          )
#
#        return result.x
#    return lbfgsb


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


lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q, ice_mask, surface, grounded = wonky_stream()



add_uv_ghost_cells, add_scalar_ghost_cells = add_ghost_cells_fcts(nr, nc, periodic=False)
gradient_function = cc_gradient_function(delta_y, delta_x)


u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_pic_iterations = 2
n_newt_iterations = 10

uc = jnp.where(thk>0, 1, 0)
uc = binary_erosion(uc)
uc = binary_erosion(uc)


#solver = make_picard_velocity_solver_function_custom_vjp(nr, nc,
#                                                         delta_y, delta_x,
#                                                         b, ice_mask,
#                                                         n_pic_iterations,
#                                                         mucoef_0,
#                                                         sliding="basic_weertman")


#solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
#                                                         delta_y,
#                                                         delta_x,
#                                                         thk, b, C,
#                                                         n_newt_iterations, mucoef_0)


def misfit(u_mod, v_mod, q, p, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    return jnp.sum(mask.reshape(-1) * (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2)


def regularised_misfit(u_mod, v_mod, q, p, speed_obs, mask):
    speed_mod = jnp.sqrt(u_mod**2 + v_mod**2 + 1e-10)
    
    misfit_term = jnp.sum(mask.reshape(-1) * \
                          (speed_mod.reshape(-1) - speed_obs.reshape(-1))**2
                         )/(nr*nc)

    #Assume that things are, on average, wrong by 100ma^-1. So, divide by 10_000:
    misfit_term = misfit_term/10_000


    phi = mucoef_0*jnp.exp(q.reshape((nr, nc)))
    dphi_dx, dphi_dy = gradient_function(phi)


    #The coefficients are at least an order of magnitude smaller than Steph's choices of:
    #alpha_phi = 1e11 (for me, that would be 1e11/10_000 ~ 1e7)
    #alpha_C   = 1e3  (for me, that would be 1e3 /10_000 ~ 1e-1)

    #maybe 1e4 a good shout?
    phi_regn_term = 5e5 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dphi_dx.reshape(-1)**2 + dphi_dy.reshape(-1)**2)
                              )/(nr*nc)
    


    C = C_0*jnp.exp(p.reshape((nr, nc)))
    dC_dx, dC_dy = gradient_function(C)

    C_regn_term = 1e-2 * jnp.sum( mask[1:-1,1:-1].reshape(-1) *\
                                (dC_dx.reshape(-1)**2 + dC_dy.reshape(-1)**2)
                              )/(nr*nc)


    jax.debug.print("misfit_term: {x}", x=misfit_term)
    jax.debug.print("phi_regn_term: {x}", x=phi_regn_term)
    jax.debug.print("C_regn_term: {x}", x=C_regn_term)

    #return misfit_term, regn_term, misfit_term + regn_term
    return misfit_term + phi_regn_term + C_regn_term

cf_cells = (thk>0) & ~binary_erosion(thk>0)

def lbfgsb_function(misfit_functional, misfit_fctl_args=(), iterations=50):
    def reduced_functional(qp):
        q = qp[:(nr*nc)]
        p = qp[(nr*nc):]
        u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
        return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)

    #get_grad_basic = jax.grad(reduced_functional)
    #def get_grad(x):
    #    grad = get_grad_basic(x)
    #    return grad*(1-cf_cells.astype(int).reshape(-1))
    get_grad = jax.grad(reduced_functional)

    def lbfgsb(initial_guess):
        print("starting opt")

        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(reduced_functional, 
                           initial_guess, 
                           jac = lambda x: get_grad(x), 
                           method="L-BFGS-B", 
                           bounds= [(-2, 0.5)] * int(initial_guess.size/2) + \
                                   [(-4, 4)] * int(initial_guess.size/2), 
                           options={"maxiter": iterations} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb



def lbfgsb_function_repeated_restart(misfit_functional, misfit_fctl_args=(), iterations=50, restart_interval=10):
    def reduced_functional(qp):
        q = qp[:(nr*nc)]
        p = qp[(nr*nc):]
        u_out, v_out = solver(q.reshape(nr, nc), p.reshape(nr, nc), u_init, v_init, thk)
        return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)

    #get_grad_basic = jax.grad(reduced_functional)
    #def get_grad(x):
    #    grad = get_grad_basic(x)
    #    return grad*(1-cf_cells.astype(int).reshape(-1))
    get_grad = jax.grad(reduced_functional)
        
    n_restarts = int(iterations/restart_interval)

    def lbfgsb(initial_guess):
        print("starting opt")

        for i in range(n_restarts):
            print(f"Restart {i}")
            #need the callback to give intermediate vals etc. will sort later.
            result = scinimize(reduced_functional,
                           initial_guess,
                           jac = get_grad,
                           method="L-BFGS-B",
                           #bounds= [(-2, 0.5)] * int(initial_guess.size/2) + \
                           #        [(-4, 4)] * int(initial_guess.size/2), 
                           bounds= [(-2, 0)] * int(initial_guess.size/2) + \
                                   [(-2, 2)] * int(initial_guess.size/2), 
                           options={"maxiter": restart_interval}
                          )
            initial_guess = result.x

        return result.x

    return lbfgsb




def lbfgsb_function_p_only(misfit_functional, misfit_fctl_args=(), iterations=50):
    def reduced_functional(p):
        u_out, v_out = solver(jnp.zeros((nr, nc)), p.reshape(nr, nc), u_init, v_init, thk)
        return misfit_functional(u_out, v_out, q, p, *misfit_fctl_args)

    #get_grad_basic = jax.grad(reduced_functional)
    #def get_grad(x):
    #    grad = get_grad_basic(x)
    #    return grad*(1-cf_cells.astype(int).reshape(-1))
    get_grad = jax.grad(reduced_functional)

    def lbfgsb(initial_guess):
        print("starting opt")

        #need the callback to give intermediate vals etc. will sort later.
        result = scinimize(reduced_functional, 
                           initial_guess, 
                           jac = lambda x: get_grad(x), 
                           method="L-BFGS-B", 
                           bounds= [(-4, 4)] * initial_guess.size,
                           options={"maxiter": iterations} #Note: disp is depricated
                          )

        return result.x
    return lbfgsb

def initial_guess_for_C(speed_obs):
    smoothed_speed_obs = gaussian_filter(speed_obs, sigma=1)

    surf_extended = add_scalar_ghost_cells(surface)
    dsdx, dsdy = gradient_function(surf_extended)

    rhs_squared = (c.RHO_I * c.g * thk)**2 * (dsdx**2 + dsdy**2)

    lhs_squared = smoothed_speed_obs ** (2/3)

    cig = jnp.minimum(jnp.sqrt(rhs_squared/(lhs_squared + 1e-12)), 1e4)

    cig = jnp.where(thk>0, jnp.where(C==0, 0, cig), 1)

    cig = cig.at[:1,:].set(1e6)
    cig = cig.at[-1:,:].set(1e6)
    cig = cig.at[:,:1].set(1e6)

    return cig


#solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
#                                                         delta_y, delta_x,
#                                                         b, ice_mask,
#                                                         n_pic_iterations,
#                                                         n_newt_iterations,
#                                                         mucoef_0, C,
#                                                         sliding="basic_weertman")
#
#u_out, v_out = solver(jnp.zeros((nr, nc)), jnp.zeros((nr, nc)), u_init, v_init, thk)
#show_vel_field(u_out, v_out, cmap="RdYlBu_r", vmin=0, vmax=3500)
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/clean_vel_high_res.npy", jnp.stack([u_out, v_out]))






vel_data = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/clean_vel.npy")
#vel_data = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/clean_vel_high_res.npy")

u_obs = vel_data[0]
v_obs = vel_data[1]


speed_obs = jnp.sqrt(u_obs**2+v_obs**2 + 1e-12)

key = jax.random.PRNGKey(0)
noise = 10*jax.random.normal(key, (nr,nc))
#noise = jnp.where(speed_obs>0, noise, 0)

speed_obs = speed_obs + noise

speed_obs = jnp.maximum(speed_obs, 0)

#plt.imshow(speed_obs)
#plt.colorbar()
#plt.show()

mucoef_0 = jnp.ones_like(mucoef_0)



dsdx, dsdy = gradient_function(surface)


C_0 = initial_guess_for_C(speed_obs)
#plt.imshow(C_0, vmin=0, vmax=2000, cmap="magma")
#plt.imshow(jnp.log10(C_0), cmap="magma")
#plt.colorbar()
#plt.show()
#raise
##C_0 = C.copy()


solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                                         delta_y, delta_x,
                                                         b, ice_mask,
                                                         n_pic_iterations,
                                                         n_newt_iterations,
                                                         mucoef_0, C_0,
                                                         sliding="basic_weertman")
#                                                       sliding="linear")


q_initial_guess = jnp.zeros_like(thk).reshape(-1)
p_initial_guess = jnp.zeros_like(thk).reshape(-1)

qp_initial_guess = jnp.zeros((2*nr*nc,))





#show_vel_field(u_obs, v_obs, cmap="RdYlBu_r", vmin=0, vmax=2500)
#raise






#grad = get_grad(initial_guess)
#
#
#plt.imshow(grad.reshape((nr,nc)))
#plt.show()
#raise



#######DOING IP FOR C AND MISFIT ON GROUNDED ICE ONLY
##p_only_lbfgsb_iterator = lbfgsb_function_p_only(regularised_misfit, (speed_obs, uc*grounded), iterations=20)
#p_only_lbfgsb_iterator = lbfgsb_function_p_only(regularised_misfit, (speed_obs, uc), iterations=20)
#
#p_ig = p_only_lbfgsb_iterator(p_initial_guess)
#qp_initial_guess = jnp.concatenate([q_initial_guess, p_ig])
#
#C_ig = C_0*jnp.exp(p_ig.reshape((nr,nc)))
#plt.imshow(C_ig, vmin=0, vmax=2000, cmap="magma")
#plt.colorbar()
#plt.show()
########################################################



#lbfgsb_iterator = lbfgsb_function(regularised_misfit, (speed_obs, uc), iterations=40)
lbfgsb_iterator = lbfgsb_function_repeated_restart(regularised_misfit, (speed_obs, uc), iterations=50, restart_interval=50)



#u_mod_ig, v_mod_ig = solver(qp_initial_guess[:(nr*nc)].reshape((nr, nc)), qp_initial_guess[(nr*nc):].reshape((nr, nc)), u_init, v_init, thk)
#show_vel_field(u_mod_ig, v_mod_ig, cmap="RdYlBu_r", vmin=0, vmax=2500)


qp_out = lbfgsb_iterator(qp_initial_guess)
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/qp_out_large_50its.npy", qp_out)

#qp_out = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/inv_prob_tests/qp_out_large_50its.npy")

q_out = qp_out[:(nr*nc)].reshape((nr,nc))
p_out = qp_out[(nr*nc):].reshape((nr,nc))


#plt.imshow(p_out)
#plt.colorbar()
#plt.show()
#
phi_out = mucoef_0*jnp.exp(q_out)
plt.imshow(phi_out, vmin=0, vmax=1, cmap="cubehelix")
plt.colorbar()
plt.show()

C_out = C_0*jnp.exp(p_out)
plt.imshow(C_out, vmin=0, vmax=2000, cmap="magma")
plt.colorbar()
plt.show()

#raise

u_mod_end, v_mod_end = solver(q_out, p_out, u_init, v_init, thk)

#full_mft = regularised_misfit(u_mod_end, v_mod_end, q_out, speed_obs, uc)

#print(full_mft)
#raise

show_vel_field(u_mod_end, v_mod_end, cmap="RdYlBu_r", vmin=0, vmax=2500)

plt.figure(figsize=(8,6))
plt.imshow(jnp.sqrt(u_mod_end**2 + v_mod_end**2) - speed_obs, vmin=-50, vmax=50, cmap="RdBu_r")
plt.colorbar()
plt.show()



