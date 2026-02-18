#1st party
from pathlib import Path
import sys
import time
from functools import partial

#local apps
sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/utils")
from sparsity_utils import (scipy_coo_to_csr,
                           basis_vectors_and_coords_2d_square_stencil,
                           make_sparse_jacrev_fct_new, make_sparse_jacrev_fct_shared_basis)

import constants_years as c
#import constants as c
from plotting_stuff import (show_vel_field, make_gif, show_damage_field,
                           create_gif_from_png_fps, create_high_quality_gif_from_pngfps,
                           create_imageio_gif, create_webp_from_pngs, create_gif_global_palette)
from grid import *

sys.path.insert(1, "/Users/eartsu/new_model/testing/nm/solvers")
from linear_solvers import create_sparse_petsc_la_solver_with_custom_vjp
from nonlinear_solvers import (forward_adjoint_and_second_order_adjoint_solvers,
        make_newton_velocity_solver_function_custom_vjp)

#3rd party
from petsc4py import PETSc
#from mpi4py import MPI

import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import matplotlib.pyplot as plt
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize
from jax import custom_vjp, custom_jvp
from jax.experimental.sparse import BCOO

import numpy as np
import scipy
from scipy.optimize import minimize as scinimize

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf)
jax.config.update("jax_enable_x64", True)


def ice_shelf():
    lx = 150_000
    ly = 200_000
    
    resolution = 1_000 #m
    
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
    thk = thk.at[:, -2:].set(0)
   
    b = jnp.zeros_like(thk)-600
    
    mucoef = jnp.ones_like(thk)
    
    C = jnp.zeros_like(thk)
    C = C.at[:4, :].set(1e4)
    C = C.at[:, :4].set(1e4)
    C = C.at[-4:,:].set(1e4)
    C = jnp.where(thk==0, 1, C)

    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q


def twisty_stream():
    lx = 180_000
    ly = 180_000
    resolution = 1_000 #m
    
    
    stencil_radius = 1
    stencil_width = 2*stencil_radius+1
    
    nr = int(ly/resolution)
    nc = int(lx/resolution)
    
    assert nc % stencil_width == 0, "domain must be tileable by stencil width in periodic bcs case"
    
    x = jnp.linspace(0, lx, nc)
    y = jnp.linspace(0, ly, nr)
    
    delta_x = x[1]-x[0]
    delta_y = y[1]-y[0]


    thk = jnp.zeros((nr,nc)) + 1000
    #want a slope of 0.5 degrees.
    sine_0pt5 = 0.0087265
    
    #b_profile = 1000 - 500*x/lx
    b_profile = 1000 - sine_0pt5*x
    b = jnp.zeros((nr, nc)) + b_profile
    
    
    xs, ys = jnp.meshgrid(x,y)
    R = ly
    m = 0.25
    C = 1e3 * (1 + 5e-3 + jnp.sin(2*jnp.pi*(ys+R/4)/R + m*jnp.sin(2*jnp.pi*xs/R)))
    C = jnp.flipud(C)

    
    #mucoef_profile = 0.5+b_profile.copy()/2000
    mucoef_profile = 1
    mucoef_0 = jnp.zeros_like(b)+mucoef_profile
    
    #plt.imshow(mucoef)
    #plt.colorbar()
    #plt.show()
    #raise
    
    #mucoef = jnp.ones_like(C)
    #mucoef_0 = jnp.ones_like(C)
    q = jnp.zeros_like(C)
    
    return lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q

lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = twisty_stream()
#lx, ly, nr, nc, x, y, delta_x, delta_y, thk, b, C, mucoef_0, q = ice_shelf()






#fig, ax = plt.subplots()
#extent = (-0.5, nc - 0.5, nr - 0.5, -0.5)  # x from -0.5..nx-0.5, y top-to-bottom
#cf = ax.contourf(jnp.log10(C), cmap='Grays', extent=extent)
#ax.set_aspect('equal', adjustable='box')
#fig.colorbar(cf, ax=ax)
#plt.tight_layout()
#plt.show()
#raise








u_init = jnp.zeros_like(b) + 100
v_init = jnp.zeros_like(b)

n_iterations = 10

mask = jnp.where(thk>0,1,0)
mask = mask.astype(int)


def functional(v_field_x, v_field_y, q):
    #NOTE: for things like this, you need to ensure the value isn't
    #zero where the argument is zero, because JAX can't differentiate
    #through the square-root otherwise. Silly JAX.
    return jnp.sum(mask.reshape(-1) * jnp.sqrt(v_field_x**2 + v_field_y**2 + 1e-10).reshape(-1))


def calculate_gradient_via_foa():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                          nr, nc, delta_y, delta_x, thk, b, C, n_iterations, mucoef_0, periodic=True
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    #show_vel_field(u_out, v_out)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    return gradient

def calculate_gradient_via_ad():

    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, b, C,
                                                             n_iterations, mucoef_0,
                                                             periodic=True)

    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    gradient = get_grad(q)

    return gradient


#calculate_gradient_via_foa()
#raise


#g_foa = calculate_gradient_via_foa()
#g_ad = calculate_gradient_via_ad()
#
#plt.imshow((2*g_foa-g_ad)/g_foa, vmin=-3, vmax=3)
#plt.colorbar()
#plt.show()
#
#raise

def calculate_hvp_via_soa():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                       nr, nc, delta_y, delta_x, thk, b, C, n_iterations, mucoef_0, periodic=True
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    #show_vel_field(u_out, v_out)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      functional)
    
    
    plt.imshow(gradient[:,:])
    plt.title("gradient via adjoint")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    print("solving second-order adjoint problem:")
    pert_dir = gradient.copy()/(jnp.linalg.norm(gradient)*10)
    #pert_dir = pert_dir.at[:,-4:].set(0)
    hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir, functional)
    
    plt.imshow(hvp[:,:], vmin=-1, vmax=1, cmap="twilight_shifted")
    plt.title("hvp via soa")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()
    

def calculate_hvp_via_ad():

    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, b, C,
                                                             n_iterations, mucoef_0,
                                                             periodic=True)

    #u_out, v_out = solver(q, u_init, v_init)
    #show_vel_field(u_out, v_out, vmax=3500)
    #raise
    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    gradient = get_grad(q)
    
    plt.imshow(gradient[:,:], vmin=-1800,)
    plt.title("gradient via ad")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()

    #plt.plot(gradient[25,:])
    #plt.ylim((-600,600))
    #plt.show()

    
    pert_dir = gradient.copy() / (jnp.linalg.norm(gradient)*10)
    #pert_dir = pert_dir.at[:,-4:].set(0)

    ##finite diff hvp for comparison
    ##plt.imshow(p)
    ##plt.colorbar()
    ##plt.show()
    ##raise
    eps = 0.01
    fd_hvp = (get_grad(q + eps*pert_dir) - get_grad(q)) / eps
    plt.imshow(fd_hvp[:,:], vmin=-5, vmax=5, cmap="twilight_shifted")
    plt.title("hvp via fd")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()


    #I'd have imagined it's ok to do the following:
    #      hvp = jax.vjp(get_grad, (q,), (pert_dir,))
    #However, JAX seemingly doesn't support reverse mode AD for functions
    #that have custom vjps defined within them. IDK why!!
    #But people seem to reverse the order of the dot product and derivative computation.
    #That's actually Lemma 2 in the original Adjoints document I wrote so
    #that, at least, seems at least to be true!

    #def hessian(q_prime):
    #    return jax.jacrev(get_grad)(q_prime)

    #hess = hessian(q)

    #plt.imshow(hess)
    #plt.show()
    #raise

    def hess_vec_product(q, perturbation):
        return jax.grad(lambda m: jnp.vdot(get_grad(m), perturbation))(q)


    #hvp = hess_vec_product(q, pert_dir)
    #hvp = jax.jvp(get_grad, (q,), (pert_dir,))[1]
    #NOTE: This contradicts the above comment, but I'm not sure I got that right first time.
    _, vjp_grad = jax.vjp(get_grad, q)
    (hvp,) = vjp_grad(pert_dir)


    #plt.imshow((hvp/jnp.linalg.norm(hvp))[:,:])
    #plt.colorbar()
    #plt.show()


    #plt.imshow((gradient/jnp.linalg.norm(gradient))[:,:])
    #plt.colorbar()
    #plt.show()


    #plt.imshow((hvp/jnp.linalg.norm(hvp) - gradient/jnp.linalg.norm(gradient))[:,:35], cmap="Spectral_r")
    #plt.colorbar()
    #plt.show()

    
    plt.imshow(hvp[:,:], vmin=-5, vmax=5, cmap="twilight_shifted")
    #plt.imshow(hvp[:,:], vmin=-50, vmax=50, cmap="twilight_shifted")
    plt.title("hvp via ad")
    plt.colorbar()
    plt.imshow(jnp.where(C>1e10, 1, jnp.nan), cmap="Grays", alpha=0.5)
    plt.show()
    
    #plt.plot(hvp[25,:])
    #plt.ylim((-10,10))
    #plt.show()
    
    #plt.imshow(jnp.where(jnp.abs(hvp)>1, ((fd_hvp - hvp)/fd_hvp[:,:]), jnp.nan), vmin=-1, vmax=1, cmap="RdBu_r")
    #plt.title("fd-ad percentage difference")
    #plt.colorbar()
    #plt.show()




#calculate_hvp_via_ad()
#calculate_hvp_via_soa()


#raise








#solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
#                                                         delta_y,
#                                                         delta_x,
#                                                         thk, b, C,
#                                                         n_iterations, mucoef_0,
#                                                         periodic=True)
#
#u_out, v_out = solver(q, u_init, v_init)
###raise
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/linear/u_big_new.npy", u_out)
#jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/linear/v_big_new.npy", v_out)
##raise
#
#show_vel_field(u_out, v_out)



#u_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/u_big.npy")
#v_data = jnp.load("/users/eetss/new_model_code/src/nm/bits_of_data/hessian_evecs_etc/shelf/v_big.npy")

u_data = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/u_big_new.npy")
v_data = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/v_big_new.npy")
speed_data = jnp.sqrt(u_data**2 + v_data**2 + 1e-10)

#show_vel_field(u_data, v_data)

#def misfit_functional(v_field_x, v_field_y, q):
#
#    return jnp.sum(mask.reshape(-1) * \
#                   jnp.sqrt((v_field_x.reshape(-1)-u_data.reshape(-1))**2 +\
#                            (v_field_y.reshape(-1)-v_data.reshape(-1))**2 +\
#                            1e-10)
#                  )

def misfit_functional(u_mod, v_mod, q):

    return jnp.sum(mask.reshape(-1) * \
                   ( jnp.sqrt(u_mod**2 + v_mod**2).flatten() - speed_data.flatten() )**2
                  )


#misfit_functional = functional


def make_hvp_ad_fct():
    solver = make_newton_velocity_solver_function_custom_vjp(nr, nc,
                                                             delta_y,
                                                             delta_x,
                                                             thk, b, C,
                                                             n_iterations, mucoef_0,
                                                             periodic=True)
    def reduced_functional(q):
        u_out, v_out = solver(q, u_init, v_init)
        return misfit_functional(u_out, v_out, q)
        #return functional(u_out, v_out, q)

    get_grad = jax.grad(reduced_functional)

    _, vjp_grad = jax.vjp(get_grad, q)
    
    def hessian_vector_product(pert_dir):
        (hvp,) = vjp_grad(pert_dir.reshape((nr,nc)))
        return hvp

    return hessian_vector_product


def make_hvp_soa_fct():
    fwd_solver, adjoint_solver, soa_solver = forward_adjoint_and_second_order_adjoint_solvers(
                    nr, nc, delta_y, delta_x, thk, b, C, n_iterations, mucoef_0, periodic=True
                                                                                             )
    
    print("solving fwd problem:")
    u_out, v_out = fwd_solver(q, u_init, v_init)
    
    #show_vel_field(u_out, v_out)
    
    print("solving adjoint problem:")
    lx, ly, gradient = adjoint_solver(q, u_out, v_out,
                                      jnp.zeros_like(u_out),
                                      jnp.zeros_like(u_out),
                                      misfit_functional)
    
    
    def hessian_vector_product(pert_dir):
        hvp = soa_solver(q, u_out, v_out, lx, ly, pert_dir.reshape((nr,nc)), misfit_functional)
        return hvp

    return hessian_vector_product
    

#function for computing hvp from perturbation direction
#hessian_vector_product = make_hvp_soa_fct()





import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh





def compute_evecs_ad():

    hessian_vector_product = make_hvp_ad_fct()

    n = nr * nc
    dtype = np.float64
    
    # JIT-compile and warm up once to avoid recompiles in the loop
    #HVP_jit = jax.jit(lambda x: HVP(x.reshape(nr, nc)).reshape(-1))
    #_ = HVP_jit(np.zeros(n))   # warmup
    
    H = LinearOperator(
        shape=(n, n), dtype=dtype,
        matvec=lambda x: np.array(hessian_vector_product(x))  # ensure ndarray to avoid device transfers
    )
    
    # Largest algebraic eigenvalues (most positive)
    k = 100  # or 1000
    w, V = eigsh(H, k=k, which='LA', tol=1e-10, maxiter=1000)  # V[:, i] is eigenvector of w[i]
    
    # If you actually want largest magnitude (could pick big negative too), use which='LM'.
    # Normalize or reshape as needed:
    eigvals = w
    eigvecs = V.reshape(nr, nc, k)
        

    jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/ad_evecs.npy", eigvecs)
    jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/ad_evals.npy", eigvals)
    
    #for i in range(k):
        #evec = eigvecs[...,i]
    
        #eval_ = eigvals[...,i]
#        #print(eval_)
#    
#        evec_norm = evec / jnp.linalg.norm(evec)
#    
#        #print(evec_norm.shape, np.array(hessian_vector_product(evec_norm)).shape, (evec_norm*eval_).shape)
#    
#        #plt.imshow(eval_*evec)
#        #plt.colorbar()
#        #plt.show()
#    
#        #plt.imshow(np.array(hessian_vector_product(evec)))
#        #plt.colorbar()
#        #plt.show()
#    
#    
#        ev_residual = np.linalg.norm(np.array(hessian_vector_product(evec)) - evec*eval_)/(np.linalg.norm(evec)*np.abs(eval_))
#        print(ev_residual)
#    
#        #eval_img = hessian_vector_product(evec) - evec*eval_
#    
#        ##plt.imshow(eval_img, vmin=-2*jnp.abs(eval_), vmax=2*jnp.abs(eval_))
#        #plt.imshow(eval_img, vmin=-100, vmax=100)
#        #plt.colorbar()
#        #plt.show()
    
    
    
        #plt.imshow(eigvecs[...,i])
        #plt.colorbar()
        #plt.title("lambda={}".format(eigvals[i]))
        #plt.savefig("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/stream/very_new/ts_evec_soa_1km_{}.png".format(i))
        ##plt.show()
        #plt.close()
    
    print("DONE AD")

#plt.plot(eigvals[::-1])
#plt.show()

def compute_evecs_sosa():

    hessian_vector_product = make_hvp_soa_fct()

    n = nr * nc
    dtype = np.float64
    
    # JIT-compile and warm up once to avoid recompiles in the loop
    #HVP_jit = jax.jit(lambda x: HVP(x.reshape(nr, nc)).reshape(-1))
    #_ = HVP_jit(np.zeros(n))   # warmup
    
    H = LinearOperator(
        shape=(n, n), dtype=dtype,
        matvec=lambda x: np.array(hessian_vector_product(x))  # ensure ndarray to avoid device transfers
    )
    
    # Largest algebraic eigenvalues (most positive)
    k = 100  # or 1000
    w, V = eigsh(H, k=k, which='LA', tol=1e-10, maxiter=1000)  # V[:, i] is eigenvector of w[i]
    
    # If you actually want largest magnitude (could pick big negative too), use which='LM'.
    # Normalize or reshape as needed:
    eigvals = w
    eigvecs = V.reshape(nr, nc, k)
        

    jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/soa_evecs.npy", eigvecs)
    jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/soa_evals.npy", eigvals)

    print("DONE SOSA")


#compute_evecs_ad()
#compute_evecs_sosa()


#raise


#soa_evecs = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/soa_evecs.npy")
#soa_evals = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/soa_evals.npy")
#
#ad_evecs = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/ad_evecs.npy")
#ad_evals = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/whys_it_gone_wrong/stream/100/ad_evals.npy")

soa_evecs = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/soa_evecs.npy")
soa_evals = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/soa_evals.npy")

ad_evecs = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/ad_evecs.npy")
ad_evals = np.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/ad_evals.npy")

k = 500

evec_residuals = []

for evcs, evls, hvp in [[soa_evecs, soa_evals, make_hvp_soa_fct()],[ad_evecs, ad_evals, make_hvp_ad_fct()]]:
#for evcs, evls, hvp in [[ad_evecs, ad_evals, make_hvp_ad_fct()]]:
    for i in range(k):
        evec = evcs[:,:,-(i+1)]
        eval_ = evls[-(i+1)]

        evec_norm = evec / np.linalg.norm(evec)

        ev_residual = np.linalg.norm(np.array(hvp(evec)) - evec*eval_)/(np.linalg.norm(evec)*np.abs(eval_))

        evec_residuals.append(ev_residual)

soa_ers = evec_residuals[:k]
ad_ers = evec_residuals[k:]
#ad_ers = evec_residuals[:k]

jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/soa_eigen_errors.npy", jnp.array(soa_ers))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/ad_eigen_errors.npy", jnp.array(ad_ers))

#print(ad_ers)
#print(soa_ers)












