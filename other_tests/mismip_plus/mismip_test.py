#1st party
import os
import sys


#3rd party
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import *
from standard_domains import mismip_domain
from plotting_stuff import show_vel_field, show_vel_field_2

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_picnewton_velocity_solver_function_full_cvjp,\
                              make_time_marcher,\
                              make_coupled_quasi_newton_solver_function,\
                              make_coupled_picnewton_solver_function,\
                              implicit_forward_solver



resolution = 1000
n_levels = 50
n_pic_iterations = 15
n_newt_iterations = 8
n_timesteps = 4

(
    lx, ly, nr, nc,
    x, y, delta_x,
    delta_y, thk, b,
    C_0, mucoef_0, q,
    ice_mask, surface,
    grounded
) = mismip_domain(resolution=resolution)
p = jnp.zeros_like(q)


#solver = make_coupled_quasi_newton_solver_function(nr, nc,delta_y,
#                                              delta_x,
#                                              b,
#                                              ice_mask,
#                                              n_pic_iterations,
#                                              mucoef_0, C_0,
#                                              sliding="basic_weertman")
#solver = make_coupled_picnewton_solver_function(nr, nc,delta_y,
#                                                delta_x,
#                                                b,
#                                                ice_mask,
#                                                n_pic_iterations,
#                                                n_newt_iterations,
#                                                mucoef_0, C_0,
#                                                sliding="basic_weertman")
solver = implicit_forward_solver(nr, nc,delta_y,
                                 delta_x,
                                 b,
                                 ice_mask,
                                 n_pic_iterations,
                                 n_newt_iterations,
                                 mucoef_0, C_0,
                                 sliding="basic_weertman")

delta_t = 50

u_trial = jnp.zeros_like(C_0)
v_trial = jnp.zeros_like(u_trial)

u_va, v_va, thk_final = solver(q, p, u_trial, v_trial, thk, delta_t, n_timesteps, accm=0.3)

show_vel_field(u_va, v_va)

grounded = jnp.where((thk_final+b)>(thk_final*(1-c.RHO_I/c.RHO_W)), 1, 0)

plt.imshow(grounded)
plt.show()

plt.imshow(thk_final-thk)
plt.colorbar()
plt.show()





raise
############ EXPLICIT:


momentum_solver, advection_stepper = make_picnewton_velocity_solver_function_full_cvjp(nr, nc,
                                              delta_y,
                                              delta_x,
                                              b,
                                              ice_mask,
                                              n_pic_iterations,
                                              n_newt_iterations,
                                              mucoef_0,
                                              C,
                                              sliding="basic_weertman",
                                              temperature_field=None,
                                            )

time_marcher = make_time_marcher(momentum_solver, advection_stepper, n_timesteps, delta_x)

u_va, v_va, thk_final, dhdt_final = time_marcher(q, p, thk)


show_vel_field(u_va, v_va)

grounded = jnp.where((thk_final+b)>(thk_final*(1-c.RHO_I/c.RHO_W)), 1, 0)

plt.imshow(grounded)
plt.show()

plt.imshow(thk_final)
plt.show()


plt.imshow(dhdt_final, vmin=-1, vmax=1, cmap="RdBu_r")
plt.colorbar()
plt.show()


raise


def functional(q_deriv):
    u_va, v_va, u_vv, v_vv, zs = momentum_solver(q_deriv, p,
                                                 jnp.zeros_like(q),
                                                 jnp.zeros_like(q),
                                                 thk)
    return jnp.sum(u_va**2)

dJ_dq = jax.grad(functional)(q)

plt.imshow(dJ_dq)
plt.colorbar()
plt.show()

raise

print(f"max |dhdt| ={float(jnp.max(jnp.abs(dhdt_final)))}")
#thk_final = jnp.load(f"{nm_home}/bits_of_data/DIVA/mismip_ss/1/thickness_{resolution}m_{n_timesteps}.npy")

grounded = jnp.where((thk_final+b)>(thk_final*(1-c.RHO_I/c.RHO_W)), 1, 0)

plt.imshow(grounded)
plt.show()

plt.imshow(thk_final)
plt.show()

raise

plt.imshow(dhdt_final, cmap="RdBu", vmin=-10, vmax=10)
plt.colorbar()
plt.show()

show_vel_field_2(u_va, v_va, vmin=0)

plt.imshow(thk)
plt.colorbar()
plt.show()

plt.imshow(h_final)
plt.colorbar()
plt.show()


z_coordinates = define_z_coordinates(b, thk, n_levels)

#diff_from_va = u_vv[64,:,:] - u_va[64,:,None]
#diff_from_va = u_vv[64,:,:] - vertically_average(u_vv, z_coordinates)[64,:, None]
diff_from_ssa = u_vv[64,:,:] - u_ssa[64,:,None]

X, Z = jnp.meshgrid(x, jnp.arange(n_levels), indexing='ij')
Z = z_coordinates[64,:,:]

#percentage_diff_from_ssa = (100/u_va_ssa[...,None])*(u_vv-u_va_ssa[...,None])

# Plot difference in vert profile from mean
plt.figure(figsize=(8, 4))
#contour = plt.contourf(X, Z, diff_from_va, levels=101, cmap='RdBu_r', vmin=-120, vmax=120)
contour = plt.contourf(X, Z, diff_from_ssa, levels=101, cmap='RdBu_r', vmin=-1000, vmax=1000)
#contour = plt.contourf(X, Z, jnp.abs(diff_from_va), levels=101, cmap='gnuplot2', vmin=0)
plt.colorbar(contour, label='Speed diff (m/a)')

plt.ylabel('Elevation (m)')
for i in range(z_coordinates.shape[-1]):
    plt.plot(x, z_coordinates[64,:,i], c="k", alpha=0.1)
plt.show()






raise


#SPIN UP STUFF THAT'S NOT QUITE WORKED!!


def interpolate_to_new_grid(x_old, y_old, field_old, x_new, y_new):

    interp = RegularGridInterpolator(
        (y_old, x_old),
        jnp.asarray(field_old),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    Xn, Yn = jnp.meshgrid(x_new, y_new)

    pts = jnp.column_stack([
        Yn.ravel(),
        Xn.ravel()
    ])

    return jnp.asarray(
        interp(pts).reshape(len(y_new), len(x_new))
    )

def spin_up():

    n_levels = 50
    n_iterations = 40

    resolutions = [8000, 4000, 2000, 1000, 500]

    h_init = None
    u_init = None
    v_init = None

    for i, resn in enumerate(resolutions):

        print(f"\nResolution = {resn} m")

        (
            lx, ly, nr, nc,
            x, y, delta_x,
            delta_y, thk, b,
            C, mucoef_0, q,
            ice_mask, surface,
            grounded
        ) = mismip_domain(resolution=resn)

        
        if i == 0:

            h_init = thk
            u_init = jnp.ones_like(thk) * 100.0
            v_init = jnp.zeros_like(thk)

        else:

            h_init = interpolate_to_new_grid(
                x_prev, y_prev, h_final,
                x, y
            )

            u_init = interpolate_to_new_grid(
                x_prev, y_prev, u_va,
                x, y
            )

            v_init = interpolate_to_new_grid(
                x_prev, y_prev, v_va,
                x, y
            )

        n_timesteps = max(3, int(10 / (2 ** i)))

        print(f"iterations = {n_iterations}")

        solver = make_diva3d_solver(nr, nc,
                                                      delta_y,
                                                      delta_x,
                                                      n_levels,
                                                      b,
                                                      ice_mask,
                                                      n_iterations,
                                                      mucoef_0,
                                                      sliding="linear",
                                                      temperature_field=None,
                                                      n_timesteps=n_timesteps
                                                    )

        (u_va, v_va, u_vv, v_vv, zs,
            h_final, dhdt_final) = solver(q,
                                          C,
                                          u_init,
                                          v_init,
                                          h_init
                                      )

        print(
            "max |dhdt| =",
            float(jnp.max(jnp.abs(dhdt_final)))
        )

        x_prev = x
        y_prev = y
        

        plt.imshow(dhdt_final, cmap="RdBu", vmin=-10, vmax=10)
        plt.colorbar()
        plt.show()
        
        show_vel_field_2(u_va, v_va, vmin=0)
        
        plt.imshow(thk)
        plt.colorbar()
        plt.show()
        
        plt.imshow(h_final)
        plt.colorbar()
        plt.show()

    return x, y, h_final, u_va, v_va, dhdt_final


spin_up()

#def spin_up():
#
#    n_levels = 50
#        
#    u_init = jnp.zeros_like(C)+100
#    v_init = jnp.zeros_like(C)
#
#    for i, resn in enumerate([8000, 4000, 2000, 1000, 500]):
#        
#        lx, ly, nr, nc,\
#        x, y, delta_x,\
#        delta_y, thk, b,\
#        C, mucoef_0, q,\
#        ice_mask, surface,\
#        grounded = mismip_domain(resolution=resn)
#        
#
#        n_iterations = 80/(2**i)
#
#
#        solver = make_diva3d_solver(nr, nc, delta_y, delta_x, n_levels,
#                                              b, ice_mask, n_iterations,
#                                              mucoef_0, sliding="linear",
#                                              temperature_field=None,
#                                              n_timesteps=1)
#
#        u_va, v_va, u_vv, v_vv, zs, h_final, dhdt_final = solver(q, C, u_init, v_init, thk)
#
#
#
#        plt.imshow(dhdt_final, cmap="RdBu", vmin=-10, vmax=10)
#        plt.colorbar()
#        plt.show()
#        
#        show_vel_field_2(u_va, v_va, vmin=0)
#        
#        plt.imshow(thk)
#        plt.colorbar()
#        plt.show()
#        
#        plt.imshow(h_final)
#        plt.colorbar()
#        plt.show()


