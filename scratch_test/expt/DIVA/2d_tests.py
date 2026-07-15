#1st party
import os
import sys


#3rd party
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

#local apps
nm_home = os.environ['NM_HOME']   

sys.path.insert(1, os.path.join(nm_home, 'utils'))
import constants_years as c
from vertical_grid import *
from standard_domains import wonky_stream
from plotting_stuff import show_vel_field

sys.path.insert(1, os.path.join(nm_home, 'solvers'))
from nonlinear_solvers import make_diva3d_velocity_solver_function,\
                              make_picnewton_velocity_solver_function_full_cvjp




lx, ly, nr, nc,\
x, y, delta_x,\
delta_y, thk, b,\
C, mucoef_0, q,\
ice_mask, surface,\
grounded = wonky_stream(resolution=1000)

n_levels = 200
n_iterations = 60
u_init = jnp.zeros_like(C)+100
v_init = jnp.zeros_like(C)


solver = make_diva3d_velocity_solver_function(nr, nc, delta_y, delta_x, n_levels,
                                              b, ice_mask, n_iterations,
                                              mucoef_0, sliding="linear",
                                              temperature_field=None)


ssa_solver = make_picnewton_velocity_solver_function_full_cvjp(nr, nc, delta_y, delta_x,
                                                               b, ice_mask, 12, 8,
                                                               mucoef_0, C, sliding="linear",
                                                               temperature_field=None)

u_ssa, v_ssa = ssa_solver(q, jnp.zeros_like(q), u_init, v_init, thk)

#show_vel_field(u_ssa, v_ssa, vmin=0, vmax=15_000, cmap="RdYlBu_r")

u_va, v_va, u_vv, v_vv, zs = solver(q, C, u_init, v_init, thk)

#show_vel_field(u_va, v_va, vmin=0, vmax=15_000, cmap="RdYlBu_r")

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





