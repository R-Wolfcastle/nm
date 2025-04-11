
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax.scipy.linalg as lalg
from jax.scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


np.set_printoptions(precision=1, suppress=False, linewidth=np.inf)


def make_fou_step(u, dt, dx, n):

    def fou_step(h, h_old):

        h_change = jnp.zeros((n,))
        h_change = h_change.at[1:].set((dt/dx)*u*(h[:n-1]-h[1:]))

        return h_old - h + h_change

    return fou_step
        

def make_naive_step(u, dt, dx, n):

    def naive_step(h):

        h_change = jnp.zeros((n,))
        h_change = h_change.at[1:n-1].set( (dt/dx)*u*0.5*(h[:n-1]-h[2:]) )

        h_new = h + h_change

        return h_new
    return fou_step

def make_ppm_problem_experimental(u, dt, dx, n, steepening=True,\
                                  eta_1=20, eta_2=0.05, epsilon=0.01):

    #I think this is actually a terrible idea and makes the thing totally unstable...

    #assume u and h are colocated
    #n = u.shape[0]

    def ppm(h, h_old):
        #follow steps laid out in:
        #Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling
        #by carpenter et al., 1989
        #and stuff learned from Colella and Woodward 1984.

        #Boundary condition is that h is prescribed at the lhs of the domain.
        



        #STEP 1: 
        #Approximate piecewise linear profile.
        #central difference, left-sided diff, right-sided diff
        #located on cell centres
        h_grad_cd = jnp.zeros((n,))
        h_grad_lsd = jnp.zeros((n,))
        h_grad_rsd = jnp.zeros((n,))

        h_grad_cd  = h_grad_cd.at[1:n-1].set((h[2:] - h[:n-2])/(2*dx))
        h_grad_cd  = h_grad_cd.at[-1].set(0)
        h_grad_cd  = h_grad_cd.at[0].set(0)
        h_grad_lsd = h_grad_lsd.at[1:].set((h[1:] - h[:n-1])/dx)
        h_grad_lsd = h_grad_lsd.at[0].set(0)
        h_grad_rsd = h_grad_rsd.at[:n-1].set((h[1:] - h[:n-1])/dx)
        h_grad_rsd = h_grad_rsd.at[-1].set(0)

        h_grad_cd_mag = jnp.abs(h_grad_cd)
        h_grad_lsd_mag = jnp.abs(h_grad_lsd)
        h_grad_rsd_mag = jnp.abs(h_grad_rsd)

        min_mag_grads_cd_lsd = jnp.where(h_grad_cd_mag <= h_grad_lsd_mag, h_grad_cd, h_grad_lsd)
        min_mag_grads_cd_rsd = jnp.where(h_grad_cd_mag <= h_grad_rsd_mag, h_grad_cd, h_grad_rsd)
        min_mag_grads = jnp.where(min_mag_grads_cd_lsd <= min_mag_grads_cd_rsd,\
                                   min_mag_grads_cd_lsd, min_mag_grads_cd_rsd)







        #STEP 2:
        #First guess at face-centred values by fitting cubic to grads and values
        h_ls = jnp.zeros((n,))
        h_ls = h_ls.at[1:].set( (1/2)*(h[:n-1]+h[1:]) - (1/6)*(min_mag_grads[1:]*dx - min_mag_grads[:n-1]*dx) )

        h_rs = jnp.zeros((n,))
        h_rs = h_rs.at[:n-1].set(h_ls[1:])
        
        h_ls = h_ls.at[0].set(h[0]-0.5*(h_rs[0]-h[0]))
        h_rs = h_rs.at[-1].set(h[-1]+0.5*(h[-1]-h_rs[-1]))
        #Note, for these fields, each h as an h_l and an h_r, but the h_l for the first
        #grid point and the h_r for the last have been modified so that the linear interps
        #just give h in the centre. But I'm not sure what do do about it. Think more carefully.






        #STEP 3:
        #STEEPENING!
        if steepening:
        

        #This stuff has given way to just copying out of the paper.
        #    #alternative L and R values given by piecewise linear distribution:
            h_ls_alt = jnp.zeros((n,))
            h_ls_alt = h_ls_alt.at[1:].set(h[:n-1] + 0.5*min_mag_grads[:n-1]*dx)
    
            h_rs_alt = jnp.zeros((n,))
            h_rs_alt = h_rs_alt.at[:n-1].set(h[1:n] - 0.5*min_mag_grads[1:n]*dx)
    
        #    print(h_ls_alt)
        #    print(h_rs_alt)
    
    
        #    #second derivs at cell centres:
        #    delta2_h = jnp.zeros((n,))
        #    delta2_h = delta2_h.at[1:n-1].set( (h[2:] - 2*h[1:n-1] + h[:n-2])/(dx**2) )
    
        #    #third derivs at cell centres
        #    delta3_h = jnp.zeros((n,))
        #    delta3_h = delta3_h.at[2:n-2].set( (h[4:] - 2*h[3:n-1] + 2*h[1:n-3] - h[:n-4])/(2*dx**3) )
    
        #    print(delta2_h)
        #    print(delta3_h)
        #    raise
        #    #remember, don't steepen any small jumps that could just be numerical noise!
    
    
            del_squared_h = jnp.zeros((n,))
            del_squared_h = del_squared_h.at[1:n-1].set((1/6)/(dx**2)*( h[2:] -2*h[1:n-1] + h[:n-2]))
    
            eta_tilde = jnp.zeros((n,))
            eta_tilde = eta_tilde.at[1:n-1].set( -((dx**2)/(h[2:] - h[:n-2])) * (del_squared_h[2:] - del_squared_h[:n-2]) )
    
            eta_tilde = eta_tilde.at[1:n-1].set(
                    jnp.where(
                        (del_squared_h[2:]*del_squared_h[:n-2] < 0) &\
                        ((jnp.abs(h[2:]-h[:n-2]) - epsilon*jnp.minimum(jnp.abs(h[2:]), jnp.abs(h[:n-2]))) > 0),
                         eta_tilde[1:n-1], 0
                    )
                                                  )
    
    
            eta = jnp.maximum(0, jnp.minimum(eta_1*(eta_tilde-eta_2), 1))
    
    
            h_ls = h_ls*(1-eta) + h_ls_alt*eta
            h_rs = h_rs*(1-eta) + h_rs_alt*eta







        #STEP 4:
        #Adjust the L and R points so that there are no overshoots (local
        #extrema) in the parabolas for each cell.

        h_six = 6*( h - 0.5*( h_ls + h_rs ) )
        delta_h = ( h_rs - h_ls )

        condition_0 = jnp.where((jnp.abs(delta_h) < jnp.abs(h_six)), 1, 0)
        condition_1 = jnp.where((condition_0==1) & (( h_rs - h )*( h - h_ls ) < 0), 1, 0)
        condition_2 = jnp.where((condition_0==1) & ( delta_h**2 < (delta_h*h_six)), 1, 0)
        condition_3 = jnp.where((condition_0==1) & (-delta_h**2 > (delta_h*h_six)), 1, 0)

        condition_2 = jnp.where(((condition_2==1) & (condition_3==0) & (condition_1==0)), 1, 0)
        condition_3 = jnp.where(((condition_3==1) & (condition_2==0) & (condition_1==0)), 1, 0)

        #TODO: check whether conditions are true at boundaries and implications!


        #where condition 1, we have an extremum, so set hl and hr to h
        #to make the thing flat.
        h_ls = jnp.where(condition_1==1, h, h_ls)
        h_rs = jnp.where(condition_1==1, h, h_rs)

        #where condition 2, there is an overshoot close to the right-hand side
        #of the cell, so h_l should be adjusted so that the gradient of h(x)
        #is zero at the right-hand side of the cell (x=1):
        h_ls = jnp.where(condition_2==1, 3*h - 2*h_rs, h_ls)

        #where condition 3, other way round
        h_rs = jnp.where(condition_3==1, 3*h - 2*h_ls, h_rs)


        #redefine these with the new h_ls and h_rs
        h_six = 6*( h - 0.5*( h_ls + h_rs ) )
        delta_h = ( h_rs - h_ls )



        

        #STEP 5:
        #COMPUTE THE UPDATES TO h!!
        #soooo. The "experimental" version is me wondering whether you should think of things
        #going backwards in time ("ooooooooo"). 
        #Answer: no. It seems to make things unstable. Should that be obvious??
        y = u * dt / dx
        h_out = h_ls + 0.5*y*(delta_h + h_six*(1 - (2/3)*y))
        h_in  = h_out.copy()
        #h_in  = h_in.at[1:n-1].set(h_out[2:]) #keeping the left with equal in and out
        h_in  = h_in.at[:n-1].set(h_out[1:])


#        #This upwinded version is pointless:
#        y = u * dt / dx
#        h_out = jnp.zeros((n,))
#        h_out = h_out.at[1:].set((h_ls + 0.5*y*(delta_h + h_six*(1 - (2/3)*y)))[:n-1])
#        h_in  = h_out.copy()
#        h_in  = h_in.at[:n-1].set(h_out[1:])



        return h_old - h + y*(h_out - h_in)

    return ppm



def make_ppm_problem(u, dt, dx, n, steepening=True,\
                       eta_1=20, eta_2=0.05, epsilon=0.01):

    #assume u and h are colocated
    #n = u.shape[0]

    def ppm(h, h_old):
        #follow steps laid out in:
        #Application of the Piecewise Parabolic Method (PPM) to Meteorological Modeling
        #by carpenter et al., 1989
        #and stuff learned from Colella and Woodward 1984.

        #Boundary condition is that h is prescribed at the lhs of the domain.
        



        #STEP 1: 
        #Approximate piecewise linear profile.
        #central difference, left-sided diff, right-sided diff
        #located on cell centres
        h_grad_cd = jnp.zeros((n,))
        h_grad_lsd = jnp.zeros((n,))
        h_grad_rsd = jnp.zeros((n,))

        h_grad_cd  = h_grad_cd.at[1:n-1].set((h[2:] - h[:n-2])/(2*dx))
        h_grad_cd  = h_grad_cd.at[-1].set(0)
        h_grad_cd  = h_grad_cd.at[0].set(0)
        h_grad_lsd = h_grad_lsd.at[1:].set((h[1:] - h[:n-1])/dx)
        h_grad_lsd = h_grad_lsd.at[0].set(0)
        h_grad_rsd = h_grad_rsd.at[:n-1].set((h[1:] - h[:n-1])/dx)
        h_grad_rsd = h_grad_rsd.at[-1].set(0)

        h_grad_cd_mag = jnp.abs(h_grad_cd)
        h_grad_lsd_mag = jnp.abs(h_grad_lsd)
        h_grad_rsd_mag = jnp.abs(h_grad_rsd)

        min_mag_grads_cd_lsd = jnp.where(h_grad_cd_mag <= h_grad_lsd_mag, h_grad_cd, h_grad_lsd)
        min_mag_grads_cd_rsd = jnp.where(h_grad_cd_mag <= h_grad_rsd_mag, h_grad_cd, h_grad_rsd)
        min_mag_grads = jnp.where(min_mag_grads_cd_lsd <= min_mag_grads_cd_rsd,\
                                   min_mag_grads_cd_lsd, min_mag_grads_cd_rsd)







        #STEP 2:
        #First guess at face-centred values by fitting cubic to grads and values
        h_ls = jnp.zeros((n,))
        h_ls = h_ls.at[1:].set( (1/2)*(h[:n-1]+h[1:]) - (1/6)*(min_mag_grads[1:]*dx - min_mag_grads[:n-1]*dx) )

        h_rs = jnp.zeros((n,))
        h_rs = h_rs.at[:n-1].set(h_ls[1:])
        
        h_ls = h_ls.at[0].set(h[0]-0.5*(h_rs[0]-h[0]))
        h_rs = h_rs.at[-1].set(h[-1]+0.5*(h[-1]-h_rs[-1]))
        #Note, for these fields, each h as an h_l and an h_r, but the h_l for the first
        #grid point and the h_r for the last have been modified so that the linear interps
        #just give h in the centre. But I'm not sure what do do about it. Think more carefully.






        #STEP 3:
        #STEEPENING!
        if steepening:
        

        #This stuff has given way to just copying out of the paper.
        #    #alternative L and R values given by piecewise linear distribution:
            h_ls_alt = jnp.zeros((n,))
            h_ls_alt = h_ls_alt.at[1:].set(h[:n-1] + 0.5*min_mag_grads[:n-1]*dx)
    
            h_rs_alt = jnp.zeros((n,))
            h_rs_alt = h_rs_alt.at[:n-1].set(h[1:n] - 0.5*min_mag_grads[1:n]*dx)
    
        #    print(h_ls_alt)
        #    print(h_rs_alt)
    
    
        #    #second derivs at cell centres:
        #    delta2_h = jnp.zeros((n,))
        #    delta2_h = delta2_h.at[1:n-1].set( (h[2:] - 2*h[1:n-1] + h[:n-2])/(dx**2) )
    
        #    #third derivs at cell centres
        #    delta3_h = jnp.zeros((n,))
        #    delta3_h = delta3_h.at[2:n-2].set( (h[4:] - 2*h[3:n-1] + 2*h[1:n-3] - h[:n-4])/(2*dx**3) )
    
        #    print(delta2_h)
        #    print(delta3_h)
        #    raise
        #    #remember, don't steepen any small jumps that could just be numerical noise!
    
    
            del_squared_h = jnp.zeros((n,))
            del_squared_h = del_squared_h.at[1:n-1].set((1/6)/(dx**2)*( h[2:] -2*h[1:n-1] + h[:n-2]))
    
            eta_tilde = jnp.zeros((n,))
            eta_tilde = eta_tilde.at[1:n-1].set( -((dx**2)/(h[2:] - h[:n-2])) * (del_squared_h[2:] - del_squared_h[:n-2]) )
    
            eta_tilde = eta_tilde.at[1:n-1].set(
                    jnp.where(
                        (del_squared_h[2:]*del_squared_h[:n-2] < 0) &\
                        ((jnp.abs(h[2:]-h[:n-2]) - epsilon*jnp.minimum(jnp.abs(h[2:]), jnp.abs(h[:n-2]))) > 0),
                         eta_tilde[1:n-1], 0
                    )
                                                  )
    
    
            eta = jnp.maximum(0, jnp.minimum(eta_1*(eta_tilde-eta_2), 1))
    
    
            h_ls = h_ls*(1-eta) + h_ls_alt*eta
            h_rs = h_rs*(1-eta) + h_rs_alt*eta







        #STEP 4:
        #Adjust the L and R points so that there are no overshoots (local
        #extrema) in the parabolas for each cell.

        h_six = 6*( h - 0.5*( h_ls + h_rs ) )
        delta_h = ( h_rs - h_ls )

        condition_0 = jnp.where((jnp.abs(delta_h) < jnp.abs(h_six)), 1, 0)
        condition_1 = jnp.where((condition_0==1) & (( h_rs - h )*( h - h_ls ) < 0), 1, 0)
        condition_2 = jnp.where((condition_0==1) & ( delta_h**2 < (delta_h*h_six)), 1, 0)
        condition_3 = jnp.where((condition_0==1) & (-delta_h**2 > (delta_h*h_six)), 1, 0)

        condition_2 = jnp.where(((condition_2==1) & (condition_3==0) & (condition_1==0)), 1, 0)
        condition_3 = jnp.where(((condition_3==1) & (condition_2==0) & (condition_1==0)), 1, 0)

        #TODO: check whether conditions are true at boundaries and implications!


        #where condition 1, we have an extremum, so set hl and hr to h
        #to make the thing flat.
        h_ls = jnp.where(condition_1==1, h, h_ls)
        h_rs = jnp.where(condition_1==1, h, h_rs)

        #where condition 2, there is an overshoot close to the right-hand side
        #of the cell, so h_l should be adjusted so that the gradient of h(x)
        #is zero at the right-hand side of the cell (x=1):
        h_ls = jnp.where(condition_2==1, 3*h - 2*h_rs, h_ls)

        #where condition 3, other way round
        h_rs = jnp.where(condition_3==1, 3*h - 2*h_ls, h_rs)


        #redefine these with the new h_ls and h_rs
        h_six = 6*( h - 0.5*( h_ls + h_rs ) )
        delta_h = ( h_rs - h_ls )



        

        #STEP 5:
        #COMPUTE THE UPDATES TO h!!
        y = u * dt / dx
        h_out = h_rs - 0.5*y*(delta_h - h_six*(1 - (2/3)*y))
        h_in  = h_out.copy()
        h_in  = h_in.at[1:].set(h_out[:n-1])


        return h - h_old + y*(h_out - h_in)

    return ppm



def solve_advection_by_newtons_method(h_init, u, dt, dx, n, num_timesteps, num_iterations):
    
    #this is not actually time dependent in this case as
    #u is not a function of h, nor is dx etc.
    #G = make_ppm_problem(u, dt, dx, n, steepening=True)
    #G = make_fou_step(u, dt, dx, n)
    G = make_ppm_problem_experimental(u, dt, dx, n, steepening=True)

    #jac_fn = jacfwd(G, argnums=(0))
    #3x speed-up
    jac_fn = jax.jit(jacfwd(G, argnums=(0)))
  
    
    #inintial initial guess
    h = h_init.copy()
    #after this, the initial guess just becomes the solution at
    #the previous timestep... I guess.

    for j in range(num_timesteps):
        print(j)
        h_old = h.copy()
    
        for i in range(num_iterations):
            jac = jac_fn(h, h_old)
            
            rhs = -G(h, h_old)
    
            delta_h = lalg.solve(jac, rhs)
    
            h = h.at[:].set(h + delta_h)
        
        if not j%5:
            plot_h(h, jnp.zeros_like(h)*jnp.nan, "implicit_experimental_ppm_{}".format(j))

    
    return h








def make_gif(filepaths, id_):
    import imageio
    with imageio.get_writer('../misc/advec_tests/ppm_test_{}.gif'.format(id_), mode='I', duration=0.5) as writer:
        for filename in filepaths:
            image = imageio.imread(filename)
            writer.append_data(image)



def plot_h(h, h_analytic, id_):
    plt.figure(figsize=(8,5))
    plt.plot(h_analytic, color='k', linestyle='dashed')
    plt.plot(h, color='b')
    plt.ylim(-0.5,1.5)
    plt.savefig("../misc/advec_tests/implicit/ppm_test_experimental_ppm_{}.png".format(id_))
    plt.close()



if __name__ == "__main__":
    x = jnp.linspace(0,1,50)
    dt = 5e-2
    u = 0.02
    
    h_init = 0.5*(1+ jnp.tanh(100*(x-0.25)))

    num_iterations = 10
    num_timesteps = 121

    h = solve_advection_by_newtons_method(h_init, u, dt, x[1]-x[0], h_init.shape[0], num_timesteps, num_iterations)
    
    h_analytic = 0.5*(1+ jnp.tanh(100*(x-0.25-u*num_timesteps*dt)))
    
    #plot_h(h, h_analytic, "implicit_experimental_ppm_100")




