#3rd party
import jax
import jax.numpy as jnp



DIRS = jnp.array([
    [-1,  0],  # N
    [ 1,  0],  # S
    [ 0, -1],  # W
    [ 0,  1],  # E
    [-1, -1],  # NW
    [-1,  1],  # NE
    [ 1, -1],  # SW
    [ 1,  1],  # SE
], dtype=jnp.int32)




def interp_cc_with_ghosts_to_fc_function(ny, nx):
    def interp_cc_to_fc(var):
        
        var_ew = 0.5*(var[1:-1, 1:]+var[1:-1, :-1])
        var_ns = 0.5*(var[:-1, 1:-1]+var[1:, 1:-1])

        return var_ew, var_ns
    return jax.jit(interp_cc_to_fc)


def interp_cc_to_fc_function(ny, nx):

    def interp_cc_to_fc(var):
        
        var_ew = jnp.zeros((ny, nx+1))
        var_ew = var_ew.at[:, 1:-1].set(0.5*(var[:, 1:]+var[:, :-1]))
        var_ew = var_ew.at[:, 0].set(var[:, 0])
        var_ew = var_ew.at[:, -1].set(var[:, -1])

        var_ns = jnp.zeros((ny+1, nx))
        var_ns = var_ns.at[1:-1, :].set(0.5*(var[:-1, :]+var[1:, :]))
        var_ns = var_ns.at[0, :].set(var[0, :])
        var_ns = var_ns.at[-1, :].set(var[-1, :])

        return var_ew, var_ns

    return jax.jit(interp_cc_to_fc)


def cc_gradient_function(dy, dx):

    def cc_gradient(var):

        dvar_dx = (0.5/dx) * (var[1:-1, 2:] - var[1:-1,:-2])
        dvar_dy = (0.5/dy) * (var[:-2,1:-1] - var[2:, 1:-1])

        return dvar_dx, dvar_dy

    return jax.jit(cc_gradient)

def fc_gradient_functions(dy, dx):

    def ew_face_gradient(var):
        
        dvar_dx_ew = (var[1:-1, 1:] - var[1:-1, :-1])/dx

        dvar_dy_ew = (var[:-2, 1:] + var[:-2, :-1] - var[2:, 1:] - var[2:, :-1])/(4*dy)
        
        return dvar_dx_ew, dvar_dy_ew
    
    def ns_face_gradient(var):
        
        dvar_dy_ns = (var[:-1, 1:-1]-var[1:, 1:-1])/dy

        dvar_dx_ns = (var[:-1, 2:] + var[1:, 2:] - var[:-1, :-2] - var[1:, :-2])/(4*dx)
        
        return dvar_dx_ns, dvar_dy_ns
    
    return jax.jit(ew_face_gradient), jax.jit(ns_face_gradient)


def add_ghost_cells_fcts(ny, nx):

    def add_reflection_ghost_cells(u_int, v_int):

        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #edges
        u = u.at[0, 1:-1].set( u[1, 1:-1])
        u = u.at[-1,1:-1].set( u[-2,1:-1])
        u = u.at[1:-1, 0].set(-u[1:-1, 1])
        u = u.at[1:-1,-1].set(-u[1:-1,-2])
        #corner points
        u = u.at[0,0].set(-u[1,1])
        u = u.at[-1,-1].set(-u[-2,-2])
        u = u.at[-1,0].set(-u[-2,1])
        u = u.at[0,-1].set(-u[1,-2])

        
        v = jnp.zeros((ny+2, nx+2))
        v = v.at[1:-1,1:-1].set(v_int)
        #edges
        v = v.at[0, 1:-1].set(-v[1, 1:-1])
        v = v.at[-1,1:-1].set(-v[-2,1:-1])
        v = v.at[1:-1, 0].set( v[1:-1, 1])
        v = v.at[1:-1,-1].set( v[1:-1,-2])
        #corner points
        v = v.at[0,0].set(-v[1,1])
        v = v.at[-1,-1].set(-v[-2,-2])
        v = v.at[-1,0].set(-v[-2,1])
        v = v.at[0,-1].set(-v[1,-2])

        return u, v

    def add_continuation_ghost_cells(h_int):

        h = jnp.zeros((ny+2, nx+2))
        h = h.at[1:-1,1:-1].set(h_int)
        #edges
        h = h.at[0, 1:-1].set(h[1, 1:-1])
        h = h.at[-1,1:-1].set(h[-2,1:-1])
        h = h.at[1:-1, 0].set(h[1:-1, 1])
        h = h.at[1:-1,-1].set(h[1:-1,-2])
        #corner points
        h = h.at[0,0].set(h[1,1])
        h = h.at[-1,0].set(h[-2,1])
        h = h.at[0,-1].set(h[1,-2])
        h = h.at[-1,-1].set(h[-2,-2])

        return h
    
    return jax.jit(add_reflection_ghost_cells), jax.jit(add_continuation_ghost_cells)
    
def add_ghost_cells_periodic_dirichlet_function(ny, nx):
    def add_ghost_cells_periodic_x_dirchlet_y(u_int):

        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #left/right edges: periodic bcs
        u = u.at[:, 0].set(u[:,-2])
        u = u.at[:,-1].set(u[:, 1])
        #top/bottom edges: dirchlet bcs
        u = u.at[0, :].set(-u[1, :])
        u = u.at[-1,:].set(-u[-2,:])

        return u
    return jax.jit(add_ghost_cells_periodic_x_dirchlet_y)

def add_ghost_cells_periodic_continuation_function(ny, nx):
    def add_ghost_cells_periodic_x_continuation_y(u_int):
        u = jnp.zeros((ny+2, nx+2))
        u = u.at[1:-1,1:-1].set(u_int)
        #left/right edges: periodic bcs
        u = u.at[:, 0].set(u[:,-2])
        u = u.at[:,-1].set(u[:, 1])
        #top/bottom edges: continuation bcs
        u = u.at[0, :].set(u[1, :])
        u = u.at[-1,:].set(u[-2,:])
        return u
    return jax.jit(add_ghost_cells_periodic_x_continuation_y)

def apply_scalar_ghost_cells_to_vector(scalar_ghost_function):
    def apply(u,v):
        u = scalar_ghost_function(u)
        v = scalar_ghost_function(v)
        return u, v
    return jnp.jit(apply)

def binary_erosion(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    kernel = jnp.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]], dtype=jnp.bool_)
    #kernel = jnp.array([[0,1,0],
    #                    [1,1,1],
    #                    [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float64)

    def erode_once(mask_float):
        out = jax.lax.conv_general_dilated(
            #shape (batch_size,channels,H,W)
            mask_float[None, None, :, :],
            #shape (out_chan,in_chan,H,W)
            kernel[None, None, :, :],
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NCHW','OIHW','NCHW')
        )
        return (out[0,0] > 8).astype(jnp.bool_)

    return erode_once(boolean_array.astype(jnp.float64))

def binary_dilation(boolean_array):
    # 3x3 cross-shaped structuring element (4-connectivity)
    kernel = jnp.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], dtype=jnp.bool_)

    kernel = kernel.astype(jnp.float64)

    def dilate_once(mask_float):
        out = jax.lax.conv_general_dilated(
            #shape (batch_size,channels,H,W)
            mask_float[None, None, :, :],
            #shape (out_chan,in_chan,H,W)
            kernel[None, None, :, :],
            window_strides=(1,1),
            padding='SAME',
            dimension_numbers=('NCHW','OIHW','NCHW')
        )
        return (out[0,0] > 0).astype(jnp.bool_)

    return dilate_once(boolean_array.astype(jnp.float64))


#def extrapolate_over_cf_function(thk):
#    def extrapolate_over_cf(cc_field):
#        return cc_field
#    return jax.jit(extrapolate_over_cf)

def safe_neighbours(cc_field):
    # cc_field shape (ny, nx)
    up    = jnp.pad(cc_field, ((0,1),(0,0)))[1: , :]   # shifted up
    down  = jnp.pad(cc_field, ((1,0),(0,0)))[:-1, :]   # shifted down
    left  = jnp.pad(cc_field, ((0,0),(0,1)))[:, 1: ]   # shifted left
    right = jnp.pad(cc_field, ((0,0),(1,0)))[:, :-1]   # shifted right
    return up, down, left, right


def stack_safe_shifted(cc_field):
    # Pad with 2 cells on all sides so we can slice safely
    cc_field_pad = jnp.pad(cc_field, ((2, 2), (2, 2)), constant_values=0)

    # 1-cell shifts (cardinal + diagonals)
    u1 = jnp.stack([
        cc_field_pad[1:-3, 2:-2],  # up 1
        cc_field_pad[3:-1, 2:-2],  # down 1
        cc_field_pad[2:-2, 1:-3],  # left 1
        cc_field_pad[2:-2, 3:-1],  # right 1
        cc_field_pad[1:-3, 1:-3],  # up-left 1
        cc_field_pad[1:-3, 3:-1],  # up-right 1
        cc_field_pad[3:-1, 1:-3],  # down-left 1
        cc_field_pad[3:-1, 3:-1],  # down-right 1
    ])

    # 2-cell shifts (cardinal + diagonals)
    u2 = jnp.stack([
        cc_field_pad[0:-4, 2:-2],  # up 2
        cc_field_pad[4:,   2:-2],  # down 2
        cc_field_pad[2:-2, 0:-4],  # left 2
        cc_field_pad[2:-2, 4:],    # right 2
        cc_field_pad[0:-4, 0:-4],  # up-left 2
        cc_field_pad[0:-4, 4:],    # up-right 2
        cc_field_pad[4:,   0:-4],  # down-left 2
        cc_field_pad[4:,   4:],    # down-right 2
    ])

    return u1, u2


def nn_extrapolate_over_cf_function(thk):
    
    cf_adjacent_zero_ice_cells = (thk==0) & binary_dilation(thk>0)

    ice_mask = (thk>0)

    ice_mask_shift_up, ice_mask_shift_down, ice_mask_shift_left, ice_mask_shift_right = safe_neighbours(ice_mask)

    def extrapolate_over_cf(cc_field):

        u_shift_up, u_shift_down, u_shift_left, u_shift_right = safe_neighbours(cc_field)
        
        neighbour_values = jnp.stack([
            jnp.where(ice_mask_shift_up   ==1, u_shift_up,    0),
            jnp.where(ice_mask_shift_down ==1, u_shift_down,  0),
            jnp.where(ice_mask_shift_left ==1, u_shift_left,  0),
            jnp.where(ice_mask_shift_right==1, u_shift_right, 0),
        ])
        
        #neighbour_counts = jnp.stack([
        #    (ice_mask_shift_up   ==1).astype(int),
        #    (ice_mask_shift_down ==1).astype(int),
        #    (ice_mask_shift_left ==1).astype(int),
        #    (ice_mask_shift_right==1).astype(int),
        #]).sum(axis=0)
        #neighbour_counts = jnp.where(neighbour_counts == 0, 1, neighbour_counts)
        #

        ##NOTE: STEPH!!
        ##Including this factor of 2 screws the gradient as computed by the HVP, but fixes
        ##the gradient as computed by the adjoint models. Make of that what you will...
        ##u_extrap_boundary = 2 * neighbour_values.sum(axis=0) / neighbour_counts.sum(axis=0)
        #u_extrap_boundary = neighbour_values.sum(axis=0) / neighbour_counts
        #u_extrap_boundary = jnp.where(neighbour_counts==0, 0, u_extrap_boundary)
        ##Think about it... 
        
        u_extrap_boundary = jnp.take_along_axis(neighbour_values,
                                                jnp.argmax(jnp.abs(neighbour_values), axis=0)[None, ...],
                                                axis=0)[0]

        return cc_field + u_extrap_boundary*cf_adjacent_zero_ice_cells.astype(jnp.float64)

    return extrapolate_over_cf

def extrapolate_over_cf_dynamic_thickness(cc_field, thk):
    cf_adjacent_zero_ice_cells = (thk==0) & binary_dilation(thk>0)

    ice_mask = (thk>0)

    ice_mask_shift_up, ice_mask_shift_down, ice_mask_shift_left, ice_mask_shift_right = safe_neighbours(ice_mask)


    u_shift_up, u_shift_down, u_shift_left, u_shift_right = safe_neighbours(cc_field)
    
    neighbour_values = jnp.stack([
        jnp.where(ice_mask_shift_up   ==1, u_shift_up,    0),
        jnp.where(ice_mask_shift_down ==1, u_shift_down,  0),
        jnp.where(ice_mask_shift_left ==1, u_shift_left,  0),
        jnp.where(ice_mask_shift_right==1, u_shift_right, 0),
    ])
    
    #neighbour_counts = jnp.stack([
    #    (ice_mask_shift_up   ==1).astype(int),
    #    (ice_mask_shift_down ==1).astype(int),
    #    (ice_mask_shift_left ==1).astype(int),
    #    (ice_mask_shift_right==1).astype(int),
    #]).sum(axis=0)
    #neighbour_counts = jnp.where(neighbour_counts == 0, 1, neighbour_counts)
    #

    ##NOTE: STEPH!!
    ##Including this factor of 2 screws the gradient as computed by the HVP, but fixes
    ##the gradient as computed by the adjoint models. Make of that what you will...
    ##u_extrap_boundary = 2 * neighbour_values.sum(axis=0) / neighbour_counts.sum(axis=0)
    #u_extrap_boundary = neighbour_values.sum(axis=0) / neighbour_counts
    #u_extrap_boundary = jnp.where(neighbour_counts==0, 0, u_extrap_boundary)
    ##Think about it... 
    
    u_extrap_boundary = jnp.take_along_axis(neighbour_values,
                                            jnp.argmax(jnp.abs(neighbour_values), axis=0)[None, ...],
                                            axis=0)[0]

    return cc_field + u_extrap_boundary*cf_adjacent_zero_ice_cells.astype(jnp.float64)


def cf_cells(thk):
    return (thk>0) & ~binary_erosion(thk>0)


def linear_extrapolate_over_cf_dynamic_thickness(cc_field, thk):
    """
    Extrapolate into ocean cells adjacent to ice by linear extrapolation along
    the direction with the longest contiguous ice run.

    u1 is value in last ice filled cell
    u2 is value one cell inside that

    u_extrp = 2*u1 - u2
    """

    ice_mask = (thk>0)
    cf_adjacent_zero_ice_cells = ~ice_mask & binary_dilation(thk>0)

    u1, u2 = stack_safe_shifted(cc_field)
    has_1, has_2 = stack_safe_shifted(ice_mask)

    score = has_1.astype(jnp.int32) + has_2.astype(jnp.int32)

    # Pick best direction
    best_k = jnp.argmax(score, axis=0)

    # Extrapolate: if has_2 then 2*u1 - u2 else u1
    u_extrap_dirs = jnp.where(has_2, 2.0*u1 - u2, u1)
    u_extrap = jnp.take_along_axis(u_extrap_dirs, best_k[None, ...], axis=0)[0]

    return jnp.where(cf_adjacent_zero_ice_cells, u_extrap, cc_field)


def linear_extrapolate_over_cf_function(thk):
    """
    Extrapolate into ocean cells adjacent to ice by linear extrapolation along
    the direction with the longest contiguous ice run.

    u1 is value in last ice filled cell
    u2 is value one cell inside that

    u_extrp = 2*u1 - u2
    """

    ice_mask = (thk>0)
    cf_adjacent_zero_ice_cells = ~ice_mask & binary_dilation(thk>0)


    
    @jax.jit
    def linear_extrapolate_over_cf(cc_field):

        cc_field_pad = jnp.pad(cc_field, ((2,2),(2,2)), constant_values=0)
        
        u1, u2 = stack_safe_shifted(cc_field)
        has_1, has_2 = stack_safe_shifted(ice_mask)
    
        # Score = number of available ice cells inward (prefer directions with both u1 and u2)
        score = has_1.astype(jnp.int32) + has_2.astype(jnp.int32)
    
        # Pick best direction
        best_k = jnp.argmax(score, axis=0)
    
        # Extrapolate: if has_2 then 2*u1 - u2 else u1
        u_extrap_dirs = jnp.where(has_2, 2.0*u1 - u2, u1)
        u_extrap = jnp.take_along_axis(u_extrap_dirs, best_k[None, ...], axis=0)[0]
    
        return jnp.where(cf_adjacent_zero_ice_cells, u_extrap, cc_field)
    return linear_extrapolate_over_cf



#test_thk = jnp.ones((10,10))
#test_thk = test_thk.at[:,-1].set(0)
#test_array = jnp.ones_like(test_thk)*jnp.linspace(0,8,10)
#test_array = test_array.at[:,-1].set(0)
#print(test_array)
#
#e = extrapolate_over_cf_function(test_thk)
#
#ta_ex = e(test_array)
#print(ta_ex)
#raise

@jax.jit
def double_dot_contraction(A, B):
    return A[:,:,0,0]*B[:,:,0,0] + A[:,:,1,0]*B[:,:,0,1] +\
           A[:,:,0,1]*B[:,:,1,0] + A[:,:,1,1]*B[:,:,1,1]


def cc_vector_field_gradient_function(ny, nx, dy, dx, cc_grad,
                                      extrp_over_cf,
                                      add_uv_ghost_cells):
    def cc_vector_field_gradient(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u = extrp_over_cf(u)
        v = extrp_over_cf(v)

        u, v = add_uv_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)

        #dudx = jnp.where(thk>0, dudx, 0)
        #dvdx = jnp.where(thk>0, dvdx, 0)
        #dudy = jnp.where(thk>0, dudy, 0)
        #dvdy = jnp.where(thk>0, dvdy, 0)

        grad_vf = jnp.zeros((ny, nx, 2, 2))

        grad_vf = grad_vf.at[:,:,0,0].set(dudx)
        grad_vf = grad_vf.at[:,:,0,1].set(dudy)
        grad_vf = grad_vf.at[:,:,1,0].set(dvdx)
        grad_vf = grad_vf.at[:,:,1,1].set(dvdy)

        return grad_vf

    return jax.jit(cc_vector_field_gradient)


def membrane_strain_rate_function(ny, nx, dy, dx, 
                                  cc_grad,
                                  extrapolate_over_cf,
                                  add_uv_ghost_cells):

    def membrane_sr_tensor(u, v):
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))

        u = extrapolate_over_cf(u)
        v = extrapolate_over_cf(v)

        u, v = add_uv_ghost_cells(u, v)

        dudx, dudy = cc_grad(u)
        dvdx, dvdy = cc_grad(v)
        
        
        msr_tensor = jnp.zeros((ny, nx, 2, 2))


        #TODO: CHECK THAT FACTOR OF 4 AND 2! ARE THEY 2 AND 1 REALLY?
        #I think this is right.
        msr_tensor = msr_tensor.at[:,:,0,0].set(4*dudx + 2*dvdy)
        msr_tensor = msr_tensor.at[:,:,0,1].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,0].set( dudy + dvdx )
        msr_tensor = msr_tensor.at[:,:,1,1].set(4*dvdy + 2*dudx)

        return msr_tensor

    return jax.jit(membrane_sr_tensor)


def divergence_of_tensor_field_function(ny, nx, dy, dx, periodic_x=False):
    def div_tensor_field(tf):
        #these have to be 2d scalar fields, of course
        tf_xx = tf[...,0,0]
        tf_xy = tf[...,0,1]
        tf_yx = tf[...,1,0]
        tf_yy = tf[...,1,1]

        shape_0, shape_1 = tf_xx.shape

        #NOTE: This is done assuming basically everything of interest and its
        #gradient is zero at the boundaries

        dx_tf_xx = jnp.zeros((shape_0, shape_1))
        dx_tf_xx = dx_tf_xx.at[:,1:-1].set((tf_xx[:,2:] - tf_xx[:,:-2])/(2*dx))
    
        dx_tf_xy = jnp.zeros((shape_0, shape_1))
        dx_tf_xy = dx_tf_xy.at[:,1:-1].set((tf_xy[:,2:] - tf_xy[:,:-2])/(2*dx))
        
        dy_tf_yx = jnp.zeros((shape_0, shape_1))
        dy_tf_yx = dy_tf_yx.at[1:-1,:].set((tf_yx[:-2,:] - tf_yx[2:,:])/(2*dy))
        
        dy_tf_yy = jnp.zeros((shape_0, shape_1))
        dy_tf_yy = dy_tf_yy.at[1:-1,:].set((tf_yy[:-2,:] - tf_yy[2:,:])/(2*dy))
        
        if periodic_x:
            dx_tf_xx = dx_tf_xx.at[:,0].set((tf_xx[:,1] - tf_xx[:,-1])/(2*dx))
            dx_tf_xx = dx_tf_xx.at[:,-1].set((tf_xx[:,0] - tf_xx[:,-2])/(2*dx))

        return dx_tf_xx+dy_tf_yx, dx_tf_xy+dy_tf_yy
    return jax.jit(div_tensor_field)


def cc_viscosity_function(ny, nx, dy, dx, cc_vector_field_gradient):
    def cc_viscosity(q, u, v):

        vfg = cc_vector_field_gradient(u, v)
        
        mu = B * mucoef_0 * jnp.exp(q) * (vfg[:,:,0,0]**2 + vfg[:,:,1,1]**2 + vfg[:,:,0,0]*vfg[:,:,1,1] + \
                           0.25*(vfg[:,:,0,1] + vfg[:,:,1,0])**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

        #NOTE: the effective viscosity isn't actually set to zero here in the ice-free regions,
        #but wherever it's used, it should be multiplied by a zero thickness there...

        return mu
    return jax.jit(cc_viscosity)









def fc_viscosity_function(ny, nx, dy, dx, extrp_over_cf, add_uv_ghost_cells,
                          add_mucoef_ghost_cells,
                          interp_cc_to_fc, ew_gradient, ns_gradient, h_1d, B):
    def fc_viscosity(q, u, v):
        mucoef = mucoef_0*jnp.exp(q)
        mucoef = add_mucoef_ghost_cells(mucoef)
        mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
        
        u = u.reshape((ny, nx))
        v = v.reshape((ny, nx))
        h = h_1d.reshape((ny, nx))

        u = extrp_over_cf(u)
        v = extrp_over_cf(v)
        #and add the ghost cells in
        u, v = add_uv_ghost_cells(u, v)

        #various face-centred derivatives
        dudx_ew, dudy_ew = ew_gradient(u)
        dvdx_ew, dvdy_ew = ew_gradient(v)
        dudx_ns, dudy_ns = ns_gradient(u)
        dvdx_ns, dvdy_ns = ns_gradient(v)
        
        #calculate face-centred viscosity:
        mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
                    0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
        mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
                    0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

        #to account for calving front boundary condition, set effective viscosities
        #of faces of all cells with zero thickness to zero:
        mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
        mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
        mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
        mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

        return mu_ew, mu_ns
 


 #def fc_viscosity_function_not_fixed_thk(ny, nx, dy, dx, extrp_over_cf, add_uv_ghost_cells,
 #                         add_mucoef_ghost_cells,
 #                         interp_cc_to_fc, ew_gradient, ns_gradient, B):
 #   def fc_viscosity(q, u, v, h):
 #       mucoef = mucoef_0*jnp.exp(q)
 #       mucoef = add_mucoef_ghost_cells(mucoef)
 #       mucoef_ew, mucoef_ns = interp_cc_to_fc(mucoef)
 #       
 #       u = u.reshape((ny, nx))
 #       v = v.reshape((ny, nx))
 #       h = h_1d.reshape((ny, nx))

 #       u = extrp_over_cf(u)
 #       v = extrp_over_cf(v)
 #       #and add the ghost cells in
 #       u, v = add_uv_ghost_cells(u, v)

 #       #various face-centred derivatives
 #       dudx_ew, dudy_ew = ew_gradient(u)
 #       dvdx_ew, dvdy_ew = ew_gradient(v)
 #       dudx_ns, dudy_ns = ns_gradient(u)
 #       dvdx_ns, dvdy_ns = ns_gradient(v)
 #       
 #       #calculate face-centred viscosity:
 #       mu_ew = B * mucoef_ew * (dudx_ew**2 + dvdy_ew**2 + dudx_ew*dvdy_ew +\
 #                   0.25*(dudy_ew+dvdx_ew)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))
 #       mu_ns = B * mucoef_ns * (dudx_ns**2 + dvdy_ns**2 + dudx_ns*dvdy_ns +\
 #                   0.25*(dudy_ns+dvdx_ns)**2 + c.EPSILON_VISC**2)**(0.5*(1/nvisc - 1))

 #       #to account for calving front boundary condition, set effective viscosities
 #       #of faces of all cells with zero thickness to zero:
 #       mu_ew = mu_ew.at[:, 1:].set(jnp.where(h==0, 0, mu_ew[:, 1:]))
 #       mu_ew = mu_ew.at[:,:-1].set(jnp.where(h==0, 0, mu_ew[:,:-1]))
 #       mu_ns = mu_ns.at[1:, :].set(jnp.where(h==0, 0, mu_ns[1:, :]))
 #       mu_ns = mu_ns.at[:-1,:].set(jnp.where(h==0, 0, mu_ns[:-1,:]))

 #       return mu_ew, mu_ns
 #   return jax.jit(fc_viscosity)   return jax.jit(fc_viscosity)
