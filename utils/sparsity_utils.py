import jax
import jax.numpy as jnp
import numpy as np
import scipy
import matplotlib.pyplot as plt


#TODO: function for translating stencil into set of basis vectors

def dodgy_coo_to_csr(values, coordinates, shape, return_decomposition=False):

    a = scipy.sparse.coo_array((values, (coordinates[:,0], coordinates[:,1])), shape=shape).tocsr()

    if return_decomposition:
        return a.indptr, a.indices, a.data
    else:
        return a

def scipy_coo_to_csr(values, coordinates, shape, return_decomposition=False):

    a = scipy.sparse.coo_array((values, (coordinates[0,:], coordinates[1,:])), shape=shape).tocsr()

    if return_decomposition:
        return a.indptr, a.indices, a.data
    else:
        return a

def make_sparse_jacrev_fct_multiprime_no_densify(basis_vectors):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)

        #I think this is true at least...
        m_range = range(len(primals))
        rows_agg = [[] for i in m_range]
        
        #need to get rid of this loop!!
        #maybe vmap or something? Maybe it's not so bad if n(bvs)~10...
        for bv in basis_vectors:
            row_tuple = vjp_fun(bv)
            for i in m_range:
                rows_agg[i].append(row_tuple[i])
        
        rows_agg = [jnp.concatenate(rows_agg[i]) for i in m_range]

        return rows_agg

    return sparse_jacrev

def make_sparse_jacrev_fct_multiprime(n, basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)

        #I think this is true at least...
        m_range = range(len(primals))
        rows_agg = [[] for i in m_range]
        
        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row_tuple = vjp_fun(bv)
            for i in m_range:
                rows_agg[i].append(row_tuple[i])
        
        rows_agg = [jnp.concatenate(rows_agg[i]) for i in m_range]

        return rows_agg

    def densify_sparse_jac(jacrows_vec):
        m = len(basis_vectors[0]) #number of columns in matrix
        jac = jnp.zeros((n, m))

        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
            # jac = jac.at[bv_is, bv_js].set(jacrow)

        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac

def make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        #Remember: vector-matrix product with a basis vector picks out a row.
        #matrix-vector product with basis vector picks out a column.
        #The indices of the things you're differentiating wrt form the columns
        #and the indices of the function form the rows.
        y, vjp_fun = jax.vjp(fun_, *primals)
        rows = []

        #need to get rid of this loop!!
        #maybe vmap or something?
        for bv in basis_vectors:
            row, _ = vjp_fun(bv)
            rows.append(row)

        rows = jnp.concatenate(rows)

        # print(rows)
        return rows

    def densify_sparse_jac(jacrows_vec):
        jac = jnp.zeros((n, n))

        # for bv_is, bv_js, jacrow in zip(i_coord_sets, j_coord_sets, jacrows):
            # jac = jac.at[bv_is, bv_js].set(jacrow)

        jac = jac.at[j_coord_sets, i_coord_sets].set(jacrows_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac



def make_sparse_jacrev_fct_new(basis_vectors, i_coord_sets, j_coord_sets, mask):

    def sparse_jacrev(fun_, primals):
        #unfortunately, there is no way to avoid retracing fun_ each
        #time you apply sparse_jacrev. It's a load of bullshit but
        #oh well. If it turns out to be really expensive, it might be
        #worth trying to add  a custom sparse Jacobian transform or
        #use JAX internals like jax.make_jaxpr or jax.interpreters.ad 
        #to extract Jacobian structure directly. (-_-)

        _, vjp_fun = jax.vjp(fun_, *primals)
        
        apply_basis = lambda bv: vjp_fun(bv)[0]
        
        basis = jnp.stack(basis_vectors).astype(jnp.float64) #int8 won't cut it apparently...
        
        rows = jax.vmap(apply_basis)(basis)

        return rows.reshape(-1)

    def densify_sparse_jac(jacrows_vec, n):
        jac = jnp.zeros((n, n))

        ics = i_coord_sets[mask].astype(jnp.int32)
        jcs = j_coord_sets[mask].astype(jnp.int32)
        jac_vec  = jacrows_vec[mask]

        jac = jac.at[ics, jcs].set(jac_vec)

        return jac

    return sparse_jacrev, densify_sparse_jac


def make_sparse_jacrev_fct_shared_basis(basis_vectors, i_coord_sets, j_coord_sets,\
                                        mask, n_outputs, active_indices=None):


    if active_indices is None:
        active_indices = list(range(n_primals))
    else:
        active_indices = list(active_indices)

    n_primals = len(active_indices)

    def sparse_jacrev(fun_, primals):
        #unfortunately, there is no way to avoid retracing fun_ each
        #time you apply sparse_jacrev. It's a load of bullshit but
        #oh well. If it turns out to be really expensive, it might be
        #worth trying to add  a custom sparse Jacobian transform or
        #use JAX internals like jax.make_jaxpr or jax.interpreters.ad 
        #to extract Jacobian structure directly.

        #This is incredibly simple and can be so due to the fact that
        #it assumes the basis vectors are shared by the primals. This
        #will be the case for u and v, but might not be the case for h
        #for example. Can certainly accomodate this, but maybe another
        #time.
    

        #You can't specify which arguments vjp_fun will differentiate
        #with respect to, so this is a little wrapper that ensures "inactive"
        #arguments are static in the function definition:
        def wrapped_function(*active_arguments):
            all_arguments = list(primals)
            for i, idx in enumerate(active_indices):
                all_arguments[idx] = active_arguments[i]
            return fun_(*all_arguments)

        _, vjp_fun = jax.vjp(wrapped_function, *[primals[k] for k in active_indices])
        #vjp_fun takes a tuple of shape (output_0, output_1, ...)

        
        basis = jnp.stack(basis_vectors).astype(jnp.float64) #int8 won't cut it apparently...
        
        #def apply_basis_for_output(out_index):
        #    return lambda bv: vjp_fun(
        #            tuple(
        #                tuple([bv] * n_primals) if i==out_index\
        #                else tuple([jnp.zeros_like(bv)] * n_primals)\
        #                for i in range(n_outputs)
        #                                     )
        #          )
        def apply_basis_for_output(out_index):
            return lambda bv: vjp_fun(
                    tuple(
                        [bv if i==out_index\
                        else jnp.zeros_like(bv)\
                        for i in range(n_outputs)]
                                             )
                  )

        rows_for_output = []
        for i in range(n_outputs):
            rows_for_output.append(
                    jax.vmap(apply_basis_for_output(i))(basis)
                                  )

        #rows_for_output looks like [(d_output0/d_primal0, d_output0/d_primal1, ...),\
        #                            (d_output1/d_primal0, d_output1/d_primal1, ...)]
        #... I think
        #and each d_outputi/d_primalj has shape (n_basis_vectors,nr*nc)


        #want to return d_output0/d_primal0, d_output0/d_primal1, ..., 
        # d_output1/d_primal0, d_output1/d_primal1, ...

        return [rows_for_output[i][j].flatten()
                for j in range(n_primals)
                for i in range(n_outputs)
               ]


    #def densify_sparse_jac(jacrows_vec, n):
    #    jac = jnp.zeros((n, n))

    #    ics = i_coord_sets[mask].astype(jnp.int32)
    #    jcs = j_coord_sets[mask].astype(jnp.int32)
    #    jac_vec  = jacrows_vec[mask]

    #    jac = jac.at[ics, jcs].set(jac_vec)

    #    return jac

    return sparse_jacrev#, densify_sparse_jac


def basis_vectors_and_coords_2d_square_stencil(nr, nc, r=2, periodic_x=False):
    #NOTE: Need to think about speed and memory efficiency in this function.
    #E.g. defining intermediate functions to limit the scope of intermediate
    #variables. I also wonder whether I need to be creating so many dense arrays.
    #I'm not sure how to do it otherwise, but might figure it out...
    
    #One thing that's obvious I suppose is that the pattern repeats, so you could
    #just do it for the smallest posible array and then extrapolate...

    #r: stencil radius
    #r of 1 gives a 9-point stencil,
    #...of 2 gives a 25-point stencil
    #...of 3 gives a 49-point stencil etc

    # width of the patch of the grid surrounding each colour
    # = stencil width plus half a stencil width plus one
    # = (2*r + 1) + r + 1
    #colour_patch_width = 3*r + 2 #NOTE: don't need this

    #You just have to colour the domain in patches of the same size as the stencil
    stencil_width = 2*r + 1

    colouring_template = jnp.arange(stencil_width**2).reshape(stencil_width, -1)

    #NOTE: It might be worth saving a copy of the coords in COO and CSR formats at this
    #point. It's annoying, but maybe better than switching between using scipy.

    n_stencils_wide = int(jnp.ceil(nc/stencil_width))
    n_stencils_tall = int(jnp.ceil(nr/stencil_width))


    tiled_domain = jnp.tile(colouring_template,\
                            (n_stencils_tall, n_stencils_wide))\
                        [:nr, :nc].astype(jnp.int8)
    td_flat = tiled_domain.flatten()

    
    #Set of basis vectors that pick out obviously orthogonal rows of
    #the jacobian. (when you do a vjp, the rows are added together)
    basis_vectors = []

    #Need to know, when you do the vjp, what the coordinates are 
    #of the resulting non-zero values of the row vectors
    #j_coordinates_guess = jnp.arange(nr*nc, dtype=jnp.int32) #NOTE: don't actually need this
    #j_coords = jnp.zeros((stencil_width**2, nr*nc), dtype=jnp.int32)
    #i_coords = jnp.zeros((stencil_width**2, nr*nc), dtype=jnp.int32)
    j_coords = []
    i_coords = []

    for colour in range(stencil_width**2):
        #indices corresponding to a particular colour
        colour_indices = jnp.where(td_flat==colour)[0].astype(jnp.int32)
        
        #basis vector is 1 where the colour is and 0 elsewhere
        #rows corresponding to same colour are "obviously orthogonal"
        basis_vectors.append((td_flat==colour).astype(jnp.int8))

        #vector of mostly zeros, then the index values where the colour is
        #(actually, the index values plus one as I'd like to reserve 0 for
        #where nodes aren't connected to a particular colour at any location (occurs
        #at the boundaries sometimes))
        #NOTE: you have to cast the zero array as 32-bit integer. If you don't, the
        #at/set will inherit the type (default float32) and you get integer overflow
        #errors if coordinate values exceed 16_777_216. Irritating bug.
        colour_coordinates = jnp.zeros((nr*nc,), dtype=jnp.int32)\
                             .at[colour_indices].set(colour_indices+1)

        #2D version, padded by a stencil radius
        if periodic_x:
            #NOTE: this will only work, unfortunately, if the domain is exactly tileable
            #in x by the stencil. Else the stencil doesn't wrap nicely and we're screwed.
            #There is a way to do it, I'm sure. But I'm fed up.
            assert nc % stencil_width == 0, "Domain must be tileable in x by stencil width"

            # Pad rows (y-direction) with zeros
            ccs_y_padded = jnp.pad(colour_coordinates.reshape(nr, nc),\
                         pad_width=((r, r),(0, 0)),\
                         mode='constant', constant_values=0)
        
            # Manually wrap columns (x-direction)
            left = ccs_y_padded[:, -r:]
            right = ccs_y_padded[:, :r]
            ccs_padded = jnp.concatenate([left, ccs_y_padded, right], axis=1)
        else:
            # Non-periodic in both directions
            ccs_padded = jnp.pad(colour_coordinates.reshape(nr, nc),\
                         pad_width=((r, r),(r, r)),\
                         mode='constant', constant_values=0)

        #fill a stencil-radius around each node of colour with the index of that colour
        i_coordinates = jnp.zeros((nr, nc), dtype=jnp.int32)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                i_coordinates += ccs_padded[(r+di):(r+di+nr), (r+dj):(r+dj+nc)]

        #don't cast as int type as the nans are important!
        #jnp.nan is of type float32 I think. So the where promotesd
        #i_coordinates to float32 also. So have to use -1 and avoid nan.
        #i_coordinates = jnp.where(i_coordinates.flatten()==0, jnp.nan, i_coordinates.flatten()-1)
        i_coordinates = jnp.where(i_coordinates.flatten()==0, -1, i_coordinates.flatten()-1)
       
        if i_coordinates.max()>=(nr*nc):
            print(colour)
            print(jnp.where(td_flat==colour)[0].max())
            print(colour_indices.max())
            print((colour_indices+1).astype(jnp.int32).max())
            
            print(jnp.zeros((nr*nc,), dtype=jnp.int32).at[colour_indices].set((colour_indices+1).astype(jnp.int32)).astype(jnp.int32).max())
            print(colour_coordinates.max())

            print(ccs_padded.max())
            
            print(i_coordinates.max())
            print((jnp.where(td_flat==colour)[0]+1).astype(jnp.int32).max())
            raise


        ##j coordinates. The only reason for doing this is the pesky edge bits which
        ##The following is possible, but I think it makes more sense to keep the nans in for later use.
        ##the stencil doesn't cover for some colours.
        #non_nan_locations = jnp.where(~jnp.isnan(i_coordinates))[0]
        #j_coordinates_slim = j_coordinates_guess[non_nan_locations]
        #i_coordinates_slim = i_coordinates[non_nan_locations]

        #i_coords.append(i_coordinates_slim.astype(jnp.int32))
        #j_coords.append(j_coordinates_slim.astype(jnp.int32))
       
        i_coords.append(i_coordinates)#.astype(jnp.int32)) #again, no casting to int!!
        #j_coords.append(j_coordinates.astype(jnp.int32))

    return basis_vectors, i_coords#, j_coords


def sparsity_pattern(nr, nc, r):
    bvs, i_coords = basis_vectors_and_coords_2d_square_stencil(nr, nc, r)

    j_coord_ar = jnp.arange(nr*nc)

    pattern = jnp.zeros((nr*nc, nr*nc))*jnp.nan
    for _, i_coord_ar in zip(bvs, i_coords):
        mask = ~jnp.isnan(i_coord_ar)

        pattern = pattern.at[i_coord_ar[mask].astype(jnp.int32),\
                             j_coord_ar[mask].astype(jnp.int32)].set(1)

    plt.imshow(np.array(pattern))
    plt.show()


#sparsity_pattern(5, 5, 1)
#raise


def create_repeated_array(base_array, n):
    repetitions = int(jnp.ceil(n / len(base_array)))

    repeated_array = jnp.tile(base_array, repetitions)

    return repeated_array[:n], repetitions


def basis_vectors_etc_nonsquare(nr, nc, case_=1, fill_from_left=True, fill_from_top=True):
    """
    create basis vectors with which to carry out jacobian-vector products and
    sets of coordinates mapping the corresponding jvps to the dense jacobian.

    I can only handle filling in from top left. JC.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian
    case 4: lower tridiagonal jaconian! (not sure it's a thing but ykwim)
    case 5: pentagonal jacobian

    """

    match case_:

        case 0: ##UNTESTED
            if nc>=nr:
                basis_vectors = [jnp.ones((nr,))]
                i_coord_sets  = [jnp.concatenate(jnp.arange(nr), jnp.zeros((nc-nr,))+nr-1)] #getting those trailing zeros in there
                j_coord_sets  = [jnp.arange(nc)]
            else:
                #basis_vectors = [jnp.ones((nc,))]
                #i_coord_sets  = [jnp.arange(nc)]
                #j_coord_sets  = [jnp.arange(nc)]
                raise NotImplementedError


        case 1:
            if nc>nr:
                base_1 = np.array([1, 0, 0])
                base_2 = np.array([0, 1, 0])
                base_3 = np.array([0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    
                    is_ = np.repeat(np.arange(k, nr, 3), 3)
                    
                    if basis[0]==1:
                        is_ = is_[1:]

                    if basis[-1]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr-1)])
                    if basis[-2]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr)])
                    if basis[-3]==1:
                        is_ = np.concatenate([is_, np.repeat(is_[-1], nc-nr+1)])

                    if basis[0]==0 and basis[1]==0:
                        is_ = np.insert(is_, 0, is_[0])

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)

            elif nc==(nr-1):
                
                base_1 = np.array([1, 0, 0])
                base_2 = np.array([0, 1, 0])
                base_3 = np.array([0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    
                    is_ = np.repeat(np.arange(k, nr, 3), 3)
                   
                    #TODO: Add functionality for things being quite different sizes.
                    #so far only works if nr-nc=1

                    if basis[0]==1:
                        is_ = is_[1:]
                    if basis[-1]==1:
                        is_ = is_[:-2]
                    if basis[-2]==1:
                        is_ = is_[:-1]

                    #remember, you need those zeros to make the things the same length!
                    if basis[0]==0 and basis[1]==0:
                        is_ = np.insert(is_, 0, is_[0])
                    

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)


            else:
                raise NotImplementedError

        case 2: ##UNTESTED
            raise NotImplementedError

        case 3: ##UNTESTED
            raise NotImplementedError
        
        case 4: ##UNTESTED
            raise NotImplementedError
        
        case 5:
            if nc==(nr-1):
                
                base_1 = np.array([1, 0, 0, 0, 0])
                base_2 = np.array([0, 1, 0, 0, 0])
                base_3 = np.array([0, 0, 1, 0, 0])
                base_4 = np.array([0, 0, 0, 1, 0])
                base_5 = np.array([0, 0, 0, 0, 1])

                basis_vectors = []
                i_coord_sets = []
                j_coord_sets = []
                k = 0
                for base in [base_1, base_2, base_3, base_4, base_5]:
                    basis, r = create_repeated_array(base, nr)
                    basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                    js_ = np.arange(nc)
                    is_ =  np.repeat(np.arange(k, nr, 5), 5)
                    
                    if basis[0]==1:
                        is_ = is_[2:]
                    elif basis[1]==1:
                        is_ = is_[1:]

                    if basis[-1]==1:
                        is_ = is_[:-3]
                    elif basis[-2]==1:
                        is_ = is_[:-2]
                    elif basis[-3]==1:
                        is_ = is_[:-1]

                    #remember, you need those zeros to make the things the same length!
                    if basis[0]==0 and basis[1]==0 and basis[2]==0:
                        is_ = np.insert(is_, 0, is_[0])
                        if basis[3]==0:
                            is_ = np.insert(is_, 0, is_[0])
                    if basis[-1]==0 and basis[-2]==0 and basis[-3]==0 and basis[-4]==0:
                        is_ = np.append(is_, is_[-1])

                    i_coord_sets.append(jnp.array(is_))
                    j_coord_sets.append(jnp.array(js_))

                    k += 1

                i_coord_sets = jnp.concatenate(i_coord_sets)
                j_coord_sets = jnp.concatenate(j_coord_sets)

            else:
                raise NotImplementedError

    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets


def basis_vectors_etc(n, case_=1):
    """
    create basis vectors with which to carry out jacobian-vector products and
    sets of coordinates mapping the corresponding jvps to the dense jacobian.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian
    case 4: lower tridiagonal jaconian! (not sure it's a thing but ykwim)
    case 5: pentagonal jacobian

    """

    match case_:

        case 0: ##UNTESTED
              basis_vectors = [jnp.ones((n,))]
              i_coord_sets  = [jnp.arange(n)]
              j_coord_sets  = [jnp.arange(n)]


        case 1:
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 3), 3)
                
                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0:
                    is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 2: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[-1]==1:
                    is_ = is_[:-1]

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 3: ##UNTESTED
            base_1 = np.array([1, 0])
            base_2 = np.array([0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 2), 2)

                if basis[0]==1:
                    is_ = is_[1:]
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

        case 4: ##UNTESTED
            base_1 = np.array([1, 0, 0])
            base_2 = np.array([0, 1, 0])
            base_3 = np.array([0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ = np.repeat(np.arange(k, n, 3), 3)

                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]

                #remember, you need those zeros to make the things the same length!
                if basis[-1]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-2]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

        case 5:
            base_1 = np.array([1, 0, 0, 0, 0])
            base_2 = np.array([0, 1, 0, 0, 0])
            base_3 = np.array([0, 0, 1, 0, 0])
            base_4 = np.array([0, 0, 0, 1, 0])
            base_5 = np.array([0, 0, 0, 0, 1])

            basis_vectors = []
            i_coord_sets = []
            j_coord_sets = []
            k = 0
            for base in [base_1, base_2, base_3, base_4, base_5]:
                basis, r = create_repeated_array(base, n)
                basis_vectors.append(jnp.array(basis).astype(jnp.float32))

                js_ = np.arange(n)
                is_ =  np.repeat(np.arange(k, n, 5), 5)
                
                if basis[0]==1:
                    is_ = is_[2:]
                elif basis[1]==1:
                    is_ = is_[1:]
                if basis[-1]==1:
                    is_ = is_[:-2]
                elif basis[-2]==1:
                    is_ = is_[:-1]

                #remember, you need those zeros to make the things the same length!
                if basis[0]==0 and basis[1]==0 and basis[2]==0:
                    is_ = np.insert(is_, 0, is_[0])
                    if basis[3]==0:
                        is_ = np.insert(is_, 0, is_[0])
                if basis[-1]==0 and basis[-2]==0 and basis[-3]==0:
                    is_ = np.append(is_, is_[-1])
                    if basis[-4]==0:
                        is_ = np.append(is_, is_[-1])

                i_coord_sets.append(jnp.array(is_))
                j_coord_sets.append(jnp.array(js_))

                k += 1

            i_coord_sets = jnp.concatenate(i_coord_sets)
            j_coord_sets = jnp.concatenate(j_coord_sets)

    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets






#basis_vectors, i_coord_sets, j_coord_sets = basis_vectors_etc(20, 5)
#print(basis_vectors)
#print(i_coord_sets)
#print(j_coord_sets)


#def colour_domain_uniform_mesh(domain, stencil_width=3):
#    #stencil_width of -1 is a special case of the 2D 5 point stencil which we can
#    #cover with 2 colours.
#
#    dim_ = len(domain.shape)
#    
#    basis_vectors = []
#    i_coord_sets = []
#    j_coord_sets = []
#
#    if dim_==1:
#        for n in range(stencil_width):
#            arr_ = np.zeros_like(domain)
#            arr_[n::stencil_width]=1
#            basis_vectors.append(arr_)
#
#            js_ = np.arange(domain.shape[0])
#            is_ = np.arange(0+n, domain.shape[0], stencil_width)
##            is_ = np.repeat(np.arange(k, domain.shape[0], stencil_width), stencil_width)
##
##            if basis[0]==1:
##                is_ = is_[1:]
##            if basis[-1]==1:
##                is_ = is_[:-1]
##
##            if basis[0]==0 and basis[1]==0:
##                is_ = np.insert(is_, 0, is_[0])
##            if basis[-1]==0 and basis[-2]==0:
##                is_ = np.append(is_, is_[-1])
##
#            i_coord_sets.append(is_)
#            j_coord_sets.append(js_)
#            
#            print(arr_)
#            print(is_)
#            print(js_)
#
#
#        return
#
#    if stencil_width==-1:
#
#        return
#
#
#    return

