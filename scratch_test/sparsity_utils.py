import jax.numpy as jnp
import numpy as np
import scipy

#TODO: function for translating stencil into set of basis vectors



def create_repeated_array(base_array, n):
    repetitions = int(jnp.ceil(n / len(base_array)))

    repeated_array = jnp.tile(base_array, repetitions)

    return repeated_array[:n], repetitions


def basis_vectors_etc(case_=1):
    """
    create basis vectors with which to carry out vector-jacobian products and
    sets of coordinates mapping the corresponding vjps to the dense jacobian.

    case 0: diagonal jacobian
    case 1: tridiagonal jacobian
    case 2: upper bidiagonal jacobian
    case 3: lower bidiagonal jacobian

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



    for i in range(len(basis_vectors)):
        assert i_coord_sets[i].shape == j_coord_sets[i].shape, \
           "is_full and js_full have different shapes of {} and {} for {}-th bv"\
           .format(i_coord_sets[i].shape, j_coord_sets[i].shape, i)


    return basis_vectors, i_coord_sets, j_coord_sets



def make_sparse_jacrev_fct(basis_vectors, i_coord_sets, j_coord_sets):
    # This can be made significantly more general, but this is just to
    # see whether the basics work and reduce demands on memory


    def sparse_jacrev(fun_, primals):
        y, jvp_fun = jax.vjp(fun_, *primals)
        rows = []
        for bv in basis_vectors:
            row, _ = jvp_fun(bv)
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



def dodgy_coo_to_csr(values, coordinates, shape, return_decomposition=False):

    a = scipy.sparse.coo_array((values, (coordinates[:,0], coordinates[:,1])), shape=shape).tocsr()

    if return_decomposition:
        return a.indptr, a.indices, a.data
    else:
        return a



