import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

soa_evecs = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/soa_evecs.npy")
soa_evals = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/soa_evals.npy")

ad_evecs = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/ad_evecs.npy")
ad_evals = jnp.load("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/ad_evals.npy")




n_evecs = ad_evals.size


indices = jnp.arange(0,n_evecs,1)[::-1]

evec_mats_soa = jnp.column_stack([soa_evecs[...,-(k+1)].flatten() for k in range(n_evecs)])
evec_mats_ad  = jnp.column_stack([ad_evecs[...,-(k+1)].flatten() for k in range(n_evecs)])

def principal_angles(A, B):
    """
    Given two matrices A, B with the same number of rows and k columns,
    compute principal angles between the column spaces of A and B.
    Returns angles in radians (sorted ascending).
    """
    # # Orthonormalise columns (QR)
    QA, _ = np.linalg.qr(A)  # QA: (n, k)
    QB, _ = np.linalg.qr(B)  # QB: (n, k)

    # # SVD of QA^T QB -> singular values are cos(theta_i)
    M = QA.T @ QB
    # M = A.T @ B
    s = la.svd(M, compute_uv=False)

    # Numerical safety
    s = np.clip(s, 0.0, 1.0)
    thetas = np.arccos(s)  # radians
    return np.sort(thetas)  # ascending


def principal_angles_efficient(A, B):
    """
    Memory-efficient principal angles between column spaces of A and B.
    Avoids forming explicit Q matrices.
    """
    # Reduced QR (only R needed; Q not stored explicitly)
    _, RA = np.linalg.qr(A, mode='reduced')  # RA: (k, k)
    _, RB = np.linalg.qr(B, mode='reduced')  # RB: (k, k)

    # Cross Gram matrix (k x k)
    G = A.T @ B

    # Compute M = R_A^{-T} G R_B^{-1}
    # Use triangular solves (more stable & memory efficient than inverse)
    X = sp.linalg.solve_triangular(RA, G, trans='T', lower=False)
    M = sp.linalg.solve_triangular(RB, X.T, trans='T', lower=False).T

    # Singular values give cos(theta)
    s = la.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)

    return jnp.sort(np.arccos(s))




#import os
#
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1"






theta_max = []
theta_min = []
theta_mean = []
grassmann_geodesic = []  # ||theta||_2 on the Grassmann manifold
chordal = []             # sqrt(sum(sin(theta)^2))
projection_dist = []     # sin(theta_max) = ||P_A - P_B||_2
all_thetas = []

for k in np.arange(1,n_evecs):
#for k in range(1,3):
    print(k)
    #thetas = principal_angles_efficient(evec_mats_soa[:, :k], evec_mats_ad[:, :k])
    thetas = principal_angles(evec_mats_soa[:, :k], evec_mats_ad[:, :k])
    all_thetas.append(thetas)
    theta_max.append(thetas[-1])
    theta_mean.append(thetas.mean())
    theta_min.append(thetas[0])
    grassmann_geodesic.append(np.linalg.norm(thetas))
    sines = jnp.sin(thetas)
    chordal.append(np.linalg.norm(sines))
    projection_dist.append(sines[-1])  # equals sin(max angle)


jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/theta_max.npy", jnp.array(theta_max))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/theta_min.npy", jnp.array(theta_min))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/theta_mean.npy", jnp.array(theta_mean))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/grassmann_geodesic.npy", jnp.array(grassmann_geodesic))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/chordal.npy", jnp.array(chordal))
jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/projection_dist.npy", jnp.array(projection_dist))


pad_val = jnp.nan

# Preallocate
all_thetas_padded_array = jnp.full((n_evecs-1, n_evecs-1), pad_val, dtype=float)

for i, a in enumerate(all_thetas):
    all_thetas_padded_array = all_thetas_padded_array.at[i, :a.size].set(a)

jnp.save("/Users/eartsu/new_model/testing/nm/bits_of_data/hessian_evecs_etc/production/stream/more/all_thetas.npy", all_thetas_padded_array)


