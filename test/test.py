import sys
sys.path.append('/mnt/d/PKU/projects/pyabacus_diago/diago/build')

from scipy.linalg.blas import zgemm
import numpy as np
import diago_dav_subspace

# h_mat = np.array(
#     [
#         4.0+0.0j, 2.0+0.0j, 2.0+0.0j,
#         2.0+0.0j, 4.0+0.0j, 2.0+0.0j,
#         2.0+0.0j, 2.0+0.0j, 4.0+0.0j
#     ], 
# dtype=np.complex128, order='C')
h_mat = np.array(
    [
        1.0+0.0j, 0.0+0.0j, 0.0+0.0j,
        0.0+0.0j, 1.0+0.0j, 0.0+0.0j,
        0.0+0.0j, 0.0+0.0j, 1.0+0.0j
    ],
dtype=np.complex128, order='C')
pre_condition = np.ones(3, dtype=np.float64, order='C')
nband = 1
nbasis = 3
dav_ndim = 2
diag_thr = 1e-2
diag_nmax = 1000
need_subspace = False
comm_info = diago_dav_subspace.diag_comm_info(0, 1)

matrix_diag = diago_dav_subspace.Matrix_DiagDav(
    h_mat,
    pre_condition,
    nband,
    nbasis,
    dav_ndim,
    diag_thr,
    diag_nmax,
    need_subspace,
    comm_info
)

res = matrix_diag.diag()

print(f'res: {res}')
print(f'eigenvalue: {matrix_diag.get_eigenvalue()}')