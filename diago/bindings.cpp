#include <complex>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

#include "matrix_diag_dav.h"

namespace py = pybind11;

PYBIND11_MODULE(diago_dav_subspace, m) {

    // m.def("gemm_op", [] (const char& transa, 
    //                      const char& transb,
    //                      const int& m,
    //                      const int& n,
    //                      const int& k,
    //                      const std::complex<double>& alpha,
    //                      const std::vector<std::complex<double>> a,
    //                      const int& lda,
    //                      const std::vector<std::complex<double>> b,
    //                      const int& ldb,
    //                      const std::complex<double>& beta,
    //                      std::vector<std::complex<double>>& c,
    //                      const int& ldc) -> std::vector<std::complex<double>>
    //     {
    //         base_device::DEVICE_CPU* ctx = {};
    //         hsolver::gemm_op<std::complex<double>, base_device::DEVICE_CPU>()(
    //             ctx, 
    //             transa,
    //             transb,
    //             m, n, k,
    //             &alpha, a.data(), lda,
    //             b.data(), ldb, 
    //             &beta, c.data(), ldc);
            
    //         return c;
    //     }
    // );

    py::class_<hsolver::diag_comm_info>(m, "diag_comm_info")
        .def(py::init<const int, const int>())
        .def_readonly("rank", &hsolver::diag_comm_info::rank)
        .def_readonly("nproc", &hsolver::diag_comm_info::nproc);

    py::class_<Matrix_DiagDav>(m, "Matrix_DiagDav")
        .def(py::init(
            [] (const py::array_t<std::complex<double>> h_mat,
                const py::array_t<double> pre_condition,
                int nband_in,
                int nbasis_in,
                int dav_ndim_in,
                double diag_thr_in,
                int diag_nmax_in,
                bool need_subspace_in,
                const hsolver::diag_comm_info& diag_comm_in)
            {
                std::vector<std::complex<double>> h_mat_vec(h_mat.size());
                std::vector<double> pre_condition_vec(pre_condition.size());
                
                for (size_t i = 0; i < h_mat.size(); i++) {
                    h_mat_vec[i] = h_mat.at(i);
                }

                for (size_t i = 0; i < pre_condition.size(); i++) {
                    pre_condition_vec[i] = pre_condition.at(i);
                }

                return Matrix_DiagDav(h_mat_vec, pre_condition_vec, nband_in, nbasis_in, dav_ndim_in, diag_thr_in, diag_nmax_in, need_subspace_in, diag_comm_in);
            }
        ))
        .def("diag", &Matrix_DiagDav::diag)
        .def("get_eigenvalue", &Matrix_DiagDav::get_eigenvalue);
}