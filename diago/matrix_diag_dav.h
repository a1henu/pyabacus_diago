#ifndef MATRIX_DIAG_DAV_H
#define MATRIX_DIAG_DAV_H

#include "diago_dav_subspace.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_base/module_device/types.h"
#include "diagh.h"

#include <complex>
#include <vector>

class Matrix_DiagDav {
public:
    Matrix_DiagDav(const std::vector<std::complex<double>>& h_mat,
                   const std::vector<double>& pre_condition,
                   int nband_in,
                   int nbasis_in,
                   int dav_ndim_in,
                   double diag_thr_in,
                   int diag_nmax_in,
                   bool need_subspace_in,
                   const hsolver::diag_comm_info& diag_comm_in) : 
                   h_mat(h_mat), pre_condition(pre_condition),
                   nband_in(nband_in), nbasis_in(nbasis_in),
                   dav_ndim_in(dav_ndim_in), diag_thr_in(diag_thr_in),
                   diag_nmax_in(diag_nmax_in), need_subspace_in(need_subspace_in),
                   diag_comm_in(diag_comm_in)
    {
        this->psi.resize(this->nband_in * this->nbasis_in);
        this->eigenvalue.resize(this->nband_in);
        this->is_occupied.resize(this->nband_in);
    }

    int diag() {
        auto hpsi_func = [this] (std::complex<double> *hpsi_out,
                            std::complex<double> *psi_in, int nband_in,
                            int nbasis_in, int band_index1,
                            int band_index2) {
            const std::complex<double> one(1.0, 0.0);
            const std::complex<double> zero(0.0, 0.0);

            base_device::DEVICE_CPU* ctx = {};

            hsolver::gemm_op<std::complex<double>, base_device::DEVICE_CPU>()(
                ctx, 'N', 'N', 
                nbasis_in, band_index2 - band_index1 + 1, nbasis_in, 
                &one, this->h_mat.data(), nbasis_in, 
                psi_in + band_index1 * nbasis_in, nbasis_in,
                &zero, hpsi_out + band_index1 * nbasis_in, nbasis_in);
        };

        psi.assign(nband_in * nbasis_in, std::complex<double>(1.0, 0.0));
        eigenvalue.assign(nband_in, 0.0);
        is_occupied.assign(nband_in, true);

        hsolver::Diago_DavSubspace<std::complex<double>, base_device::DEVICE_CPU> diag_dav_subspace
            {pre_condition, nband_in, nbasis_in, dav_ndim_in, diag_thr_in, diag_nmax_in, need_subspace_in, diag_comm_in};

        return diag_dav_subspace.diag(
            hpsi_func, nullptr,
            this->psi.data(), this->nbasis_in, 
            this->eigenvalue.data(), this->is_occupied, false);
    }

    std::vector<double> get_eigenvalue() const {
        return this->eigenvalue;
    }

private:
    std::vector<std::complex<double>> h_mat;
    std::vector<double> pre_condition;
    int nband_in;
    int nbasis_in;
    int dav_ndim_in;
    double diag_thr_in;
    int diag_nmax_in;
    bool need_subspace_in;
    hsolver::diag_comm_info diag_comm_in;

    std::vector<std::complex<double>> psi;
    std::vector<double> eigenvalue;
    std::vector<bool> is_occupied;
};

#endif