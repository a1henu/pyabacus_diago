#ifndef MATRIX_DIAG_DAV_H
#define MATRIX_DIAG_DAV_H

#include "diago_dav_subspace.h"
#include "module_hsolver/kernels/math_kernel_op.h"
#include "module_base/module_device/types.h"
#include "diagh.h"

#include <complex>

class Matrix_DiagDav {
public:
    Matrix_DiagDav(const std::vector<std::complex<double>> h_mat,
                   const std::vector<double> pre_condition,
                   const int nband_in,
                   const int nbasis_in,
                   const int dav_ndim_in,
                   const double diag_thr_in,
                   const int diag_nmax_in,
                   const bool need_subspace_in,
                   const hsolver::diag_comm_info diag_comm_in) : 
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

    // Matrix_DiagDav(const Matrix_DiagDav& other) : diag_comm_in(other.diag_comm_in) {
    //     std::cout << "Copy constructor called" << std::endl;
    // }

    // Matrix_DiagDav(Matrix_DiagDav&& other) : diag_comm_in(other.diag_comm_in) {
    //     std::cout << "Move constructor called" << std::endl;
    // }

    int diag() {
        auto hpsi_func = [this] (std::complex<double> *hpsi_out,
                            std::complex<double> *psi_in, const int nband_in,
                            const int nbasis_in, const int band_index1,
                            const int band_index2) {
            const std::complex<double> *one_ = nullptr, *zero_ = nullptr;

            one_ = new std::complex<double>(1.0, 0.0);
            zero_ = new std::complex<double>(0.0, 0.0);

            base_device::DEVICE_CPU *ctx = {};

            hsolver::gemm_op<std::complex<double>, base_device::DEVICE_CPU>()(
                ctx, 'N', 'N', 
                nbasis_in, band_index2 - band_index1 + 1, nbasis_in, 
                one_, this->h_mat.data(), nbasis_in, 
                psi_in + band_index1 * nbasis_in, nbasis_in,
                zero_, hpsi_out + band_index1 * nbasis_in, nbasis_in);
        };

        psi = std::vector<std::complex<double>>(nband_in * nbasis_in, std::complex<double>(1.0, 0.0));
        eigenvalue = std::vector<double>(nband_in, 0.0);
        is_occupied = std::vector<bool>(nband_in, true);

        hsolver::Diago_DavSubspace<std::complex<double>, base_device::DEVICE_CPU> diag_dav_subspace
            {pre_condition, nband_in, nbasis_in, dav_ndim_in, diag_thr_in, diag_nmax_in, need_subspace_in, diag_comm_in};

        return diag_dav_subspace.diag(
            hpsi_func, nullptr,
            this->psi.data(), this->nbasis_in, 
            this->eigenvalue.data(), this->is_occupied, false);
    }

    std::vector<double> get_eigenvalue() {
        return this->eigenvalue;
    }

private:
    std::vector<std::complex<double>> h_mat;
    hsolver::diag_comm_info diag_comm_in;

    std::vector<double> pre_condition;

    int nband_in;
    int nbasis_in;
    int dav_ndim_in;
    double diag_thr_in;
    int diag_nmax_in;
    bool need_subspace_in;

    std::vector<std::complex<double>> psi;
    std::vector<double> eigenvalue;
    std::vector<bool> is_occupied;
};

#endif