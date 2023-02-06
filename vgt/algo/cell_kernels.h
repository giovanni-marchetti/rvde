#pragma once

#include "../utils.h"
#include "../RandomEngine.h"

class CellKernel {
public:
    explicit CellKernel(int dim);

    /**
     * Returns Vol(S^{n-1}) * \int_0^{length} K(x) x^{n-1} dx.
     */
    virtual ftype cone_integral(int index, ftype length) const = 0;

    virtual ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const = 0;  // (x + ut - p0)^2 = t^2 + 2at + b

    virtual std::string latex() const = 0;
protected:
    int dim;
};

/**
 * Describes a radial kernel function K(x), x >= 0.
 */
class RadialCellKernel : public CellKernel {
public:
    explicit RadialCellKernel(int dim, ftype sigma);

    virtual ftype general_cone_integral(ftype sigma, ftype length) const = 0;

    virtual ftype cone_integral(int index, ftype length) const override;

    /**
     * Returns K(radius).
     */
    virtual ftype kernel_value(int index, ftype radius) const;

    virtual ftype base_kernel_value(ftype sigma, ftype radius) const = 0;

    /**
     * Returns K'(radius).
     */
    virtual ftype derivative_value(int index, ftype radius) const = 0;

    /**
     * Returns Vol(S^{n-1}) * \int_0^{length} K'(x) x^{n-1} dx.
     */
    virtual ftype cone_integral_derivative(int index, ftype length) const = 0;

    virtual ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                 ftype t0, ftype t1, RandomEngine &re) const = 0;

    virtual ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u,
                                        ftype t0, ftype t1) const = 0;

    ftype get_sigma() const;

protected:
    ftype sigma;
};

class UniformCellKernel : public RadialCellKernel {
public:
    explicit UniformCellKernel(int dim);
    std::string latex() const override;

    /**
     * Returns 1 at 0.
     */
    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;
};

class GaussianCellKernel : public RadialCellKernel {
public:
    GaussianCellKernel(int dim, ftype sigma);
    std::string latex() const override;

    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;
};

class LaplaceCellKernel : public RadialCellKernel {
public:
    LaplaceCellKernel(int dim, ftype sigma);
    std::string latex() const override;

    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;
};

class PolynomialCellKernel : public RadialCellKernel {
public:
    PolynomialCellKernel(int dim, ftype sigma, ftype k);
    std::string latex() const override;

    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;

private:
    ftype k;
};


class AdaptiveGaussianCellKernel : public RadialCellKernel {
public:
    AdaptiveGaussianCellKernel(int dim, ftype sigma);
    std::string latex() const override;

    AdaptiveGaussianCellKernel(int dim, ftype sigma, const vec<ftype> &local_sigma);

    bool is_initialized() const;

    void update_local_bandwidths(const vec<ftype> &new_local_sigma);

    ftype kernel_value(int index, ftype radius) const override;

    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype cone_integral(int index, ftype length) const override;

    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;

private:
    bool initialized;
    vec<ftype> local_sigma;
};

class FixedVolumeCellKernel {
public:
    virtual void compute_local_bandwidths(const vec<vec<ftype>> &lengths) = 0;
};

class FixedVolumeGaussianCellKernel : public RadialCellKernel, public FixedVolumeCellKernel {
public:
    FixedVolumeGaussianCellKernel(int dim, ftype volume);
    std::string latex() const override;

    void compute_local_bandwidths(const vec<vec<ftype>> &lengths) override;

    ftype kernel_value(int index, ftype radius) const override;

    ftype base_kernel_value(ftype sigma, ftype radius) const override;

    ftype derivative_value(int index, ftype radius) const override;

    ftype cone_integral(int index, ftype length) const override;
    ftype general_cone_integral(ftype sigma, ftype length) const override;

    ftype cone_integral_derivative(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    ftype sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                         RandomEngine &re) const override;

    ftype sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                ftype t1) const override;

//    ftype normalization_constant(int index) const override;

private:
    bool initialized;
    ftype volume;
    vec<ftype> local_sigma;
};


class BalancedCellKernel : public CellKernel { // abstract
public:
    /**
     * alpha is positive constant such that Vol(Cell(p)) = alpha * Vol(S^{n-1})
     */
    BalancedCellKernel(int dim, ftype alpha);

    virtual ftype kernel_value(int index, ftype radius, ftype length) const;

    ftype cone_integral(int index, ftype length) const override;

    ftype cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const override;

    virtual ftype sample_on_cone(int index, ftype length, RandomEngine &re) const;

    virtual ftype compute_beta(int index, ftype length) const;

    virtual ftype cone(ftype beta, ftype length) const = 0;
protected:
    ftype alpha;

    virtual ftype base_kernel_value(ftype beta, ftype radius) const = 0;
};

class SmoothBalancedCellKernel : virtual public BalancedCellKernel { // abstract
public:
    SmoothBalancedCellKernel(int dim, ftype alpha);

    virtual ftype dbeta_dlength(ftype beta, ftype length);

    virtual ftype d_dradius(ftype beta, ftype radius) const = 0;
    virtual ftype d_dbeta(ftype beta, ftype radius) const = 0;
    virtual ftype cone_of_d_dbeta(ftype beta, ftype length) const = 0;

};

class BalancedExponentialCellKernel : public SmoothBalancedCellKernel {
public:
    BalancedExponentialCellKernel(int dim, ftype alpha, ftype k);
    std::string latex() const override;

    ftype d_dradius(ftype beta, ftype radius) const override;
    ftype d_dbeta(ftype beta, ftype radius) const override;
    ftype cone_of_d_dbeta(ftype beta, ftype length) const override;

    ftype cone(ftype beta, ftype length) const override;

protected:
    ftype k;
    ftype base_kernel_value(ftype beta, ftype radius) const override;
};


ftype _F(ftype m, ftype k, ftype beta, ftype length);

class BalancedPolynomialCellKernel : public SmoothBalancedCellKernel {
public:
    BalancedPolynomialCellKernel(int dim, ftype alpha, ftype k);
    std::string latex() const override;

    ftype d_dradius(ftype beta, ftype radius) const override;
    ftype d_dbeta(ftype beta, ftype radius) const override;
    ftype cone_of_d_dbeta(ftype beta, ftype length) const override;

    ftype cone(ftype beta, ftype length) const override;

protected:
    ftype k;
    ftype base_kernel_value(ftype beta, ftype radius) const override;
};


class BalancedSecondPolynomialCellKernel : public BalancedCellKernel {
public:
    BalancedSecondPolynomialCellKernel(int dim, ftype alpha);
    std::string latex() const override;

    ftype cone(ftype beta, ftype length) const override;

protected:
    ftype k;
    ftype base_kernel_value(ftype beta, ftype radius) const override;
};


class BalancedSigmoidalGaussian : public BalancedCellKernel {
public:
    BalancedSigmoidalGaussian(int dim, ftype alpha);
    std::string latex() const override;

    ftype cone(ftype beta, ftype length) const override;

protected:
    ftype k;
    ftype base_kernel_value(ftype beta, ftype radius) const override;
};

class BalancedLinear : public BalancedCellKernel {
public:
    BalancedLinear(int dim, ftype alpha);
    std::string latex() const override;

    ftype cone(ftype beta, ftype length) const override;

protected:
    ftype k;
    ftype base_kernel_value(ftype beta, ftype radius) const override;
};

class PuncturedCellKernel : virtual public BalancedCellKernel {
public:
    PuncturedCellKernel(int dim, ftype alpha);

    ftype compute_beta(int index, ftype length) const override;

    ftype cone(ftype beta, ftype length) const override;
protected:
    ftype base_kernel_value(ftype beta, ftype radius) const override;
    virtual ftype base_kernel_value(ftype radius) const = 0;

    virtual ftype cone(ftype length) const = 0;
};

class SmoothPuncturedCellKernel : public PuncturedCellKernel, public SmoothBalancedCellKernel {
public:
    SmoothPuncturedCellKernel(int dim, ftype alpha);

    ftype d_dradius(ftype beta, ftype radius) const override;

    ftype d_dbeta(ftype beta, ftype radius) const override;

    ftype cone_of_d_dbeta(ftype beta, ftype length) const override;
protected:
    virtual ftype base_derivative_value(ftype radius) const = 0;
};

class PuncturedConstantCellKernel : public SmoothPuncturedCellKernel {
public:
    PuncturedConstantCellKernel(int dim, ftype alpha);

    std::string latex() const override;

    ftype cone(ftype length) const override;


protected:
    ftype k;
    ftype base_kernel_value(ftype radius) const override;
    ftype base_derivative_value(ftype radius) const override;
};

class PuncturedExponentialCellKernel : public SmoothPuncturedCellKernel {
public:
    PuncturedExponentialCellKernel(int dim, ftype alpha, ftype scale, ftype k);

    std::string latex() const override;

    ftype cone(ftype length) const override;

protected:
    ftype scale;
    ftype k;
    ftype base_kernel_value(ftype radius) const override;

    ftype base_derivative_value(ftype radius) const override;
};

ftype gaussian_bw_to_alpha(ftype sigma, int dim);

template<class CellKernelT>
CellKernelT* as(const ptr<CellKernel> &cell_kernel);
