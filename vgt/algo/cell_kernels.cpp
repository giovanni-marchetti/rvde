#include "cell_kernels.h"

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/distributions/normal.hpp>

CellKernel::CellKernel(int dim) : dim(dim) {
}

RadialCellKernel::RadialCellKernel(int dim, ftype sigma) : CellKernel(dim), sigma(sigma) {}

ftype RadialCellKernel::cone_integral(int index, ftype length) const {
    return general_cone_integral(sigma, length);
}

ftype RadialCellKernel::kernel_value(int index, ftype radius) const {
    return base_kernel_value(sigma, radius);
}

ftype RadialCellKernel::get_sigma() const {
    return sigma;
}

GaussianCellKernel::GaussianCellKernel(int dim, ftype sigma) : RadialCellKernel(dim, sigma) {}

AdaptiveGaussianCellKernel::AdaptiveGaussianCellKernel(int dim, ftype sigma) : RadialCellKernel(dim, sigma),
                                                                               local_sigma(0), initialized(false) {}

AdaptiveGaussianCellKernel::AdaptiveGaussianCellKernel(int dim, ftype sigma, const vec<ftype> &local_sigma) :
        RadialCellKernel(dim, sigma), local_sigma(local_sigma), initialized(true) {
}

bool AdaptiveGaussianCellKernel::is_initialized() const {
    return initialized;
}

UniformCellKernel::UniformCellKernel(int dim) : RadialCellKernel(dim, 1) {
}

ftype UniformCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    if (math::isinf(length)) {
        return std::numeric_limits<ftype>::infinity();
    }
    return nsphere_volume(dim - 1) * qpow(length, dim) / dim;
}

ftype UniformCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                        ftype t0, ftype t1, RandomEngine &re) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    if (math::isinf(t0) || math::isinf(t1)) {
        return std::numeric_limits<ftype>::quiet_NaN();
    }
    return t0 + re.rand_float() * (t1 - t0);
}

ftype UniformCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    return 1;
}

ftype UniformCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                               ftype t1) const {
    if (math::isinf(t0) || math::isinf(t1)) {
        return std::numeric_limits<ftype>::quiet_NaN();
    }
    return static_cast<ftype>(0.5) * (t0 + t1);
}

ftype UniformCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    return cone_integral(index, length);
}

ftype UniformCellKernel::cone_integral_derivative(int index, ftype length) const {
    return 0;
}

ftype UniformCellKernel::derivative_value(int index, ftype radius) const {
    return 0;
}

ftype GaussianCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
            boost::math::gamma_p(dim * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype
GaussianCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0, ftype t1,
                                   RandomEngine &re) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = cdf0 + re.rand_float() * (cdf1 - cdf0);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
        t_ret = t0 + re.rand_float() * (t1 - t0);
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype GaussianCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    return math::exp(-static_cast<ftype>(0.5) * radius * radius / (sigma * sigma));
}

//ftype GaussianCellKernel::normalization_constant(int index) const {
//    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5));
//}

ftype GaussianCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                                ftype t1) const {
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = static_cast<ftype>(0.5) * (cdf0 + cdf1);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
//        t_ret = static_cast<ftype>(0.5) * (t0 + t1);
        return NAN_ftype;
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype GaussianCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    ftype sq_2 = math::sqrt(2);
    ftype sq_2pi = math::sqrt(2 * PI_ftype);
    ftype o5 = static_cast<ftype>(0.5);

    ftype vol_sphere = 2 * math::pow(PI_ftype, o5 * subspace_dim) / boost::math::tgamma(o5 * subspace_dim);

    ftype upper = (length + a) / (sq_2 * sigma);
    ftype lower = a / (sq_2 * sigma);

    ftype erf_upper = boost::math::erf(upper);
    ftype erf_lower = boost::math::erf(lower);

    ftype exp_upper2 = math::exp(-upper * upper);
    ftype exp_lower2 = math::exp(-lower * lower);

    // setting for n=1
    ftype phi_n0 = o5 * sq_2pi * sigma * (erf_upper - erf_lower);
    ftype phi_n1 = sigma * sigma * (-exp_upper2 + exp_lower2) -
                   o5 * sq_2pi * sigma * a * (erf_upper - erf_lower);

    if (subspace_dim == 1) {
        phi_n1 = phi_n0;
    }

    ftype beta_n = math::isinf(length) ? 0 : sigma * sigma * exp_upper2 * length;

    for (int n = 1; n + 2 <= subspace_dim; n++) {
        ftype phi_n2 = -a * phi_n1 + sigma * sigma * n * phi_n0 - beta_n;

        phi_n0 = phi_n1;
        phi_n1 = phi_n2;
        if (!math::isinf(length)) {
            beta_n *= length;
        }
    }


    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
            vol_sphere * math::exp(-(b + a * a) / (2 * sigma * sigma)) * phi_n1;
}

ftype double_factorial_ratio(int n) {
    // returns (n-1)!! / (n-2)!!
    ensure(n >= 2, "Dimensionality too low");
    static vec<ftype> a = {static_cast<ftype>(1), static_cast<ftype>(1), static_cast<ftype>(1)};
    while (a.size() < n + 1) {
        int k = a.size();
        a.push_back(a[k - 2] * static_cast<ftype>(k - 1) / static_cast<ftype>(k - 2));
    }
    return a[n];
}

ftype GaussianCellKernel::cone_integral_derivative(int index, ftype length) const {
    return - math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
           math::pow(static_cast<ftype>(2), 1.5) / sigma * double_factorial_ratio(dim) *
           boost::math::gamma_p((dim + 1) * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype GaussianCellKernel::derivative_value(int index, ftype length) const {
    return (-length / (sigma * sigma)) * math::exp(-static_cast<ftype>(0.5) * length * length / (sigma * sigma));
}

LaplaceCellKernel::LaplaceCellKernel(int dim, ftype sigma) : RadialCellKernel(dim, sigma) {
}

std::string LaplaceCellKernel::latex() const {
    return "K(t) = e^{\\frac{-t}{" + std::to_string(sigma) + "}}";
}

ftype LaplaceCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    return math::exp(-radius / sigma);
}

ftype LaplaceCellKernel::derivative_value(int index, ftype radius) const {
    throw std::runtime_error("not implemented");
}

ftype LaplaceCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    return nsphere_volume(dim - 1) * math::pow(sigma, dim) * boost::math::tgamma_lower(dim, length / sigma);
}

ftype LaplaceCellKernel::cone_integral_derivative(int index, ftype length) const {
    throw std::runtime_error("not implemented");
}

ftype LaplaceCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    throw std::runtime_error("not implemented");
}

ftype
LaplaceCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                  ftype t1, RandomEngine &re) const {
    throw std::runtime_error("not implemented");
}

ftype LaplaceCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u,
                                               ftype t0, ftype t1) const {
    throw std::runtime_error("not implemented");
}

// F(m, k) = \int_0^l {t^m (t + 1)^{-k} dt}
ftype _F_int(int m, int k, ftype length) {
    if (length <= -1) {
        return INF_ftype;
    }

    if (m == 0) {
        return ftype(1) / (-k + 1) * (math::pow(length + 1, -k + 1) - 1);

    } else if (math::isinf(length)) {
//        return factorial(m) * factorial(k - m - 1) / factorial(k - 1) / (k - 1);
        return ftype(m) / ftype(k - 1) * _F_int(m - 1, k - 1, length);
    }

    if (k < 30) {
        return ftype(1) / (k - 1) * (
                //- math::pow(length, m) * math::pow(beta * length + 1, -k + 1)
                - math::pow(length / (length + 1), m) * math::pow(length + 1, -k+1+m)
                + m * _F_int(m - 1, k - 1, length)
        );
    }

    // Numeric integration
    auto integrand = [=](ftype t) {
        return qpow(t, m) / qpow(t + 1, k);
    };

    int n = 1;
    ftype tol = static_cast<ftype>(0);

    ftype a = 0;
    ftype b = length;
    ftype h = (b - a);
    ftype I = (integrand(a) + integrand(b)) * (h / 2);

    for(int it = 0; it < 12; it++) {
        h /= 2;

        ftype sum(0);
        for(int j = 1U; j <= n; j++)
        {
            sum += integrand(a + (ftype((j * 2) - 1) * h));
        }

        const ftype I0 = I;
        I = (I / 2) + (h * sum);

        const ftype ratio     = I0 / I;
        const ftype delta     = ratio - 1;
        const ftype delta_abs = ((delta < 0) ? -delta : delta);

        if((it > 1) && (delta_abs < tol)) {
            break;
        }

        n *= 2;
    }

    return I;

//    if (m == 0) {
//        return ftype(1) / (-k + 1) * (math::pow(length + 1, -k + 1) - 1);
//    } else if (math::isinf(length)) {
//        return factorial(m) * factorial(k - m - 1) / factorial(k - 1) / (k - 1);
////        return ftype(1) / (k - 1) * (
////                m * _F(m - 1, k - 1, beta, length)
////        );
//    } else {
////        ftype val = math::pow(length, m+1) * boost::math::hypergeometric_pFq({k, m + 1}, {m + 2}, -length) / (m+1);
////        std::cerr << val << std::endl;
////        exit(0);
////        return val;
//    }
}

// F(m, k) = \int_0^l {t^m (\beta t + 1)^{-k} dt}
ftype _F(ftype m, ftype k, ftype beta, ftype length) {
    ensure(m == static_cast<int>(m) && k == static_cast<int>(k), "only supporting int degree");
    if (beta == 0) {
        return math::pow(length, m + 1) / (m + 1);
    }
    return math::pow(beta, -(m+1)) * _F_int(static_cast<int>(m), static_cast<int>(k), beta * length);
}

PolynomialCellKernel::PolynomialCellKernel(int dim, ftype sigma, ftype k) : RadialCellKernel(dim, sigma), k(k) {
}

std::string PolynomialCellKernel::latex() const {
    return std::string("(\\frac{x}{\\sigma} + 1)^{-k}");
}

ftype PolynomialCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    return math::pow(radius / sigma + 1, -k);
}

ftype PolynomialCellKernel::derivative_value(int index, ftype radius) const {
    throw std::runtime_error("not implemented");
}

ftype PolynomialCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    return _F(dim - 1, k, 1 / sigma, length) * nsphere_volume(dim - 1);
}

ftype PolynomialCellKernel::cone_integral_derivative(int index, ftype length) const {
    throw std::runtime_error("not implemented");
}

ftype
PolynomialCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    throw std::runtime_error("not implemented");
}

ftype
PolynomialCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u, ftype t0,
                                     ftype t1, RandomEngine &re) const {
    throw std::runtime_error("not implemented");
}

ftype
PolynomialCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center, const dvector &u,
                                            ftype t0, ftype t1) const {
    throw std::runtime_error("not implemented");
}

void AdaptiveGaussianCellKernel::update_local_bandwidths(const vec<ftype> &new_local_sigma) {
    initialized = true;
    this->local_sigma = new_local_sigma;
}

ftype AdaptiveGaussianCellKernel::cone_integral(int index, ftype length) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    return general_cone_integral(sigma * local_sigma[index], length);
}

ftype AdaptiveGaussianCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
            boost::math::gamma_p(dim * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype
AdaptiveGaussianCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                           ftype t0, ftype t1, RandomEngine &re) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = sigma * local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = cdf0 + re.rand_float() * (cdf1 - cdf0);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
        t_ret = t0 + re.rand_float() * (t1 - t0);
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype AdaptiveGaussianCellKernel::kernel_value(int index, ftype radius) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = sigma * local_sigma[index];
    return base_kernel_value(sigma, radius);
}

ftype AdaptiveGaussianCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    return math::exp(-static_cast<ftype>(0.5) * radius * radius / (sigma * sigma));
}

//ftype AdaptiveGaussianCellKernel::normalization_constant(int index) const {
//    ensure(!local_sigma.empty(), "Local bandwidths not updated");
//    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
//    ftype sigma = global_sigma * local_sigma[index];
//    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5));
//}

ftype AdaptiveGaussianCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center,
                                                        const dvector &u, ftype t0, ftype t1) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = sigma * local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = static_cast<ftype>(0.5) * (cdf0 + cdf1);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
//        t_ret = static_cast<ftype>(0.5) * (t0 + t1);
        return NAN_ftype;
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype AdaptiveGaussianCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = sigma * local_sigma[index];
    ftype sq_2 = math::sqrt(2);
    ftype sq_2pi = math::sqrt(2 * PI_ftype);
    ftype o5 = static_cast<ftype>(0.5);

    ftype vol_sphere = 2 * math::pow(PI_ftype, o5 * subspace_dim) / boost::math::tgamma(o5 * subspace_dim);

    ftype upper = (length + a) / (sq_2 * sigma);
    ftype lower = a / (sq_2 * sigma);

    ftype erf_upper = boost::math::erf(upper);
    ftype erf_lower = boost::math::erf(lower);

    ftype exp_upper2 = math::exp(-upper * upper);
    ftype exp_lower2 = math::exp(-lower * lower);

    // setting for n=1
    ftype phi_n0 = o5 * sq_2pi * sigma * (erf_upper - erf_lower);
    ftype phi_n1 = sigma * sigma * (-exp_upper2 + exp_lower2) -
                   o5 * sq_2pi * sigma * a * (erf_upper - erf_lower);

    if (subspace_dim == 1) {
        phi_n1 = phi_n0;
    }

    ftype beta_n = math::isinf(length) ? 0 : sigma * sigma * exp_upper2 * length;

    for (int n = 1; n + 2 <= subspace_dim; n++) {
        ftype phi_n2 = -a * phi_n1 + sigma * sigma * n * phi_n0 - beta_n;

        phi_n0 = phi_n1;
        phi_n1 = phi_n2;
        if (!math::isinf(length)) {
            beta_n *= length;
        }
    }


    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
            vol_sphere * math::exp(-(b + a * a) / (2 * sigma * sigma)) * phi_n1;
}

ftype AdaptiveGaussianCellKernel::cone_integral_derivative(int index, ftype length) const {
    throw std::runtime_error("cone integral of the derivative is not implemented for this cell kernel");
}

ftype AdaptiveGaussianCellKernel::derivative_value(int index, ftype radius) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = sigma * local_sigma[index];
    return (-radius / (sigma * sigma)) * math::exp(-static_cast<ftype>(0.5) * radius * radius / (sigma * sigma));
}

FixedVolumeGaussianCellKernel::FixedVolumeGaussianCellKernel(int dim, ftype volume) :
        RadialCellKernel(dim, 0), volume(volume) {
}

void FixedVolumeGaussianCellKernel::compute_local_bandwidths(const vec<vec<ftype>> &lengths) {
    std::cout << "Computing local bandwidths" << std::endl;
    int n_cells = lengths.size();
    local_sigma = vec<ftype>(n_cells);

    my_tqdm bar(n_cells);
    #pragma omp parallel for
    for (int i = 0; i < n_cells; i++) {
        bar.atomic_iteration();

        {
            local_sigma[i] = INF_ftype;
            ftype cur_volume = 0;
            for (int j = 0; j < lengths[i].size(); j++) {
                cur_volume += cone_integral(i, lengths[i][j]);
            }
            cur_volume /= lengths[i].size();
            if (cur_volume < volume) {
                continue;
            }
        }


        ftype L = ftype(0.01);
        ftype R = ftype(100);
        int BS_IT = 100;
        ftype BS_EPS = 1e-9;
        for (int it = 0; it < BS_IT; it++) {
            local_sigma[i] = (L + R) / 2;

            ftype cur_volume = 0;
            for (int j = 0; j < lengths[i].size(); j++) {
                cur_volume += cone_integral(i, lengths[i][j]);
            }
            cur_volume /= lengths[i].size();
            if (math::abs(cur_volume - volume) < BS_EPS) {
                break;
            }
            if (cur_volume < volume) {
                L = local_sigma[i];
            } else {
                R = local_sigma[i];
            }
        }
    }
    bar.bar().finish();

}

ftype FixedVolumeGaussianCellKernel::kernel_value(int index, ftype radius) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    return base_kernel_value(sigma, radius);
}

ftype FixedVolumeGaussianCellKernel::base_kernel_value(ftype sigma, ftype radius) const {
    if (math::isinf(sigma)) {
        return 1;
    }
    return math::exp(-static_cast<ftype>(0.5) * radius * radius / (sigma * sigma));
}

ftype FixedVolumeGaussianCellKernel::derivative_value(int index, ftype radius) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    return (-radius / (sigma * sigma)) * math::exp(-static_cast<ftype>(0.5) * radius * radius / (sigma * sigma));
}

ftype FixedVolumeGaussianCellKernel::cone_integral(int index, ftype length) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    return general_cone_integral(sigma, length);
}

ftype FixedVolumeGaussianCellKernel::general_cone_integral(ftype sigma, ftype length) const {
    if (math::isinf(sigma)) {
        return nsphere_volume(dim - 1) * math::pow(length, dim) / dim;
    }
    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
           boost::math::gamma_p(dim * static_cast<ftype>(0.5), length * length / (2 * sigma * sigma));
}

ftype FixedVolumeGaussianCellKernel::cone_integral_derivative(int index, ftype length) const {
    throw std::runtime_error("cone integral of the derivative is not implemented for this cell kernel");
}

ftype FixedVolumeGaussianCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b,
                                                              int subspace_dim) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    ftype sq_2 = math::sqrt(2);
    ftype sq_2pi = math::sqrt(2 * PI_ftype);
    ftype o5 = static_cast<ftype>(0.5);

    ftype vol_sphere = 2 * math::pow(PI_ftype, o5 * subspace_dim) / boost::math::tgamma(o5 * subspace_dim);

    ftype upper = (length + a) / (sq_2 * sigma);
    ftype lower = a / (sq_2 * sigma);

    ftype erf_upper = boost::math::erf(upper);
    ftype erf_lower = boost::math::erf(lower);

    ftype exp_upper2 = math::exp(-upper * upper);
    ftype exp_lower2 = math::exp(-lower * lower);

    // setting for n=1
    ftype phi_n0 = o5 * sq_2pi * sigma * (erf_upper - erf_lower);
    ftype phi_n1 = sigma * sigma * (-exp_upper2 + exp_lower2) -
                   o5 * sq_2pi * sigma * a * (erf_upper - erf_lower);

    if (subspace_dim == 1) {
        phi_n1 = phi_n0;
    }

    ftype beta_n = math::isinf(length) ? 0 : sigma * sigma * exp_upper2 * length;

    for (int n = 1; n + 2 <= subspace_dim; n++) {
        ftype phi_n2 = -a * phi_n1 + sigma * sigma * n * phi_n0 - beta_n;

        phi_n0 = phi_n1;
        phi_n1 = phi_n2;
        if (!math::isinf(length)) {
            beta_n *= length;
        }
    }


    return math::pow(2 * PI_ftype * sigma * sigma, dim * static_cast<ftype>(0.5)) *
           vol_sphere * math::exp(-(b + a * a) / (2 * sigma * sigma)) * phi_n1;
}

ftype
FixedVolumeGaussianCellKernel::sample_on_line(int index, const dvector &ref, const dvector &center, const dvector &u,
                                              ftype t0, ftype t1, RandomEngine &re) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = cdf0 + re.rand_float() * (cdf1 - cdf0);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
        t_ret = t0 + re.rand_float() * (t1 - t0);
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}

ftype FixedVolumeGaussianCellKernel::sample_on_line_median(int index, const dvector &ref, const dvector &center,
                                                           const dvector &u, ftype t0, ftype t1) const {
    ensure(!local_sigma.empty(), "Local bandwidths not updated");
    ensure(0 <= index && index < local_sigma.size(), "Cell index out of range (" + std::to_string(index) + ")");
    ftype sigma = local_sigma[index];
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    ftype mean = u.dot(center - ref);
    boost::math::normal_distribution distr(mean, sigma);
    ftype cdf0 = boost::math::cdf(distr, t0);
    ftype cdf1 = boost::math::cdf(distr, t1);
    ftype cdf_ret = static_cast<ftype>(0.5) * (cdf0 + cdf1);
    ftype t_ret;
    try {
        t_ret = quantile(distr, cdf_ret);
    } catch (const std::overflow_error& e) {
        std::cout << "Warning: overflow error in sampling on a segment [" << e.what() << "]" << std::endl;
//        t_ret = static_cast<ftype>(0.5) * (t0 + t1);
        return NAN_ftype;
    }
    t_ret = std::max(t0, std::min(t1, t_ret));
    return t_ret;
}


// ========================== BALANCED KERNELS ========================== //

BalancedCellKernel::BalancedCellKernel(int dim, ftype alpha) :
        CellKernel(dim), alpha(alpha) {
}

ftype BalancedCellKernel::kernel_value(int index, ftype radius, ftype length) const {
    ftype beta = compute_beta(index, length);
    return base_kernel_value(beta, radius);
}

ftype SmoothBalancedCellKernel::dbeta_dlength(ftype beta, ftype length) {
    return - math::pow(length, dim - 1) * base_kernel_value(beta, length) /
            cone_of_d_dbeta(beta, length);
}

ftype BalancedCellKernel::cone_integral(int index, ftype length) const {
    return alpha * nsphere_volume(dim - 1);
}

ftype
BalancedCellKernel::cone_integral_uncentered(int index, ftype length, ftype a, ftype b, int subspace_dim) const {
    throw std::runtime_error("not implemented");
}

SmoothBalancedCellKernel::SmoothBalancedCellKernel(int dim, ftype alpha) : BalancedCellKernel(dim, alpha) {
}

BalancedExponentialCellKernel::BalancedExponentialCellKernel(int dim, ftype alpha, ftype k) :
        SmoothBalancedCellKernel(dim, alpha), BalancedCellKernel(dim, alpha), k(k) { }

ftype BalancedExponentialCellKernel::base_kernel_value(ftype beta, ftype radius) const {
    if (beta >= 0) {
        return math::exp(-math::pow(math::abs(beta) * radius, k));
    } else {
        return math::exp(math::pow(math::abs(beta) * radius, k));
    }
}

ftype BalancedCellKernel::compute_beta(int index, ftype length) const {
    static bool printed = false;
    if (!printed) {
        ftype beta = math::pow(alpha / cone(1, INF_ftype), ftype(1) / dim);
        printed = true;
        std::cout << "BETA: " << beta << std::endl;
    }

    // todo this must be ok
    if (math::isinf(length)) {
        ftype beta = math::pow(alpha / cone(1, length), ftype(1) / dim);
	    return beta;
    }

//    ftype max_integral = cone(0, length);
//    ftype min_val = static_cast<ftype>(0);
    ftype max_val = static_cast<ftype>(1e+3);
    int NEWTON_MIN_IT = 100;
    int NEWTON_MAX_IT = 1000;
    ftype NEWTON_DIFF_EPS = static_cast<ftype>(1e-15);
//    if (max_integral - alpha < 0) {
//        ftype l = -max_val, r = -min_val;
//        ftype beta;
//        for (int it = 0; it < NEWTON_MAX_IT && abs(r - l) > NEWTON_DIFF_EPS; it++) {
//            beta = (l + r) / 2;
//            ftype F = cone(beta, length) - alpha;
//            if (F < 0) {
//                r = beta;
//            } else {
//                l = beta;
//            }
//        }
//
//        return beta;
//
//    }
    // newton
//    ftype beta = 1;
//    ftype F = lambda * cone(beta, length) - alpha;
//    int NEWTON_MAX_IT = 1000;
//    ftype NEWTON_DIFF_EPS = 1e-9;
//    for (int it = 0; it < NEWTON_MAX_IT && abs(F) > NEWTON_DIFF_EPS; it++) {
//        ftype F_prime = lambda * cone_derivative(beta, length);
//        beta -= 0.1 * F / F_prime;
//        if (beta < 0) beta = 0.00001;
//        F = lambda * cone(beta, length) - alpha;
//    }
    // bin search
    ftype l = -max_val, r = max_val;
    ftype beta;
//    for (int it = 0; it < NEWTON_MIN_IT; it++) {
    for (int it = 0; it < NEWTON_MIN_IT || it < NEWTON_MAX_IT && abs(r - l) > NEWTON_DIFF_EPS; it++) {
        beta = (l + r) / 2;
        ftype F = cone(beta, length) - alpha;
        if (F < 0) {
            r = beta;
        } else {
            l = beta;
        }
    }

    return beta;
}

ftype BalancedCellKernel::sample_on_cone(int index, ftype length, RandomEngine &re) const {
    ftype beta = compute_beta(index, length);
    ftype vol = re.rand_float() * alpha;
    ftype l = 0;
    ftype r = std::min(length, ftype(1000));

    int BS_MAX_IT = 1000;
    ftype BS_DIFF_EPS = 1e-9;
    for (int it = 0; it < BS_MAX_IT && abs(r - l) > BS_DIFF_EPS; it++) {
        ftype m = (l + r) / 2;
        ftype cur = cone(beta, m);
        if (cur < vol) {
            l = m;
        } else {
            r = m;
        }
    }

    return l;

}

ftype BalancedExponentialCellKernel::d_dradius(ftype beta, ftype radius) const {
    return base_kernel_value(beta, radius) * (-k) * radius * math::pow(math::abs(beta) * radius, k - 1);
}

ftype BalancedExponentialCellKernel::d_dbeta(ftype beta, ftype radius) const {
    return base_kernel_value(beta, radius) * (-k) * beta * math::pow(math::abs(beta) * radius, k - 1);
}

// \int e^t t^m dt from 0 to l
ftype _Int0(int m, ftype l) {
    if (m == 0) {
        return math::exp(l) - 1;
    } else {
        return math::exp(l) * math::pow(l, m) - m * _Int0(m - 1, l);
    }
}

ftype BalancedExponentialCellKernel::cone(ftype beta, ftype length) const {
    if (beta < 0) {
        beta = -beta;
        int k0 = int(k);
        ensure(k0 == k, "k non-integer is not supported");
        ensure(dim % k0 == 0, "dim % k != 0 is not supported");
        ensure(k0 == 1, "TEMPORARY only k=1 is supported!");
//        return _Int0(dim / k0 - 1, math::pow(beta * length, k)) / math::pow();
        return _Int0(dim - 1, beta * length) * math::pow(beta, -dim); // TODO formula is wrong for k != 1
    } else if (beta == 0) {
        return math::pow(length, dim) / dim;
    } else { // beta > 0
        return math::pow(beta, -dim) / k * boost::math::tgamma_lower(ftype(dim) / k, math::pow(beta * length, k));
    }
}

ftype BalancedExponentialCellKernel::cone_of_d_dbeta(ftype beta, ftype length) const {
    if (beta < 0) {
        beta = -beta;
        int k0 = int(k);
        int n = dim + k0;
        ensure(k0 == k, "k non-integer is not supported");
        ensure(n % k0 == 0, "dim % k != 0 is not supported");
        return (-k) * math::pow(beta, k - 1) *
                _Int0(n / k0 - 1, math::pow(beta * length, k)) ;
    } else if (beta == 0) {
        return (-k) * math::pow(beta, k - 1) * math::pow(length, dim + k) / (dim + k);
    } else { // beta > 0
        ftype n = dim + k;
        return (-k) * math::pow(beta, k - 1) *
                math::pow(beta, -n) / k * boost::math::tgamma_lower(ftype(n) / k, math::pow(beta * length, k));
    }
}

BalancedPolynomialCellKernel::BalancedPolynomialCellKernel(int dim, ftype alpha, ftype k) :
        SmoothBalancedCellKernel(dim, alpha), BalancedCellKernel(dim, alpha), k(k) {
    ensure(k > dim, "bad kernel");
}

ftype BalancedPolynomialCellKernel::d_dradius(ftype beta, ftype radius) const {
    return -k * beta * math::pow(beta * radius + 1, -k - 1);
}

ftype BalancedPolynomialCellKernel::d_dbeta(ftype beta, ftype radius) const {
    return -k * radius * math::pow(beta * radius + 1, -k - 1);
}

ftype BalancedPolynomialCellKernel::base_kernel_value(ftype beta, ftype radius) const {
//    return math::exp(beta);
    return math::pow(beta * radius + 1, -k);
}

ftype BalancedPolynomialCellKernel::cone(ftype beta, ftype length) const {
    if (beta < 0 && length >= -1 / beta) {
        return INF_ftype;
    }
    return _F(dim - 1, k, beta, length);
}

ftype BalancedPolynomialCellKernel::cone_of_d_dbeta(ftype beta, ftype length) const {
    return -k * _F(dim, k + 1, beta, length);
}

BalancedSecondPolynomialCellKernel::BalancedSecondPolynomialCellKernel(int dim, ftype alpha)
        : BalancedCellKernel(dim, alpha) {
}

ftype BalancedSecondPolynomialCellKernel::base_kernel_value(ftype beta, ftype radius) const {
    return math::pow(radius + 1, -beta);
}

// m is assumed a non-negative integer!
// F(m, beta) = \int_0^l {t^m (t + 1)^{-beta} dt}
ftype _F2(ftype m, ftype beta, ftype length) {
    if (beta == 0) {
        return math::pow(length, m + 1) / (m + 1);
    } else if (m == 0) {
        if (beta == 1) {
            return math::log(length + 1);
        } else {
            return (math::pow(length + 1, -beta + 1) - 1) / (-beta + 1);
        }
    } else {
        if (beta == 1) {
            throw std::runtime_error("case of beta=1 not implemented");
        } else {
            if (math::isinf(length)) {
                if (m > beta - 1) {
                    return INF_ftype;
                } else if (m == -beta + 1) {
                    throw std::runtime_error("case not implemented");
                } else {
                    return 1 / (-beta + 1) * ( - m * _F2(m - 1, beta - 1, length) );
                }
            }
            return 1 / (-beta + 1) * (
                    math::pow(length, m) * math::pow(length + 1, -beta + 1)
                    - m * _F2(m - 1, beta - 1, length)
                    );
        }
    }
}

ftype BalancedSecondPolynomialCellKernel::cone(ftype beta, ftype length) const {
    return _F2(dim - 1, beta, length);
}

BalancedSigmoidalGaussian::BalancedSigmoidalGaussian(int dim, ftype alpha) :
        BalancedCellKernel(dim, alpha) {

}

ftype BalancedSigmoidalGaussian::base_kernel_value(ftype beta, ftype radius) const {
    return std::max(math::exp(-radius * radius) * (1 + beta) - beta, ftype(0));
}

ftype BalancedSigmoidalGaussian::cone(ftype beta, ftype length) const {
    if (beta > 0) {
        length = std::min(length, math::sqrt(math::log((1 + beta) / beta)));
    }
    return 0.5 * (1 + beta) * boost::math::tgamma_lower(ftype(dim) / 2, length * length)
    - beta * math::pow(length, dim) / ftype(dim);

}

BalancedLinear::BalancedLinear(int dim, ftype alpha) : BalancedCellKernel(dim, alpha) {
}

ftype BalancedLinear::base_kernel_value(ftype beta, ftype radius) const {
    return std::max(1 - beta * radius, ftype(0));
}

ftype BalancedLinear::cone(ftype beta, ftype length) const {
    if (beta > 0) {
        length = std::min(length, 1 / beta);
    }
    ftype l_n = math::pow(length, dim);
    return l_n / dim - beta * l_n * length / (dim + 1);
}

PuncturedCellKernel::PuncturedCellKernel(int dim, ftype alpha) : BalancedCellKernel(dim, alpha) {}

ftype PuncturedCellKernel::compute_beta(int index, ftype length) const {
    if (length == 0) {
        return 0;
    }
    return alpha / cone(length);
}

ftype PuncturedCellKernel::base_kernel_value(ftype beta, ftype radius) const {
    if (beta <= 0) return 0;
    return beta * base_kernel_value(radius);
}

ftype PuncturedCellKernel::cone(ftype beta, ftype length) const {
    if (beta <= 0) return 0;
    return beta * cone(length);
}

SmoothPuncturedCellKernel::SmoothPuncturedCellKernel(int dim, ftype alpha) :
        SmoothBalancedCellKernel(dim, alpha), PuncturedCellKernel(dim, alpha), BalancedCellKernel(dim, alpha) {
}

ftype SmoothPuncturedCellKernel::d_dradius(ftype beta, ftype radius) const {
    if (beta <= 0) return 0;
    return beta * base_derivative_value(radius);
}

ftype SmoothPuncturedCellKernel::d_dbeta(ftype beta, ftype radius) const {
    if (beta <= 0) return 0;
    return base_kernel_value(radius);
}

ftype SmoothPuncturedCellKernel::cone_of_d_dbeta(ftype beta, ftype length) const {
    if (beta <= 0) return 0;
    return cone(length);
}

PuncturedExponentialCellKernel::PuncturedExponentialCellKernel(int dim, ftype alpha, ftype scale, ftype k) :
        SmoothPuncturedCellKernel(dim, alpha), BalancedCellKernel(dim, alpha), scale(scale), k(k) {}

ftype PuncturedExponentialCellKernel::base_kernel_value(ftype radius) const {
    return math::exp(-math::pow(scale * radius, k));
}

ftype PuncturedExponentialCellKernel::cone(ftype length) const {
    return math::pow(scale, -dim) / k * boost::math::tgamma_lower(ftype(dim) / k, math::pow(scale * length, k));
}

ftype PuncturedExponentialCellKernel::base_derivative_value(ftype radius) const {
    return -k * math::pow(scale * radius, k - 1) * scale * base_kernel_value(radius);
}

PuncturedConstantCellKernel::PuncturedConstantCellKernel(int dim, ftype alpha) : BalancedCellKernel(dim, alpha),
                                                                                 SmoothPuncturedCellKernel(dim, alpha) {
}

ftype PuncturedConstantCellKernel::base_kernel_value(ftype radius) const {
    return 1;
}

ftype PuncturedConstantCellKernel::cone(ftype length) const {
    return math::pow(length, dim) / dim;
}

ftype PuncturedConstantCellKernel::base_derivative_value(ftype radius) const {
    return 0;
}

std::string GaussianCellKernel::latex() const {
    return "K(t) = e^{\\frac{-t^2}{" + std::to_string(sigma) + "^2}}";
}

std::string UniformCellKernel::latex() const {
    return "K(t) = 1";
}

std::string AdaptiveGaussianCellKernel::latex() const {
    return "K(t) = e^{\\frac{-t^2}{\\sigma_i^2}}";
}

std::string FixedVolumeGaussianCellKernel::latex() const {
    return "K(t) = e^{\\frac{-t^2}{\\sigma_i^2}}";
}

std::string BalancedExponentialCellKernel::latex() const {
    std::string k_ = (k - math::round(k) == 0 ? std::to_string(int(k)) : std::to_string(k));
    return "K(t, \\beta) = e^{-sgn(\\beta)(|\\beta|t)^" + k_ + "}";
}

std::string BalancedPolynomialCellKernel::latex() const {
    std::string k_ = (k - math::round(k) == 0 ? std::to_string(int(k)) : std::to_string(k));
    return "K(t, \\beta) = (\\beta t + 1)^{-" + k_ + "}";
}

std::string BalancedSecondPolynomialCellKernel::latex() const {
    return "K(t, \\beta) = (t + 1)^{-\\beta}";
}

std::string BalancedSigmoidalGaussian::latex() const {
    return "K(t, \\beta) = \\max\\{e^{-t^2}(1+\\beta)-\\beta, 0\\}";
}

std::string BalancedLinear::latex() const {
    return "K(t, \\beta) = \\max\\{1 - \\beta t, 0\\}";
}

std::string PuncturedExponentialCellKernel::latex() const {
    std::string k_ = (k - math::round(k) == 0 ? std::to_string(int(k)) : std::to_string(k));
    return "K(t, \\lambda) = \\lambda e^{-t^" + k_ + "}";
}

std::string PuncturedConstantCellKernel::latex() const {
    return "K(t, \\lambda) = \\lambda";
}

template<class CellKernelT>
CellKernelT *as(const ptr<CellKernel> &cell_kernel) {
    return dynamic_cast<CellKernelT*>(cell_kernel.get());
}

ftype gaussian_bw_to_alpha(ftype sigma, int dim) {
    ftype beta = 1 / sigma;
    ftype val = math::pow(beta, -dim) / 2 * boost::math::tgamma(ftype(dim) / 2);
    return val;
}

template RadialCellKernel* as<RadialCellKernel>(const ptr<CellKernel> &cell_kernel);
template BalancedCellKernel* as<BalancedCellKernel>(const ptr<CellKernel> &cell_kernel);
template SmoothBalancedCellKernel* as<SmoothBalancedCellKernel>(const ptr<CellKernel> &cell_kernel);

