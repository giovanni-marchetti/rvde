#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "algo/cell_kernels.h"
#include "algo/kernels.h"
#include "algo/VoronoiDensityEstimator.h"
#include "extra/KDE.h"
#include "extra/GabrielGraph.h"

namespace py = boost::python;
namespace numpy = boost::python::numpy;

void copy_numpy_data_float(const std::string &dtype, const numpy::ndarray &array, ftype *out) {
    char *src = array.get_data();
    int nd = array.get_nd();
    const Py_intptr_t *shape = array.get_shape();
    const Py_intptr_t *strides = array.get_strides();
    vec<Py_intptr_t> index(nd, 0);
    int dst_shift = 0;
    int jump = 0;
    while (!jump) {
        Py_intptr_t src_shift = 0;
        for (int i = nd - 1; i >= 0; i--) {
            src_shift += index[i] * strides[i];
        }
        if (dtype == "float32") {
            assert(sizeof(float) == 4);
            out[dst_shift++] = static_cast<ftype>(*reinterpret_cast<float *>(src + src_shift));
        } else if (dtype == "float64") {
            assert(sizeof(double) == 8);
            out[dst_shift++] = static_cast<ftype>(*reinterpret_cast<double *>(src + src_shift));
        } else if (dtype == "float128") {
            assert(sizeof(long double) == 16);
            out[dst_shift++] = static_cast<ftype>(*reinterpret_cast<long double *>(src + src_shift));
        } else {
            throw std::runtime_error("Unknown dtype: " + dtype);
        }
        jump = 1;
        for (int i = nd - 1; jump && i >= 0; i--) {
            index[i]++;
            jump = index[i] >= shape[i];
            if (jump) {
                index[i] = 0;
            }
        }
    }
}

dmatrix numpy_to_dmatrix(const numpy::ndarray &array) {
    assert(array.get_nd() == 2);
    auto shape = array.get_shape();
    dmatrix res(shape[1], shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, res.data());
    return res;
}

dvector numpy_to_dvector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    dvector res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, res.data());
    return res;
}

vec<ftype> numpy_to_vector(const numpy::ndarray &_array) {
    const numpy::ndarray &array = _array.squeeze();
    assert(array.get_nd() == 1);
    auto shape = array.get_shape();
    vec<ftype> res(shape[0]);
    std::string dtype(py::extract<char const *>(py::str(array.get_dtype())));
    copy_numpy_data_float(dtype, array, res.data());
    return res;
}

numpy::ndarray vector_to_numpy(const vec<ftype> &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[1] = { static_cast<Py_intptr_t>(data.size()) };
    numpy::ndarray result = numpy::zeros(1, shape, numpy::dtype(py::object("float128")));
    std::transform(data.begin(), data.end(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

numpy::ndarray vector_vector_to_numpy(const vec<vec<ftype>> &data) {
    // At the moment, always saves to np.float128
    // assume a matrix!!!
//    std::cerr << "HERE" << std::endl;
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.size()), static_cast<Py_intptr_t>(data[0].size())};
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype(py::object("float128")));
    for (int i = 0; i < data.size(); i++) {
//        std::cerr << "HERE " << i << std::endl;
        ensure(data[i].size() == data[0].size(), "Not a matrix!");
        std::transform(data[i].begin(), data[i].end(), reinterpret_cast<long double*>(result.get_data()) + (i * data[0].size()),
                       [](ftype a) {return static_cast<long double>(a);});
    }
    return result;
}

numpy::ndarray dvector_to_numpy(const dvector &data) {
    return vector_to_numpy(vec<ftype>(data.begin(), data.end()));
}

numpy::ndarray dmatrix_to_numpy(const dmatrix &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.cols()), static_cast<Py_intptr_t>(data.rows()) };
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype(py::object("float128")));
    std::transform(data.data(), data.data() + data.size(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

numpy::ndarray dynmatrix_to_numpy(const dynmatrix &data) {
    // At the moment, always saves to np.float128
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.cols()), static_cast<Py_intptr_t>(data.rows()) };
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype(py::object("float128")));
    std::transform(data.data(), data.data() + data.size(), reinterpret_cast<long double*>(result.get_data()),
                   [](ftype a) {return static_cast<long double>(a);});
    return result;
}

numpy::ndarray edge_matrix_to_numpy(const edge_matrix &data) {
    Py_intptr_t shape[2] = { static_cast<Py_intptr_t>(data.cols()), static_cast<Py_intptr_t>(data.rows()) };
    numpy::ndarray result = numpy::zeros(2, shape, numpy::dtype::get_builtin<int>());
    std::transform(data.data(), data.data() + data.size(), reinterpret_cast<int*>(result.get_data()),
                   [](int a) {return a;});
    return result;
}

numpy::ndarray vec_dmatrix_to_numpy(const vec<dmatrix> &data) {
    // At the moment, always saves to np.float128
    // asserting that add dmatrix have the same size
    Py_intptr_t shape[3] = { static_cast<Py_intptr_t>(data.size()),
                             static_cast<Py_intptr_t>(data[0].cols()),
                             static_cast<Py_intptr_t>(data[0].rows()) };
    numpy::ndarray result = numpy::zeros(3, shape, numpy::dtype(py::object("float128")));
    long double *result_ptr = reinterpret_cast<long double*>(result.get_data());
    for (int i = 0; i < shape[0]; i++) {
        ensure(data[i].cols() == data[0].cols(), "All dmatrices should have the same size to be stacked");
        std::transform(data[i].data(), data[i].data() + data[i].size(), result_ptr,
                       [](ftype a) {return static_cast<long double>(a);});
        result_ptr += data[i].size();
    }
    return result;
}

py::object scalar_to_numpy(ftype a) {
    return vector_to_numpy({a}).reshape(py::tuple()).scalarize();
}

class WrapperFuncs {
public:
    static ptr<VoronoiDensityEstimator> VoronoiDensityEstimator_constructor(
            const numpy::ndarray &points, const ptr<CellKernel> &cell_kernel, int seed,
            int njobs, int nrays_weights, int nrays_sampling, RayStrategyType strategy, const ptr<Bounds> &bounds) {
        return std::make_shared<VoronoiDensityEstimator>(numpy_to_dmatrix(points), cell_kernel, seed, njobs,
                                                         nrays_weights, nrays_sampling, strategy, bounds);
    }

    static void VoronoiDensityEstimator_constructor_wrapper(py::object &self,
                                                            const numpy::ndarray &points, const ptr<CellKernel> &cell_kernel, int seed,
                                                            int njobs, int nrays_weights, int nrays_sampling, RayStrategyType strategy, const ptr<Bounds> &bounds) {

        auto constructor = py::make_constructor(&VoronoiDensityEstimator_constructor);
        constructor(self, points, cell_kernel, seed, njobs,
                    nrays_weights, nrays_sampling, strategy, bounds);
    }

    static numpy::ndarray VoronoiDensityEstimator_dvol_dp(VoronoiDensityEstimator *self) {
        return vec_dmatrix_to_numpy(self->dvol_dp());
    }

    static numpy::ndarray VoronoiDensityEstimator_dlogf_dp(VoronoiDensityEstimator *self, const numpy::ndarray &queries) {
        return dmatrix_to_numpy(self->dlogf_dp(numpy_to_dmatrix(queries)));
    }

    static numpy::ndarray VoronoiDensityEstimator_estimate(VoronoiDensityEstimator *self, const numpy::ndarray &points) {
        return vector_to_numpy(self->estimate(numpy_to_dmatrix(points)));
    }

    static numpy::ndarray VoronoiDensityEstimator_get_points(VoronoiDensityEstimator *self) {
        return dmatrix_to_numpy(self->get_points());
    }

    static numpy::ndarray VoronoiDensityEstimator_sample(VoronoiDensityEstimator *self, int size) {
        return dmatrix_to_numpy(self->sample(size));
    }

    static numpy::ndarray VoronoiDensityEstimator_sample_masked(VoronoiDensityEstimator *self, int size, const numpy::ndarray &mask) {
        return dmatrix_to_numpy(self->sample_masked(size, numpy_to_dvector(mask)));
    }

    static numpy::ndarray VoronoiDensityEstimator_get_volumes(VoronoiDensityEstimator *self) {
        return vector_to_numpy(self->get_volumes());
    }

    static void VoronoiDensityEstimator_initialize_volumes_uncentered(VoronoiDensityEstimator *self,
                                                                      const numpy::ndarray &centroids) {
        self->initialize_weights_uncentered(numpy_to_dmatrix(centroids));
    }

    static numpy::ndarray VoronoiDensityEstimator_weights_getter(VoronoiDensityEstimator *self) {
        return vector_to_numpy(self->get_weights());
    }

    static numpy::ndarray VoronoiDensityEstimator_lengths_debug_getter(VoronoiDensityEstimator *self) {
        return vector_vector_to_numpy(self->lengths_debug);
    }

    static void VoronoiDensityEstimator_weights_setter(VoronoiDensityEstimator *self, const numpy::ndarray &weights) {
        self->set_weights(numpy_to_vector(weights));
    }

    static ptr<AdaptiveGaussianCellKernel> AdaptiveGaussianCellKernel_constructor(
            int dim, ftype global_sigma) {
        return std::make_shared<AdaptiveGaussianCellKernel>(dim, global_sigma);
    }

    static void AdaptiveGaussianCellKernel_constructor_wrapper(py::object &self,
                                                               int dim, ftype global_sigma) {

        auto constructor = py::make_constructor(&AdaptiveGaussianCellKernel_constructor);
        constructor(self, dim, global_sigma);
    }

    static ptr<AdaptiveGaussianCellKernel> AdaptiveGaussianCellKernel_constructor2(
            int dim, ftype global_sigma, const numpy::ndarray &local_sigma) {
        return std::make_shared<AdaptiveGaussianCellKernel>(dim, global_sigma, numpy_to_vector(local_sigma));
    }

    static void AdaptiveGaussianCellKernel_constructor2_wrapper(py::object &self,
                                                               int dim, ftype global_sigma, const numpy::ndarray &local_sigma) {

        auto constructor = py::make_constructor(&AdaptiveGaussianCellKernel_constructor2);
        constructor(self, dim, global_sigma, local_sigma);
    }

    static void AdaptiveGaussianCellKernel_update_local_bandwidths(AdaptiveGaussianCellKernel *self,
                                                                   const numpy::ndarray &bandwidths) {
        self->update_local_bandwidths(numpy_to_vector(bandwidths));
    }

    static ptr<BoundingBox> BoundingBox_constructor(const numpy::ndarray &min, const numpy::ndarray &max) {
        return std::make_shared<BoundingBox>(numpy_to_dvector(min), numpy_to_dvector(max));
    }

    static void BoundingBox_constructor_wrapper(py::object &self, const numpy::ndarray &min, const numpy::ndarray &max) {
        auto constructor = py::make_constructor(&BoundingBox_constructor);
        constructor(self, min, max);
    }

    static numpy::ndarray BoundingBox_lower_getter(BoundingBox *self) {
        return dvector_to_numpy(self->get_mn());
    }

    static numpy::ndarray BoundingBox_upper_getter(BoundingBox *self) {
        return dvector_to_numpy(self->get_mx());
    }

    static bool Bounds_contains(Bounds *self, const numpy::ndarray &ref) {
        return self->contains(numpy_to_dvector(ref));
    }

    static ftype Bounds_max_length(Bounds *self, const numpy::ndarray &ref, const numpy::ndarray &u) {
        return self->max_length(numpy_to_dvector(ref), numpy_to_dvector(u));
    }

    static numpy::ndarray BoundingSphere_radius_getter(BoundingSphere *self) {
        return dvector_to_numpy(self->get_center());
    }

    // =============== KDE ===============
    static ptr<KDE> KDE_constructor(const numpy::ndarray &points, ftype global_bw, int seed, int njobs) {
        return std::make_shared<KDE>(numpy_to_dmatrix(points), global_bw, seed, njobs);
    }

    static void KDE_constructor_wrapper(py::object &self, const numpy::ndarray &points, ftype global_bw, int seed, int njobs) {
        auto constructor = py::make_constructor(&KDE_constructor);
        constructor(self, points, global_bw, seed, njobs);
    }

    static ptr<KDE> KDE_constructor2(const numpy::ndarray &points, const ptr<RadialCellKernel> &ck, int seed, int njobs) {
        return std::make_shared<KDE>(numpy_to_dmatrix(points), ck, seed, njobs);
    }

    static void KDE_constructor2_wrapper(py::object &self, const numpy::ndarray &points, const ptr<RadialCellKernel> &ck, int seed, int njobs) {
        auto constructor = py::make_constructor(&KDE_constructor2);
        constructor(self, points, ck, seed, njobs);
    }

    static numpy::ndarray KDE_estimate(KDE *self, const numpy::ndarray &points) {
        return vector_to_numpy(self->estimate(numpy_to_dmatrix(points)));
    }

    static numpy::ndarray KDE_get_points(KDE *self) {
        return dmatrix_to_numpy(self->get_points());
    }

    static numpy::ndarray KDE_sample(KDE *self, int size) {
        return dmatrix_to_numpy(self->sample(size));
    }

    static numpy::ndarray KDE_weights_getter(KDE *self) {
        return vector_to_numpy(self->get_weights());
    }

    static void KDE_weights_setter(KDE *self, const numpy::ndarray &weights) {
        self->set_weights(numpy_to_vector(weights));
    }

    static py::object gaussian_bw_to_alpha_wrapper(ftype sigma, int dim) {
        return scalar_to_numpy(gaussian_bw_to_alpha(sigma, dim));
    }

    static py::object BalancedCellKernel_cone(BalancedCellKernel* self, ftype beta, ftype length) {
        return scalar_to_numpy(self->cone(beta, length));
    }
    // F(m, k) = \int_0^l {t^m (\beta t + 1)^{-k} dt}
    static py::object _F_wrapper(ftype m, ftype k, ftype beta, ftype length) {
	    return scalar_to_numpy(_F(m, k, beta, length));
    }

    static numpy::ndarray get_gabriel_edges_wrapper(const numpy::ndarray &points) {
        return edge_matrix_to_numpy(get_gabriel_edges(numpy_to_dmatrix(points)));
    }

    static py::tuple get_gabriel_edges_subset_wrapper(const numpy::ndarray &points, int size, int seed) {
        ftype ne;
        auto result = edge_matrix_to_numpy(get_gabriel_edges_subset(
                numpy_to_dmatrix(points), size, seed, &ne
                ));
        return py::make_tuple(result, scalar_to_numpy(ne));
    }
};


#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-unused-raii"
BOOST_PYTHON_MODULE(vgt) {
    numpy::initialize();

    py::def("set_silent", &set_silent);

    py::class_<CellKernel, boost::noncopyable>("CellKernel", py::no_init)
            .def("latex", &CellKernel::latex);
    py::class_<RadialCellKernel, py::bases<CellKernel>, boost::noncopyable>("RadialCellKernel", py::no_init)
            .def("__call__", &RadialCellKernel::kernel_value)
            .def("value", &RadialCellKernel::kernel_value)
            .def("derivative_value", &RadialCellKernel::derivative_value);
    py::class_<UniformCellKernel, py::bases<RadialCellKernel>>("UniformCellKernel", py::init<int>());
    py::class_<GaussianCellKernel, py::bases<RadialCellKernel>>("GaussianCellKernel", py::init<int, ftype>());
    py::class_<LaplaceCellKernel, py::bases<RadialCellKernel>>("LaplaceCellKernel", py::init<int, ftype>());
    py::class_<PolynomialCellKernel, py::bases<RadialCellKernel>>("PolynomialCellKernel", py::init<int, ftype, ftype>());
    py::class_<AdaptiveGaussianCellKernel, py::bases<RadialCellKernel>>("AdaptiveGaussianCellKernel", py::no_init)
            .def("__init__", &WrapperFuncs::AdaptiveGaussianCellKernel_constructor_wrapper)
            .def("__init__", &WrapperFuncs::AdaptiveGaussianCellKernel_constructor2_wrapper)
            .def("update_local_bandwidths", &WrapperFuncs::AdaptiveGaussianCellKernel_update_local_bandwidths);

    py::class_<FixedVolumeGaussianCellKernel, py::bases<RadialCellKernel>>(
            "FixedVolumeGaussianCellKernel", py::init<int, ftype>());


    py::class_<BalancedCellKernel, py::bases<CellKernel>, boost::noncopyable>("BalancedCellKernel", py::no_init)
            .def("__call__", &BalancedCellKernel::kernel_value)
            .def("value", &BalancedCellKernel::kernel_value)
            .def("compute_beta", &BalancedCellKernel::compute_beta)
            .def("cone", &WrapperFuncs::BalancedCellKernel_cone);

    py::class_<BalancedExponentialCellKernel, py::bases<BalancedCellKernel>>(
            "BalancedExponentialCellKernel", py::init<int, ftype, ftype>());
    py::class_<BalancedPolynomialCellKernel, py::bases<BalancedCellKernel>>(
            "BalancedPolynomialCellKernel", py::init<int, ftype, ftype>());
    py::def("polynomial_integral", &WrapperFuncs::_F_wrapper);
    py::class_<BalancedSecondPolynomialCellKernel, py::bases<BalancedCellKernel>>(
            "BalancedSecondPolynomialCellKernel", py::init<int, ftype>());
    py::class_<BalancedSigmoidalGaussian, py::bases<BalancedCellKernel>>(
            "BalancedSigmoidalGaussian", py::init<int, ftype>());
    py::class_<BalancedLinear, py::bases<BalancedCellKernel>>(
            "BalancedLinear", py::init<int, ftype>());
    py::class_<PuncturedConstantCellKernel, py::bases<BalancedCellKernel>>(
            "PuncturedConstantCellKernel", py::init<int, ftype>());
    py::class_<PuncturedExponentialCellKernel, py::bases<BalancedCellKernel>>(
            "PuncturedExponentialCellKernel", py::init<int, ftype, ftype, ftype>());

    py::def("gaussian_bw_to_alpha", &WrapperFuncs::gaussian_bw_to_alpha_wrapper);

    py::enum_<RayStrategyType>("RayStrategyType")
            .value("BRUTE_FORCE", BRUTE_FORCE)
            .value("BIN_SEARCH", BIN_SEARCH)
            .value("BRUTE_FORCE_GPU", BRUTE_FORCE_GPU)
            ;

    py::class_<Bounds, boost::noncopyable>("Bounds", py::no_init)
            .def("contains", &WrapperFuncs::Bounds_contains)
            .def("max_length", &WrapperFuncs::Bounds_max_length);
    py::class_<Unbounded, py::bases<Bounds>>("Unbounded", py::init<>());
    py::class_<BoundingBox, py::bases<Bounds>>("BoundingBox", py::init<int, ftype>())
            .def("__init__", &WrapperFuncs::BoundingBox_constructor_wrapper)
            .add_property("lower", &WrapperFuncs::BoundingBox_lower_getter)
            .add_property("upper", &WrapperFuncs::BoundingBox_upper_getter);
    py::class_<BoundingSphere, py::bases<Bounds>>("BoundingSphere", py::init<int, ftype>())
            .add_property("center", &WrapperFuncs::BoundingSphere_radius_getter)
            .add_property("radius", &BoundingSphere::get_radius);

    py::class_<VoronoiDensityEstimator/*, py::bases<AbstractDensityEstimator>*/>("VoronoiDensityEstimator", py::no_init)
            .def("__init__", &WrapperFuncs::VoronoiDensityEstimator_constructor_wrapper,
                 (py::arg("points"), py::arg("cell_kernel"), py::arg("seed"), py::arg("njobs"),
                         py::arg("nrays_si"), py::arg("nrays_hr"),
                         py::arg("strategy"), py::arg("bounds")),
                 py::return_internal_reference<2, py::return_internal_reference<3, py::return_internal_reference<9>>>())
            .def("initialize_volumes", &VoronoiDensityEstimator::initialize_volumes)
            .def("dvol_dp", &WrapperFuncs::VoronoiDensityEstimator_dvol_dp)
            .def("dlogf_dp", &WrapperFuncs::VoronoiDensityEstimator_dlogf_dp)
            .def("initialize_volumes_uncentered", &WrapperFuncs::VoronoiDensityEstimator_initialize_volumes_uncentered)
            .def("estimate", &WrapperFuncs::VoronoiDensityEstimator_estimate)
            .def("centroid_smoothing", &VoronoiDensityEstimator::centroid_smoothing)
            .def("get_points", &WrapperFuncs::VoronoiDensityEstimator_get_points)
            .def("sample", &WrapperFuncs::VoronoiDensityEstimator_sample)
//            .def("set_max_block_size", &VoronoiDensityEstimator::set_max_block_size)
            .def("sample_masked", &WrapperFuncs::VoronoiDensityEstimator_sample_masked)
            .def("get_volumes", &WrapperFuncs::VoronoiDensityEstimator_get_volumes)
            .def("get_initial_seed", &VoronoiDensityEstimator::get_initial_seed)
            .add_property("weights", &WrapperFuncs::VoronoiDensityEstimator_weights_getter,
                          &WrapperFuncs::VoronoiDensityEstimator_weights_setter)
            .add_property("lengths_debug", &WrapperFuncs::VoronoiDensityEstimator_lengths_debug_getter);

    py::class_<KDE>("KDE", py::no_init)
            .def("__init__", &WrapperFuncs::KDE_constructor_wrapper,
                 (py::arg("points"), py::arg("global_bw"), py::arg("seed"), py::arg("njobs")),
                 py::return_internal_reference<2>())
            .def("__init__", &WrapperFuncs::KDE_constructor2_wrapper,
                 (py::arg("points"), py::arg("cell_kernel"), py::arg("seed"), py::arg("njobs")),
                 py::return_internal_reference<2, py::return_internal_reference<3>>())
            .def("estimate", &WrapperFuncs::KDE_estimate)
            .def("get_points", &WrapperFuncs::KDE_get_points)
            .def("sample", &WrapperFuncs::KDE_sample)
            .add_property("weights", &WrapperFuncs::KDE_weights_getter, &WrapperFuncs::KDE_weights_setter)
            .def("make_adaptive", &KDE::make_adaptive)
            ;

    py::def("get_gabriel_edges", &WrapperFuncs::get_gabriel_edges_wrapper);
    py::def("get_gabriel_edges_subset", &WrapperFuncs::get_gabriel_edges_subset_wrapper);
}
#pragma clang diagnostic pop
