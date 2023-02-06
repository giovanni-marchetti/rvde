#include "KDE.h"

KDE::KDE(const dmatrix &points, ftype global_bw, int seed, int njobs) :
        points(points), global_bw(global_bw), re(seed), njobs(njobs),
        local_bw(points.cols(), 1), data_n(points.cols()), dim(points.rows()),
        cell_kernel(nullptr) {
    set_weights(vec<ftype>(data_n, 1));
}

KDE::KDE(const dmatrix &points, const ptr<RadialCellKernel> &cell_kernel, int seed, int njobs) :
        points(points), global_bw(cell_kernel->get_sigma()), re(seed), njobs(njobs),
        local_bw(points.cols(), 1), data_n(points.cols()), dim(points.rows()),
        cell_kernel(cell_kernel) {
    set_weights(vec<ftype>(data_n, 1));
}

vec<ftype> KDE::estimate(const dmatrix &queries) const {
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    int q_n = queries.cols();
    vec<ftype> result(q_n, 0);
    ftype global_normalization = cell_kernel->general_cone_integral(global_bw, INF_ftype);

    my_tqdm bar(q_n);
    #pragma omp parallel for
    for (int q_i = 0; q_i < q_n; q_i++) {
        bar.atomic_iteration();
        dvector q = queries.col(q_i);
        for (int j = 0; j < data_n; j++) {
            ftype local_normalization = global_normalization;
            ftype bw = global_bw * local_bw[j];
            if (local_bw[j] != 1) {
                local_normalization = cell_kernel->general_cone_integral(bw, INF_ftype);
            }
            if (cell_kernel) {
                ftype dist = (q - points.col(j)).norm();
                result[q_i] += weights[j] * cell_kernel->base_kernel_value(bw, dist) / local_normalization;
//                std::cout << cell_kernel->base_kernel_value(bw, dist) << std::endl;
            } else { // gaussian
                ftype dist_squared = (q - points.col(j)).squaredNorm();
                ftype norm = math::pow(2 * PI_ftype * sqr(bw), -ftype(dim) / 2);
                result[q_i] += weights[j] * norm * math::exp(-ftype(0.5) * dist_squared / sqr(bw));
            }
        }
//        std::cout << result[q_i] << std::endl;
    }
    bar.bar().finish();

    omp_set_num_threads(old_threads);

    return result;
}

dmatrix KDE::sample(int sample_size) const {
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    dmatrix result(dim, sample_size);

    #pragma omp parallel
    re.fix_random_engines();

    #pragma omp parallel for
    for (int i = 0; i < sample_size; i++) {
        RandomEngine &engine = re.current();
        dvector q(dim, 1);

        int point_index = point_distribution(engine.generator());
        if (cell_kernel) {
            throw std::runtime_error("Sampling from a kernel within KDE is not supported yet");
        } else { // gaussian
            ftype bw = global_bw * local_bw[point_index];

            for (int j = 0; j < dim; j++) {
                q(j, 0) = engine.rand_normal();
            }
            q = points.col(point_index) + q * bw;
            result.col(i) = q;
        }
    }

    omp_set_num_threads(old_threads);
    return result;
}

const dmatrix &KDE::get_points() const {
    return points;
}

void KDE::set_weights(const vec<ftype> &weights) {
    ensure(weights.size() == data_n, "The number of weights should be equal to the number of generators");
    this->weights = weights;
    ftype sum = 0;
    for (int i = 0; i < data_n; i++) {
        sum += this->weights[i];
    }
    for (int i = 0; i < data_n; i++) {
        this->weights[i] /= sum;
    }

    point_distribution = std::discrete_distribution<int>(this->weights.begin(), this->weights.end());
}

const vec<ftype> &KDE::get_weights() {
    return weights;
}

void KDE::make_adaptive(ftype alpha) {
//    ensure(!cell_kernel, "Cannot use adaptive with an arbitrary cell kernel");
//    ensure(alpha >= 0 && alpha <= 1, "alpha should be in range [0; 1]");
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    std::cout << "Making adaptive:" << std::endl;
    my_tqdm bar(data_n);

    vec<ftype> base_values(data_n, 0);
    ftype global_normalization = cell_kernel->general_cone_integral(global_bw, INF_ftype);
    #pragma omp parallel for
    for (int i = 0; i < data_n; i++) {
        bar.atomic_iteration();
        dvector q = points.col(i);
        for (int j = 0; j < data_n; j++) {
            if (cell_kernel) {
                ftype dist = (q - points.col(j)).norm();
                base_values[i] += weights[j] * cell_kernel->base_kernel_value(global_bw, dist) / global_normalization;
            } else { // gaussian
                ftype bw = global_bw;
                ftype dist_squared = (q - points.col(j)).squaredNorm();
                ftype norm = math::pow(2 * PI_ftype * math::pow(bw, dim), ftype(-0.5));
                base_values[i] += weights[j] * norm * math::exp(-dist_squared / sqr(bw));
            }
        }
    }
    bar.bar().finish();

    ftype g = 0;
    for (int i = 0; i < data_n; i++) {
        g += math::log(base_values[i]);
    }
    g = math::exp(g / data_n);
    for (int i = 0; i < data_n; i++) {
        local_bw[i] = math::pow(base_values[i] / g, -alpha);
    }
    is_adaptive = true;

    omp_set_num_threads(old_threads);
}
