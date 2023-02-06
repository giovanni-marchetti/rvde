#include <tqdm.h>
#include <memory>
#include <chrono>

#include "VoronoiDensityEstimator.h"

VoronoiDensityEstimator::VoronoiDensityEstimator(const dmatrix &points, const ptr<CellKernel> &cell_kernel, int seed,
                                                 int njobs, int nrays_weights, int nrays_sampling,
                                                 RayStrategyType strategy,
                                                 const ptr<Bounds> &bounds)
        : points(points), initial_seed(seed), re(seed), njobs(njobs), cell_kernel(cell_kernel),
          nrays_si(nrays_weights), nrays_hr(nrays_sampling), strategy(strategy),
          volumes(points.cols(), 0), is_initialized(points.cols(), false),
          data_n(points.cols()), dim(points.rows()) {
    set_weights(vec<ftype>(data_n, 1));
    if (strategy == BRUTE_FORCE_GPU) {
        geometry_kernel = std::make_shared<EuclideanKernelGPU>(this->points, bounds);
    } else {
        geometry_kernel = std::make_shared<EuclideanKernel>(this->points, bounds);
    }

    if (BalancedCellKernel* ck = dynamic_cast<BalancedCellKernel*>(cell_kernel.get())) {
        if (nrays_sampling != 1) {
            std::cout << "W: Using 1 \"hit&run\" step instead of " << nrays_sampling << " for sampling from a balanced cell kernel" << std::endl;
            nrays_hr = 1;
        }
    }

    if(AdaptiveGaussianCellKernel* ck = dynamic_cast<AdaptiveGaussianCellKernel*>(cell_kernel.get())) {
        if (!ck->is_initialized()) {
            std::cout << "Initializing the adaptive kernel...";
            std::cout.flush();

            vec<ftype> local_bw(data_n);

            #pragma omp parallel
            re.fix_random_engines();
            #pragma omp parallel for
            for (int i = 0; i < data_n; i++) {
                geometry_kernel->nearest_point_extra(points.col(i), -1, {i}, &local_bw[i]);
            }
            ck->update_local_bandwidths(local_bw);

            std::cout << " done" << std::endl;
        } else {
            std::cout << "Adaptive kernel was already initialized" << std::endl;
        }
    }
}

vec<ftype> VoronoiDensityEstimator::estimate(const dmatrix &queries) const {
    int n = queries.cols();
    vec<ftype> results(n, 0);

    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    if (strategy == BRUTE_FORCE_GPU && as<BalancedCellKernel>(cell_kernel)) {
        auto *ck = as<BalancedCellKernel>(cell_kernel);
        // obtain nearest neighbors
        std::cout << "Estimating: finding nearest neighbors" << std::endl;
        my_tqdm bar(n);
        dmatrix ref_mat(dim, n);
        vec<int> j0_list(n);
        dmatrix u_mat(dim, n);
        vec<int> u_indices(n);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            bar.atomic_iteration();
            dvector q = queries.col(i);
            int nearest = geometry_kernel->nearest_point(queries.col(i));
            ref_mat.col(i) = points.col(nearest);
            j0_list[i] = nearest;
            u_mat.col(i) = (q - ref_mat.col(i)).normalized();
            u_indices[i] = i;
        }
        bar.bar().finish();

        std::cout << "Estimating: finding ray intersections" << std::endl;
        EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());
        kernel->reset_reference_points_gpu(ref_mat, j0_list);
        kernel->reset_rays_gpu(u_mat);

        vec<int> best_j_pos(n);
        vec<int> best_j_neg(n);
        kernel->intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);

        std::cout << "Estimating: computing values" << std::endl;
        bar = my_tqdm(n);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            bar.atomic_iteration();
            int nearest = j0_list[i];
            ftype length;
            geometry_kernel->intersect_ray_precomputed(
                    ref_mat.col(i), u_mat.col(i), j0_list[i], best_j_pos[i], &length);
            ftype ck_val = ck->kernel_value(j0_list[i], (queries.col(i) - ref_mat.col(i)).norm(), length);
            results[i] = ck_val / volumes[nearest] * weights[nearest];
        }
        bar.bar().finish();
    } else {
        my_tqdm bar(n);
        #pragma omp parallel
        {
            re.fix_random_engines();
            #pragma omp for
            for (int i = 0; i < n; i++) {
                bar.atomic_iteration();
                results[i] = estimate_single(static_cast<dvector>(queries.col(i)));
            }
        }
        bar.bar().finish();
    }

    omp_set_num_threads(old_threads);

    return results;
}

const vec<ftype> &VoronoiDensityEstimator::get_volumes() const {
    return volumes;
}

const dmatrix &VoronoiDensityEstimator::get_points() const {
    return points;
}


void VoronoiDensityEstimator::initialize_volumes() {
    if (auto *ck = as<BalancedCellKernel>(cell_kernel)) {
        ftype vol = ck->cone_integral(0, 1);
        ray_sample_sums = vec<ftype>(data_n, vol);
        volumes = vec<ftype>(data_n, vol);
        ray_sample_counts = vec<int>(data_n, 1);
        is_initialized = vec<bool>(data_n, true);
        return;
    }
    int max_block_size = -1;
    if (strategy == BRUTE_FORCE_GPU) {
        EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());
        max_block_size = kernel->estimate_max_block_size();
        if (nrays_si < max_block_size) {
            max_block_size = kernel->estimate_max_block_size(nrays_si);
        }
        std::cout << "Maximum block size estimated as " << max_block_size << std::endl;
    }

//    ftype norm_const = cell_kernel->normalization_constant();

    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    if (ray_sample_sums.empty()) {
        ray_sample_sums = vec<ftype>(data_n, 0);
        ray_sample_counts = vec<int>(data_n, 0);
    }
    is_initialized = vec<bool>(data_n, false);

    vec<vec<ftype>> lengths(data_n, vec<ftype>(nrays_si, -1));

    int u_block_n = max_block_size < 0 ? 1 : (nrays_si + max_block_size - 1) / max_block_size;
    int ref_block_n = max_block_size < 0 ? 1 : (data_n + max_block_size - 1) / max_block_size;
    int block_size = max_block_size < 0 ? math::max(data_n, nrays_si) : max_block_size;

    for (int u_block_i = 0; u_block_i < u_block_n; u_block_i++) {
        int u_block_start = u_block_i * block_size;
        int u_block_end = math::min(u_block_start + block_size, nrays_si);
        // generate directions
        dmatrix u_mat = dmatrix(dim, u_block_end - u_block_start);
        for (int i = 0; i < u_mat.cols(); i++) {
            u_mat.col(i) = re.current().rand_on_sphere(dim);
        }

        for (int ref_block_i = 0; ref_block_i < ref_block_n; ref_block_i++) {
            std::cout << " * Block (" << u_block_i + 1 << ", " << ref_block_i + 1
                      << ") out of (" << u_block_n << ", " << ref_block_n << ") ..." << std::endl;
            int ref_block_start = ref_block_i * block_size;
            int ref_block_end = math::min(ref_block_start + block_size, data_n);
//            dmatrix ref_mat = points(Eigen::all, Eigen::seq(ref_block_start, ref_block_end - 1));
            dmatrix ref_mat = points.block(0, ref_block_start, points.rows(), ref_block_end - ref_block_start);

            // precalculations
            switch (strategy) {
                case RayStrategyType::BRUTE_FORCE:
                    std::cout << "Running CPU precalculations" << std::endl;
                    geometry_kernel->precompute(ref_mat, u_mat);
                    break;
                case RayStrategyType::BRUTE_FORCE_GPU: {
                    geometry_kernel->reset_ref_mat(ref_mat.cols());
                    geometry_kernel->reset_u_mat(u_mat.cols());
                    std::cout << "Running GPU precalculations" << std::endl;
                    EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                    vec<int> j0_list(ref_mat.cols());
                    for (int i = 0; i < j0_list.size(); i++) {
                        j0_list[i] = ref_block_start + i;
                    }
                    kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
                    kernel_gpu->reset_rays_gpu(u_mat);
                    break;
                }
                case RayStrategyType::BIN_SEARCH:
                    break;
                default:
                    throw std::runtime_error("Unknown strategy");
            }

            bool distances_precomputed = false;
            vec<int> best_j_pos((ref_block_end - ref_block_start) * (u_block_end - u_block_start));
            vec<int> best_j_neg((ref_block_end - ref_block_start) * (u_block_end - u_block_start));

            if (strategy == BRUTE_FORCE_GPU) {
                std::cout << "Running raycasting on GPU" << std::endl;
                EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                kernel_gpu->intersect_ray_bruteforce_gpu(&best_j_pos, &best_j_neg);
                distances_precomputed = true;
                std::cout << "Computing integrals" << std::endl;
            }

            my_tqdm bar(ref_block_end - ref_block_start);

            #pragma omp parallel
            {
                #pragma omp master
                {
                    if (u_block_i == 0 && ref_block_i == 0) {
                        std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
                    }
                }
                re.fix_random_engines();

                #pragma omp for
                for (int ref_i = ref_block_start; ref_i < ref_block_end; ref_i++) {
                    bar.atomic_iteration();
                    if (!is_initialized[ref_i]) {
                        dvector ref = dvector::col(points, ref_i);
                        ref.index -= ref_block_start;   // needed for precomputations
                        for (int j = 0; j < u_mat.cols(); j++) {
                            const dvector &u = dvector::col(u_mat, j);
                            int tmp;
                            ftype length = INF_ftype;
                            if (!distances_precomputed) {
                                geometry_kernel->intersect_ray(strategy, ref, u, ref_i, {ref_i}, &tmp, &length,
                                                               Kernel::IntersectionSearchType::RAY_INTERSECTION);
                            } else {
                                geometry_kernel->intersect_ray_precomputed(
                                        ref, u, ref_i, best_j_pos[(ref_i - ref_block_start) * u_mat.cols() + j], &length);
                            }

                            lengths[ref_i][u_block_start + j] = length;
//                            if (length < 0)
                        }
                    }
                }
            }
            bar.bar().finish();
            // ## end of ref loop ##
        }

        // ## end of u loop ##
    }

    lengths_debug = lengths;

    if (FixedVolumeCellKernel *ck = dynamic_cast<FixedVolumeCellKernel*>(cell_kernel.get())) {
        ck->compute_local_bandwidths(lengths);
    }

    #pragma omp parallel for
    for (int ref_i = 0; ref_i < data_n; ref_i++) {
        for (int j = 0; j < nrays_si; j++) {
            ftype length = lengths[ref_i][j];
            if (length < 0) length = 0;
            ensure(length >= 0, "Negative length: " + std::to_string(length));
            ftype vol = cell_kernel->cone_integral(ref_i, length);
            if (math::isinf(vol)) {
                throw std::runtime_error("Integral diverges for ref_i=" + std::to_string(ref_i) +
                " and length=" + std::to_string(length) + ". Perhaps your space is not compactified?");
            }
            ray_sample_sums[ref_i] += vol;
            ray_sample_counts[ref_i]++;
        }
    }

    for (int i = 0; i < data_n; i++) {
        volumes[i] = ray_sample_sums[i] / ray_sample_counts[i];
        is_initialized[i] = true;
    }

    omp_set_num_threads(old_threads);
}


vec<dmatrix> VoronoiDensityEstimator::dvol_dp() {  // also updates volumes
    if (as<BalancedCellKernel>(cell_kernel)) {
        return vec<dmatrix>(data_n, dmatrix::Zero(dim, data_n));
    }
    RadialCellKernel *ck = as<RadialCellKernel>(cell_kernel);
    int max_block_size = -1;
    if (strategy == BRUTE_FORCE_GPU) {
        EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());
        max_block_size = kernel->estimate_max_block_size();
        if (nrays_si < max_block_size) {
            max_block_size = kernel->estimate_max_block_size(nrays_si);
        }
        std::cout << "Maximum block size estimated as " << max_block_size << std::endl;
    }

//    ftype norm_const = cell_kernel->normalization_constant();

    vec<dmatrix> interior_sums;
    vec<vec<ftype>> interior_counts;
    vec<dmatrix> boundary_sums;
    vec<vec<ftype>> boundary_counts;
    for (int i = 0; i < data_n; i++) {
        interior_sums.push_back(dmatrix::Zero(dim, data_n));
        interior_counts.push_back(vec<ftype>(data_n, 0));
        boundary_sums.push_back(dmatrix::Zero(dim, data_n));
        boundary_counts.push_back(vec<ftype>(data_n, 0));
    }

    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    if (ray_sample_sums.empty()) {
        ray_sample_sums = vec<ftype>(data_n, 0);
        ray_sample_counts = vec<int>(data_n, 0);
    }

    int u_block_n = max_block_size < 0 ? 1 : (nrays_si + max_block_size - 1) / max_block_size;
    int ref_block_n = max_block_size < 0 ? 1 : (data_n + max_block_size - 1) / max_block_size;
    int block_size = max_block_size < 0 ? math::max(data_n, nrays_si) : max_block_size;

    for (int u_block_i = 0; u_block_i < u_block_n; u_block_i++) {
        int u_block_start = u_block_i * block_size;
        int u_block_end = math::min(u_block_start + block_size, nrays_si);
        // generate directions
        dmatrix u_mat = dmatrix(dim, u_block_end - u_block_start);
        for (int i = 0; i < u_mat.cols(); i++) {
            u_mat.col(i) = re.current().rand_on_sphere(dim);
        }

        for (int ref_block_i = 0; ref_block_i < ref_block_n; ref_block_i++) {
            std::cout << " * Block (" << u_block_i + 1 << ", " << ref_block_i + 1
                      << ") out of (" << u_block_n << ", " << ref_block_n << ") ..." << std::endl;
            int ref_block_start = ref_block_i * block_size;
            int ref_block_end = math::min(ref_block_start + block_size, data_n);
//            dmatrix ref_mat = points(Eigen::all, Eigen::seq(ref_block_start, ref_block_end - 1));
            dmatrix ref_mat = points.block(0, ref_block_start, points.rows(), ref_block_end - ref_block_start);

            // precalculations
            switch (strategy) {
                case RayStrategyType::BRUTE_FORCE:
                    std::cout << "Running CPU precalculations" << std::endl;
                    geometry_kernel->precompute(ref_mat, u_mat);
                    break;
                case RayStrategyType::BRUTE_FORCE_GPU: {
                    geometry_kernel->reset_ref_mat(ref_mat.cols());
                    geometry_kernel->reset_u_mat(u_mat.cols());
                    std::cout << "Running GPU precalculations" << std::endl;
                    EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                    vec<int> j0_list(ref_mat.cols());
                    for (int i = 0; i < j0_list.size(); i++) {
                        j0_list[i] = ref_block_start + i;
                    }
                    kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
                    kernel_gpu->reset_rays_gpu(u_mat);
                    break;
                }
                case RayStrategyType::BIN_SEARCH:
                    break;
                default:
                    throw std::runtime_error("Unknown strategy");
            }

            bool distances_precomputed = false;
            vec<int> best_j_pos((ref_block_end - ref_block_start) * (u_block_end - u_block_start));
            vec<int> best_j_neg((ref_block_end - ref_block_start) * (u_block_end - u_block_start));

            if (strategy == BRUTE_FORCE_GPU) {
                std::cout << "Running raycasting on GPU" << std::endl;
                EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                kernel_gpu->intersect_ray_bruteforce_gpu(&best_j_pos, &best_j_neg);
                distances_precomputed = true;
                std::cout << "Computing integrals" << std::endl;
            }

            my_tqdm bar(ref_block_end - ref_block_start);

            #pragma omp parallel
            {
                #pragma omp master
                {
                    if (u_block_i == 0 && ref_block_i == 0) {
                        std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
                    }
                }
                re.fix_random_engines();

                #pragma omp for
                for (int ref_i = ref_block_start; ref_i < ref_block_end; ref_i++) {
                    bar.atomic_iteration();
//                    if (!is_initialized[ref_i]) {
                        dvector ref = dvector::col(points, ref_i);
                        ref.index -= ref_block_start;   // needed for precomputations
                        for (int j = 0; j < u_mat.cols(); j++) {
                            const dvector &u = dvector::col(u_mat, j);
                            int other_i = -1;
                            ftype length = INF_ftype;
                            if (!distances_precomputed) {
                                geometry_kernel->intersect_ray(strategy, ref, u, ref_i, {ref_i}, &other_i, &length,
                                                               Kernel::IntersectionSearchType::RAY_INTERSECTION);
                            } else {
                                other_i = best_j_pos[(ref_i - ref_block_start) * u_mat.cols() + j];
                                geometry_kernel->intersect_ray_precomputed(
                                        ref, u, ref_i, other_i, &length);
                            }

//                            std::cout << ref_i << " " << length << std::endl;
                            ftype vol = ck->cone_integral(ref_i, length);
                            if (math::isinf(vol)) {
                                throw std::runtime_error("Integral diverges. Perhaps your space is not compactified?");
                            }
                            ray_sample_sums[ref_i] += vol;
                            ray_sample_counts[ref_i]++;

                            // UPDATE TO THE DERIVATIVE
                            ftype interior_value = ck->cone_integral_derivative(ref_i, length);
                            interior_sums[ref_i].col(ref_i) += u * interior_value;
                            interior_counts[ref_i][ref_i] += 1;

                            if (math::isfinite(length) && other_i >= 0) {
                                ftype scalar_product = u.dot((points.col(other_i) - points.col(ref_i)).normalized());
                                ftype ck_val = ck->kernel_value(ref_i, length);
                                ftype boundary_value = nsphere_volume(dim - 1) * ck_val
                                                       * math::pow(length, dim - 1) / scalar_product;
                                ftype distance = (points.col(other_i) - points.col(ref_i)).norm();
                                dvector y = ref + u * length;
                                boundary_sums[ref_i].col(ref_i) += (y - ref) / distance * boundary_value;

                                boundary_counts[ref_i][ref_i] += 1;
                                boundary_sums[ref_i].col(other_i) += (points.col(other_i) - y) / distance * boundary_value;
                                boundary_counts[ref_i][other_i] += 1;
                            }
                        }
//                    }
                }
            }
            bar.bar().finish();
            // ## end of ref loop ##
        }

        // ## end of u loop ##
    }

    for (int i = 0; i < data_n; i++) {
//        volumes[i] = ray_sample_counts[i] / (data_n * ray_sample_sums[i]);
        volumes[i] = ray_sample_sums[i] / ray_sample_counts[i];
        is_initialized[i] = true;
    }

    vec<dmatrix> jacobian(data_n, dmatrix::Zero(dim, data_n));
    #pragma omp parallel for
    for (int i = 0; i < data_n; i++) {
        for (int j = 0; j < data_n; j++) {
            jacobian[i].col(j) =
                    interior_sums[i].col(j) / nrays_si
                    + boundary_sums[i].col(j) / nrays_si;
        }
    }

    omp_set_num_threads(old_threads);

    return jacobian;
}

dvector ddist_dp(const dvector &p, const dvector &x) {
    return (p - x).normalized();
}

dmatrix VoronoiDensityEstimator::dlogf_dp(const dmatrix &queries) {
    int q_n = queries.cols();
    vec<int> nearest(q_n, -1);
    #pragma omp parallel for
    for (int i = 0; i < q_n; i++) {
        nearest[i] = geometry_kernel->nearest_point(queries.col(i));
    }

    dmatrix dlogf_dp_v = dmatrix::Zero(dim, data_n);
    if (auto *ck = as<RadialCellKernel>(cell_kernel)) {
        {
            if (dynamic_cast<AdaptiveGaussianCellKernel*>(cell_kernel.get())) {
                throw std::runtime_error("Derivatives for AdaptiveGaussianCellKernel not implemented");
            }
            // kernel derivative
            vec<int> indices(q_n);
            vec<dvector> additives(q_n);
            #pragma omp parallel for
            for (int i = 0; i < q_n; i++) {
                int j = nearest[i];
                ftype length = (points.col(j) - queries.col(i)).norm();
                indices[i] = j;
                additives[i] =
                        ck->derivative_value(j, length) / ck->kernel_value(j, length)
                        * ddist_dp(points.col(j), queries.col(i));
            }
            for (int i = 0; i < q_n; i++) {
                dlogf_dp_v.col(indices[i]) += additives[i];
            }
        }

        {
            // geometric derivative
            vec<dmatrix> dvol_dp_v = dvol_dp();
            vec<ftype> vol_v = get_volumes();
            vec<int> num_points(data_n, 0);
            for (int i = 0; i < q_n; i++) {
                num_points[nearest[i]]++;
            }
            #pragma omp parallel for
            for (int j = 0; j < data_n; j++) {
                for (int i = 0; i < data_n; i++) {
                    dlogf_dp_v.col(j) -=
                            num_points[i] * dvol_dp_v[i].col(j) / vol_v[i];
                }
            }
        }
    } else if (auto *ck = as<SmoothBalancedCellKernel>(cell_kernel)) {
        // FIRST obtain lengths
        vec<int> second_nearest(q_n, -1);
        vec<ftype> lengths(q_n, 0);
        if (strategy == BRUTE_FORCE_GPU) {
            // obtain nearest neighbors
            std::cout << "Estimating: finding nearest neighbors" << std::endl;
            dmatrix ref_mat(dim, q_n);
            vec<int> j0_list(q_n);
            dmatrix u_mat(dim, q_n);
            vec<int> u_indices(q_n);
//        #pragma omp parallel for
            for (int i = 0; i < q_n; i++) {
                dvector q = queries.col(i);
                ref_mat.col(i) = points.col(nearest[i]);
                j0_list[i] = nearest[i];
                u_mat.col(i) = (q - ref_mat.col(i)).normalized();
                u_indices[i] = i;
            }

            std::cout << "Estimating: finding ray intersections" << std::endl;
            EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());
            kernel->reset_reference_points_gpu(ref_mat, j0_list);
            kernel->reset_rays_gpu(u_mat);

            vec<int> best_j_neg(q_n);
            kernel->intersect_ray_indexed_gpu(u_indices, &second_nearest, &best_j_neg);

            std::cout << "Estimating: computing values" << std::endl;
            my_tqdm bar(q_n);
            #pragma omp parallel for
            for (int i = 0; i < q_n; i++) {
                bar.atomic_iteration();
                // this is needed to potentially update best_j_pos in a boundary case
                geometry_kernel->intersect_ray_precomputed(
                        ref_mat.col(i), u_mat.col(i),
                        j0_list[i], second_nearest[i], &lengths[i]);
            }
            bar.bar().finish();
        } else {
            #pragma omp parallel for
            for (int i = 0; i < q_n; i++) {
                dvector q = queries.col(i);
                int j0 = nearest[i];
                dvector u = (q - points.col(j0)).normalized();
                geometry_kernel->intersect_ray(
                        strategy, q, u, j0, {j0}, &second_nearest[i],
                        &lengths[i], Kernel::RAY_INTERSECTION);
            }
        }

        // Actual derivatives
        vec<dvector> additives_p(q_n, dvector::Zero(dim, 1));
        vec<dvector> additives_q(q_n, dvector::Zero(dim, 1));
        #pragma omp parallel for
        for (int i = 0; i < q_n; i++) {
            int j = nearest[i];
            dvector p = points.col(j);
            dvector x = queries.col(i);
            ftype beta = ck->compute_beta(j, lengths[i]);
            ftype radius = (p - x).norm();
            ftype length = lengths[i];

            additives_p[i] += ck->d_dradius(beta, radius) * ddist_dp(p, x);

            if (second_nearest[i] >= 0) {
                dvector q = points.col(second_nearest[i]);
                dvector a = p - q;
                dvector b = p - x;

                ftype ata = a.dot(a);
                ftype atb = a.dot(b);
                ftype btb = b.dot(b);
                ftype _b_ = b.norm();

                dvector dlength_dp =
                        (a * (2 * atb * btb - ata * btb) + b * (ata * atb - ata * btb))
                                                    /
                                          (2 * atb * atb * _b_);
                dvector dlength_dq =
                        (a * (-2 * atb * _b_) + b * (ata * _b_))
                                            /
                                    (2 * atb * atb);

                ftype dK_dlength = ck->d_dbeta(beta, radius) * ck->dbeta_dlength(beta, length);
                additives_p[i] += dK_dlength * dlength_dp;
                additives_q[i] += dK_dlength * dlength_dq;
            }
        }
        for (int i = 0; i < q_n; i++) {
            dlogf_dp_v.col(nearest[i]) += additives_p[i];
            if (second_nearest[i] >= 0) {
                dlogf_dp_v.col(second_nearest[i]) += additives_q[i];
            }
        }
    } else {
        throw std::runtime_error("the cell kernel does not support derivatives");
    }

    return dlogf_dp_v;
}

void VoronoiDensityEstimator::initialize_weights_uncentered(const dmatrix &centroids) {
    int max_block_size = -1;
    if (strategy == BRUTE_FORCE_GPU) {
        EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());
        max_block_size = kernel->estimate_max_block_size();
        if (nrays_si < max_block_size) {
            max_block_size = kernel->estimate_max_block_size(nrays_si);
        }
        std::cout << "Maximum block size estimated as " << max_block_size << std::endl;
    }

    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    if (ray_sample_sums.empty()) {
        ray_sample_sums = vec<ftype>(data_n, 0);
        ray_sample_counts = vec<int>(data_n, 0);
    }
    is_initialized = vec<bool>(data_n, false);

    int u_block_n = max_block_size < 0 ? 1 : (nrays_si + max_block_size - 1) / max_block_size;
    int ref_block_n = max_block_size < 0 ? 1 : (data_n + max_block_size - 1) / max_block_size;
    int block_size = max_block_size < 0 ? math::max(data_n, nrays_si) : max_block_size;

    for (int u_block_i = 0; u_block_i < u_block_n; u_block_i++) {
        int u_block_start = u_block_i * block_size;
        int u_block_end = math::min(u_block_start + block_size, nrays_si);
        // generate directions
        dmatrix u_mat = dmatrix(dim, u_block_end - u_block_start);
        for (int i = 0; i < u_mat.cols(); i++) {
            u_mat.col(i) = re.current().rand_on_sphere(dim);
        }

        for (int ref_block_i = 0; ref_block_i < ref_block_n; ref_block_i++) {
            std::cout << " * Block (" << u_block_i + 1 << ", " << ref_block_i + 1
                      << ") out of (" << u_block_n << ", " << ref_block_n << ") ..." << std::endl;
            int ref_block_start = ref_block_i * block_size;
            int ref_block_end = math::min(ref_block_start + block_size, data_n);
//            dmatrix ref_mat = points(Eigen::all, Eigen::seq(ref_block_start, ref_block_end - 1));
            dmatrix ref_mat = centroids.block(0, ref_block_start,
                                              points.rows(), ref_block_end - ref_block_start);

            // precalculations
            switch (strategy) {
                case RayStrategyType::BRUTE_FORCE:
                    std::cout << "Running CPU precalculations" << std::endl;
                    geometry_kernel->precompute(ref_mat, u_mat);
                    break;
                case RayStrategyType::BRUTE_FORCE_GPU: {
                    geometry_kernel->reset_ref_mat(ref_mat.cols());
                    geometry_kernel->reset_u_mat(u_mat.cols());
                    std::cout << "Running GPU precalculations" << std::endl;
                    EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                    vec<int> j0_list(ref_mat.cols());
                    for (int i = 0; i < j0_list.size(); i++) {
                        j0_list[i] = ref_block_start + i;
                    }
                    kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
                    kernel_gpu->reset_rays_gpu(u_mat);
                    break;
                }
                case RayStrategyType::BIN_SEARCH:
                    break;
                default:
                    throw std::runtime_error("Unknown strategy");
            }

            bool distances_precomputed = false;
            vec<int> best_j_pos((ref_block_end - ref_block_start) * (u_block_end - u_block_start));
            vec<int> best_j_neg((ref_block_end - ref_block_start) * (u_block_end - u_block_start));

            if (strategy == BRUTE_FORCE_GPU) {
                std::cout << "Running raycasting on GPU" << std::endl;
                EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
                kernel_gpu->intersect_ray_bruteforce_gpu(&best_j_pos, &best_j_neg);
                distances_precomputed = true;
                std::cout << "Computing integrals" << std::endl;
            }

            my_tqdm bar(ref_block_end - ref_block_start);

            #pragma omp parallel
            {
                #pragma omp master
                {
                    if (u_block_i == 0 && ref_block_i == 0) {
                        std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
                    }
                }
                re.fix_random_engines();

                #pragma omp for
                for (int ref_i = ref_block_start; ref_i < ref_block_end; ref_i++) {
                    bar.atomic_iteration();
                    if (!is_initialized[ref_i]) {
                        dvector ref = dvector::col(points, ref_i);
                        ref.index -= ref_block_start;   // needed for precomputations
                        for (int j = 0; j < u_mat.cols(); j++) {
                            const dvector &u = dvector::col(u_mat, j);
                            int tmp;
                            ftype length = INF_ftype;
                            if (!distances_precomputed) {
                                geometry_kernel->intersect_ray(strategy, ref, u, ref_i, {ref_i}, &tmp, &length,
                                                               Kernel::IntersectionSearchType::RAY_INTERSECTION);
                            } else {
                                geometry_kernel->intersect_ray_precomputed(
                                        ref, u, ref_i, best_j_pos[(ref_i - ref_block_start) * u_mat.cols() + j], &length);
                            }

                            ftype vol = cell_kernel->cone_integral_uncentered(ref_i, length,
                                                                              u.dot(ref - points.col(ref_i)),
                                                                              (ref - points.col(ref_i)).squaredNorm(), dim);
                            if (math::isinf(vol)) {
                                throw std::runtime_error("Integral diverges. Perhaps your space is not compactified?");
                            }
                            ray_sample_sums[ref_i] += vol;
                            ray_sample_counts[ref_i]++;
                        }
                    }
                }
            }
            bar.bar().finish();
            // ## end of ref loop ##
        }

        // ## end of u loop ##
    }

    for (int i = 0; i < data_n; i++) {
        volumes[i] = ray_sample_sums[i] / ray_sample_counts[i];
        is_initialized[i] = true;
    }

    omp_set_num_threads(old_threads);
}

void VoronoiDensityEstimator::update_volumes(int nrays) {
    throw std::runtime_error("not implemented");
}

ftype VoronoiDensityEstimator::estimate_single(const dvector &point) const {
    int nearest = geometry_kernel->nearest_point(point);
    if (!is_initialized[nearest]) {
        throw std::runtime_error("lazy weight initialization not implemented");
    }

    ftype radius = (point - points.col(nearest)).norm();
    ftype ck_val = NAN_ftype;
    if (auto *ck = as<RadialCellKernel>(cell_kernel)) {
        ck_val = ck->kernel_value(nearest, radius);
    } else if (auto *ck = as<BalancedCellKernel>(cell_kernel)) {
        static bool flag = false;
        if (!flag && strategy == BRUTE_FORCE_GPU) {
            flag = true;
            std::cout << "W: USING NON GPU ESTIMATION" << std::endl;
        }

        int tmp;
        ftype length;
        geometry_kernel->intersect_ray(strategy, points.col(nearest), (point - points.col(nearest)).normalized(),
                                       nearest, {nearest}, &tmp, &length,
                                       Kernel::IntersectionSearchType::RAY_INTERSECTION);
        ck_val = as<BalancedCellKernel>(cell_kernel)->kernel_value(nearest, radius, length);
    } else {
        throw std::runtime_error("Unknown cell kernel type");
    }
    return ck_val / volumes[nearest] * weights[nearest];
}

dvector VoronoiDensityEstimator::sample_single() const {
    // hit and run sampling method is used, it does NOT require volume computation
    RandomEngine &engine = re.current();
    // pick a cell according to weighting
//    int cell_index = engine.rand_int(data_n);
    int cell_index = cell_distribution(engine.generator());
    return sample_within_cell(cell_index, engine);
}

dvector VoronoiDensityEstimator::sample_within_cell(int cell_index, RandomEngine &engine) const {
    if (RadialCellKernel *ck = dynamic_cast<RadialCellKernel*>(cell_kernel.get())) {
        dvector center = points.col(cell_index);
        dvector ref = center;
        IndexSet dual = {cell_index};
        for (int it = 0; it < nrays_hr; it++) {
            dvector u = engine.rand_on_sphere(dim);
            int tmp_j[2];
            ftype lengths[2];
            geometry_kernel->intersect_ray(strategy, ref, u, cell_index, dual,
                                           tmp_j, lengths, Kernel::LINE_INTERSECTION);
            ftype len = ck->sample_on_line(cell_index, ref, center, u, lengths[1], lengths[0], engine);
            if (isnan(len)) {
                throw std::runtime_error("Hit&run went to infinity. Perhaps your space is not compactified?");
            }
            ref = ref + u * len;
        }
        return ref;
    } else if (BalancedCellKernel *ck = dynamic_cast<BalancedCellKernel*>(cell_kernel.get())) {
        dvector ref = points.col(cell_index);
        IndexSet dual = {cell_index};
        dvector u = engine.rand_on_sphere(dim);
        int tmp_j;
        ftype length;
        geometry_kernel->intersect_ray(strategy, ref, u, cell_index, dual,
                                       &tmp_j, &length, Kernel::RAY_INTERSECTION);
        ftype len = ck->sample_on_cone(cell_index, length, engine);
        return ref + u * len;
    } else {
        throw std::runtime_error("unsupported cell kernel");
    }
}

void VoronoiDensityEstimator::centroid_smoothing(int smoothing_steps, int points_per_centroid) {
    RadialCellKernel *ck = dynamic_cast<RadialCellKernel*>(cell_kernel.get());
    if (!ck) {
        throw std::runtime_error("Centroid smoothing only supported for RadialCellKernel");
    }
    using namespace std::chrono;
    bool gpu = strategy == BRUTE_FORCE_GPU;
    ensure(gpu, "Only GPU smoothing is supported by now.");

    int u_n = std::max(1000, int(math::sqrt(1.0 * points_per_centroid * nrays_hr * data_n)));
    std::cout << "Using " << u_n << " precomputed rays per smoothing iteration" << std::endl;

    vec<int> j0_list(data_n);
    for (int i = 0; i < j0_list.size(); i++) {
        j0_list[i] = i;
    }

    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    for (int gs = 0; gs < smoothing_steps; gs++) {
        std::cout << "Smoothing #" << gs + 1 << std::endl;
        EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());
        dmatrix ref_mat = points;
        // generate directions
        dmatrix u_mat = dmatrix(dim, u_n);
        for (int i = 0; i < u_n; i++) {
            u_mat.col(i) = re.current().rand_on_sphere(dim);
        }

        kernel_gpu->reset_ref_mat(ref_mat.cols());
        kernel_gpu->reset_u_mat(u_mat.cols());
        kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
        kernel_gpu->reset_rays_gpu(u_mat);

        dmatrix centroids = dmatrix::Zero(dim, data_n);
        vec<int> u_indices(data_n);
        my_tqdm bar(points_per_centroid);
        for (int i = 0; i < points_per_centroid; i++) {
            bar.atomic_iteration();
            for (int j = 0; j < nrays_hr; j++) {
//                auto t0 = high_resolution_clock::now();
                // assign rays to points randomly
                for (int k = 0; k < u_indices.size(); k++) {
                    u_indices[k] = re.current().rand_int(u_n);
                }
                // obtain indices of intersections
                vec<int> best_j_pos(data_n);
                vec<int> best_j_neg(data_n);
                kernel_gpu->intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);
                vec<ftype> t_picked(data_n);
//                auto t1 = high_resolution_clock::now();
                #pragma omp parallel
                {
                    re.fix_random_engines();
                    #pragma omp for
                    for (int k = 0; k < data_n; k++) {
                        ftype t_pos, t_neg;
//                        auto t0 = high_resolution_clock::now();
                        kernel_gpu->intersect_ray_precomputed(
                                ref_mat.col(k),   // note: do not use precomputations for ref points
                                dvector::col(u_mat, u_indices[k]),
                                k, best_j_pos[k], &t_pos
                        );
                        kernel_gpu->intersect_ray_precomputed(
                                ref_mat.col(k),
                                dvector::col(u_mat, u_indices[k]),
                                k, best_j_neg[k], &t_neg
                        );
//                        auto t1 = high_resolution_clock::now();
                        if (t_pos < 0) {
                            t_pos = INF_ftype;
                        }
                        if (t_neg > 0) {
                            t_neg = -INF_ftype;
                        }
                        t_picked[k] = ck->sample_on_line(k,
                                ref_mat.col(k),
                                dvector::col(points, k),
                                dvector::col(u_mat, u_indices[k]),
                                t_neg, t_pos, re.current());
//                        auto t2 = high_resolution_clock::now();

                        ref_mat.col(k) = ref_mat.col(k) +
                                u_mat.col(u_indices[k]) * t_picked[k];
//                        auto t3 = high_resolution_clock::now();

//                std::cout
//                        << " " << duration_cast<nanoseconds>(t1 - t0).count()
//                        << " " << duration_cast<nanoseconds>(t2 - t1).count()
//                        << " " << duration_cast<nanoseconds>(t3 - t2).count()
//                        << " " << duration_cast<seconds>(t4 - t0).count()
//                        << std::endl;

                    }
                };
//                auto t2 = high_resolution_clock::now();
                kernel_gpu->move_reference_points(t_picked);
//                auto t3 = high_resolution_clock::now();
//                std::cout
//                        << " " << duration_cast<milliseconds>(t1 - t0).count()
//                        << " " << duration_cast<milliseconds>(t2 - t1).count()
//                        << " " << duration_cast<milliseconds>(t3 - t2).count()
//                        << " " << duration_cast<seconds>(t4 - t0).count()
//                        << std::endl;
            }
            centroids = centroids + ref_mat;
        }
        bar.bar().finish();
        centroids /= ftype(points_per_centroid);
        reset_points(centroids);
    }

    omp_set_num_threads(old_threads);
}

void VoronoiDensityEstimator::reset_points(const dmatrix &points) {
    ensure(points.cols() == data_n && points.rows() == dim,
           "Number of points or dimensionality changed!");
    this->points = points;
    volumes = vec<ftype>(points.cols(), 0);
    is_initialized = vec<bool>(points.size(), false);

    ray_sample_sums = vec<ftype>();
    ray_sample_counts = vec<int>();
    if (strategy == BRUTE_FORCE_GPU) {
        geometry_kernel = std::make_shared<EuclideanKernelGPU>(this->points, geometry_kernel->get_bounds());
    } else {
        geometry_kernel = std::make_shared<EuclideanKernel>(this->points, geometry_kernel->get_bounds());
    }

}

dmatrix VoronoiDensityEstimator::sample(int sample_size) const {
    if (as<BalancedCellKernel>(cell_kernel)) {
        ensure(nrays_hr == 1, "\"hit&run steps should be set to 1\"");
    }
    if (strategy == BRUTE_FORCE || strategy == BIN_SEARCH) {
        // CPU
        dmatrix results(dim, sample_size);
        int old_threads = omp_get_num_threads();
        omp_set_num_threads(njobs);
        my_tqdm bar(sample_size);
        #pragma omp parallel
        {
            re.fix_random_engines();
            #pragma omp for
            for (int i = 0; i < sample_size; i++) {
                bar.atomic_iteration();
                results.col(i) = sample_single();
            }
        }
        bar.bar().finish();
        omp_set_num_threads(old_threads);
        return results;
    }
    ensure(strategy == BRUTE_FORCE_GPU, "Wrong sampling strategy");

    // GPU
    int old_threads = omp_get_num_threads();
    omp_set_num_threads(njobs);

    EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(geometry_kernel.get());

    int u_n = std::max(1000, int(math::sqrt(static_cast<ftype>(1.0) * sample_size * nrays_hr)));

    int max_block_size = kernel_gpu->estimate_max_block_size();
    if (u_n < max_block_size) {
        max_block_size = kernel_gpu->estimate_max_block_size(u_n);
    } else {
        u_n = max_block_size;
    }
    std::cout << "Using " << u_n << " precomputed rays per block for sampling" << std::endl;
    std::cout << "Maximum block size estimated as " << max_block_size << std::endl;

    if (max_block_size < 0) {
        max_block_size = sample_size;
    }

    dmatrix result(dim, sample_size);

    for (int block_start = 0; block_start < sample_size; block_start += max_block_size) {
        std::cout << " * Block " << block_start / max_block_size + 1
                << " out of " << (sample_size + max_block_size - 1) / max_block_size << std::endl;
        int block_end = math::min(block_start + max_block_size, sample_size);
        int block_size = block_end - block_start;

        dmatrix u_mat = dmatrix(dim, u_n);
        vec<int> j0_list(block_size);
        dmatrix ref_mat(dim, block_size);

        #pragma omp parallel
        {
            re.fix_random_engines();
            #pragma omp for
            for (int i = 0; i < u_n; i++) {
                u_mat.col(i) = re.current().rand_on_sphere(dim);
            }
            #pragma omp for
            for (int i = 0; i < block_size; i++) {
//                int cell_index = re.current().rand_int(data_n);
                int cell_index = cell_distribution(re.current().generator());
                j0_list[i] = cell_index;
                ref_mat.col(i) = points.col(cell_index);
            }
        };
        kernel_gpu->reset_ref_mat(ref_mat.cols());
        kernel_gpu->reset_u_mat(u_mat.cols());
        kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
        kernel_gpu->reset_rays_gpu(u_mat);

        vec<int> u_indices(block_size);
        my_tqdm bar(nrays_hr);
        for (int j = 0; j < nrays_hr; j++) {
            bar.atomic_iteration();
            // assign rays to points randomly
            for (int k = 0; k < block_size; k++) {
                u_indices[k] = re.current().rand_int(u_n);
            }
            // obtain indices of intersections
            vec<int> best_j_pos(block_size);
            vec<int> best_j_neg(block_size);
            kernel_gpu->intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);
            vec<ftype> t_picked(block_size);
            #pragma omp parallel
            {
                re.fix_random_engines();
                #pragma omp for
                for (int k = 0; k < block_size; k++) {
                    ftype t_pos, t_neg;
                    kernel_gpu->intersect_ray_precomputed(
                            ref_mat.col(k),   // note: do not use precomputations for ref points
                            dvector::col(u_mat, u_indices[k]),
                            j0_list[k], best_j_pos[k], &t_pos
                    );
                    kernel_gpu->intersect_ray_precomputed(
                            ref_mat.col(k),
                            dvector::col(u_mat, u_indices[k]),
                            j0_list[k], best_j_neg[k], &t_neg
                    );
                    if (t_pos < 0) {
                        t_pos = INF_ftype;
                    }
                    if (t_neg > 0) {
                        t_neg = -INF_ftype;
                    }
                    if (auto *ck = as<RadialCellKernel>(cell_kernel)) {
                        t_picked[k] = ck->sample_on_line(j0_list[k],
                                                         ref_mat.col(k),
                                                         dvector::col(points, j0_list[k]),
                                                         dvector::col(u_mat, u_indices[k]),
                                                         t_neg, t_pos, re.current());
                    } else if (auto *ck = as<BalancedCellKernel>(cell_kernel)) {
                        t_picked[k] = ck->sample_on_cone(j0_list[k], t_pos, re.current());
                    } else {
                        throw std::runtime_error("cell kernel not supported");
                    }

                    ref_mat.col(k) = ref_mat.col(k) +
                                     u_mat.col(u_indices[k]) * t_picked[k];
                }
            }
            kernel_gpu->move_reference_points(t_picked);
        }
        bar.bar().finish();
        result.block(0, block_start, dim, block_size) = ref_mat;

    }

    omp_set_num_threads(old_threads);

    return result;
}

dvector _qr(const const_dmatrix_ref &mat, int col_i) {
    dynmatrix temp2 = dynmatrix::Zero(mat.rows(), 1);
    temp2(col_i, 0) = 1;
    auto qr = mat.householderQr();
    return (qr.householderQ() * temp2).col(0) * qr.matrixQR().triangularView<Eigen::Upper>()(col_i, col_i);
}

dmatrix VoronoiDensityEstimator::sample_masked(int sample_size, const dvector &mask) const {
    RadialCellKernel *ck = dynamic_cast<RadialCellKernel*>(cell_kernel.get());
    if (!ck) {
        throw std::runtime_error("Masked sampling only supported for RadialCellKernel");
    }
    // I.   find reference points -- intersections of cells with the subspace
    // II.  exit the corners
    // III. find centroids (optional)
    // IV.  perform non-isotropic spherical integration
    // V.   sample from non-uniformly weighted voronoi cells

    auto mask_nan = mask.array().isNaN().cast<ftype>();
    int nan_n = int(mask_nan.sum());
//    int mc_dim = -1;
//    for (int i = 0; i < dim; i++) {
//        if (math::isnan(mask(i))) {
//            if (mc_dim < 0) {
//                mc_dim = i;
//            }
//        } else if (mc_dim >= 0) {
//            throw std::runtime_error("Please provide the mask in the form (a0, a1, a2, ..., ak, nan, nan, ..., nan)"
//                                     " [you may permute the data in advance]");
//        }
//    }

    // I. Find the reference points
    dmatrix ref_mat(dim, data_n);
    for (int i = 0; i < data_n; i++) {
        ref_mat.col(i) = points.col(i);
    }

    int local_data_n = 0;
    vec<int> global_indices;
    dmatrix local_points;
    vec<vec<int>> local_duals;  // note: duals have "global" indices!
    {
        int old_threads = omp_get_num_threads();
        omp_set_num_threads(njobs);
        int non_nan_d = -1;
        dmatrix u_mat(dim, data_n);
        for (int i = 0; i < data_n; i++) {
            for (int d = 0; d < dim; d++) {
                if (math::isnan(mask(d))) {
                    u_mat(d, i) = 0;
                } else {
                    if (non_nan_d < 0) {
                        non_nan_d = d;
                    }
                    u_mat(d, i) = mask(d) - ref_mat(d, i);
                }
            }
            u_mat.col(i) /= u_mat.col(i).norm();
        }

        vec<vec<int>> duals;   // dual: [p, q_last, ...]
        vec<vec<int>> final_duals(data_n);
        for (int i = 0; i < data_n; i++) {
            duals.push_back({i});
        }

//    EuclideanKernelGPU* kernel = dynamic_cast<EuclideanKernelGPU*>(geometry_kernel.get());

        int step = 0;

        // Perform simplex walk
        while (!duals.empty()) {
            step++;
            std::cout << "Step: " << step << ", number of alive walks: " << duals.size() << std::endl;
            vec<vec<int>> new_duals;
            geometry_kernel->reset_ref_mat(ref_mat.cols());
//        geometry_kernel->reset_u_mat(u_mat.cols());
            #pragma omp parallel
            {
                re.fix_random_engines();

                #pragma omp for
                for (int ii = 0; ii < duals.size(); ii++) {
                    vec<int> dual = duals[ii];
                    int i = dual[0];
                    int bc_dim = int(dual.size()) - 1;
                    dvector u = u_mat.col(i);

                    // construct B^c
                    dmatrix B_c(dim, bc_dim + 1);
                    for (int j = 1; j <= bc_dim; j++) {
                        B_c.col(j - 1) = points.col(dual[j]) - points.col(dual[0]);
                    }

                    // project B^c onto <M, u>
                    for (int j = 0; j < bc_dim; j++) {
                        dvector cur = B_c.col(j);
                        B_c.col(j) = (cur.array() * mask_nan).matrix() + cur.dot(u) * u;
                        B_c.col(j) = B_c.col(j) / B_c.col(j).norm();
                    }

                    // QR
                    vec<int> new_dual = {dual[0]};
                    bool decreasing_dim = false;
                    dvector new_u = dvector::Zero(dim);

                    for (int j = 0; j < bc_dim; j++) {
                        // remove jth generator
                        dvector tmp = B_c.col(j);
                        B_c.col(j) = B_c.col(bc_dim - 1);
                        B_c.col(bc_dim - 1) = u;
                        dvector cur_u = _qr(B_c.block(0, 0, dim, bc_dim), bc_dim - 1);
                        if (cur_u.dot(tmp) > 0) {
                            new_dual.push_back(dual[j + 1]);
                        } else {
                            decreasing_dim = true;
                            new_u += cur_u;
                        }
                        B_c.col(bc_dim - 1) = B_c.col(j);
                        B_c.col(j) = tmp;
                    }
                    if (nan_n + 1 <
                        new_dual.size()) {  // dim(<M, u>) + dim(boundary) < dim  ===>  no overlap in general case
                        // negative stop
                        continue;
                    }

                    if (!decreasing_dim) {
                        B_c.col(bc_dim) = u;
                        u = _qr(B_c, bc_dim);
                    } else {
                        u = new_u / new_u.norm();
                    }

                    // intersection search
                    // todo gpu support
                    int best_j;
                    ftype best_l;
                    geometry_kernel->intersect_ray(strategy, ref_mat.col(i), u, i, dual,
                                                   &best_j, &best_l, Kernel::IntersectionSearchType::RAY_INTERSECTION);
                    ftype max_l = (mask(non_nan_d) - ref_mat(non_nan_d, i)) / u(non_nan_d);
                    bool done = best_j < 0 || max_l < best_l;
                    if (done) {
                        // positive stop
                        best_l = max_l;
                        final_duals[i] = new_dual;
                        #pragma omp atomic
                        local_data_n++;
                    }
                    ref_mat.col(i) = geometry_kernel->move_along_ray(ref_mat.col(i), u, best_l);

                    if (!done) {
                        if (new_dual.size() >= 2) {
                            new_dual.push_back(new_dual[1]);
                            new_dual[1] = best_j;
                        } else {
                            new_dual.push_back(best_j);
                        }
                        #pragma omp critical
                        new_duals.push_back(new_dual);
                    }
                }
            }

            duals = new_duals;
        }

        // Collect all projected points
        global_indices = vec<int>(local_data_n, -1);
        local_points = dmatrix(dim, local_data_n);
        local_duals = vec<vec<int>>(local_data_n);
        {
            dmatrix new_ref_mat(dim, local_data_n);
            int j = 0;
            for (int i = 0; i < data_n; i++) {
                if (!final_duals[i].empty()) {
                    new_ref_mat.col(j) = ref_mat.col(i);
                    local_points.col(j) = points.col(i);
                    local_duals[j] = final_duals[i];
                    global_indices[j] = i;
                    j++;
                }
            }
            ensure(j == local_data_n, "should not be here");
            ref_mat = new_ref_mat;
        }
        omp_set_num_threads(old_threads);
    }

    ptr<EuclideanKernel> local_geometry_kernel = strategy == BRUTE_FORCE_GPU
            ? std::make_shared<EuclideanKernelGPU>(local_points, geometry_kernel->get_bounds())
            : std::make_shared<EuclideanKernel>(local_points, geometry_kernel->get_bounds());

    std::cout << "Number of cells at the intersection: " << local_data_n << std::endl;

    // II. Exit the corner
    std::cout << "Making a step into the interior" << std::endl;
    #pragma omp parallel for
    for (int i = 0; i < local_data_n; i++) {
        const vec<int> &dual = local_duals[i];
        ensure(dual[0] == global_indices[i], "Check final_dual construction");
        int bc_dim = int(dual.size()) - 1;
        ensure(bc_dim >= 0, "check delaunay simplices");

        // if we are strictly inside the cell -- all good
        if (bc_dim == 0) {
            continue;
        }

        // construct B^c
        dmatrix B_c(dim, bc_dim);
        for (int j = 1; j <= bc_dim; j++) {
            B_c.col(j - 1) = local_points.col(i) - points.col(dual[j]);    // note: points inside the cell
        }

        // project B^c onto <M>
        for (int j = 0; j < bc_dim; j++) {
            dvector cur = B_c.col(j);
            cur = (cur.array() * mask_nan).matrix().normalized();
            B_c.col(j) = cur;
        }

        // project each vector onto the intersection of other hyperplanes, u - their unit-sum
        dvector u = dvector::Zero(dim);
        for (int j = 0; j < bc_dim; j++) {
            // place jth generator last
            dvector col_j = B_c.col(j);
            B_c.col(j) = B_c.col(bc_dim - 1);
            B_c.col(bc_dim - 1) = col_j;
            dvector cur_u = _qr(B_c, bc_dim - 1);
            u += cur_u;
            B_c.col(bc_dim - 1) = B_c.col(j);
            B_c.col(j) = col_j;
        }
        u /= ftype(bc_dim);

        // step away
        int best_j = -1;
        ftype best_l;
        local_geometry_kernel->intersect_ray(strategy, ref_mat.col(i), u, i, {i}, &best_j, &best_l,
                                       Kernel::IntersectionSearchType::RAY_INTERSECTION);
        if (best_l < 1e-7) {
            std::cout << "WARNING: VERIFY INTERSECTION (" << best_l << ")" << std::endl;
        }
        ensure(best_l >= 0, "negative ray length");
        ftype t = ck->sample_on_line_median(global_indices[i], ref_mat.col(i), local_points.col(i), u, 0, best_l);
        if (math::isinf(t) || math::isnan(t)) {
            ref_mat.col(i).setConstant(NAN_ftype);
        } else {
            ref_mat.col(i) = local_geometry_kernel->move_along_ray(ref_mat.col(i), u, t);
        }
//        std::cout << i << std::endl;
    }

//    // Verify results
//    for (int i = 0; i < local_data_n; i++) {
//        if (ref_mat.col(i).array().isNaN().sum() > 0) {
//            std::cout << "Bad reference: " << i << std::endl;
//        }
//        ftype dst1, dst2;
//        geometry_kernel->nearest_point(ref_mat.col(i), &dst1);
//        dst2 = (ref_mat.col(i) - points.col(global_indices[i])).norm();
//        if (dst1 + 1e-3 < dst2) {
//            std::cout << "BAD CELL AFTER [II] ... " << global_indices[i] << " " << dst1
//                      << " " << dst2 << " " << dst2 - dst1 << std::endl;
//            std::cout << ref_mat.col(i) << std::endl;
//        }
//    }

    // III. Find centroids
    // skipping...

    // IV. Spherical integration
    std::cout << "Performing spherical integration" << std::endl;
    // edited copy of initialize_weights_uncentered
    dmatrix local_centroids = ref_mat;
    vec<ftype> local_cell_weights(local_data_n);
    {
        int max_block_size = -1;
        if (strategy == BRUTE_FORCE_GPU) {
            EuclideanKernelGPU *kernel = dynamic_cast<EuclideanKernelGPU *>(local_geometry_kernel.get());
            max_block_size = kernel->estimate_max_block_size();
            if (nrays_si < max_block_size) {
                max_block_size = kernel->estimate_max_block_size(nrays_si);
            }
            std::cout << "Maximum block size estimated as " << max_block_size << std::endl;
        }

        int old_threads = omp_get_num_threads();
        omp_set_num_threads(njobs);

        vec<ftype> local_weight_sums(local_data_n, 0);
        vec<int> local_weight_counts(local_data_n, 0);

        int u_block_n = max_block_size < 0 ? 1 : (nrays_si + max_block_size - 1) / max_block_size;
        int ref_block_n = max_block_size < 0 ? 1 : (local_data_n + max_block_size - 1) / max_block_size;
        int block_size = max_block_size < 0 ? math::max(local_data_n, nrays_si) : max_block_size;

        for (int u_block_i = 0; u_block_i < u_block_n; u_block_i++) {
            int u_block_start = u_block_i * block_size;
            int u_block_end = math::min(u_block_start + block_size, nrays_si);
            // generate directions
            dmatrix u_mat = dmatrix(dim, u_block_end - u_block_start);
            for (int i = 0; i < u_mat.cols(); i++) {
                dvector u = re.current().rand_on_sphere(dim);
                u = (u.array() * mask_nan).matrix().normalized();
                u_mat.col(i) = u;

            }

            for (int ref_block_i = 0; ref_block_i < ref_block_n; ref_block_i++) {
                std::cout << " * Block (" << u_block_i + 1 << ", " << ref_block_i + 1
                          << ") out of (" << u_block_n << ", " << ref_block_n << ") ..." << std::endl;
                int ref_block_start = ref_block_i * block_size;
                int ref_block_end = math::min(ref_block_start + block_size, local_data_n);
                ref_mat = local_centroids.block(0, ref_block_start,
                                                local_centroids.rows(), ref_block_end - ref_block_start);

                // precalculations
                switch (strategy) {
                    case RayStrategyType::BRUTE_FORCE:
                        std::cout << "Running CPU precalculations" << std::endl;
                        local_geometry_kernel->precompute(ref_mat, u_mat);
                        break;
                    case RayStrategyType::BRUTE_FORCE_GPU: {
                        local_geometry_kernel->reset_ref_mat(ref_mat.cols());
                        local_geometry_kernel->reset_u_mat(u_mat.cols());
                        std::cout << "Running GPU precalculations" << std::endl;
                        EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(local_geometry_kernel.get());
                        vec<int> j0_list(ref_mat.cols());
                        for (int i = 0; i < j0_list.size(); i++) {
                            j0_list[i] = ref_block_start + i;
                        }
                        kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
                        kernel_gpu->reset_rays_gpu(u_mat);
                        break;
                    }
                    case RayStrategyType::BIN_SEARCH:
                        break;
                    default:
                        throw std::runtime_error("Unknown strategy");
                }

                bool distances_precomputed = false;
                vec<int> best_j_pos((ref_block_end - ref_block_start) * (u_block_end - u_block_start));
                vec<int> best_j_neg((ref_block_end - ref_block_start) * (u_block_end - u_block_start));

                if (strategy == BRUTE_FORCE_GPU) {
                    std::cout << "Running raycasting on GPU" << std::endl;
                    EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(local_geometry_kernel.get());
                    kernel_gpu->intersect_ray_bruteforce_gpu(&best_j_pos, &best_j_neg);
                    distances_precomputed = true;
                    std::cout << "Computing integrals" << std::endl;
                }

                my_tqdm bar(ref_block_end - ref_block_start);

                #pragma omp parallel
                {
                    #pragma omp master
                    {
                        if (u_block_i == 0 && ref_block_i == 0) {
                            std::cout << "Using " << omp_get_num_threads() << " threads" << std::endl;
                        }
                    }
                    re.fix_random_engines();

                    #pragma omp for
                    for (int ref_i = ref_block_start; ref_i < ref_block_end; ref_i++) {
                        bar.atomic_iteration();

                        dvector ref = dvector::col(local_centroids, ref_i);
                        if (ref.array().isNaN().sum() > 0) {
                            continue;
                        }
                        ref.index -= ref_block_start;   // needed for precomputations
                        for (int j = 0; j < u_mat.cols(); j++) {
                            const dvector &u = dvector::col(u_mat, j);
                            int tmp;
                            ftype length = INF_ftype;
                            if (!distances_precomputed) {
                                local_geometry_kernel->intersect_ray(strategy, ref, u, ref_i, {ref_i}, &tmp, &length,
                                                                     Kernel::IntersectionSearchType::RAY_INTERSECTION);
                            } else {
                                local_geometry_kernel->intersect_ray_precomputed(
                                        ref, u, ref_i, best_j_pos[(ref_i - ref_block_start) * u_mat.cols() + j],
                                        &length);
                            }

                            ftype vol = cell_kernel->cone_integral_uncentered(global_indices[ref_i],
                                    length, u.dot(ref - local_points.col(ref_i)),
                                    (ref - local_points.col(ref_i)).squaredNorm(), nan_n);
                            ensure(!math::isnan(vol), "Cone integral is nan");
                            if (math::isinf(vol)) {
                                throw std::runtime_error("Integral diverges. Perhaps your space is not compactified?");
                            }

                            local_weight_sums[ref_i] += vol;
                            local_weight_counts[ref_i]++;
                        }
                    }
                }
                bar.bar().finish();
                // ## end of ref loop ##
            }
            // ## end of u loop ##
        }
        for (int i = 0; i < local_data_n; i++) {
            if (local_weight_counts[i] > 0) {
                local_cell_weights[i] = 1 / (data_n * volumes[global_indices[i]]) * local_weight_sums[i] / local_weight_counts[i];
            } else {
                local_cell_weights[i] = 0;
            }
//            if (isnan(local_cell_weights[i])) {
//                std::cout << "NAN " << i << " " << weights[global_indices[i]] << " " << local_weight_sums[i]  << " " << local_weight_counts[i] << std::endl;
//                exit(0);
//            }
//            local_cell_weights[i] = 1;
        }

        omp_set_num_threads(old_threads);
    }

    std::cout << "Cell weights are: [" << std::endl;
    for (int i = 0; i < local_data_n; i++) {
        std::cout << "    (" << global_indices[i] << ") local=" << local_cell_weights[i] << " global=" << 1 / (volumes[global_indices[i]] * data_n) << std::endl;
    }
    std::cout << "]" << std::endl;

    // V. Sample from non-uniformly weighted voronoi cells
    std::cout << "Sampling!" << std::endl;
    dmatrix result(dim, sample_size);
    {
        // todo refactor!
        std::discrete_distribution cell_distribution(local_cell_weights.begin(), local_cell_weights.end());

        ensure(strategy == BRUTE_FORCE_GPU, "We do GPU sampling until refactored");

        int old_threads = omp_get_num_threads();
        omp_set_num_threads(njobs);

        EuclideanKernelGPU *kernel_gpu = dynamic_cast<EuclideanKernelGPU *>(local_geometry_kernel.get());

        int u_n = std::max(1000, int(math::sqrt(static_cast<ftype>(1.0) * sample_size * nrays_hr)));

        int max_block_size = kernel_gpu->estimate_max_block_size();
        if (u_n < max_block_size) {
            max_block_size = kernel_gpu->estimate_max_block_size(u_n);
        } else {
            u_n = max_block_size;
        }
        std::cout << "Using " << u_n << " precomputed rays per block for sampling" << std::endl;
        std::cout << "Maximum block size estimated as " << max_block_size << std::endl;

        if (max_block_size < 0) {
            max_block_size = sample_size;
        }

        for (int block_start = 0; block_start < sample_size; block_start += max_block_size) {
            std::cout << " * Block " << block_start / max_block_size + 1
                      << " out of " << (sample_size + max_block_size - 1) / max_block_size << std::endl;
            int block_end = math::min(block_start + max_block_size, sample_size);
            int block_size = block_end - block_start;

            dmatrix u_mat = dmatrix(dim, u_n);
            vec<int> j0_list(block_size);
            ref_mat = dmatrix(dim, block_size);

            vec<int> cnt(local_data_n, 0);

            #pragma omp parallel
            {
                re.fix_random_engines();
                #pragma omp for
                for (int i = 0; i < u_n; i++) {
                    dvector u = re.current().rand_on_sphere(dim);
                    u = (u.array() * mask_nan).matrix().normalized();
                    u_mat.col(i) = u;
                }
                #pragma omp for
                for (int i = 0; i < block_size; i++) {
                    int local_cell_index = cell_distribution(re.current().generator());
                    j0_list[i] = local_cell_index;
                    ref_mat.col(i) = local_centroids.col(local_cell_index);
                    #pragma omp atomic
                    cnt[local_cell_index]++;
                }
            };

            std::cout << "Sampled cell counts (>0): [" << std::endl;
            for (int i = 0; i < cnt.size(); i++) {
                if (cnt[i] > 0)
                    std::cout << "    (" << global_indices[i] << ") " << cnt[i] << std::endl;
            }
            std::cout << "]" << std::endl;

            kernel_gpu->reset_ref_mat(ref_mat.cols());
            kernel_gpu->reset_u_mat(u_mat.cols());
            kernel_gpu->reset_reference_points_gpu(ref_mat, j0_list);
            kernel_gpu->reset_rays_gpu(u_mat);

            vec<int> u_indices(block_size);
            my_tqdm bar(nrays_hr);
            for (int j = 0; j < nrays_hr; j++) {
                bar.atomic_iteration();
                // assign rays to points randomly
                for (int k = 0; k < block_size; k++) {
                    u_indices[k] = re.current().rand_int(u_n);
                }
                // obtain indices of intersections
                vec<int> best_j_pos(block_size);
                vec<int> best_j_neg(block_size);
                kernel_gpu->intersect_ray_indexed_gpu(u_indices, &best_j_pos, &best_j_neg);
                vec<ftype> t_picked(block_size);
                #pragma omp parallel
                {
                    re.fix_random_engines();
                    #pragma omp for
                    for (int k = 0; k < block_size; k++) {
                        ftype t_pos, t_neg;
                        kernel_gpu->intersect_ray_precomputed(
                                ref_mat.col(k),   // note: do not use precomputations for ref points
                                dvector::col(u_mat, u_indices[k]),
                                j0_list[k], best_j_pos[k], &t_pos
                        );
                        kernel_gpu->intersect_ray_precomputed(
                                ref_mat.col(k),
                                dvector::col(u_mat, u_indices[k]),
                                j0_list[k], best_j_neg[k], &t_neg
                        );
                        if (t_pos < 0) {
                            t_pos = INF_ftype;
                        }
                        if (t_neg > 0) {
                            t_neg = -INF_ftype;
                        }
                        t_picked[k] = ck->sample_on_line(global_indices[j0_list[k]],
                                ref_mat.col(k),
                                dvector::col(local_points, j0_list[k]),
                                dvector::col(u_mat, u_indices[k]),
                                t_neg, t_pos, re.current());

                        ref_mat.col(k) = ref_mat.col(k) +
                                         u_mat.col(u_indices[k]) * t_picked[k];
                    }
                };
                kernel_gpu->move_reference_points(t_picked);
            }
            bar.bar().finish();
            result.block(0, block_start, dim, block_size) = ref_mat;

        }

        omp_set_num_threads(old_threads);
    }

    return result;
}

void VoronoiDensityEstimator::set_weights(const vec<ftype> &weights) {
    ensure(weights.size() == data_n, "The number of weights should be equal to the number of generators");
    this->weights = weights;
    ftype sum = 0;
    for (int i = 0; i < data_n; i++) {
        sum += this->weights[i];
    }
    for (int i = 0; i < data_n; i++) {
        this->weights[i] /= sum;
    }

    cell_distribution = std::discrete_distribution<int>(this->weights.begin(), this->weights.end());
}

const vec<ftype> &VoronoiDensityEstimator::get_weights() {
    return weights;
}

int VoronoiDensityEstimator::get_initial_seed() const {
    return initial_seed;
}
