#pragma once

#include "VoronoiGraph.h"
#include "cell_kernels.h"
#include "kernels_gpu.h"

class VoronoiDensityEstimator {
public:
    VoronoiDensityEstimator(const dmatrix &points, const ptr<CellKernel> &cell_kernel, int seed,
                            int njobs, int nrays_weights, int nrays_sampling=5,
                            RayStrategyType strategy = BRUTE_FORCE,
                            const ptr<Bounds> &bounds = std::make_shared<Unbounded>());


    vec<ftype> estimate(const dmatrix &queries) const;

    const vec<ftype> &get_volumes() const;

    const dmatrix &get_points() const;

    dmatrix sample(int sample_size) const;

    void initialize_volumes();

    void set_weights(const vec<ftype> &weights);

    const vec<ftype> &get_weights();

    vec<dmatrix> dvol_dp();

    dmatrix dlogf_dp(const dmatrix &queries);

    void initialize_weights_uncentered(const dmatrix &ref_mat);

    void update_volumes(int nrays);

    ftype estimate_single(const dvector &point) const;

    dvector sample_single() const;

    void centroid_smoothing(int smoothing_steps, int points_per_centroid);

    void reset_points(const dmatrix &points);

//    void set_max_block_size(int _max_block_size);

    dmatrix sample_masked(int sample_size, const dvector &mask) const;  // mask has NaN in free directions

    int get_initial_seed() const;

    vec<vec<ftype>> lengths_debug;
private:
    dvector sample_within_cell(int cell_index, RandomEngine &engine) const;

    int initial_seed;

    dmatrix points;
    const int data_n;
    const int dim;


    mutable RandomEngineMultithread re;

    vec<ftype> volumes;
    vec<bool> is_initialized;

    vec<ftype> weights;
    mutable std::discrete_distribution<int> cell_distribution;


    int njobs;

    vec<ftype> ray_sample_sums;
    vec<int> ray_sample_counts;

    RayStrategyType strategy;
    ptr<EuclideanKernel> geometry_kernel;

    ptr<CellKernel> cell_kernel;

    int nrays_si;
    int nrays_hr;
};
