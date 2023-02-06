#pragma once

#include "../utils.h"
#include "../RandomEngine.h"
#include "../algo/cell_kernels.h"

class KDE {
public:
    KDE(const dmatrix &points, ftype global_bw, int seed, int njobs);
    KDE(const dmatrix &points, const ptr<RadialCellKernel> &cell_kernel, int seed, int njobs);

    vec<ftype> estimate(const dmatrix &queries) const;

    dmatrix sample(int sample_size) const;

    const dmatrix &get_points() const;

    void set_weights(const vec<ftype> &weights);

    const vec<ftype> &get_weights();

    /**
     * An implementation of:
     *      B. Wang and X. Wang, "Bandwidth Selection for Weighted Kernel Density Estimation"
     * Original implementation can be found at: https://github.com/mennthor/awkde
     */
    void make_adaptive(ftype alpha);

private:
    dmatrix points;
    const int data_n;
    const int dim;

    ftype global_bw;
    vec<ftype> local_bw;

    bool is_adaptive = false;

    const ptr<RadialCellKernel> cell_kernel;

    vec<ftype> weights;
    mutable std::discrete_distribution<int> point_distribution;

    mutable RandomEngineMultithread re;

    int njobs;
};
