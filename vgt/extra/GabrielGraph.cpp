#include "GabrielGraph.h"
#include "../algo/kernels.h"
#include "../RandomEngine.h"

edge_matrix get_gabriel_edges(const dmatrix &points) {
    EuclideanKernel geometry(points);
    int npoints = points.cols();
    int npairs = npoints * (npoints - 1) / 2;

    my_tqdm bar(npairs);
    vec<int> edge_indices;

    #pragma omp parallel for
    for (int ipair = 0; ipair < npairs; ipair++) {
        bar.atomic_iteration();
        int iu = int(math::floor(0.5 * (2 * npoints - 1 - math::sqrt((2 * npoints - 1) * (2 * npoints - 1) - 8 * ipair))));
        int iv = int(ipair - (2 * npoints - iu - 3) * iu / 2 + 1);
        auto &u = points.col(iu);
        auto &v = points.col(iv);
        dvector mp = (u + v) / 2;
        ftype radius = (mp - u).norm();
        int nearest = geometry.nearest_point_extra(mp, radius, {iu, iv});
        if (nearest < 0) {
            #pragma omp critical
            edge_indices.push_back(ipair);
        }
    }
    bar.bar().finish();

    int nedges = edge_indices.size();
    edge_matrix edges(2, nedges);
    for (int i = 0; i < nedges; i++) {
        int ipair = edge_indices[i];
        int iu = int(math::floor(0.5 * (2 * npoints - 1 - math::sqrt((2 * npoints - 1) * (2 * npoints - 1) - 8 * ipair))));
        int iv = ipair - (2 * npoints - iu - 3) * iu / 2 + 1;
        edges(0, i) = iu;
        edges(1, i) = iv;
    }

    return edges;
}


edge_matrix get_gabriel_edges_subset(const dmatrix &points, int size, int seed, ftype *nedges_estimate) {
    // final size might differ
    EuclideanKernel geometry(points);
    using ipair_t = std::pair<int, int>;
    struct pair_hash {
        size_t operator()(const ipair_t &p) const {
            return p.first * 31 + p.second;
        }
    };

    int npoints = points.cols();

    my_tqdm bar(size);
    vec<ipair_t> edge_indices;
    std::unordered_set<ipair_t, pair_hash> was;
    int good = 0;
    int bad = 0;

    RandomEngineMultithread re(seed);
    #pragma omp parallel
    re.fix_random_engines();

    #pragma omp parallel
    while (true) {
        if (good >= size) {
            break;
        }
        int iu = re.current().rand_int(npoints);
        int iv = re.current().rand_int(npoints);
        if (iu == iv) {
            continue;
        }
        if (iu > iv) {
            std::swap(iu, iv);
        }
        ipair_t ipair = std::make_pair(iu, iv);
        if (was.find(ipair) != was.end()) {
            continue;
        }
        auto &u = points.col(iu);
        auto &v = points.col(iv);
        dvector mp = (u + v) / 2;
        ftype radius = (mp - u).norm();
        int nearest = geometry.nearest_point_extra(mp, radius, {iu, iv});
        if (nearest < 0) {
            #pragma omp critical
            {
                if (good < size && was.find(ipair) == was.end()) {
                    good++;
                    was.insert(ipair);
                    edge_indices.push_back(ipair);
                    bar.iteration();
                }
            }
        } else {
            #pragma omp critical
            {
                bad++;
                was.insert(ipair);
            }
        }
    }
    bar.bar().finish();

    int nedges = edge_indices.size();
    edge_matrix edges(2, nedges);
    for (int i = 0; i < nedges; i++) {
        ipair_t ipair = edge_indices[i];
        int iu = ipair.first;
        int iv = ipair.second;
        edges(0, i) = iu;
        edges(1, i) = iv;
    }

    if (nedges_estimate) {
        *nedges_estimate = ftype(good) / ftype(good + bad) * ftype(npoints) * ftype(npoints - 1) / 2;
    }

    return edges;
}