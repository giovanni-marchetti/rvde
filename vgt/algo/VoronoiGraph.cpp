#include "VoronoiGraph.h"

#include <memory>

#include "cnpy.h"

const VoronoiGraph::Polytope VoronoiGraph::NONE = VoronoiGraph::Polytope();

VoronoiGraph::Polytope::Polytope(const IndexSet &dual, const dvector &ref) :
        dual(dual), ref(ref) {}

VoronoiGraph::Polytope::Polytope() : dual({}) {}

bool VoronoiGraph::Polytope::is_none() const {
    return dual.empty();
}

VoronoiGraph::VoronoiGraph(RayStrategyType strategy, DataType data_type) :
        strategy(strategy), data_type(data_type) {
}

Kernel& VoronoiGraph::get_kernel() const {
    return *kernel;
}

bool VoronoiGraph::is_vertex(const VoronoiGraph::Polytope &p) const {
    return p.dual.dim() == data_dim;
}

void VoronoiGraph::read_points(const std::string &filename, ptr<Bounds> bounds) {
    cnpy::NpyArray data_npy = cnpy::npy_load(filename);
    points = npy2matrix(data_npy);
    n_points = points.cols();
    ambient_dim = points.rows();

    // Initialize kernel
    switch (data_type) {
        case EUCLIDEAN:
            if (!bounds) {
                bounds = std::make_shared<Unbounded>();
            }
            kernel = std::make_shared<EuclideanKernel>(points, bounds);
            data_dim = ambient_dim;
            break;
        case SPHERICAL:
            ensure(!bounds, "Data bounds are not allowed for the spherical data");
            kernel = std::make_shared<SphericalKernel>(points);
            data_dim = ambient_dim - 1;
            break;
        case TOROIDAL:
        default:
            throw std::runtime_error("Current data type is not supported");
    }
}

int VoronoiGraph::get_containing_voronoi_cell(const dvector &ref) const {
    return get_kernel().nearest_point(ref);
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(int point_idx, RandomEngine &re) const {
    return retrieve_vertex_nearby(points.col(point_idx), re, point_idx);
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex_nearby(const dvector &ref, RandomEngine &re,
                                                            int nearest_idx) const {
    // `cur` is the current reference point, it changes after every ray projection
    dvector cur(ref);
    // determine the initial Voronoi cell
    if (nearest_idx < 0) {
        nearest_idx = get_containing_voronoi_cell(cur);
    }
    // `dual` contains vertices of a delaunay simplex, dual of which contains `cur`
    IndexSet dual = {nearest_idx};
    // `normals` describes the basis of the orthogonal complement of the current polytope
    vec<dvector> normals;
    bool ok = true;
    for (int cur_dim = 0; ok && cur_dim < data_dim; cur_dim++) {
        ok = false;
        // try to find a finite direction within the polytope
        // should happen with probability 1, but just in case -- retried a few times
        for (int retries = 0; !ok && retries < MAX_RETRIES; retries++) {
            // random direction `u` is picked uniformly ...
            dvector u = re.rand_on_sphere(ambient_dim);
            // ..., projected onto the polytope...
            for (const dvector &norm : normals) {
                u = u - u.dot(norm) * norm;
            }
            // ... including the projection onto the tangent hyperplane of the data manifold
            // (needed for spherical data) ...
            get_kernel().project_to_tangent_space_inplace(cur, u);
            // ..., and normalized
            u.normalize();

            int best_j = -1;    // index of the data point that generates the next boundary
            ftype best_l = 0;   // length of the ray until the intersection, may be negative
            get_kernel().intersect_ray(strategy, cur, u, nearest_idx, dual,
                                       &best_j, &best_l, Kernel::ANY_INTERSECTION);

            if (best_j < 0) {
                // the polytope is infinite in the direction `u`
                continue;
            }

            // the dual of the next boundary
            IndexSet new_dual = dual.append(best_j);
            // a new reference point on that boundary
            dvector new_cur = get_kernel().move_along_ray(cur, u, best_l);

            // check that the new boundary is valid
            if (!validate(new_dual, new_cur)) {
                #pragma omp atomic
                validations_failed++;
                continue;
            } else {
                #pragma omp atomic
                validations_ok++;
            }

            // update the current polytope
            dual = new_dual;
            cur = new_cur;

            // add a new vector to the basis of the complement
            dvector new_norm = points.col(best_j) - points.col(nearest_idx);
            for (const dvector &norm : normals) {
                new_norm = new_norm - new_norm.dot(norm) * norm;
            }
            new_norm.normalize();
            normals.push_back(new_norm);
            ok = true;
        }
    }

    if (ok) {
        return Polytope(dual, cur);
    } else {
        return NONE;
    }
}

/**
 * Returns argsort of negative eigenvalues.
 */
vec<int> test_lambda(const svector &lambda) {
    vec<int> res;
    for (int i = 0; i < lambda.size(); i++) {
        if (lambda[i] < 0) {
            res.push_back(i);
        }
    }
    std::sort(res.begin(), res.end(), [&](int a, int b) {
        return lambda[a] < lambda[b];
    });
    return res;
}

VoronoiGraph::Polytope VoronoiGraph::retrieve_vertex(const dvector &point, RandomEngine &re,
                                                     svector *coordinates) const {
    // todo fix for spherical!!!!
    // Get an initial vertex nearby
    Polytope vertex = retrieve_vertex_nearby(point, re, -1);
    if (vertex.is_none()) { // potentially due to numeric issues
        return NONE;
    }

    svector q_vector(ambient_dim + 1);
    q_vector.head(ambient_dim) = point;
    q_vector[ambient_dim] = 1;

    svector lambda;
    for (int step = 0; step < VISIBILITY_WALK_MAX_STEPS; step++) {
        Eigen::Matrix<ftype, dim_delta<DATA_DIM, 1>(), dim_delta<DATA_DIM, 1>()> coords(ambient_dim + 1, ambient_dim + 1);
        for (int i = 0; i <= ambient_dim; i++) {
            coords.col(i).head(ambient_dim) = points.col(vertex.dual[i]);
            coords(ambient_dim, i) = 1;
        }
        lambda = coords.colPivHouseholderQr().solve(q_vector);
        vec<int> walk_directions = test_lambda(lambda);

        if (walk_directions.empty()) {
            break;
        }

        Polytope new_vertex;
        for (int i = 0; i < walk_directions.size() && new_vertex.is_none(); i++) {
            new_vertex = get_neighbor(vertex, walk_directions[i], re);
        }
        if (new_vertex.is_none()) {
            return NONE;
        }
        vertex = new_vertex;
    }

    if (coordinates) {
        *coordinates = lambda;
    }
    return vertex;
}

// todo re is not really needed, may be removed from the method definition later
VoronoiGraph::Polytope VoronoiGraph::get_neighbor(const VoronoiGraph::Polytope &vertex, int index,
                                                  RandomEngine &re) const {
    ensure(is_vertex(vertex), "`v` should be a Voronoi vertex");
    IndexSet edge = vertex.dual.remove_at(index);

    // Find the direction vector. May be done faster (right now - d^3)
    // find all normalizers
    vec<dvector> normals;
    for (int p = 1; p < data_dim; p++) {
        dvector v = points.col(edge[p]) - points.col(edge[0]);
        for (const dvector &norm : normals) {
            v = v - v.dot(norm) * norm;
        }
        v.normalize();
        normals.push_back(v);
    }
    // u -- direction vector
    dvector u = re.rand_on_sphere(ambient_dim);
    for (const dvector &norm : normals) {
        u = u - u.dot(norm) * norm;
    }
    get_kernel().project_to_tangent_space_inplace(vertex.ref, u);
    u.normalize();
    // determine the correct direction of u
    if (u.dot(points.col(vertex.dual[index]) - points.col(edge[0])) > 0) {
        u = -u;
    }

    // find the other end of the edge
    int best_j = -1;
    ftype best_l = -1;

    get_kernel().intersect_ray(strategy, vertex.ref, u, edge[0], vertex.dual,
                               &best_j, &best_l, Kernel::RAY_INTERSECTION);

    if (best_j >= 0) {
        // move to the next point (otherwise, stay)
        IndexSet next_vertex = edge.append(best_j);
        dvector next_ref = get_kernel().move_along_ray(vertex.ref, u, best_l);
        if (validate(next_vertex, next_ref)) {
            #pragma omp atomic
            validations_ok++;
            return Polytope(next_vertex, next_ref);
        } else {
            #pragma omp atomic
            validations_failed++;
            return NONE;
        }
    }

    return NONE;
}

vec<VoronoiGraph::Polytope> VoronoiGraph::get_neighbors(const VoronoiGraph::Polytope &vertex,
                                                        RandomEngine &re) const {
    vec<Polytope> result;
    for (int i = 0; i <= data_dim; i++) {
        result.push_back(get_neighbor(vertex, i, re));
    }
    return result;
}

VoronoiGraph::Polytope
VoronoiGraph::cast_ray(const VoronoiGraph::Polytope &p, const dvector &direction,
                       ptr<const vec<dvector>> orthogonal_complement, bool any_direction,
                       ftype *length) const {
    ensure(!p.is_none(), "p should not be NONE");
    if (!orthogonal_complement) {
        vec<dvector> normals;
        for (int i = 1; i < p.dual.dim(); i++) {
            dvector v = points.col(p.dual[i]) - points.col(p.dual[0]);
            for (const dvector &norm : normals) {
                v = v - v.dot(norm) * norm;
            }
            v.normalize();
            normals.push_back(v);
        }
        orthogonal_complement = std::make_shared<const vec<dvector>>(normals);
    }
    dvector u = direction;
    for (const dvector &norm : *orthogonal_complement) {
        u = u - u.dot(norm) * norm;
    }
    get_kernel().project_to_tangent_space_inplace(p.ref, u);
    u.normalize();

    int best_j = -1;
    ftype best_l = 0;
    get_kernel().intersect_ray(strategy, p.ref, u, p.dual[0], p.dual,
                               &best_j, &best_l,
                               any_direction ? Kernel::ANY_INTERSECTION : Kernel::RAY_INTERSECTION);

    if (length) {
        *length = best_l;
    }

    if (best_j < 0) {
        return NONE;
    }

    IndexSet new_dual = p.dual.append(best_j);
    dvector new_ref = get_kernel().move_along_ray(p.ref, u, best_l);

    if (!validate(new_dual, new_ref)) {
        #pragma omp atomic
        validations_failed++;
        return NONE;
    } else {
        #pragma omp atomic
        validations_ok++;
        return Polytope(new_dual, new_ref);
    }
}

bool VoronoiGraph::validate(const vec<int> &is, const dvector &ref) const {
    Kernel &k = get_kernel();
    ftype min_dist = k.distance(ref, points.col(is[0]));
    ftype max_dist = min_dist;
    for (size_t i = 1; i < is.size(); i++) {
        ftype dst = k.distance(ref, points.col(is[i]));
        min_dist = std::min(min_dist, dst);
        max_dist = std::max(max_dist, dst);
    }
//    std::cout << min_dist << " " << max_dist << std::endl;
    if (max_dist - min_dist > VALIDATION_EPS) {
        return false;
    }
    return get_kernel().nearest_point_extra(ref, max_dist - VALIDATION_EPS, is) < 0;
}

const dmatrix& VoronoiGraph::get_data() const {
    return points;
}

int VoronoiGraph::get_data_size() const {
    return n_points;
}

int VoronoiGraph::get_data_dim() const {
    return data_dim;
}

void VoronoiGraph::reset_failure_rate_counters() const {
    validations_ok = 0;
    validations_failed = 0;
}

std::pair<long long, long long> VoronoiGraph::get_failure_rate() const {
    return std::make_pair(validations_ok, validations_failed);
}

void VoronoiGraph::print_validations_info() const {
    printf("Validations failed: %lld out of %lld (%.2f%%)\n",
           validations_failed, validations_ok + validations_failed,
           100 * float(validations_failed)/float(validations_ok + validations_failed));
    fflush(stdout);
}

RayStrategyType VoronoiGraph::get_strategy() const {
    return strategy;
}

DataType VoronoiGraph::get_data_type() const {
    return data_type;
}
