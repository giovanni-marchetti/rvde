#pragma once

#include "../utils.h"

using edge_matrix = Eigen::Matrix<int, 2, Eigen::Dynamic, Eigen::ColMajor>;

edge_matrix get_gabriel_edges(const dmatrix &points);

edge_matrix get_gabriel_edges_subset(const dmatrix &points, int size, int seed=21203, ftype *nedges_estimate=nullptr);

