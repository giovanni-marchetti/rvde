#pragma once

#include "utils.h"
#include "IndexSet.h"

class Lookup {
public:
    explicit Lookup(const dmatrix &data);

    virtual void init();

    virtual int find_nn(const dvector &x, ftype *best_dist_sqr,
                        ftype margin_sqr = -1, const vec<int> &ignore = vec<int>()) const = 0;

    virtual void update_inserted_points();

protected:
    const dmatrix &data;

};

class KDTree : public Lookup {
private:
    class Node;
    typedef int pNode;

public:
    explicit KDTree(const dmatrix &data, int leaf_len = 5);

    void init() override;

    int find_nn(const dvector &x, ftype *best_dist_sqr, ftype margin_sqr = -1,
                const vec<int> &ignore = vec<int>()) const override;

    void update_inserted_points() override;

private:
    pNode build_tree(int l, int r, int depth);
    void find_nn(const dvector &x, pNode node, int *best, ftype *best_dist_sqr,
                 ftype margin_sqr, ftype cur_bin_dist_sqr, const vec<int> &ignore,
                 vec<ftype> &partial_dist_sqr) const;

    pNode get_containing_node(const dvector &x) const;

    pNode make_node(int lidx, int ridx);

    pNode make_node(int dim, ftype m);

private:
    int n;
    int d;
    pNode head;
    int leaf_len;

    vec<int> indices;
    vec<Node> nodes;

    struct Node {
        Node(int lidx, int ridx);
        Node(int dim, ftype m);

        int lidx, ridx;
        vec<int> extra;

        int dim;
        ftype m;

        pNode left = -1, right = -1;

        int size();
    };
};

class BruteForceLookup : public Lookup {
public:
    BruteForceLookup(const dmatrix &data);

    int find_nn(const dvector &x, ftype *best_dist_sqr, ftype margin_sqr, const vec<int> &ignore) const override;
};