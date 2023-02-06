import scipy.special

from estimators import *
from distributions import *
import numpy as np
from embeddings import *
from tqdm import tqdm
import argparse
import os

# np.random.seed(243)

def alpha_by_gabriel(points, percentile):
    import vgt
    edges = vgt.get_gabriel_edges(points)
    halflengths = np.linalg.norm(points[edges[:, 0], :] - points[edges[:, 1], :], axis=1) / 2
    dim = points.shape[1]
    length = np.percentile(halflengths, percentile * 100)
    return (length ** dim) / dim

def alpha_by_gabriel2(points, percentile):
    import vgt
    edges = vgt.get_gabriel_edges(points)
    print(edges.shape)
    edges = np.vstack((edges, edges[:, [1, 0]]))
    print(edges.shape)
    halflengths = np.linalg.norm(points[edges[:, 0], :] - points[edges[:, 1], :], axis=1) / 2
    dim = points.shape[1]
    length = np.percentile(halflengths, percentile * 100)
    return (length ** dim) / dim


def alpha_by_gabriel3(points, _):
    import vgt
    nv = points.shape[0]
    if nv < 10000:
        edges = vgt.get_gabriel_edges(points)
        ne = edges.shape[0]
    else:
        edges, ne = vgt.get_gabriel_edges_subset(points, 100000, np.random.randint(0, 65536))
    halflengths = np.linalg.norm(points[edges[:, 0], :] - points[edges[:, 1], :], axis=1) / 2
    dim = points.shape[1]
    percentile = min(1, nv / ne)
    print(f'{percentile=}')
    length = np.percentile(halflengths, percentile * 100)
    return (length ** dim) / dim


def alpha_poly(points, k):
    import vgt
    dim = points.shape[1]

    # mean = np.mean(points, axis=0, keepdims=True)
    # norm_sq = points - mean
    # norm_sq = np.sum(norm_sq * norm_sq, axis=1)
    # cov = np.mean(norm_sq)

    edges = vgt.get_gabriel_edges(points)
    lengths = np.linalg.norm(points[edges[:, 0], :] - points[edges[:, 1], :], axis=1)
    cov = np.mean(lengths ** 2)

    # import scipy.spatial.distance
    # dist_mat = scipy.spatial.distance.cdist(points, points)
    # dist_mat = dist_mat**2
    # n = dist_mat.shape[0]
    # dist_mat[range(n), range(n)] = np.inf
    # cov = np.mean(np.min(dist_mat, axis=1))

    # import scipy.spatial.distance
    # dist_mat = scipy.spatial.distance.pdist(points)
    # dist_mat = dist_mat**2
    # cov = np.mean(dist_mat)

    # int1 = vgt.polynomial_integral(dim - 1, k, 1., np.float128('inf'))
    # int2 = vgt.polynomial_integral(dim + 1, k, 1., np.float128('inf'))
    int1 = vgt.polynomial_integral(dim - 1, k, 1., np.sqrt(cov))
    int2 = vgt.polynomial_integral(dim + 1, k, 1., np.sqrt(cov))
    print(f'{cov=} {int1=} {int2=}')
    return (cov * int1 / int2)**(dim/2) * int1
    # return (cov ** (dim/2)) * (int1 ** (dim/2 + 1)) / (int2 ** (dim/2))

def alpha_exp(points):
    import vgt
    dim = points.shape[1]
    mean = np.mean(points, axis=0, keepdims=True)
    norm_sq = points - mean
    norm_sq = np.sum(norm_sq * norm_sq, axis=1)
    cov = np.mean(norm_sq)
    int1 = scipy.special.gamma(dim - 0)
    int2 = scipy.special.gamma(dim + 2)
    print(f'{cov=} {int1=} {int2=}')
    return (cov * int1 / int2)**(dim/2) * int1
    # return (cov ** (dim/2)) * (int1 ** (dim/2 + 1)) / (int2 ** (dim/2))

# def alpha_to_bw(alpha, dim):
#     return (alpha * 2 / (scipy.special.gamma(dim / 2))) ** (1 / dim)

def bw_to_alpha(kind, band, dim):
    if kind in ['exponential', 'e', 'exp', 'contvde_e']:
        ck = vgt.BalancedExponentialCellKernel(dim, 1., 1.)
    elif kind in ['rational', 'r', 'rat', 'contvde_r']:
        ck = vgt.BalancedPolynomialCellKernel(dim, 1., dim + 3)
    else:
        raise Exception(f'Unknown kernel: {kind}')
    # alpha = ck.cone(1/band, np.longdouble('inf'))
    alpha = (band ** dim) * ck.cone(1., np.longdouble('inf'))
    return alpha

def alpha_to_bw(kind, alpha, dim):
    if kind in ['exponential', 'e', 'exp', 'contvde_e']:
        ck = vgt.BalancedExponentialCellKernel(dim, 1., 1.)
    elif kind in ['rational', 'r', 'rat', 'contvde_r']:
        ck = vgt.BalancedPolynomialCellKernel(dim, 1., dim + 3)
    else:
        raise Exception(f'Unknown kernel: {kind}')
    bw = (alpha / ck.cone(1., np.longdouble('inf'))) ** (1 / dim)
    return bw

parser = argparse.ArgumentParser()
parser.add_argument('estimator', type=str)
# parser.add_argument('estimator', choices=['kde', 'cvde', 'awkde', 'contvde_e', 'contvde_r', 'orig_kde'])
parser.add_argument('dataset', type=str)
# parser.add_argument('dataset', choices=['gaussian', 'gaussians', 'mnist', 'mnist1', 'frogs', 'mf', 'sphere'])
parser.add_argument('--reduce', choices=['none', 'pca', 'isomap', 'hog', 'resize', 'pick'], default='none')
parser.add_argument('--kind', choices=['r', 'e', 'g'], default='r')  # g = gaussian1/1
parser.add_argument('--dim', type=int, default=None)
parser.add_argument('--band_low', type=float, default=None)
parser.add_argument('--band_high', type=float, default=None)
parser.add_argument('--extra', action='store_true')
parser.add_argument('--no-extra', dest='extra', action='store_false')
parser.set_defaults(extra=False)
parser.add_argument('--alpharange', action='store_true')
parser.add_argument('--no-alpharange', dest='alpharange', action='store_false')
parser.set_defaults(alpharange=False)
parser.add_argument('--logscale', action='store_true')
parser.add_argument('--no-logscale', dest='logscale', action='store_false')
parser.set_defaults(logscale=False)
args = parser.parse_args()
estimator_name = args.estimator
kind = args.kind
dataset_name = args.dataset
embedding_name = args.reduce
dim = args.dim
band_low = args.band_low
band_high = args.band_high
is_alpha_range = args.alpharange
is_logscale = args.logscale

# estimator_name = 'contvde'  # kde | cvde | awkde | contvde | orig_kde | orig_awkde
# dataset_name = 'mnist'  # gaussians | mnist | frogs
# embedding_name = 'none'  # none | pca | hog | isomap
# dim = None
# band_low = None
# band_high = None

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

num_bands = 100
runs = 5
suggested = args.extra
alpha_dict = {}
alpha_dict['suggested_alpha'] = []
alpha_dict['suggested_bw'] = []

if dataset_name == 'gaussian':
    n_train = 1000
    n_test = 1000
    if band_low is None: band_low = .1
    if band_high is None: band_high = 5
    distr = Gaussian(10, 1.)
elif dataset_name == 'gaussians':
    n_train = 1000
    n_test = 1000
    if band_low is None: band_low = .1
    if band_high is None: band_high = 10
    # distr = TwoGaussians(10, 10, .1, 1, alpha=0.5)
    distr = TwoGaussians(10, 10, .1, 1, alpha=0.5)
elif dataset_name == 'laplace':
    n_train = 1000
    n_test = 1000
    if band_low is None: band_low = .1
    if band_high is None: band_high = 10
    distr = Laplace(10, 1)
elif dataset_name == 'dirichlet':
    n_train = 1000
    n_test = 1000
    if band_low is None: band_low = .1
    if band_high is None: band_high = 10
    distr = Dirichlet(10, 1)
elif dataset_name == 'mnist':
    n_train = 30000  # 30000
    n_test = -1  # -1 # 10000
    if band_low is None: band_low = .1
    if band_high is None: band_high = 1.
    # band_list = np.linspace(start=.01, stop=.6, num=num_bands)
    distr = mnist()
elif dataset_name == 'frogs':
    n_train = 3238
    n_test = 719
    n_train = -1
    n_test = -1
    if band_low is None: band_low = .01
    if band_high is None: band_high = .2
    distr = frogs()
elif dataset_name == 'sphere':
    n_train = 1000
    n_test = 1000
    if band_low is None: band_low = .001
    if band_high is None: band_high = 1000.
    R = 1
    r = 0.05
    D = 5
    os.makedirs('datasets/mf/', exist_ok=True)
    distr = FatSphere(D, R, r)
else:
    raise Exception(f'Unknown dataset: {dataset_name}')

assert band_low is not None
assert band_high is not None

if is_logscale:
    band_list = np.geomspace(start=band_low, stop=band_high, num=num_bands)
else:
    band_list = np.linspace(start=band_low, stop=band_high, num=num_bands)


estimators = {
    'kde': make_kde(False, kind),
    'cvde': make_cvde(kind),
    'awkde': make_kde(True, kind),
    'contvde': make_contvde(kind),
    'orig_kde': Orig_KDE,
    'orig_awkde': Orig_AWKDE
}

embeddings = {
    None: Identity,
    'none': Identity,
    'pca': PCA,
    'isomap': Isomap,
    'hog': HOG,
    'resize': Resize,
    'pick': PickFirstN
}

embedding = embeddings[embedding_name](dim)

out_fname = f'{output_dir}/{estimator_name}_{kind}_{dataset_name}_{embedding_name}_{dim}.npz'
alpha_out_fname = f'{output_dir}/alphas_{estimator_name}_{kind}_{dataset_name}_{embedding_name}_{dim}.npz'

# Sample test once
test_data_orig = distr.sample(n_test, train=False)

scores = []
silencing_flag = False
for run in range(runs):
    scores.append([])

    train_data = distr.sample(n_train, train=True)
    # test_data = distr.sample(n_test, train=False)
    test_data = test_data_orig
    d = train_data.shape[-1]

    n = train_data.shape[0]
    reduced = embedding.fit_transform(np.concatenate((train_data, test_data), axis=0))
    train_data = reduced[:n]
    test_data = reduced[n:]

    if suggested:
        if estimator_name not in ['contvde']:
            print('Skipping')
            exit(0)
        d = train_data.shape[1]
        if kind == 'r':
            # suggested_alpha = alpha_poly(train_data, d+3)
            suggested_alpha = alpha_by_gabriel3(train_data, 0.01)
            # suggested_alpha = alpha_by_gabriel3(train_data, 0.01)
        elif kind == 'e':
            # suggested_alpha = alpha_exp(train_data)
            suggested_alpha = alpha_by_gabriel3(train_data, 0.01)
        suggested_bw = alpha_to_bw(kind, suggested_alpha, d)
        alpha_dict['suggested_alpha'].append(suggested_alpha)
        alpha_dict['suggested_bw'].append(suggested_bw)
        print(f' -------->>> {suggested_alpha=} bw={suggested_bw} <<<--------')

        if run == 0:
            # alpha_list = [vgt.gaussian_bw_to_alpha(band, d) for band in band_list]
            alpha_list = [bw_to_alpha(kind, band, d) for band in band_list]
            alpha_dict['alpha_list'] = alpha_list
            alpha_dict['band_list'] = band_list

    for b_i, band in enumerate(tqdm(band_list, desc=f'run {run + 1}/{runs}')):
        if not silencing_flag: silencing_flag = True
        else: vgt.set_silent(True)

        try:
            if is_alpha_range:
                est = estimators[estimator_name](train_data, band, band_is_alpha=True)
            else:
                est = estimators[estimator_name](train_data, band)

            if suggested:
                continue

            scores_cur = est.logdensity(test_data)
            scores[run].append(scores_cur)
        except BaseException as ex:
            print(ex)
            if estimator_name == 'cvde':
                logname = f'log_{np.random.randint(0, 65535)}.npz'
                print(f'Log saved at "{logname}"')
                np.savez(logname,
                         seed=est.est.get_initial_seed(),
                         train_data=train_data, band=band, test_data=test_data)
            exit()


        try:
            np.savez(out_fname,
                     scores=np.array(scores),
                     band=band_list,
                     dim=dim,
                     runs=run + 1)
        except:
            pass


if suggested:
    alpha_m = np.mean(alpha_dict['suggested_alpha'])
    alpha_sd = np.std(alpha_dict['suggested_alpha'])
    alpha_dict['suggested_bw_m'] = alpha_to_bw(kind, alpha_m, d)
    alpha_dict['suggested_bw_l'] = alpha_to_bw(kind, alpha_m - alpha_sd, d)
    alpha_dict['suggested_bw_r'] = alpha_to_bw(kind, alpha_m + alpha_sd, d)
    alpha_dict['data_dim'] = d
    np.savez(alpha_out_fname, **alpha_dict)
    vgt.set_silent(False)
    exit(0)


scores = np.array(scores)

np.savez(out_fname,
         scores=scores,
         band=band_list,
         dim=dim,
         runs=runs)

vgt.set_silent(False)
