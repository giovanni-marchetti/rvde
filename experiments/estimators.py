import numpy as np
import vgt
from numpy import random
from sklearn.neighbors import NearestNeighbors
from scipy.special import gamma

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

def gen_seed(seed=None):
    if seed is None:
        seed = int(np.random.randint(0, 65536))
    return seed

class Estimator:
    def __init__(self):
        pass

    def logdensity(self, data):
        pass

    def sample(self, n):
        pass


class VGTEstimator(Estimator):
    def __init__(self, est):
        self.est = est

    def logdensity(self, data):
        # return np.log(self.est.estimate(np.copy(data)))
        density = self.est.estimate(np.copy(data).astype(np.float128))
        # print(density.dtype, density)
        return np.log(density)

    def sample(self, n):
        return self.est.sample(n)


class VDE(VGTEstimator):
    def __init__(self, train_data, ck=None, band=None, kind=None, seed=None, num_threads=16, num_rays=1000, num_steps=10): 
        dim = train_data.shape[1]
        if ck is None:
            if kind in ['exponential', 'e', 'exp']:
                ck = vgt.LaplaceCellKernel(dim, band)
            elif kind in ['rational', 'r', 'rat']:
                ck = vgt.PolynomialCellKernel(dim, band, dim + 3)
            elif kind in ['gaussian', 'g']:
                ck = vgt.GaussianCellKernel(dim, band)
            else:
                raise Exception(f'Unknown kernel: {kind}')
        super(VDE, self).__init__(
            vgt.VoronoiDensityEstimator(np.copy(train_data), ck, gen_seed(seed), num_threads, num_rays, num_steps,
                                        vgt.RayStrategyType.BRUTE_FORCE_GPU, vgt.Unbounded())) # todo gpu here
        self.est.initialize_volumes()


def make_cvde(kind):
    #dim = train_data.shape[1]
    # return VDE(train_data, vgt.GaussianCellKernel(dim, band))
    # return VDE(train_data, vgt.LaplaceCellKernel(dim, band/5))
    #return VDE(train_data, vgt.PolynomialCellKernel(dim, band, dim + 3))
    
    def constructor(train_data, band):
        return VDE(train_data, band=band, kind=kind)

    return constructor



def make_contvde(kind):
    def constructor(train_data, band, band_is_alpha=False):
        dim = train_data.shape[1]
        if band_is_alpha:
            alpha = band
        else:
            alpha = bw_to_alpha(kind, band, dim)
        # print(f'{band=} {alpha=}')

        if kind in ['exponential', 'e', 'exp']:
            ck = vgt.BalancedExponentialCellKernel(dim, alpha, 1.)
        elif kind in ['rational', 'r', 'rat']:
            ck = vgt.BalancedPolynomialCellKernel(dim, alpha, dim + 3)
        else:
            raise Exception(f'Unknown kernel: {kind}')
        # Some options:
        # vgt.BalancedPolynomialCellKernel(dim, alpha, k) : (1 + t)^{-k}
        # vgt.BalancedExponentialCellKernel(dim, alpha, k) : exp(-sgn(beta) * |t|^k)
        # ^^^^^ NOTE: currently only works when k divides dim ^^^^^
        # ck = vgt.BalancedExponentialCellKernel(dim, alpha, 1)
        # ck = vgt.BalancedPolynomialCellKernel(dim, alpha, dim + 1)
        # ck = vgt.BalancedSecondPolynomialCellKernel(dim, alpha)
        # print(f'${ck.latex()}$')

        return VDE(train_data, ck)
    return constructor


class KDE(VGTEstimator):
    def __init__(self, train_data, band, seed=None, num_threads=16, kind=None):
        if kind is None:
            super(KDE, self).__init__(vgt.KDE(np.copy(train_data).astype(np.float128),
                                              band,
                                              # ck,
                                              gen_seed(seed), num_threads))
            return
        dim = train_data.shape[1]
        if kind in ['exponential', 'e', 'exp']:
            ck = vgt.LaplaceCellKernel(dim, band)
        elif kind in ['rational', 'r', 'rat']:
            ck = vgt.PolynomialCellKernel(dim, band, dim + 3)
        elif kind in ['gaussian', 'g']:
            ck = vgt.GaussianCellKernel(dim, band)
        else:
            raise Exception(f'Unknown kernel: {kind}')
        # ck = vgt.GaussianCellKernel(dim, band)
        # ck = vgt.LaplaceCellKernel(dim, band/5)
        # ck = vgt.PolynomialCellKernel(dim, band, dim+3)
        # super(KDE, self).__init__(vgt.KDE(np.copy(train_data).astype(np.float128), band, gen_seed(seed), num_threads))
        super(KDE, self).__init__(vgt.KDE(np.copy(train_data).astype(np.float128),
                                          # band,
                                          ck,
                                          gen_seed(seed), num_threads))


class AWKDE(KDE):
    def __init__(self, train_data, band, seed=None, num_threads=16, alpha=0.5, kind=None):
        super().__init__(train_data, band, gen_seed(seed), num_threads, kind=kind)
        self.est.make_adaptive(alpha)


def make_kde(adaptive, kind=None):
    DE = AWKDE if adaptive else KDE

    def constructor(train_data, band):
        return DE(train_data, band, kind=kind)

    return constructor


class Orig_KDE(Estimator):
    def __init__(self, train_data, band):
        from sklearn.neighbors import KernelDensity
        self.kde = KernelDensity(kernel='gaussian', bandwidth=band).fit(train_data)

    def logdensity(self, samples):
        logd = self.kde.score_samples(samples)
        #print(np.exp(logd))
        #exit()
        return logd

    def sample(self, n):
        return self.kde.sample(n)


class Orig_AWKDE(Estimator):
    def __init__(self, train_data, band):
        import awkde
        self.kde = awkde.GaussianKDE(glob_bw=band)
        self.kde.fit(train_data)

    def logdensity(self, samples):
        return np.log(self.kde.predict(samples))

    def sample(self, n):
        return self.kde.sample(n)

