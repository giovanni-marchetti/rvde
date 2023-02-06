import os.path

import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import eig
import math
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, SVHN
from torchvision import transforms
import torch
from numpy import random
from scipy.optimize import linear_sum_assignment


def bw(data, factor):
    # mean = np.mean(data, axis=0)
    # tmp = (data - mean[None, :])
    # std = np.mean(np.sqrt(np.sum(tmp * tmp, axis=0) / (data.shape[0])))
    # return std * factor
    cov = np.cov(data.T)
#    print('covariance', cov.shape)
    eigval, _ = eig(cov)
    #print('eigen', eigval)
    tmp = np.real(np.sqrt(np.mean(eigval)))
    #print('tmp', tmp)
    return factor * tmp



def scott_bw(data):
    n = data.shape[0]
    d = data.shape[1]
    factor = n ** (-1. / (d + 4))
    """Computes the covariance matrix for each Gaussian kernel using
    covariance_factor().
    """
    return bw(data, factor)



def Wasserstein(set1, set2):
    distance_matrix = ((np.expand_dims(set1, axis=1) - np.expand_dims(set2, axis=0))**2).sum(-1)
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    cost = distance_matrix[row_ind, col_ind].mean()

    return cost


class Laplace:
    def __init__(self, dim, sigma):
        self.name = 'single'
        self.dim = dim
        self.sigma = sigma

    def sample(self, n, train=False, ood=False):
        return np.random.laplace(size=(n, self.dim))

    def logpdf(self, points):
        return 0


class Dirichlet:
    def __init__(self, dim, sigma):
        self.name = 'single'
        self.dim = dim
        self.sigma = sigma

    def sample(self, n, train=False, ood=False):
        return np.random.dirichlet(np.ones((self.dim,)) / (self.dim + 1), n)

    def logpdf(self, points):
        return 0


class Gaussian:
    def __init__(self, dim, sigma):
        self.name = 'single'
        self.dim = dim
        self.sigma = sigma

        self.distr = multivariate_normal(np.zeros(dim), np.eye(dim) * self.sigma * self.sigma)

    def sample(self, n, train=False, ood=False):
        return np.random.normal(0, self.sigma, size=(n, self.dim))

    def logpdf(self, points):
        return np.log(self.distr.pdf(points))


class TwoGaussians:
    def __init__(self, dim, s1, s2, dst, alpha=0.5):
        self.name = 'double'
        self.dim = dim
        self.s1 = s1
        self.s2 = s2
        self.dst = dst
        self.alpha = alpha

        mean1 = np.zeros(dim)
        mean1[0] -= dst * 0.5
        mean2 = np.zeros(dim)
        mean2[0] += dst * 0.5
        self.distr1 = multivariate_normal(mean1, np.eye(dim) * self.s1 * self.s1)
        self.distr2 = multivariate_normal(mean2, np.eye(dim) * self.s2 * self.s2)

    def sample(self, n, train=False, ood=False):
        which = np.random.random(n)
        result = np.zeros((n, self.dim))
        g1 = np.random.normal(0, self.s1, size=result.shape) + self.distr1.mean[None, :]
        g2 = np.random.normal(0, self.s2, size=result.shape) + self.distr2.mean[None, :]
        result[which < self.alpha] = g1[which < self.alpha]
        result[which >= self.alpha] = g2[which >= self.alpha]
        return result

    def logpdf(self, points):
        pdf1 = self.distr1.pdf(points)
        pdf2 = self.distr2.pdf(points)
        return np.log(self.alpha * pdf1 + (1 - self.alpha) * pdf2)


class FatSphere:
    def __init__(self, dim, R, r):
        self.name = 'sphere'
        self.dim = dim
        self.R = R
        self.r = r

        self.vol = None
        if dim % 2 == 0:
            k = dim // 2
            self.logvol = k * math.log(math.pi) - math.log(math.factorial(k)) + math.log((R+r)**dim - (R-r)**dim)
        else:
            k = (dim - 1) // 2
            self.logvol = math.log(2 * math.factorial(k)) + k * math.log(4 * math.pi) - math.log(math.factorial(dim)) +\
                math.log((R+r)**dim - (R-r)**dim)

    def sample(self, n, train=False, ood=False):
        def gen_sphere(n, d, r):  # (d-1)-dim sphere
            directions = np.random.normal(0, 1, (n, d))
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            radii = r

            data_gpu = directions * radii
            return data_gpu

        def gen_ball(n, d, r):
            directions = np.random.normal(0, 1, (n, d))
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            radii = r * (np.random.random((n, 1)) ** (1. / d))

            data = directions * radii
            return data

        sphere = gen_sphere(n, self.dim, self.R)
        noise = gen_ball(n, self.dim, self.r)
        return sphere + noise

    def logpdf(self, points):
        rad = np.linalg.norm(points, axis=1)
        result = np.ones(points.shape[0]) * (-np.inf)
        result[(rad > self.R - self.r) & (rad < self.R + self.r)] = -self.logvol
        return result


class mnist:
    def __init__(self, dim_red=None, label=None):
        transform = transforms.Compose([transforms.ToTensor()])
        def parse_data(dset):
            images, labels = [], []
            for i in range(len(dset)):
                img, lbl = dset[i]
                images.append(img.flatten().numpy())
                labels.append(lbl)
            images, labels = np.array(images), np.array(labels)
            if label is not None:
                images = images[labels == label]
                labels = labels[labels == label]
            return images, labels
        self.train_data, self.train_labels = parse_data(MNIST('./datasets/MNIST/', download=True,
                                                              transform=transform, train=True))
        self.test_data, self.test_labels = parse_data(MNIST('./datasets/MNIST/', download=True,
                                                            transform=transform, train=False))
        # self.test_data_ood = FashionMNIST('./datasets/FashionMNIST/', download=True,
        #     transform=transform, train=False)

        print(self.train_data.shape)

    def sample(self, n, train=False, ood=False):
        if train:
            dset = self.train_data
        else:
            dset = self.test_data

        idxs = random.permutation(dset.shape[0])
        res = dset[idxs]
        if n > 0:
            return res[:n]
        else:
            return res

class frogs:
    def __init__(self, lbl1=-1, lbl2=-1):
        lbls = np.load('./datasets/frog_calls/frogs_lbls.npy')
        data = np.load('./datasets/frog_calls/frogs_data.npy')
        # print(np.unique(lbls, return_counts=True))
        # exit()
        # import matplotlib.pyplot as plt
        # plt.plot(range(lbls.shape[0]), lbls)
        # plt.show()
        indices = np.random.permutation(data.shape[0])
        data = data[indices]
        lbls = lbls[indices]
        if lbl1 >= 0:
            id_data = data[lbls==lbl1]
            if lbl2 >= 0:
                ood_data = data[lbls==lbl2]
            else:
                ood_data = data[~(lbls == lbl1)]
        else:
            id_data = data
            ood_data = data  # hopefully not used
        l1 = len(id_data)
        print(f'{l1=}')
        l2 = len(ood_data)
        self.train_data = id_data[: -int(l1/10)]
        self.test_data = id_data[-int(l1/10) :]
        self.test_data_ood = ood_data[-int(l1/10) :]

    def sample(self, n, train=False, ood=False):
        res = []
        if train:
            dset_full = self.train_data
            l = len(dset_full)
            dset = dset_full[random.permutation(l)][:int(l/2)]

        else:
            dset = self.test_data
        if ood:
            dset = self.test_data_ood

        if n > 0:
            return dset[:n]
        else:
            return dset
