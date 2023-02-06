import skimage.feature
import numpy as np
import sklearn


class AbstractEmbedding:
    def __init__(self, dim):
        self.dim = dim

    def fit_transform(self, data):
        pass


class Identity(AbstractEmbedding):
    def __init__(self, dim):
        super().__init__(dim)

    def fit_transform(self, data):
        return data


class HOG(AbstractEmbedding):
    def __init__(self, dim, orientations=9):
        super().__init__(dim)
        self.orientations = orientations

    def fit_transform(self, data):
        images = data.view().reshape((data.shape[0], 28, 28))
        features = np.array([np.sum(skimage.feature.hog(im, feature_vector=False, orientations=self.orientations),
                                    axis=(0, 1, 2, 3)) for im in images])
        return features / np.max(features)

class Isomap(AbstractEmbedding):
    def __init__(self, dim):
        from sklearn.manifold import Isomap as SkIsomap
        super().__init__(dim)
        self.iso = SkIsomap(n_components=dim)

    def fit_transform(self, data):
        data = self.iso.fit_transform(data)
        print(np.max(data))
        return np.copy(data.astype(np.float128))

class PickFirstN(AbstractEmbedding):
    def __init__(self, dim):
        super().__init__(dim)
        self.n = dim

    def fit_transform(self, data):
        return data[:, :self.n]


class PCA(AbstractEmbedding):
    def __init__(self, dim):
        super().__init__(dim)
        self.pca = sklearn.decomposition.PCA(dim)

    def fit_transform(self, data):
        data = self.pca.fit_transform(data)
        return data

class Resize(AbstractEmbedding):
    def __init__(self, dim):
        super().__init__(dim)
        self.side = int(np.sqrt(dim))

    def fit_transform(self, data):
        orig_side = int(np.sqrt(data.shape[1]))
        from skimage.transform import resize as skresize
        n = data.shape[0]
        return np.reshape(skresize(np.reshape(data, (n, orig_side, orig_side)), (n, self.side, self.side)), (n, self.side*self.side))

