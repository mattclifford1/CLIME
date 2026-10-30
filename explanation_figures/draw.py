'''Drawing helpers shared by the figures, so a style change made here reaches every figure.'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>
import numpy as np
from sklearn.decomposition import PCA

import style


class Projection:
    '''2D view of the data: identity for 2D data, else PCA oriented so the query line runs left to right.'''

    def __init__(self, X, query_points=None):
        self.pca = None
        self.sign = 1.0
        if X.shape[1] > 2:
            self.pca = PCA(n_components=2, svd_solver='full').fit(X)
            if query_points is not None:
                ends = self.pca.transform(query_points[[0, -1]])
                self.sign = 1.0 if ends[0, 0] <= ends[1, 0] else -1.0
        self.labels = ('Feature 1', 'Feature 2') if self.pca is None else (
            'PCA component 1', 'PCA component 2')

    def transform(self, X):
        if self.pca is None:
            return np.asarray(X)
        Z = self.pca.transform(X)
        Z[:, 0] *= self.sign
        return Z

    def inverse(self, Z):
        if self.pca is None:
            return Z
        Z = np.array(Z, copy=True)
        Z[:, 0] *= self.sign
        return self.pca.inverse_transform(Z)


def limits(Z, pad=0.6):
    lo, hi = Z.min(axis=0) - pad, Z.max(axis=0) + pad
    return (lo[0], hi[0]), (lo[1], hi[1])


def surface(ax, predict_proba, proj, lims, res=200, boundary=True):
    '''Shade P(class 1) of the black box over the 2D view, with its 0.5 contour.'''
    (x0, x1), (y0, y1) = lims
    xx, yy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    p = predict_proba(proj.inverse(np.c_[xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap=style.PROBA_CMAP,
                alpha=style.SURFACE_ALPHA, zorder=0)
    if boundary:
        ax.contour(xx, yy, p, levels=[0.5], colors=style.INK, linewidths=0.9, zorder=1)
    ax.set(xlim=(x0, x1), ylim=(y0, y1))


def classes(ax, Z, y, size=10, alpha=0.8):
    for label in (0, 1):
        z = Z[np.asarray(y) == label]
        ax.scatter(z[:, 0], z[:, 1], s=size, color=style.CLASS[label], alpha=alpha,
                   edgecolors=style.INK, linewidths=0.3, zorder=2)


def query_points(ax, Z, label_every=4, size=18):
    ax.scatter(Z[:, 0], Z[:, 1], s=size, color=style.QUERY, edgecolors=style.INK,
               linewidths=0.6, zorder=4)
    for i in range(0, len(Z), label_every):
        ax.annotate(str(i), Z[i], xytext=(0, 4), textcoords='offset points', ha='center',
                    va='bottom', fontsize=7, zorder=5)


def fidelity_lines(ax, curves, ylim=(0.4, 1.0)):
    '''curves: {explainer registry name: per-query-point scores}.'''
    for name, scores in curves.items():
        x = np.arange(len(scores))
        ax.plot(x, scores, color=style.SCHEME_COLOURS[name], marker='o', ms=2.5,
                label=style.SCHEME_LABELS[name])
    ax.set(ylim=ylim, xlabel='Query point', ylabel='Local fidelity')
    ax.set_xticks(np.arange(0, len(scores), 4))
