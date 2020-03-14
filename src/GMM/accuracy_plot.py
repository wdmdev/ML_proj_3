import sys
sys.path.insert(1, '../')
from data_cleaner import create_dataset

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from matplotlib import rcParams
rcParams['figure.figsize'] = 16, 8

def plot_accuracy(seed):
    colors = ['navy', 'darkorange']

    target_names = ['High Risk', 'Low Risk']

    def make_ellipses(gmm, ax):
        for n, color in enumerate(colors):
            if gmm.covariance_type == 'full':
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == 'tied':
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == 'diag':
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == 'spherical':
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                      180 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect('equal', 'datalim')

    X, y = create_dataset()
    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # Only take the first fold.
    train_index, test_index = next(iter(skf.split(X,y)))


    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    n_classes = len(np.unique(y_train))

    # Try GMMs using different types of covariances.
    estimators = {cov_type: GaussianMixture(n_components=n_classes,
                  covariance_type=cov_type, max_iter=20, random_state=0)
                  for cov_type in ['spherical', 'diag', 'tied', 'full']}

    n_estimators = len(estimators)

    plt.figure(figsize=(3 * n_estimators // 2, 6))
    plt.subplots_adjust(bottom=.05, top=0.95, hspace=.15, wspace=.05,
                        left=.02, right=.98)


    for index, (name, estimator) in enumerate(estimators.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                        for i in range(n_classes)])

        # Train the other parameters using the EM algorithm.
        estimator.fit(X_train)

        h = plt.subplot(2, n_estimators // 2, index + 1)
        make_ellipses(estimator, h)

        for n, color in enumerate(colors):
            data = X[y == n]
            plt.scatter(data[:, 0], data[:, 1], s=2, color=color,
                        label=target_names[n])
        # Plot the test data with crosses
        for n, color in enumerate(colors):
            data = X_test[y_test == n]
            plt.scatter(data[:, 0], data[:, 1], s=20*8, marker='x', color=color)

        y_train_pred = estimator.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes, fontsize=(18))

        y_test_pred = estimator.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 transform=h.transAxes, fontsize=(18))

        plt.title('GMM: K=2, Cv-type = {}'.format(name))

    plt.legend(scatterpoints=40, loc='lower right', prop=dict(size=25))


    plt.show()