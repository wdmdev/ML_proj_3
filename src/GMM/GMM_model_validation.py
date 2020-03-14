#Adding parent folder to path
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../../Documents/DTU/semester3/MachineLearning/PythonToolbox/Tools/')
from toolbox_02450 import clusterval

from plot_data_fit import plot_results

import numpy as np
import itertools

from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from sklearn import mixture
from data_cleaner import create_dataset
from nested_CV import my_cv
import plot_data_fit as pdfit
from bic_plot import plot_bic
from weighted_GMM_plot import plot_weighted_gmm
from accuracy_plot import plot_accuracy
from matplotlib import colors
from matplotlib import rcParams
rcParams.update({   'figure.figsize': (16, 8),
                    'font.size': 20 
                })

def create_gmm_models(cv_types, n_components):
    models = []
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            models.append(mixture.GaussianMixture(n_components=n_components, 
                                                    covariance_type=cv_type))
    
    return models

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    # Convert covariance to principal axes
    U, s, Vt = np.linalg.svd(covariance)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    result = s/np.linalg.norm(s) #normalize
    width = result[0]
    height = result[1]
    
    # Draw the Ellipse
    ax.add_patch(Ellipse(position,  width, height,
                         angle, **kwargs))

def plot_gmms(cv_types, n_components_range, best_gmm):
    color_iter = colors._colors_full_map
    # Plot the winner
    fig, ax = plt.subplots()
    Y_ = best_gmm.predict(X)
    for i, (mean, cov, weights, color) in enumerate(zip(best_gmm.means_, best_gmm.covariances_, 
                                                        best_gmm.weights_, color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 40, color=color)
        ax.plot(mean[0], mean[1], '*', markersize=60, markeredgecolor='k', markerfacecolor=color, 
                    markeredgewidth=2, zorder=3)
        ax.text(mean[0]-0.025, mean[1]-0.05, str(i+1), fontsize=20, color='white')

        w_factor = 0.5 / weights.max()
        draw_ellipse(mean, cov, alpha=weights * w_factor,
                     edgecolor=color, facecolor=color, linewidth=4)

    ax.set(title='Plot of the best fitting GMM')
    plt.show()

if __name__ == '__main__':
    # Create data set
    seed = 56
    np.random.seed(seed)
    X, y = create_dataset()

    n_components_range = range(1, 3)
    cv_types = ['full']

    models = create_gmm_models(cv_types, n_components_range)
    best_gmm, score = my_cv(X, y, models, K_out=10, K_in=10, seed=seed)
    best_gmm.fit(X)
    clf = best_gmm.predict(X)
    cent = best_gmm.means_
    covars = best_gmm.covariances_

    plot_accuracy(seed)
    plot_gmms(cv_types,n_components_range,best_gmm)

    #Cluster validity
    rand, jaccard, NMI = clusterval(y,clf)
    print('''\n ----- Quality Check of GMM ----- \n
    Rand index score: {} \n
    Jaccard similarity score: {} \n
    Normalized Mutual Information score: {} \n
    '''.format(rand, jaccard, NMI))
