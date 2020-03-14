# exercise 11.1.1
from matplotlib.pyplot import figure, show
import numpy as np
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
# Load Matlab data file and extract variables of interest

from data_cleaner import create_dataset

data, y_true = create_dataset()

X = data[:,[0,4]]

#y = mat_data['y'].squeeze()
attributeNames =  ['Property value', 'tertiary education']  #[name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = ['Class 1','Class 2', 'Class 3', 'Class 4']   #[name[0][0] for name in mat_data['classNames']]
#X_old = X
#X = np.hstack([X,X])
N, M = X.shape
C = len(classNames)
# Number of clusters
K = 4
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'random'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14,9))
plt.scatter(X[:,0],X[:,1])
#clusterplot(X, clusterid=cls, centroids=cds, covars=covs) #, y=y)
show()

## In case the number of features != 2, then a subset of features most be plotted instead.
#figure(figsize=(14,9))
#idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
#show()

print('Ran Exercise 11.1.1')