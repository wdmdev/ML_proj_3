import os, sys
os.chdir(sys.path[0])

import matplotlib.pyplot as plt
import numpy as np

from data_cleaner import create_dataset
from scipy.stats import multivariate_normal as mvn

def gdk(data: np.ndarray):

  # Returns array of gaussian kernel densities
  mus = data
  N = len(mus)

  lambdas = np.logspace(-6, 1, 21)  # Standard deviations
  # Performs leave-one-out cross-validation to find best lambdas
  ps = np.zeros_like(lambdas)
  for k, l in enumerate(lambdas):
    print("lambda: %.4f" % l)
    for i in range(N):
      s = np.zeros(N)
      for j in range(N):
        if i == j:
          continue
        m = mvn(mean=mus[j], cov=l**2*np.diag(np.ones(data.shape[1])))
        s[j] = m.pdf(data[i])
      s = s.sum() / (N - 1)
      s = np.log(s)
      ps[k] += s
  ps /= N

  l = lambdas[np.argmax(ps)]
  print("Best lambda (std): %.4f" % l)
  
  print(ps)
  plt.semilogx(lambdas, ps, "ro-")
  plt.title("GDK Log Likelyhood")
  plt.xlabel(r"$\log_{10}(\lambda)$")
  plt.ylabel(r"$\log (P(X))$")
  plt.grid(True)
  plt.show()

  densities = np.zeros(N)
  for i in range(N):
    for j in range(N):
      s = np.zeros(N)
      if i == j:
        continue
      m = mvn(mean=mus[j], cov=l**2*np.diag(np.ones(data.shape[1])))
      s[j] = m.pdf(data[i])
    s = s.sum() / (N - 1)
    densities[i] = s
  
  densities.sort()
  print(densities)
  plt.bar(np.linspace(1, N, N), densities)
  plt.semilogy()
  plt.title("Gaussian Kernel Density Estimation")
  plt.xlabel("Observation")
  plt.ylabel("GKD")
  plt.show()



def kde(data: np.ndarray, K: int):

  d = lambda x, y: np.sqrt(((x-y)**2).sum())

  N = len(data)
  densities = np.zeros(N)
  for i, obs in enumerate(data):
    distances = np.zeros(N)
    for j, obs2 in enumerate(data):
      if i == j:
        continue
      distances[j] += d(obs, obs2)
    distances.sort()
    densities[i] = distances[:K].sum()
  
  densities *= K
  densities.sort()

  plt.bar(np.linspace(1, N, N), densities)
  plt.title("KNN Density Estimation")
  plt.xlabel("Observation")
  plt.ylabel(r"KNNDE")
  plt.show()

  return densities

def ard(data: np.ndarray, K: int):

  d = lambda x, y: np.sqrt(((x-y)**2).sum())

  densities = kde(data, K)

  N = len(data)
  ards = np.zeros(N)
  for i, obs in enumerate(data):
    distances = np.zeros(N)
    for j, obs2 in enumerate(data):
      if i == j:
        continue
      distances[j] += d(obs, obs2)
    distances.sort()
    ards[i] = distances[:K].sum()
  
  ards = densities / ards * K
  ards.sort()


  plt.bar(np.linspace(1, N, N), ards)
  plt.title("KNN Average relative density")
  plt.xlabel("Observation")
  plt.ylabel("ARD")
  plt.show()

if __name__ == "__main__":

  data, target = create_dataset()
  # gdk(data)
  ard(data, 10)
