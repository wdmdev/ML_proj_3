import os
from os.path import dirname, realpath
os.chdir(realpath(dirname(__file__)))

import json
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib import colors as mcolors
import numpy as np

from data_cleaner import create_dataset

from time import time

def do_pca(x: np.ndarray):

	"""
	Udfører PCA for et givent datasæt og år
	:param data: Observationer nedad, variabel henad
	:return:
	"""
	
	# Laver eigenvektorer og -værdier
	lambdas, V = np.linalg.eig(x.T @ x)
	sort_arr = lambdas.argsort()[::-1]
	lambdas = lambdas[sort_arr]
	V = V[:, sort_arr]

	# Projicerer ind i egenbasis
	z = V.T @ x.T

	return x.T, z, lambdas, V

def le_plot(z: np.ndarray, m: list or tuple, nyears: int):

    fs = 16
    # Gets first two principal components of z and sorts to make masking easier
    z = z[:2]
    for i in range(z.shape[0]):
    	z[i] = sort_by_municipality(z[i])
    
    # Gets municipalities
    with open("attrs.out", encoding="utf-8") as f:
    	muns = f.readline().strip().split(";")
    
    indices = [muns.index(x.lower()) for x in m if x.lower() in muns]
    # Splits data into municipalities
    mundat = []
    markers = list(MarkerStyle.markers)
    for i in reversed(indices):
    	mundat.append(z[:, i*nyears:(i+1)*nyears])
    	z = np.concatenate((z[:, :i*nyears], z[:, (i+1)*nyears:]), axis=1)
    mundat.reverse()
    
    #Plots stuff
    plt.scatter(*z, 4, c="black")
    for i, md in enumerate(mundat):
    	plt.plot(*md, markers[i+2], markersize=12, label=muns[i].capitalize() + " municipality")
    plt.title('Projection of Municipality Key Figures into PCA Space',fontsize=fs+8)
    plt.xlabel('PC 1', fontsize=fs)
    plt.ylabel('PC 2', fontsize=fs)
    plt.legend(labels=[str.capitalize(mu) + ' kommune' for mu in m], fontsize=fs)
    plt.show()

def sort_by_municipality(x: np.ndarray):

	"""
	Sorts a 1D array x such that (k1-2007, k2-2007, ..., k1-2008, ...)
	becomes (k1-2007, k1-2008, ..., k2-2007, ...)
	"""

	return x.reshape(98, x.size//98).ravel(order="F")


if __name__ == "__main__":

	data, target = create_dataset()

	x, z, l, V = do_pca(data)
	low = z[:, ~target][:2]
	high = z[:, target][:2]

	plt.scatter(*low, 6, label="Low third")
	plt.scatter(*high, 6, label="Mid third")
	plt.title("Different classes in PCA space")
	plt.xlabel("PC dimension 1")
	plt.ylabel("PC dimension 2")
	plt.legend()
	plt.show()

	from hier_clust import d1, d2
	plt.clf()
	plt.scatter(*d1[:2], 6, label="Low third")
	plt.scatter(*d2[:2], 6, label="Mid third")
	plt.title("Different classes in PCA space")
	plt.xlabel("PC dimension 1")
	plt.ylabel("PC dimension 2")
	plt.legend()
	plt.show()

