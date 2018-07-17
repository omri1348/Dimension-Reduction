import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance_matrix as distance_matrix
from scipy.misc import imread as imread
from scipy.misc import imrotate as imrotate

"""
LLE Algorithm Steps - 
1. Compute the KNN graph of the data matrix
2. Compute W using the inverse of the Gram Matrix of each points nearest neighbors
3. Decompose M = (I − W)^T(I − W) into its eigenvectors and return the ones corresponding
to the 2...(d + 1) lowest eigenvalues.
"""


def LLE(X, d, k):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the LLE algorithm.
    :param X: NxD data matrix.
    :param d: the dimension.
    :param k: the number of neighbors for the weight extraction.
    :return: Nxd reduced data matrix.
    '''
    # building the k nearest neighbors graph
    distances = distance_matrix(X, X)
    sort_distances = np.argsort(distances, axis=1)[:, 1:k+1]
    # compute the W matrix
    n = X.shape[0]
    w = np.zeros((n, n))
    for i in range(n):
        # creating the k by p neighbors matrix and centering using the translation invariance
        k_indexes = sort_distances[i, :]
        neighbors = X[k_indexes, :] - X[i, :]
        # computing the corresponding gram matrix
        gram_inv = np.linalg.pinv(np.dot(neighbors, np.transpose(neighbors)))
        # setting the weight values according to the lagrangian
        lambda_par = 2/np.sum(gram_inv)
        w[i, k_indexes] = lambda_par*np.sum(gram_inv, axis=1)/2
    m = np.subtract(np.eye(n), w)
    values, u = np.linalg.eigh(np.dot(np.transpose(m), m))
    return u[:, 1:d+1]


"""
MDS Algorithm Steps - 
1. Compute the squared euclidean distance matrix ∆.
2. From the distance matrix, form the matrix S = −0.5H∆H
3. Diagonalize S to form S = UΛU^T
4. Return the n × d matrix of columns √λiui for i = 1...d.
"""

def MDS(X, d):
    '''
    Given a NxN pairwise distance matrix and the number of desired dimensions,
    return the dimensionally reduced data points matrix after using MDS.
    :param X: NxN distance matrix.
    :param d: the dimension.
    :return: Nxd reduced data point matrix.
    '''
    # creating the data centering matrix
    n = X.shape[0]
    data_center = np.eye(n) - ((1/n) * np.ones((n, n)))
    # computing the similarity matrix
    s = -0.5 * np.dot(np.dot(data_center, np.power(X, 2)), data_center)
    # extracting the eigenfunctions and eigenvalues
    values, u = np.linalg.eigh(s)
    values = values[::-1]
    u = np.fliplr(u)
    # values = np.sqrt(values[:d])
    # u = u[:, :d]
    return u[:, :d]*np.transpose(np.sqrt(values[:d]))


"""
Diffusion Maps Algorithm Steps - 
1. Given a data set X, compose the kernel matrix K using a kernel function
2. Fix alpha in the [0,1] range and preform anisotropic normalization (used for manifold reductions)
3. Normalize the rows of the kernel matrix in order to get the transition matrix P.
4. Decompose P into its eigenfunctions and select only the ones corresponding to the 2...d+1 highest eigenvalues
5. Return the n × d matrix of columns λiui for i = 2...d+1.


"""

def heat_kernel(X,sigma):
    """
    Compute the kernel matrix using the heat kernel.
    :param X: NxP data matrix
    :param sigma: the kernel width.
    :return: kernel: NxN kernel matrix
    """
    distances = distance_matrix(X, X)
    kernel = np.exp(-np.power(distances, 2) / sigma)
    return kernel


def DiffusionMap(X, d, sigma, t, kernel, alpha=0):
    '''
    Given a NxD data matrix, return the dimensionally reduced data matrix after
    using the Diffusion Map algorithm. The k parameter allows restricting the
    gram matrix to only the k nearest neighbor of each data point.
    :param X: NxP data matrix.
    :param d: the reduced dimension.
    :param sigma: the kernel width.
    :param t: time scale.
    :param kernel: a kernel function that return the affinity matrix.
    :param alpha: the anisotropic parameter (alpha in the [0,1] range)
    :return: Nxd reduced data matrix.
    '''
    # building the kernel matrix
    k_mat = kernel(X,sigma)
    # anisotropic normalization
    if alpha != 0:
        tmp = np.linalg.inv(np.diag(np.power(np.sum(k_mat, axis=1),alpha)))
        k_mat = np.dot(np.dot(tmp,k_mat),tmp)
    # row-normalize
    a = np.dot(np.linalg.inv(np.diag(np.sum(k_mat, axis=1))), k_mat)
    # extracting the eigenfunctions and eigenvalues
    values, u = np.linalg.eig(a)
    sort_index = np.argsort(values)
    values = values[sort_index[::-1]]
    values = np.power(values[1:d+1], t)
    u = u[:,sort_index[::-1]]
    u = u[:, 1:d+1]
    return u*np.transpose(values)