import numpy as np
from typing import List, Dict, Tuple
from glob import glob
import random
from PIL import Image
from scipy.ndimage import gaussian_filter
from common import SEED
from scipy.linalg import svd, sqrtm, inv, cholesky, cho_factor, cho_solve
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from scipy.linalg import svd
np.random.seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

def normalise_data(patterns: np.array, backgrounds: np.array) -> Tuple[np.array, np.array]:
    patterns /= np.linalg.norm(patterns, ord='fro')
    d_norm = np.linalg.norm(backgrounds, ord='fro')
    distractor_term = backgrounds if 0 == d_norm else backgrounds / d_norm
    return patterns, distractor_term

def scale_to_bound(row, scale):
    return row * scale

def regularize_cov(cov, threshold=1e-16, increase_factor=1):
    eigs = np.linalg.eigvalsh(cov)
    
    min_eig = np.min(eigs)
    
    # If the smallest eigenvalue is negative or very close to zero
    if min_eig < threshold:
        # Increase the regularization threshold
        new_threshold = threshold * increase_factor
        # Add a value slightly larger than the absolute minimum eigenvalue to the diagonal
        regularizer = abs(min_eig) + new_threshold
        print(f"Added regularization: {regularizer}")
        cov += np.eye(cov.shape[0]) * regularizer
    
    return cov

def generate_backgrounds(sample_size: int, mean_data: int, var_data: float, image_shape: list=[8,8]) -> np.array:
    backgrounds = np.zeros((sample_size, image_shape[0] * image_shape[1]))

    for i in range(sample_size):
        samples = np.random.normal(mean_data, var_data, size=image_shape)
        backgrounds[i] = np.reshape(samples, (image_shape[0] * image_shape[1]))   

    return backgrounds

def get_patterns(params: Dict) -> List:
    manip = params['manipulation']
    scale = params['pattern_scale']
    t = np.array([
            [manip,0],
            [manip,manip],
            [manip,0]
        ])
    
    l = np.array([
            [manip,0],
            [manip,0],
            [manip,manip]
        ])
    
    pattern_dict = {
        't': np.kron(t, np.ones((scale,scale))),
        'l': np.kron(l, np.ones((scale,scale))),
    }

    chosen_patterns = []
    for pattern_name in params['patterns']:
        chosen_patterns.append(pattern_dict[pattern_name])
    return chosen_patterns

def generate_fixed(params: Dict, image_shape: list):
    patterns = np.zeros((params['sample_size'], image_shape[0], image_shape[1]))
    chosen_patterns = get_patterns(params)
    j = 0
    for k, pattern in enumerate(chosen_patterns):
        pos = params['positions'][k]
        for i in range(int(params['sample_size']/len(params['patterns']))):
            patterns[j][pos[0]:pos[0]+pattern.shape[0], pos[1]:pos[1]+pattern.shape[1]] = pattern
            if params['pattern_scale'] > 3:
                patterns[j] = gaussian_filter(patterns[j], 1.5)
            j+=1 

    return np.reshape(patterns, (params['sample_size'], image_shape[0] * image_shape[1]))

def generate_translations_rotations(params: Dict, image_shape: list) -> np.array:
    patterns = np.zeros((params['sample_size'], image_shape[0], image_shape[1]))    
    chosen_patterns = get_patterns(params)

    j = 0
    for pattern in chosen_patterns:
        for i in range(int(params['sample_size']/len(params['patterns']))):                        
            pattern_adj = pattern

            rand = np.random.randint(0, high=4)
            if rand > 0:
                pattern_adj = np.rot90(pattern, k=rand)
        
            rand_y = np.random.randint(0, high= (image_shape[0])-pattern_adj.shape[0] + 1)
            rand_x = np.random.randint(0, high= (image_shape[0])-pattern_adj.shape[1] + 1)
            pos = (rand_y, rand_x)

            patterns[j][pos[0]:pos[0]+pattern_adj.shape[0], pos[1]:pos[1]+pattern_adj.shape[1]] = pattern_adj
            if params['pattern_scale'] > 3:
                patterns[j] = gaussian_filter(patterns[j], 1.5)
            j+=1     

    return np.reshape(patterns, (params['sample_size'], image_shape[0] * image_shape[1]))

def generate_xor(params: Dict, image_shape: list) -> np.array:
    patterns = np.zeros((params['sample_size'], image_shape[0], image_shape[1]))
    chosen_patterns = get_patterns(params)
    poses = params['positions']

    manips = [
        [1,1],
        [-1,-1],
        [1,-1],
        [-1,1],
    ]

    k = 0
    for ind in range(0, params['sample_size'], int(params['sample_size']/4)):
        pat = np.zeros((image_shape[0], image_shape[1]))
        pat[poses[0][0]:poses[0][0]+chosen_patterns[0].shape[0], poses[0][1]:poses[0][1]+chosen_patterns[0].shape[1]] = chosen_patterns[0] * manips[k][0]
        pat[poses[1][0]:poses[1][0]+chosen_patterns[1].shape[0], poses[1][1]:poses[1][1]+chosen_patterns[1].shape[1]] = chosen_patterns[1] * manips[k][1]
        patterns[ind:ind+int(params['sample_size']/4)] = pat
        k+=1
    
    for j, pattern in enumerate(patterns):
        if params['pattern_scale'] > 3:
            patterns[j] = gaussian_filter(pattern, 1.5)

    return np.reshape(patterns, (params['sample_size'], image_shape[0] * image_shape[1]))

# WHITENING
def symmetric_orthogonalise_helper(A, maintainMagnitudes=False, return_W=False):
    L = None
    W = None
    if maintainMagnitudes:
        D = np.sqrt(np.diag(A.T @ A))
        D = np.diag(D)
        if return_W:
            Lnorm, W = symmetric_orthogonalise_helper(A @ D, maintainMagnitudes=False, return_W=True)
            if Lnorm is not None:
                L = Lnorm @ D
                W = D @ W @ D
        else:
            L = symmetric_orthogonalise_helper(A @ D, maintainMagnitudes=False, return_W=False) @ D
            W = None
    else:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        tol = max(A.shape) * S[0] * np.finfo(A.dtype).eps
        r = np.sum(S > tol)

        if r >= A.shape[1]:
            L = U @ Vt
            if return_W:
                W = Vt.T @ np.diag(1.0 / S) @ Vt
            else:
                W = None
        else:
            print("Skipping. Matrix is not full rank.")

    return L, W

def symmetric_orthogonalization(X):
    # (n_samples, n_features)
    # Ensure data is zero-centered
    X_centered = X - np.mean(X, axis=0)
    
    # Call the orthogonalization helper on the data matrix itself
    L, P = symmetric_orthogonalise_helper(X_centered, maintainMagnitudes=True, return_W=True)
    if P is not None:
        X_new = (P @ X_centered.T).T 
        return X_new, P
    else:
        print("returning the same X")
        P = np.eye(X.shape[1])  # dummy transformation matrix, note the change to shape[1] assuming X is samples x features
        return X, P

def sphering(X):
    # (n_samples, n_features)
    # Ensure data is zero-centered
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar = False, bias=True)
    
    # Regularize covariance matrix if necessary
    covariance_matrix = regularize_cov(covariance_matrix)
    
    # Compute inverse square root of covariance matrix
    P = np.linalg.inv(sqrtm(covariance_matrix))

    # Apply the transformation
    X_whitened = np.dot(X_centered, P.T)
    
    return X_whitened, P

def cholesky_whitening(X):
    # Ensure data is zero-centered
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False, bias=True)

    # Regularize covariance matrix if necessary
    covariance_matrix = regularize_cov(covariance_matrix)

    # Compute the Cholesky decomposition
    L = np.linalg.cholesky(covariance_matrix)

    # Invert the lower triangular matrix L
    L_inv = np.linalg.inv(L)

    # Apply the transformation
    X_whitened = np.dot(X_centered, L_inv.T)

    return X_whitened, L_inv

def partial_regression(X):
    # (n_samples, n_features)
    # Ensure data is zero-centered
    X_centered = X - np.mean(X, axis=0)

    nsamples, nfeatures = X_centered.shape
    X_whitened = np.empty_like(X_centered)
    for ii in range(nfeatures):
        xin = X[:, ii]
        xout = X[:, [j for j in range(nfeatures) if j != ii]]
        
        # Compute the weights to regress xin on xout
        weights = np.linalg.pinv(xout) @ xin
        
        # Compute the residuals of the regression
        residuals = xin - xout @ weights
        
        X_whitened[:, ii] = residuals
        
    P = np.eye(nfeatures)  # dummy transformation matrix
    return X_whitened, P

def optimal_signal_preserving_whitening(X):
    # (n_samples, n_features)
    # Ensure data is zero-centered
    X_centered = X - np.mean(X, axis=0)
    
    # Compute the correlation matrix of X's transpose
    corr = np.corrcoef(X_centered, rowvar=False, bias=True)
    
    # Regularize correlation matrix if necessary
    corr = regularize_cov(corr)
    
    # Compute transformation matrix P
    P = np.linalg.inv(sqrtm(corr)) @ np.diag(1 / np.sqrt(np.var(X_centered, axis=0)))
    
    # Apply the transformation
    X_whitened = np.dot(X_centered, P.T)
    
    
    return X_whitened, P