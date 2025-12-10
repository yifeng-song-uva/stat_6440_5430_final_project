import numpy as np
import math
import glob
import os

def normalize_vector(arr):
    # such that the components of the vector sum up to 1
    arr_sum = np.sum(arr)
    return arr / arr_sum

def exp_normalize(arr):
    # exponentiate, and then normalize to 1 (using the idea of log-sum-exp)
    M = np.max(arr)
    arr = arr - M
    exp_arr = np.exp(arr)
    return normalize_vector(exp_arr)

def exp_normalize_by_row(arr):
    # exponentiate, and then normalize to 1 by each row (using the idea of log-sum-exp)
    arr_rowmax = arr.max(axis=1)
    arr_new = np.exp(arr.T - arr_rowmax)
    return (arr_new / arr_new.sum(axis=0)).T

def linear_time_natural_gradient(g, h, z):
    # Gradient g, Hessian H = diag(h) + 1 * z * 1^T (both already scaled by batch size if needed)
    # based on the special structure of Hessian matrix w.r.t alpha or xi, using Woodbury matrix identity
    # (See section A.2 of the LDA paper)
    c = np.sum(g/h) / (1/z + np.sum(1/h))
    return (g-c)/h

def stochastic_variational_update(old_value, value_hat, rho):
    # update rule for the global variational parameters in one step, in stochastic optimization
    new_value = (1 - rho) * old_value + rho * value_hat
    return new_value

def stochastic_hyperparameter_update(old_value, value_hat, rho):
    # update rule for global hyperparameters in one step, in stochastic optimization
    new_value = old_value - rho * value_hat
    return new_value

def merge_dict(dict_list):
    merged = {}
    for dict in dict_list:
        merged.update(dict)
    return merged

def create_gamma_matrix(new_gamma_dict):
    # convert gamma from dictionary into matrix format, which is required in efficient running of M step
    return np.vstack([new_gamma_dict[i] for i in range(len(new_gamma_dict))])

def pct_diff(x1,x2):
    # compute percentage difference of a metric between two successive iterations
    # especially useful for keeping track of improvement in ELBO when we want to determine convergence of variational EM
    if x1 == -math.inf:
        return math.inf
    else:
        return (x2-x1)/np.abs(x1) * 100

def step_size(t, tau, kappa):
    # rho_t in stochastic optimization
    return (t+tau)**(-kappa)

def delete_all_files(fpath):
    # delete all files in the given temporary folder
    fn_list = glob.glob(fpath + "/*")
    for fn in fn_list:
        os.remove(fn)

def predictive_R2(test_y, pred_y):
    return 1 - np.mean((test_y - pred_y)**2) / np.var(test_y)