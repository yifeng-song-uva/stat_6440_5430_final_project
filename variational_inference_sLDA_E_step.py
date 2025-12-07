from variational_inference_utils import *
from scipy.special import polygamma
import math
import pickle

class VI_sLDA_E_Step:
    '''
    Note: 
    - E step does not depend on the global hyperparameter xi
    - Local variational parameters are updated document-wise
    - This class is written for parallelized processing
    '''
    def __init__(self, fragment_indx, fpath, K, bow, y, alpha, eta, delta, Lambda, epsilon=1e-4):
        self.fpath = fpath
        self.fragment_indx = fragment_indx
        self.K = K # number of topics
        self.bow = bow # dictionaries of arrays, with length D: each array represents the bag of words in the d^th document, with the length of array being N_d
        self.doc_len = {d:len(v) for d,v in bow.items()} # number of words within each document
        self.DI = list(self.bow.keys()) # document indices in the original minibatch (batch)
        self.y = y # length = D
        self.alpha = alpha # K-dimensional vector
        self.eta = eta # K-dimensional vector
        self.delta = delta # scalar
        self.Lambda = Lambda # size: K x V
        self.Lambda_rowsum = np.sum(Lambda, axis=1)
        self.gamma = {d:np.ones(shape=(K,)) for d in self.DI} # initialize local variational parameter gamma (size: D x K)
        self.phi = {d:np.empty(shape=(self.doc_len[d], K)) for d in self.DI} # initialize local variational parameter phi (for each document, size is N_d x K)
        self.epsilon = epsilon

    def update_gamma(self, d):
        # update rule for local variational parameter gamma
        sum_phi = np.sum(self.phi[d], axis=0) # K-dimensional
        self.gamma[d] = self.alpha + sum_phi

    def update_phi_unsupervised(self, d):
        # update rule for local variational parameter phi in case y is not observed (prediction mode): same as naive LDA
        # can use vectorized operations to update each phi_d
        log_phi = polygamma(0, self.Lambda[:, self.bow[d]]).T + polygamma(0, self.gamma[d]) - polygamma(0, self.Lambda_rowsum) # the first term has size N_d x K, the 2nd & 3rd terms are K-dimensional vectors, so broadcasting is applicable
        self.phi[d] = exp_normalize_by_row(log_phi) # use log-sum-exp normalization, as the raw values of log_phi could be very negative
        
    def update_phi_supervised(self, d):
        # update rule for local variational parameter phi when y is observed (training mode): Eq (33) of the sLDA paper
        N_d = self.doc_len[d]
        temp_var_1 = (self.y[d]/N_d/self.delta) * self.eta
        temp_var_2 = 1/(2*N_d**2*self.delta)
        temp_var_3 = self.eta**2
        for j,v in enumerate(self.bow[d]):
            log_phi_j = polygamma(0, self.Lambda[:, v]) + polygamma(0, self.gamma[d]) - polygamma(0, self.Lambda_rowsum) # first 2 terms same as the unsupervised case (see Eq (31) of the SVI paper)
            phi_minus_j = self.phi[d].sum(axis=0) - self.phi[d][j,:] # K-dimensional vector
            log_phi_j += temp_var_1 - temp_var_2 * (2*np.dot(self.eta, phi_minus_j)*self.eta + temp_var_3) # Eq (33) of sLDA paper
            self.phi[d][j,:] = exp_normalize(log_phi_j) # use log-sum-exp normalization, as the raw values of log_phi_j could be very negative

    def coordinate_ascent_training(self, prediction=False):
        if prediction == False: # Under the supervised training mode, we ought to assign some reasonable initial values to phi 
            for d in self.DI:
                self.update_phi_unsupervised(d) # initialize phi without using y: based on values of initial values of gamma
        for d in self.DI:
            change_in_gamma = math.inf
            while change_in_gamma > self.epsilon: # stopping criteria for convergence: average change in every component of gamma is <= epsilon
                if prediction == True:
                    self.update_phi_unsupervised(d)
                else:
                    self.update_phi_supervised(d)
                previous_gamma = self.gamma[d].copy()
                self.update_gamma(d)
                change_in_gamma = np.mean(np.abs(self.gamma[d] - previous_gamma))

    def save_parameters(self):   
        pickle.dump(self.gamma, open(self.fpath + "gamma_{}.pickle".format(self.fragment_indx), "wb"))
        pickle.dump(self.phi, open(self.fpath + "phi_{}.pickle".format(self.fragment_indx), "wb"))