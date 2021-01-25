# EL2320 course project
# Author: Michaela Söderström

"""
Particle filter algorithm based on color histograms.
---------------------------------------------------
"""

# Load libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

# Load classes and functions
from initialparams import getParams
#from framefunctions import drawFrame
from color_model import makeHist_GRAY, makeHist_RGB, distance, epanechnikovKernel

class particleFilter():
    def __init__(self, coord_target, frame_size, roi_size, method, color_hist, init_hypotheses, frame_no, M=100):
        '''
        Initialization of the particle filter
        ----------------------------------------
        INPUT: 
            coord_target: list
                contains the x, y top left coordinates and the width and height of the target region
            frame_size: tuple
                specifies the dimensions of the original input frame
            roi_size: tuple
                specifies the dimensions of the target frame
            color_hist: bool
                RGB histogram or grayscale
            init_hypotheses: bool
                initialize particles over the whole frame or over a chosen region
            M: int
                no. of particles
            frame_no: int
                Current frame
        OTHER PARAMETERS:
            S: array
                prior particle set
            var_R: int
                Value of the variance in the process noise covariance matrix
            mu: array
                Gaussian mean value
            Sigma_R: array
                covariance matrix of the motion model
            Sigma_Q: float
                covariance matrix (1-dim) of the measurement model
            A: array
                Transition matrix
            weights: array
                particle weights
            Nt: float
                threshold value used in resampling
        ----------------------------------------
        '''   
        self.M = M
        # initialize state
        S = getParams(coord_target, frame_size, init_hypotheses, self.M) 
        self.S = S 
        self.var_R = 10 #5 or 10
        mu = np.zeros(np.size(S, 0))
        self.mu = mu.reshape(len(S), 1)
        self.Sigma_R = np.eye(np.size(S, 0))*self.var_R
        self.Sigma_Q = 0.1 #0.1
        self.A = [[1, 0, 1, 0, 1, 0], 
            [0, 1, 0, 1, 0 , 1], 
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]]
        self.frame_size = frame_size
        self.roi_size = roi_size
        self.weights = np.full((1, self.M), 1/self.M)
        self.Nt = 0.96
        self.frame_no = frame_no
        self.method = method
        self.color_hist = color_hist
        self.init_hypotheses = init_hypotheses

    def predict(self):
        '''
        Predict the next state of the object.
        -----------------------------------------
        OUTPUT:
            S_hat: array
                predicted state
        -----------------------------------------
        '''

        std = np.sqrt(np.diag(self.Sigma_R))
        std = np.reshape(std, (np.size(self.S, 0), 1))
        rep_tile = np.tile(std, (1, self.M)) # works like repmat in Matlab
        randn_mtx = np.random.normal(self.mu, std, np.size(rep_tile)).astype(int) # matrix with random integers from a normal distribution (Gaussian white noise)
        randn_mtx = np.reshape(randn_mtx, (np.size(rep_tile, 0), np.size(rep_tile, 1)))
        diffusion = np.multiply(rep_tile, randn_mtx) # drawn samples from normal distribution with zero mean and std = sqrt(Sigma_R)

        #new state
        S_hat = np.dot(self.A, self.S) + diffusion 
        S_hat = S_hat.astype(int) # to follow pixel values

        # Force particle positions to be within the frame
        S_hat[0, :] =  np.where(S_hat[0, :]>self.frame_size[0], self.frame_size[0], S_hat[0, :])
        S_hat[0, :] = np.where(S_hat[0, :]<0, 0, S_hat[0, :])
        S_hat[1, :] = np.where(S_hat[1, :]>self.frame_size[1], self.frame_size[1], S_hat[1, :])
        S_hat[1, :] = np.where(S_hat[1, :]<0, 0, S_hat[1, :])

        # reset velocity and acceleration between predictions
        S_hat[2:, :] = np.zeros(self.M)
        
        self.S = S_hat

        return self.S

    def measurementUpdate(self, S_hat, hist_target, frame, frame_no):
        '''
        Measurement update based on color histograms. The weights are updated based on the observations.
        ----------------------------------------
        INPUT:
            S_hat: array
                the mean of each particle
            hist_target: array
                reference histogram of the object
            frame: array
                original input frame
            frame_no: int
                current frame
        OUTPUT:
            frame: array
                original frame with drawn estimted region
            S_mean: array
                the estimated state
            RMSE: array
                Root mean square error of the target and candidate histogram
        ----------------------------------------
        '''
        frame_candidate = np.copy(frame)
        # define the observation region of the particles
        x = S_hat[0]
        y = S_hat[1]
        w = self.roi_size[0] # width of bbox
        h = self.roi_size[1] # height of bbox
        W = self.frame_size[0] # width of frame
        H = self.frame_size[1] # height of frame
        
        # create an array to store values
        likelihood = np.zeros(self.M)
        p = np.zeros(self.M)
        D = np.zeros([len(hist_target), self.M])
        Euclidian = np.zeros([len(hist_target), self.M])
        histogram_vec = []
        X_min = np.zeros(np.size(S_hat[0]))  
        X_max = np.zeros(np.size(S_hat[0]))
        Y_min = np.zeros(np.size(S_hat[1]))
        Y_max = np.zeros(np.size(S_hat[1]))
        
        # region coordinates
        X_min = S_hat[0] - w//2
        X_max = S_hat[0] + w//2
        Y_min = S_hat[1] - h//2
        Y_max = S_hat[1] + h//2

        # Force roi candidate pixel values to be bounded by the frame
        X_min = np.where(X_min<0, 0, X_min)
        X_max = np.where(X_max>W, W, X_max)
        Y_min = np.where(Y_min<0, 0, Y_min)
        Y_max = np.where(Y_max>H, H, Y_max)

        # Epanechnikov kernel for target histogram
        k_target = epanechnikovKernel(self.roi_size, self.M)

        for i in range(self.M):
            # compute the histograms of the candidate regions for each particle 
            roi_candidate = frame_candidate[Y_min[i]: Y_max[i], X_min[i]: X_max[i]]
            
            if self.color_hist:
                hist_candidate, bins = makeHist_RGB(roi_candidate)
                if frame_no==self.frame_no:
                    histogram_vec.append(hist_candidate)
            elif self.color_hist==0:
                hist_candidate, bins = makeHist_GRAY(roi_candidate)
            else:
                continue
            # Compute the similarity between two regions
            D[:, i], Euclidian[:, i] = distance(hist_target, hist_candidate, roi_candidate, k_target, self.M, self.method) 
        
        Euclidian_avg = np.sum(Euclidian, axis=1)/self.M
        RMSE = np.sqrt(Euclidian_avg)

        if frame_no==self.frame_no:
            hist_candidate = np.sum(histogram_vec, axis=1)


            plt.figure()
            plt.hist(hist_candidate[0], bins=bins[0], density=True, color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r'])
            plt.hist(hist_candidate[1], bins=bins[1], density=True, color=['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g'])
            plt.hist(hist_candidate[2], bins=bins[2], density=True, color=['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'])
            plt.title('RGB histogram frame' + str(self.frame_no))
            plt.show()

        # Importance weights
        p = 1
        for j in range(len(D)):
            p = (1/math.sqrt(2*math.pi*self.Sigma_Q**2))*np.exp(-D[j]**2/(2*self.Sigma_Q**2))*p
        
        likelihood = p
        self.weights = np.multiply(self.weights, likelihood)

        # Normalize weights
        self.weights = np.divide(self.weights, np.sum(self.weights))
        #print(np.sum(self.weights), 'sum weights') #OK

        # Resample
        Neff = 1/(self.M*np.sum(self.weights**2))

        if Neff < self.Nt:
            # Systematic resampling:
            S_bar = np.zeros(self.S.shape)
            # construct CDF
            cdf = np.cumsum(self.weights)
            # starting point
            r_0 = np.random.random_sample()/self.M

            for m in range(self.M):
                idx = np.argmax(cdf>= r_0 + (m - 1)/self.M)
                S_bar[:, m] = self.S[:, idx]

            self.weights = np.full((1, self.M), 1/self.M)
            self.S = S_bar

        # Estimated mean state:
        S_mean = np.sum(np.multiply(self.weights, self.S), axis=1)
        S_mean = S_mean.astype(int) # to closest pixel values
        cv2.rectangle(frame, (S_mean[0] - w//2, S_mean[1] - h//2), (S_mean[0] + w//2, S_mean[1] + h//2), (255, 0, 0), 3) # estimated region
        
        return frame, S_mean, RMSE