# EL2320 course project
# Author: Michaela Söderström

"""
Functions that are used in the color measurement model.
------------------------------------------------------
"""

import cv2 
import numpy as np
import math

def makeHist_RGB(frame):
    '''
    Compute the histogram in HSV space of a given frame.
    --------------------------------
    INPUT:
        frame: array
            of which the desired histogram is computed from
    OUTPUT: 
        hist: array
            normalized histogram of the input frame
    --------------------------------
    '''
    
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(frame_RGB)

    channel_id = (0, 1, 2)
    no_bins = (8, 8, 8)

    #for ch_id, bins in zip(channel_id, no_bins):
        #print('id', ch_id, 'bins',  bins)


    #     hist, bin_edges = np.histogram(frame_HSV[:, :, ch_id],
    #                                     bins,
    #                                     range=(0, 255),
    #                                     density=True)
    # print(np.size(hist), 'for hist------', hist)
        # Weighing function: Kernel density estimate by the Epanechnikov kernel
    

    # Weighted histograms
    hist_r, bin_edges_r = np.histogram(r, no_bins[0], range=(0, 255), density=True)
    hist_g, bin_edges_g = np.histogram(g, no_bins[1], range=(0, 255), density=True)
    hist_b, bin_edges_b = np.histogram(b, no_bins[2], range=(0, 255), density=True)
    
    hist = np.array([[hist_r], [hist_g], [hist_b]])
    hist = np.divide(hist, np.sum(hist))

    bin_edges = np.array([[bin_edges_r], [bin_edges_g], [bin_edges_b]])

    return hist, bin_edges

def makeHist_GRAY(frame):
    '''
    Compute the histogram in gray-scale of a given frame.
    --------------------------------
    INPUT:
        frame: array
            of which the desired histogram is computed from
    OUTPUT: 
        hist: array
            normalized histogram of the input frame
    --------------------------------
    '''
    #print('IN greay ', frame)
    bins = 8
    frame_GRAY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hist, bin_edges = np.histogram(frame_GRAY,
                                    bins,
                                    (0, 255),
                                    density=True)
    # plt.hist(hist, bins=bins, density=True)
    # plt.show()
    #print('hist', hist)
    return hist, bins

def distance(hist_target, hist_candidate, roi_candidate, k_target, M, method):
    '''
    Computes the distance between two distributions
    ---------------------------
    INPUT:
        hist_target: array

        hist_candidate: array

        method: bool
            the method choice can be either True (Bhattacharyya) or False (Euclidian)
    OUTPUT:
        dist: float
            The computed distance between the distributions
    ----------------------------
    '''

    roi_size = (np.size(roi_candidate, 0), np.size(roi_candidate, 1)) # size of candidate frame
    
    if method:
        # create a storage vector in RGB case
        dist = np.zeros([len(hist_target)])

        k_candidate = epanechnikovKernel(roi_size, M, init=True)

        f_target = 1/np.sum(k_target)
        f = 1/(np.sum(k_candidate))

        q = f_target*np.multiply(np.sum(k_target), hist_target)
        p = f*np.multiply(np.sum(k_candidate), hist_candidate)

        # Bhattaracharyya distance
        if len(hist_target)==3:

            for i in range(len(hist_target)):
                # rho = np.sum(np.multiply(hist_target[i], hist_candidate[i]))
                # dist[i] = math.sqrt(1 - rho) 
                rho = np.sum(np.multiply(q[i], p[i]))
                #print(rho, '-----rho')
                dist[i] = math.sqrt(1 - rho) 
        else:
            rho = np.sum(np.sqrt(np.multiply(hist_target[0], hist_candidate[0])))
            dist = math.sqrt(1 - rho)
            #print(rho,'---------rho')

        # Bayes error
        D, Q, P = 1, 1, 1 
        # for i in range(len(dist)):
        #     D = dist[i]*D
        #     Q = q[i]*Q
        #     P = p[i]*P
        # Ebay_l = (1 - np.sqrt(1 - 4*np.multiply(np.multiply(Q, P), np.exp(-2*D))))/2
        # Ebay_l = np.sum(Ebay_l)
        # Ebay_u = np.multiply(np.exp(-D), np.sqrt(Q, P))
        # Ebay_u = np.sum(Ebay_u)
        Ebay_l = np.zeros((len(dist), np.size(hist_target[1])))
        Ebay_u = np.zeros((len(dist), np.size(hist_target[1])))

        for i in range(len(dist)):
            Ebay_l[i, :] = (1 - np.sqrt(1 - 4*np.multiply(np.multiply(q[i], p[i]), np.exp(-2*dist[i]))))/2
            Ebay_u[i, :] = np.multiply(np.exp(-dist[i]), np.sqrt(q[i], p[i]))

        Ebay_l = np.sum(Ebay_l, axis=1)
        Ebay_u = np.sum(Ebay_u, axis=1)
        
    # elif method == False:
    #     # Euclidian distance
    #     dist = math.sqrt(np.sum((hist_target - hist_candidate)**2, axis=0)*1/3)
    else:
        raise ValueError
    
    return dist, Ebay_l, Ebay_u

def epanechnikovKernel(frame_size, M, init=False):
    
    '''


    '''

    k = np.zeros([frame_size[1], frame_size[0]])
    min = np.minimum(frame_size[0], frame_size[1])//2 # minimum windows radius
    h = np.sqrt(min**2 + min**2)

    x_max = frame_size[0]
    y_max = frame_size[1]

    x_axis = np.arange(0, x_max)
    y_axis = np.arange(0, y_max)
    x =  np.tile(x_axis, (y_max, 1))
    y = np.tile(y_axis, (x_max, 1))
    y = np.reshape(y, (y_max, x_max))

    if not init:
        x_diff = 0
        y_diff = 0
        #x_diff = x_max // 2
        #y_diff = y_max // 2
        #x_diff = np.subtract(x, x_max // 2)
        #y_diff = np.subtract(y, y_max // 2)
    else:
        x_diff = np.subtract(x, x_max // 2)
        y_diff = np.subtract(y, y_max // 2)

    r_norm = np.sqrt(y_diff**2 + x_diff**2)/h 
    k = np.where(r_norm < 1, (1 - r_norm**2), k)
    #print(k,'------------------------k')

    return k
