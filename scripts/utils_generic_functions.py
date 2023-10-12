"""
Generic functions 

10/12/2023
_______________
    
Summary: The script provides generic utility functions

DISCLAIMER: THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING 
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""

####^^ Generic imports
import os
import pandas as pd
from math import sqrt
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.spatial.distance import cdist # to compute the medoid center of each class

##^ settings
np.set_printoptions(precision=2)

####^^ Global variables
n_outputs_ae = 3 # this constant tells how many outputs has the AutoEncoder 
half_wind_size = 5 # we pick "half_wind_size" to the left and same to the right of a policy change

####^^ Generic Functions

##^^ this function unstacks an np array as illustrated below: output_autoencoder=unstack(output_ae[k],1) - use axis=1 for a linear list
def unstack(a, axis = 0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]
 
##^^ calculate minkowski distance
def minkowskiDistance(a, b, p):
	return sum(abs(e1-e2)**p for e1, e2 in zip(a,b))**(1/p)

##^^ normalize -1 and 1
def scalingMinusOnetoOne(X):
	X_sca = 2* ( (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) ) -1
	return X_sca 

##^^ normalize -1 and 1
def customScalingMinusOnetoOne(X,max_v,min_v):
	X_sca = 2* ( (X - min_v) / (max_v - min_v) ) -1
	return X_sca

##^^ stacking options
def stackOutputAEPRBs(ouae_slice_a,prev_prb_slice_a):
    slice_a = []
    for x, y in zip(ouae_slice_a, prev_prb_slice_a):
        x_l = x.tolist()
        z = []
        for a in x_l:
            z.append(a)
        z.append(y)
        slice_a.append(z)
    return slice_a
     
def stackOutputAE(ouae_slice_a):
    slice_a = []
    for x in ouae_slice_a:
        x_l = x.tolist()
        z = []
        for a in x_l:
            z.append(a)
        slice_a.append(z)
    return slice_a

def cosineSimilarity(a,b):
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    return cos_sim


def mostFrequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def computeMedoid(x,y):

    #^ Combine the arrays into a single feature matrix
    data = np.column_stack((x, y))

    #^ Calculate pairwise distances between all data points
    distances = cdist(data, data, metric='euclidean')

    #^ Compute the total distance for each data point
    total_distances = np.sum(distances, axis=1)

    #^ Find the index of the data point with the minimum total distance
    medoid_index = np.argmin(total_distances)

    #^ Get the medoid center
    medoid_center = data[medoid_index]

    # #^ Print
    # print("Medoid center:", medoid_center)

    return medoid_center
