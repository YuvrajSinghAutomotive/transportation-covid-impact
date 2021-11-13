# This file contains the regression code/functions that will be called by ipynb files.
# This can be used as a general purpose regression library

import numpy as np
import pandas as pd
import os, os.path
import warnings
import sys

'''
Helper functions
'''
def genGaussianDataCholesky(n,mu,cov,weights=True):
    '''
    Generate multivariate gaussian data of a certain mean and covariance
    '''
    from scipy.linalg import cholesky
    x_uncorr = np.random.normal(0,1,(n,cov.shape[0]))
    c = cholesky(cov,lower=True)
    if weights==True:
        w = np.random.random(n)
    else:
        w = np.ones(n)
    return np.dot(c,x_uncorr.T).T + mu , w

def normalizeData(data):
    return (data-np.mean(data,0))/np.std(data,0)

def interceptCol(data):
    # appends a column of ones at the beginning of the predictors data
    return np.concatenate((np.ones((data.shape[0],1)),data),axis=1)

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

'''
Cross-Validation Splits
'''
class splitCV:
    '''
    Split data for cross validation
    Data should be from 
    '''
    def __init__(self,X,Y,w=None,returnIdx=False):
        self.X = X
        self.Y = Y
        self.w = w
        self.returnIdx = returnIdx
    
    def testTrain(self,testRatio=0.2):
        '''
        Returns a tuple (Xtrain,Xtest,Ytrain,Ytest,wtrain,wtest)
        '''
        num = np.random.permutation(np.arange(self.X.shape[0]))
        idxTest = num[0:np.ceil(testRatio*self.X.shape[0]).astype(int)]
        idxTrain = num[np.ceil(testRatio*self.X.shape[0]).astype(int):]
        if type(self.w)==type(None):
            if self.returnIdx==True: return ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest], idxTrain,idxTest )
            else: return ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest] )
        else:
            if self.returnIdx==True: return ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest],self.w[idxTrain],self.w[idxTest], idxTrain,idxTest )
            else: return ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest],self.w[idxTrain],self.w[idxTest] )
    
    def monteCarlo(self,numSplits=10,testRatio=0.2):
        '''
        Returns a list of tuples (Xtrain,Xtest,Ytrain,Ytest,wtrain,wtest)
        '''
        splits=[]
        for i in range(numSplits):
            splits.append(self.testTrain(testRatio))
        return splits
    
    def KFold(self,numFolds=5):
        '''
        Returns a list of tuples (Xtrain,Xtest,Ytrain,Ytest,wtrain,wtest)
        '''
        order = np.random.permutation(np.arange(self.X.shape[0]))
        splits=[]
        for i in range(numFolds):
            if i < numFolds-1:
                idxTest = order[i*np.ceil(self.X.shape[0]/numFolds).astype(int):(i+1)*np.ceil(self.X.shape[0]/numFolds).astype(int)]
            else:
                idxTest = order[i*np.ceil(self.X.shape[0]/numFolds).astype(int):]
            idxTrain = np.array(list(set(order).difference(set(idxTest))))
            if type(self.w)==type(None):
                if self.returnIdx==True: splits.append( ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest],idxTrain,idxTest ) )
                else: splits.append( ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest] ) )
            else:
                if self.returnIdx==True: splits.append( ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest],self.w[idxTrain],self.w[idxTest],idxTrain,idxTest ) )
                else: splits.append( ( self.X[idxTrain],self.X[idxTest],self.Y[idxTrain],self.Y[idxTest],self.w[idxTrain],self.w[idxTest] ) )
        return splits



