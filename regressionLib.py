# This file contains the regression code/functions that will be called by ipynb files.
# This can be used as a general purpose regression library

import numpy as np
import pandas as pd
import os, os.path
import warnings
import sys
import matplotlib.pyplot as plt

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

'''
Model Validation
'''
from scipy.stats import beta
def plotBetaAccuracy(accuracy,numTestSamples):
    '''
    Test Accuracy and Confidence: beta distribution of accuracy scores
    '''
    betaEval = beta.pdf(np.linspace(start=0,stop=1,num=1000), 
                        a=np.floor((accuracy*numTestSamples))+1 , b=numTestSamples - np.floor((accuracy*numTestSamples)) +1)
    plt.plot(np.linspace(start=0,stop=1,num=1000),betaEval,color='r')
    plt.xlim(0,1)
    plt.title("Test Accuracy: \nBeta Distribution: a = {}, b = {}".format(np.floor((accuracy*numTestSamples))+1 , numTestSamples - np.floor((accuracy*numTestSamples)) +1 ) )
    plt.xlabel('Test Accuracy')
    plt.ylabel('Probability Density')
    plt.show()
    
def credibleInterval():
    pass

def bayesianSignificanceTest():
    pass

'''
Performance Metrics
'''
def confusionMatrix(classificationTest,Ytest):
    # classificationTest: numpy array or list
    # Ytest: pandas Series
    # Confusion Matrix: predictedLabel (rows) and actualLabel (columns)
    cMatrix = np.zeros(( len(Ytest.unique()) , len(Ytest.unique()) ))
    for idxPredictedLabel,predictedLabel in enumerate(Ytest.unique()):
        for idxActualLabel,actualLabel in enumerate(Ytest.unique()):
            predForActualLabel = classificationTest[Ytest==actualLabel]
            cMatrix[idxPredictedLabel,idxActualLabel] = np.sum(predForActualLabel == predictedLabel)
    return cMatrix

def metrics(M):
    # M: confusion matrix: predictedLabel (rows) and actualLabel (columns)
    overallAccuracy = np.sum(np.diag(M))/np.sum(M)
    userAccuracy = []
    producerAccuracy = []
    for i in range(M.shape[0]):
        userAccuracy.append(np.diag(M)[i]/np.sum(M[i,:]))
    for j in range(M.shape[1]):
        producerAccuracy.append(np.diag(M)[j]/np.sum(M[:,j]))
    N = np.sum(M)
    Mii = np.sum(np.diag(M))
    sumMplus = 0
    for i in range(M.shape[0]):
        Miplus = np.sum(M[i,:])
        Mplusi = np.sum(M[:,i])
        sumMplus = sumMplus + Miplus*Mplusi
    kappaCoeff = (N*Mii - sumMplus)/(N**2 - sumMplus)
    return overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff

'''
Exploration
'''
def corrMatrix(data,title='Correlation Matrix'):
    dataNorm = ( data - data.mean() ) / data.std()
    fig = plt.figure(figsize=(15,15),dpi=200)
    plt.imshow(np.corrcoef(data.T),cmap='PiYG')
    plt.colorbar(shrink=0.3)
    plt.xticks(ticks=np.arange(len(data.columns)), labels=data.columns, fontsize=4, rotation=90)
    plt.yticks(ticks=np.arange(len(data.columns)), labels=data.columns, fontsize=4)
    plt.tight_layout()
    plt.title(title)
    plt.close()
    return fig
    
def corrMatrixHighCorr(data,corrThresh=0.5,title='Correlation Matrix (Strong Correlations, absolute value > 0.5)'):
    dataNorm = ( data - data.mean() ) / data.std()
    fig = plt.figure(figsize=(15,15),dpi=200)
    plt.imshow(np.abs(np.corrcoef(data.T))>corrThresh,cmap='gray')
    plt.colorbar(shrink=0.3)
    plt.xticks(ticks=np.arange(len(data.columns)), labels=data.columns, fontsize=4, rotation=90)
    plt.yticks(ticks=np.arange(len(data.columns)), labels=data.columns, fontsize=4)
    plt.tight_layout()
    plt.title(title)
    plt.close()
    return fig

'''
Plot Functions
'''
def plotPredictorVsResponse(predictorsDataFrame,predictorName,actualResponse,predictedResponse,hueVarName,labels=['Pre-Covid','Post-Covid']):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,3.5),dpi=100)
    if type(hueVarName) != type(None):
        hueVar = predictorsDataFrame[hueVarName].values
        ax[0].scatter(predictorsDataFrame[predictorName].loc[hueVar==1],actualResponse[np.where(hueVar==1)]-0.1,label=labels[0],color='green') # ,edgecolors='k')
        ax[0].scatter(predictorsDataFrame[predictorName].loc[hueVar==0],actualResponse[np.where(hueVar==0)]+0.1,label=labels[1],color='red') # ,edgecolors='k')
        ax[0].legend(loc='lower left',ncol=2)
    else:
        ax[0].scatter(predictorsDataFrame[predictorName],actualResponse,color='green')
    ax[0].set_xlabel(predictorName)
    ax[0].set_ylabel('severity')
    ax[0].set_yticks(np.arange(5))
    ax[0].set_title('{} vs actual severity'.format(predictorName)) 
    if type(hueVarName) != type(None):
        ax[1].scatter(predictorsDataFrame[predictorName].loc[hueVar==1],predictedResponse[np.where(hueVar==1)]-0.1,label=labels[0],color='green') # ,edgecolors='k')
        ax[1].scatter(predictorsDataFrame[predictorName].loc[hueVar==0],predictedResponse[np.where(hueVar==0)]+0.1,label=labels[1],color='red') # ,edgecolors='k')
        ax[1].legend(loc='lower left',ncol=2)
    else:
        ax[1].scatter(predictorsDataFrame[predictorName],predictedResponse,color='green')    
    ax[1].set_xlabel(predictorName)
    ax[1].set_ylabel('severity')
    ax[1].set_yticks(np.arange(5))
    ax[1].set_title('{} vs predicted severity'.format(predictorName))
    fig.tight_layout()
    plt.close()
    return fig
