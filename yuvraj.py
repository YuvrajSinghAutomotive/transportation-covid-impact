from cleanData import cleanData
import time

start = time.time()
data,dataPreCovid,dataPostCovid = cleanData(verbose=0)
end = time.time()
print('Time: Data Extraction: {} seconds'.format(end - start) ); 

'''
Import libraries needed
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## General regression and classification functions: validation
from regressionLib import splitCV, plotBetaAccuracy

## Exploration and cluster analysis
from sklearn.cluster import KMeans,MeanShift

## Models
from sklearn.linear_model import LogisticRegression


'''
Training models
'''
def prepDataForTraining(data):
    predictorColNames = data.columns[1:]
    X = np.array(data[predictorColNames])
    targetColNames = data.columns[0]
    Y = np.array(data[data.columns[0]])

    dataDict = {'X':X,
                'Y':Y,
                'predictorNames':predictorColNames,
                'targetName':targetColNames}
    return dataDict

dataDict = prepDataForTraining(data)
dataDictPreCovid = prepDataForTraining(dataPreCovid)
dataDictPostCovid = prepDataForTraining(dataPostCovid)

XTrain,XTest,YTrain,YTest,idxTrain,idxTest = splitCV(dataDict['X'],
                                                     dataDict['Y'],
                                                     returnIdx=True).testTrain(testRatio=0.5)
XTrainPreCovid,XTestPreCovid,YTrainPreCovid,YTestPreCovid,idxTrainPreCovid,idxTestPreCovid = splitCV(dataDictPreCovid['X'],
                                                                                                     dataDictPreCovid['Y'], 
                                                                                                     returnIdx=True).testTrain(testRatio=0.5)

## base model: logistic regression
logisticRegr = LogisticRegression()

start = time.time()
logisticRegr.fit(XTrain, YTrain)    ## Train Logistic Regression
end = time.time()
print('Time: Logistic Regression (sklearn): {} seconds'.format(end - start) ); 

pred = []
for i in range(YTest.shape[0]):
    pred.append(logisticRegr.predict(XTest[i].reshape(1,-1)))   ## Predict using Logistic Regression

pred = np.array(pred).reshape(YTest.shape)
print('Logistic Regression Accuracy: {}'.format( np.mean(pred == YTest) ) )

accuracy = np.mean(pred == YTest)
plotBetaAccuracy(accuracy,XTest.shape[0])
