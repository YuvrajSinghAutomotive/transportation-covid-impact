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
XTrainPostCovid,XTestPostCovid,YTrainPostCovid,YTestPostCovid,idxTrainPostCovid,idxTestPostCovid = splitCV(dataDictPostCovid['X'], 
                                                                                                           dataDictPostCovid['Y'], 
                                                                                                           returnIdx=True).testTrain(testRatio=0.5)

## base model: logistic regression
Mdls = {'MdlName': ['Logistic Regression'],
        'Mdl': [ LogisticRegression(max_iter=500) ],
        'Predictions': [[]],
        'Test Accuracy': [[]] }

MdlsPreCovid = {'MdlName': ['Logistic Regression: Pre-Covid'],
                'Mdl':[LogisticRegression(max_iter=500)],
                'Predictions': [[]],
                'Test Accuracy': [[]] }

MdlsPostCovid = {'MdlName': ['Logistic Regression: Post-Covid'],
                'Mdl':[LogisticRegression(max_iter=500)],
                'Predictions': [[]],
                'Test Accuracy': [[]] }


def fitTestModel(Mdl,MdlName,XTrain,YTrain,XTest,YTest):
    start = time.time()
    Mdl.fit(XTrain, YTrain)
    end = time.time()
    print('Time: {}: {} seconds'.format(MdlName,end - start) )
    pred = []
    for i in range(XTest.shape[0]):
        pred.append(Mdl.predict(XTest[i].reshape(1,-1)))
    pred = np.array(pred).reshape(YTest.shape)
    accuracy = np.mean(pred == YTest)
    print('Accuracy: {}'.format(accuracy) )
    plotBetaAccuracy(accuracy,XTest.shape[0])
    return Mdl,pred,accuracy

for i in range(len(Mdls['Mdl'])):
    Mdls['Mdl'][i] , Mdls['Predictions'][i], Mdls['Test Accuracy'][i] = fitTestModel(Mdl=Mdls['Mdl'][i],
                                                                                     MdlName=Mdls['MdlName'][i],
                                                                                     XTrain=XTrain,
                                                                                     YTrain=YTrain,
                                                                                     XTest=XTest,
                                                                                     YTest=YTest)

for i in range(len(MdlsPreCovid['Mdl'])):
    MdlsPreCovid['Mdl'][i] , MdlsPreCovid['Predictions'][i], MdlsPreCovid['Test Accuracy'][i] = fitTestModel(Mdl=MdlsPreCovid['Mdl'][i],
                                                                                                             MdlName=MdlsPreCovid['MdlName'][i],
                                                                                                             XTrain=XTrainPreCovid,
                                                                                                             YTrain=YTrainPreCovid,
                                                                                                             XTest=XTestPreCovid,
                                                                                                             YTest=YTestPreCovid)

for i in range(len(MdlsPostCovid['Mdl'])):
    MdlsPostCovid['Mdl'][i] , MdlsPostCovid['Predictions'][i], MdlsPostCovid['Test Accuracy'][i] = fitTestModel(Mdl=MdlsPostCovid['Mdl'][i],
                                                                                                                MdlName=MdlsPostCovid['MdlName'][i],
                                                                                                                XTrain=XTrainPostCovid,
                                                                                                                YTrain=YTrainPostCovid,
                                                                                                                XTest=XTestPostCovid,
                                                                                                                YTest=YTestPostCovid)

# pred = []
# predPreCovid = []
# predPostCovid = []
# for i in range(YTest.shape[0]):
#     pred.append(logisticRegr.predict(XTest[i].reshape(1,-1)))   ## Predict using Logistic Regression
# for i in range(YTestPreCovid.shape[0]):
#     predPreCovid.append(logisticRegrPreCovid.predict(XTestPreCovid[i].reshape(1,-1)))   ## Predict using Logistic Regression
# for i in range(YTestPostCovid.shape[0]):
#     predPostCovid.append(logisticRegrPostCovid.predict(XTestPostCovid[i].reshape(1,-1)))   ## Predict using Logistic Regression

# pred = np.array(pred).reshape(YTest.shape)
# print('Logistic Regression Accuracy: {}'.format( np.mean(pred == YTest) ) )

# accuracy = np.mean(pred == YTest)
# plotBetaAccuracy(accuracy,XTest.shape[0])
