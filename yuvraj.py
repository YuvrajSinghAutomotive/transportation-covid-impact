from cleanData import cleanData
import time
import sys
plotBool = sys.argv[2] if int(len(sys.argv)>2) else 0

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
from regressionLib import confusionMatrix, metrics

## Exploration and cluster analysis
from sklearn.cluster import KMeans,MeanShift
from regressionLib import corrMatrix, corrMatrixHighCorr

## Models
from sklearn.linear_model import LogisticRegression

## Plots
from regressionLib import plotPredictorVsResponse


'''
Data Dictionaries
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

'''
Correlation matrix
'''
if plotBool==1:
    predictors = pd.DataFrame(dataDict['X'], columns=dataDict['predictorNames'])
    fig = corrMatrixHighCorr(predictors)
    fig.savefig('Plots/CorrMatrixHighThresh.svg')
    fig = corrMatrix(predictors)
    fig.savefig('Plots/CorrMatrix.svg')

    predictorsPreCovid = pd.DataFrame(dataDictPreCovid['X'], columns=dataDictPreCovid['predictorNames'])
    fig = corrMatrixHighCorr(predictorsPreCovid)
    fig.savefig('Plots/CorrMatrixHighThreshPreCovid.svg')
    fig = corrMatrix(predictorsPreCovid)
    fig.savefig('Plots/CorrMatrixPreCovid.svg')

    predictorsPostCovid = pd.DataFrame(dataDictPostCovid['X'], columns=dataDictPostCovid['predictorNames'])
    fig = corrMatrixHighCorr(predictorsPostCovid)
    fig.savefig('Plots/CorrMatrixHighThreshPostCovid.svg')
    fig = corrMatrix(predictorsPostCovid)
    fig.savefig('Plots/CorrMatrixPostCovid.svg')

'''
Training models: Base model
'''
XTrain,XTest,YTrain,YTest,idxTrain,idxTest = splitCV(dataDict['X'],
                                                     dataDict['Y'],
                                                     returnIdx=True).testTrain(testRatio=0.5)
XTrainPreCovid,XTestPreCovid,YTrainPreCovid,YTestPreCovid,idxTrainPreCovid,idxTestPreCovid = splitCV(dataDictPreCovid['X'],
                                                                                                     dataDictPreCovid['Y'], 
                                                                                                     returnIdx=True).testTrain(testRatio=0.5)
XTrainPostCovid,XTestPostCovid,YTrainPostCovid,YTestPostCovid,idxTrainPostCovid,idxTestPostCovid = splitCV(dataDictPostCovid['X'], 
                                                                                                           dataDictPostCovid['Y'], 
                                                                                                           returnIdx=True).testTrain(testRatio=0.5)
'''
Train Models and Test: Draw beta distribution of accuracy.
## base model: logistic regression (location 0)
'''
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

if plotBool==1:
    predictorsTest = pd.DataFrame(XTest, columns=dataDict['predictorNames'])
    for i in range(len(predictorsTest.columns)):
        fig = plotPredictorVsResponse(predictorsDataFrame=predictorsTest,
                                    predictorName=predictorsTest.columns[i],
                                    actualResponse=YTest,
                                    predictedResponse=Mdls['Predictions'][0],
                                    hueVarName='preCovid',
                                    labels=['Pre-Covid','Post-Covid'])
        fig.savefig('./Plots/Logistic results/complete data/fig_{}.jpg'.format(i),dpi=300)

    predictorsTestPreCovid = pd.DataFrame(XTestPreCovid, columns=dataDictPreCovid['predictorNames'])
    for i in range(len(predictorsTestPreCovid.columns)):
        fig = plotPredictorVsResponse(predictorsDataFrame=predictorsTestPreCovid,
                                    predictorName=predictorsTestPreCovid.columns[i],
                                    actualResponse=YTestPreCovid,
                                    predictedResponse=MdlsPreCovid['Predictions'][0],
                                    hueVarName=None,
                                    labels=['Pre-Covid','Post-Covid'])
        fig.savefig('./Plots/Logistic results/preCovid/fig_{}.jpg'.format(i),dpi=300)

cMatrix = confusionMatrix(classificationTest = Mdls['Predictions'][0],
                          Ytest = pd.Series(YTest))
overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
print('Kappa Coefficient: {}'.format(np.round(kappaCoeff,6)))

if plotBool==1:
    cMatrixLabels = list(pd.Series(YTest).unique())
    fig = plt.figure()
    plt.imshow(cMatrix,cmap='gray')
    plt.xticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
    plt.yticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
    plt.xlabel('Severity Class (Predicted)')
    plt.ylabel('Severity Class (Actual)')
    plt.colorbar()
    plt.show()
