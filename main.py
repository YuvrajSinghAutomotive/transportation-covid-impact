'''
Key take-away: feature engineering is important. Garbage in = Garbage Out
'''

from cleanData import cleanData
import time
import sys

plotBool = int(sys.argv[1]) if len(sys.argv)>1 else 0
resampleDataBool = int(sys.argv[2]) if len(sys.argv)>2 else 1
MISelectorBool = int(sys.argv[3]) if len(sys.argv)>3 else 0

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
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

## Plots
from regressionLib import plotPredictorVsResponse

'''
Data Dictionaries
'''
## Only select predictors highly correlated with severity
print('Correlation with severity')
def predictorsCorrelatedWithTarget(data):
    correlation = [1]
    for i in range(1,len(data.columns)):
        correlation.append(np.corrcoef(data[[data.columns[0],data.columns[i]]].T)[0,1])
    correlation = np.array(correlation)
    sortedCorr = np.sort(np.abs(correlation))
    sortedCorrIdx = np.argsort(np.abs(correlation))
    cols = list(data.columns[sortedCorrIdx[sortedCorr>0.05]])     ## at least 5% correlation needed
    return cols

def prepDataForTraining(data):
    predictorColNames = list(data.columns)
    predictorColNames.remove('Severity')
    X = np.array(data[predictorColNames])
    targetColNames = ['Severity']
    Y = np.array(data['Severity'])
    dataDict = {'X':X,
                'Y':Y,
                'predictorNames':predictorColNames,
                'targetName':targetColNames}
    return dataDict

dataDict = prepDataForTraining(data[predictorsCorrelatedWithTarget(data)])
dataDictPreCovid = prepDataForTraining(dataPreCovid[predictorsCorrelatedWithTarget(dataPreCovid)])
dataDictPostCovid = prepDataForTraining(dataPostCovid[predictorsCorrelatedWithTarget(dataPostCovid)])

## Mutual information between selected predictors and target
# Mutual information: MI(X,Y) = Dkl( P(X,Y) || Px \crossproduct Py)
from sklearn.feature_selection import mutual_info_classif
def mutualInfoPredictorsTarget(dataDict):
    MI = mutual_info_classif(dataDict['X'],dataDict['Y'])
    return ['{}: {}'.format(name,MI[i]) for i,name in enumerate(dataDict['predictorNames']) ] 

if MISelectorBool != 0:
    print('Mutual Information: data\n{}\n'.format( mutualInfoPredictorsTarget(dataDict) ) )
    print('Mutual Information: dataPreCovid\n{}\n'.format( mutualInfoPredictorsTarget(dataDictPreCovid) ) )
    print('Mutual Information: dataPostCovid\n{}\n'.format( mutualInfoPredictorsTarget(dataDictPostCovid) ) )

if resampleDataBool != 0:
    from regressionLib import resampleData
    dataDict = resampleData(dataDict)
    dataDictPreCovid = resampleData(dataDictPreCovid)
    dataDictPostCovid = resampleData(dataDictPostCovid)

'''
Correlation matrix
'''
if plotBool != 0:
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
                                                     returnIdx=True).testTrain(testRatio=0.05)
XTrainPreCovid,XTestPreCovid,YTrainPreCovid,YTestPreCovid,idxTrainPreCovid,idxTestPreCovid = splitCV(dataDictPreCovid['X'],
                                                                                                     dataDictPreCovid['Y'], 
                                                                                                     returnIdx=True).testTrain(testRatio=0.05)
XTrainPostCovid,XTestPostCovid,YTrainPostCovid,YTestPostCovid,idxTrainPostCovid,idxTestPostCovid = splitCV(dataDictPostCovid['X'], 
                                                                                                           dataDictPostCovid['Y'], 
                                                                                                           returnIdx=True).testTrain(testRatio=0.05)
'''
Train Models and Test: Draw beta distribution of accuracy.
## base model: logistic regression (location 0)
## All multiclass classifiers are declared here and fit(), predict() methods form sklearn model classes are used 
'''
Mdls = {'MdlName': ['Logistic Regression','Random Forest'],
        'Mdl': [ LogisticRegression(max_iter=5000) , 
                 RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True) ],
        'Predictions': np.zeros(shape=(2,),dtype='object'),
        'Confusion Matrix': np.zeros(shape=(2,),dtype='object') }

MdlsPreCovid = {'MdlName': ['Logistic Regression: Pre-Covid','Random Forest: Pre-Covid'],
                'Mdl':[LogisticRegression(max_iter=5000) ,
                       RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True) ],
                'Predictions': np.zeros(shape=(2,),dtype='object'),
                'Confusion Matrix': np.zeros(shape=(2,),dtype='object') }

MdlsPostCovid = {'MdlName': ['Logistic Regression: Post-Covid','Random Forest: Post-Covid'],
                'Mdl':[LogisticRegression(max_iter=5000) ,
                       RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True) ],
                'Predictions': np.zeros(shape=(2,),dtype='object'),
                'Confusion Matrix': np.zeros(shape=(2,),dtype='object') }


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

    cMatrix = confusionMatrix(classificationTest = pred,
                              Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))
    return Mdl,pred,cMatrix

def cMatrixPlots(cMatrixList,YTest,MdlNames):
    ## DO NOT CALL THIS FUNCTION IN SCRIPT. Use it only in jupyter to plot confusion matrices
    fig,ax = plt.subplots(nrows=1,ncols=len(cMatrixList),figsize=(6*len(cMatrixList),5))
    cMatrixLabels = list(pd.Series(YTest).unique())
    for i,cMatrix in enumerate(cMatrixList):
        img = ax[i].imshow(cMatrix,cmap='gray')
        ax[i].set_xticks(np.arange(len(cMatrixLabels)))
        ax[i].set_xticklabels(cMatrixLabels)
        ax[i].set_xticks(np.arange(len(cMatrixLabels)))
        ax[i].set_xticklabels(cMatrixLabels)
        ax[i].set_xlabel('Severity Class (Predicted)')
        ax[i].set_ylabel('Severity Class (Actual)')
        ax[i].set_title(MdlNames[i])
        fig.colorbar(mappable=img,ax = ax[i], fraction=0.1)
        fig.tight_layout()
    return fig,ax

for i in range(len(Mdls['Mdl'])):
    Mdls['Mdl'][i] , \
    Mdls['Predictions'][i], \
    Mdls['Confusion Matrix'][i] = fitTestModel(Mdl=Mdls['Mdl'][i],MdlName=Mdls['MdlName'][i], 
                                                    XTrain=XTrain, YTrain=YTrain, XTest=XTest, YTest=YTest)

for i in range(len(MdlsPreCovid['Mdl'])):
    MdlsPreCovid['Mdl'][i] , \
    MdlsPreCovid['Predictions'][i], \
    MdlsPreCovid['Confusion Matrix'][i] = fitTestModel(Mdl=MdlsPreCovid['Mdl'][i],MdlName=MdlsPreCovid['MdlName'][i], 
                                                            XTrain=XTrainPreCovid, YTrain=YTrainPreCovid, XTest=XTestPreCovid, YTest=YTestPreCovid)

for i in range(len(MdlsPostCovid['Mdl'])):
    MdlsPostCovid['Mdl'][i] , \
    MdlsPostCovid['Predictions'][i], \
    MdlsPostCovid['Confusion Matrix'][i] = fitTestModel(Mdl=MdlsPostCovid['Mdl'][i],MdlName=MdlsPostCovid['MdlName'][i], 
                                                             XTrain=XTrainPostCovid, YTrain=YTrainPostCovid, XTest=XTestPostCovid, YTest=YTestPostCovid)

if plotBool != 0:
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


'''
Perceptron
'''
def predictPerceptron(Wx):
    predictions = []
    for val in Wx:
        if val>0: predictions.append(1)
        else: predictions.append(0)
    return predictions

## One vs All perceptron multi-class classifier
def perceptronOnevsAll(XTrain,YTrain,XTest,YTest,plotcMatrix=True):
    ## One vs All
    YTrainDummies = pd.get_dummies(YTrain)
    YTestDummies = pd.get_dummies(YTest)

    perceptronDict = {'Classes':YTrainDummies.columns,
                      'Wx': [[]]*len(YTrainDummies.columns),
                      'Predictions': [[]]*len(YTrainDummies.columns)}

    for i,targetClass in enumerate(YTrainDummies.columns):
        target = np.array( YTrainDummies[targetClass] )
        clf = Perceptron()
        clf.fit(XTrain,target)
        W = clf.coef_
        Wx = (XTest @ W.T).reshape(-1,)
        predictions = predictPerceptron(Wx)
        perceptronDict['Wx'][i] = Wx
        perceptronDict['Predictions'][i] = np.array(predictions)
    WxBinary = np.array(perceptronDict['Wx']).T
    predictionsBinary = np.array(perceptronDict['Predictions']).T

    classification = []
    Wx_pred = np.multiply(WxBinary,predictionsBinary)
    for i in range(len(WxBinary)):
        classification.append(perceptronDict['Classes'][np.argmax(Wx_pred[i])])
    classification = np.array(classification)
    
    cMatrix = confusionMatrix(classificationTest = classification,
                              Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))

    if plotcMatrix:
        fig = plt.figure()
        cMatrixLabels = list(pd.Series(YTest).unique())
        plt.imshow(cMatrix,cmap='gray')
        plt.xticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
        plt.yticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
        plt.xlabel('Severity Class (Predicted)')
        plt.ylabel('Severity Class (Actual)')
        plt.colorbar()
        plt.close()
    return perceptronDict, classification, cMatrix, metrics(cMatrix),fig

## One vs One perceptron multiclass classifier
def perceptronOnevsOne(XTrain,YTrain,XTest,YTest,plotcMatrix=True):
    ## One vs One
    YTrainDummies = pd.get_dummies(YTrain)
    YTestDummies = pd.get_dummies(YTest)

    perceptronDict = {'Mdl':np.zeros((len(YTrainDummies.columns),len(YTrainDummies.columns)),dtype='object'),
                      'Wx': np.zeros((len(YTrainDummies.columns),len(YTrainDummies.columns)),dtype='object'),
                      'Predictions': np.zeros((len(YTrainDummies.columns),len(YTrainDummies.columns)),dtype='object')}

    for c1,label1 in enumerate(YTrainDummies.columns):
        for c2,label2 in enumerate(YTrainDummies.columns):
            if c1<c2:
                y1 = YTrainDummies[YTrainDummies.columns[c1]]
                y2 = YTrainDummies[YTrainDummies.columns[c2]]
                y = y1.iloc[ list(np.where( ((y1==1).astype(int) + (y2==1).astype(int))==1 )[0]) ]
                x = XTrain[list(y.index.astype(int))]
                clf = Perceptron().fit(x,y)
                perceptronDict['Mdl'][c1][c2] = clf
                W = clf.coef_
                Wx = (XTest @ W.T).reshape(-1,)
                predictions = predictPerceptron(Wx)
                perceptronDict['Wx'][c1][c2] = Wx
                perceptronDict['Predictions'][c1][c2] = np.array(predictions)

    ## Predicitons from each model
    pred = pd.DataFrame(np.zeros(len(YTestDummies)))
    for c1,label1 in enumerate(YTestDummies.columns):
        for c2,label2 in enumerate(YTestDummies.columns):
            if c1<c2:
                col = '{}_{}'.format(label1,label2)
                pred[col] = perceptronDict['Mdl'][c1][c2].predict(XTest)
    pred = pred.drop(pred.columns[0],axis=1)

    ## Assign labels to every model's prediction
    predLabels = pred.copy()
    for c1,label1 in enumerate(YTestDummies.columns):
        for c2,label2 in enumerate(YTestDummies.columns):
            if c1<c2:
                col = '{}_{}'.format(label1,label2)
                vector = pred[col]
                vector[vector==1] = label1
                vector[vector==0] = label2
                predLabels[col] = vector

    # Voting for classification
    classification = []
    from scipy.stats import mode
    for i in range(len(predLabels)):
        classification.append( ( mode(predLabels.iloc[i])[0].reshape(-1) )[0] )
    classification = np.array(classification)
    cMatrix = confusionMatrix(classificationTest = classification,
                                  Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))

    if plotcMatrix:
        fig = plt.figure()
        cMatrixLabels = list(pd.Series(YTest).unique())
        plt.imshow(cMatrix,cmap='gray')
        plt.xticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
        plt.yticks(np.arange(len(cMatrixLabels)),labels=cMatrixLabels)
        plt.xlabel('Severity Class (Predicted)')
        plt.ylabel('Severity Class (Actual)')
        plt.colorbar()
        plt.close()
    return perceptronDict, classification, cMatrix, metrics(cMatrix),fig


## Perceptrons
def perceptronsTrainTest(XTrain,YTrain,XTest,YTest):
    perceptrons = []

    perceptronsDict = {'Binary Perceptrons': [],
                    'Classification': [],
                    'Confusion Matrix': [],
                    'Confusion Matrix Metrics': [],
                    'Confusion Matrix Plot': []}

    perceptronsDict['Binary Perceptrons'], \
    perceptronsDict['Classification'], \
    perceptronsDict['Confusion Matrix'],\
    perceptronsDict['Confusion Matrix Metrics'],\
    perceptronsDict['Confusion Matrix Plot'] = perceptronOnevsAll(XTrain,YTrain,XTest,YTest)

    perceptrons.append(perceptronsDict)

    perceptronsDict = {'Binary Perceptrons': [],
                    'Classification': [],
                    'Confusion Matrix': [],
                    'Confusion Matrix Metrics': [],
                    'Confusion Matrix Plot': []}

    perceptronsDict['Binary Perceptrons'], \
    perceptronsDict['Classification'], \
    perceptronsDict['Confusion Matrix'],\
    perceptronsDict['Confusion Matrix Metrics'],\
    perceptronsDict['Confusion Matrix Plot'] = perceptronOnevsOne(XTrain,YTrain,XTest,YTest)

    perceptrons.append(perceptronsDict)
    return perceptrons

perceptrons = perceptronsTrainTest(XTrain,YTrain,XTest,YTest)
perceptronsPreCovid = perceptronsTrainTest(XTrainPreCovid,YTrainPreCovid,XTestPreCovid,YTestPreCovid)
perceptronsPostCovid = perceptronsTrainTest(XTrainPostCovid,YTrainPostCovid,XTestPostCovid,YTestPostCovid)
