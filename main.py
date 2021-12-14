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
from regressionLib import flatten

## Exploration and cluster analysis
from sklearn.cluster import KMeans,MeanShift
from regressionLib import corrMatrix, corrMatrixHighCorr

## Models
from sklearn.linear_model import LogisticRegression,Perceptron
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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

#################################################################################################################
# ### TEMP CODE: DELETE LATER
# dataDict = prepDataForTraining(data)
# dataDictPreCovid = prepDataForTraining(dataPreCovid)
# dataDictPostCovid = prepDataForTraining(dataPostCovid)

# # Correlation matrix: ALL VARIABLES
# if plotBool == 0:
#     predictors = pd.DataFrame(dataDict['X'], columns=dataDict['predictorNames'])
#     fig = corrMatrixHighCorr(predictors)
#     fig.savefig('Plots/CorrMatrixHighThreshRAW.svg')
#     fig = corrMatrix(predictors)
#     fig.savefig('Plots/CorrMatrixRAW.svg')

#     predictorsPreCovid = pd.DataFrame(dataDictPreCovid['X'], columns=dataDictPreCovid['predictorNames'])
#     fig = corrMatrixHighCorr(predictorsPreCovid)
#     fig.savefig('Plots/CorrMatrixHighThreshPreCovidRAW.svg')
#     fig = corrMatrix(predictorsPreCovid)
#     fig.savefig('Plots/CorrMatrixPreCovidRAW.svg')

#     predictorsPostCovid = pd.DataFrame(dataDictPostCovid['X'], columns=dataDictPostCovid['predictorNames'])
#     fig = corrMatrixHighCorr(predictorsPostCovid)
#     fig.savefig('Plots/CorrMatrixHighThreshPostCovidRAW.svg')
#     fig = corrMatrix(predictorsPostCovid)
#     fig.savefig('Plots/CorrMatrixPostCovidRAW.svg')
# #################################################################################################################

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
Correlation matrix: Features
'''
if plotBool != 0:
    predictors = pd.DataFrame(dataDict['X'], columns=dataDict['predictorNames'])
    fig = corrMatrixHighCorr(predictors)
    fig.savefig('Plots/CorrMatrixHighThreshfeat.svg')
    fig = corrMatrix(predictors)
    fig.savefig('Plots/CorrMatrixfeat.svg')

    predictorsPreCovid = pd.DataFrame(dataDictPreCovid['X'], columns=dataDictPreCovid['predictorNames'])
    fig = corrMatrixHighCorr(predictorsPreCovid)
    fig.savefig('Plots/CorrMatrixHighThreshPreCovidfeat.svg')
    fig = corrMatrix(predictorsPreCovid)
    fig.savefig('Plots/CorrMatrixPreCovidfeat.svg')

    predictorsPostCovid = pd.DataFrame(dataDictPostCovid['X'], columns=dataDictPostCovid['predictorNames'])
    fig = corrMatrixHighCorr(predictorsPostCovid)
    fig.savefig('Plots/CorrMatrixHighThreshPostCovidfeat.svg')
    fig = corrMatrix(predictorsPostCovid)
    fig.savefig('Plots/CorrMatrixPostCovidfeat.svg')

# #############################################################################
# sys.exit("Just wanted correlation matrices lol")
# #############################################################################

## Initial model selection study: using testTrain split and credible intervals, binomial significance

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
Mdls = {'MdlName': ['Logistic Regression',
                    'Random Forest: Bootstrap Aggregation',
                    'Random Forest: AdaBoost',
                    'Neural Network: 3 hidden layers, 50 hidden units'],
        'Mdl': [ LogisticRegression(max_iter=5000) , 
                 RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True),
                 AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy',max_depth=5) , n_estimators=50, learning_rate=1),
                 MLPClassifier(hidden_layer_sizes=(50,50,50,), alpha=0.1 , max_iter=2000, activation = 'logistic', solver='adam') ],
        'Predictions': np.zeros(shape=(4,),dtype='object'),
        'Confusion Matrix': np.zeros(shape=(4,),dtype='object') }

MdlsPreCovid = {'MdlName': ['Logistic Regression: Pre-Covid',
                            'Random Forest: Bootstrap Aggregation: Pre-Covid',
                            'Random Forest: AdaBoost: Pre-Covid',
                            'Neural Network: 3 hidden layers, 10 hidden units'],
                'Mdl':[LogisticRegression(max_iter=5000) ,
                       RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True),
                       AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy',max_depth=5) , n_estimators=50, learning_rate=1),
                       MLPClassifier(hidden_layer_sizes=(50,50,50,), alpha=0.1 , max_iter=2000, activation = 'logistic', solver='adam') ],
                'Predictions': np.zeros(shape=(4,),dtype='object'),
                'Confusion Matrix': np.zeros(shape=(4,),dtype='object') }

MdlsPostCovid = {'MdlName': ['Logistic Regression: Post-Covid',
                             'Random Forest: Bootstrap Aggregation: Post-Covid',
                             'Random Forest: AdaBoost: Post-Covid',
                             'Neural Network: 3 hidden layers, 10 hidden units'],
                'Mdl':[LogisticRegression(max_iter=5000) ,
                       RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=10,min_samples_leaf=100,min_samples_split=150,bootstrap=True),
                       AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy',max_depth=5) , n_estimators=50, learning_rate=1),
                       MLPClassifier(hidden_layer_sizes=(50,50,50,), alpha=0.1 , max_iter=2000, activation = 'logistic', solver='adam')  ],
                'Predictions': np.zeros(shape=(4,),dtype='object'),
                'Confusion Matrix': np.zeros(shape=(4,),dtype='object') }

## Fit sklearn models
def fitTestModel(Mdl,MdlName,XTrain,YTrain,XTest,YTest,saveLocation=None):
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
    if type(saveLocation)!=type(None):
        plotBetaAccuracy(accuracy,XTest.shape[0],saveLocation)
    else:
        plotBetaAccuracy(accuracy,XTest.shape[0])

    cMatrix = confusionMatrix(classificationTest = pred,
                              Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))
    print('########################################################\n')
    return Mdl,pred,cMatrix

def cMatrixPlots(cMatrixList,YTest,MdlNames):
    ## DO NOT CALL THIS FUNCTION IN SCRIPT. Use it only in jupyter to plot confusion matrices
    fig,axs = plt.subplots(nrows=2,ncols=np.ceil(len(cMatrixList)/2).astype(int),figsize=(3*len(cMatrixList),8))
    ax = axs.reshape(-1)
    cMatrixLabels = list(pd.Series(YTest).unique())
    if len(cMatrixList)<=1:
        ax = [ax]
    for i,cMatrix in enumerate(cMatrixList):
        img = ax[i].imshow(cMatrix,cmap='gray')
        ax[i].set_xticks(np.arange(len(cMatrixLabels)))
        ax[i].set_xticklabels(cMatrixLabels)
        ax[i].set_yticks(np.arange(len(cMatrixLabels)))
        ax[i].set_yticklabels(cMatrixLabels)
        ax[i].set_xlabel('Severity Class (Actual)')
        ax[i].set_ylabel('Severity Class (Predicted)')
        ax[i].set_title(MdlNames[i])
        for j in range(len(cMatrixLabels)):
            for k in range(len(cMatrixLabels)):
                ax[i].text(j-0.25,k,int(cMatrix[k,j]),color='blue',fontweight='semibold',fontsize=18)
        fig.colorbar(mappable=img,ax = ax[i], fraction=0.1)
    fig.tight_layout()
    return fig,ax

def cMatrixPlot_single(cMatrix,YTest,MdlName):
    ## DO NOT CALL THIS FUNCTION IN SCRIPT. Use it only in jupyter to plot confusion matrices
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.5,3.5))
    cMatrixLabels = list(pd.Series(YTest).unique())
    img = ax.imshow(cMatrix,cmap='gray')
    ax.set_xticks(np.arange(len(cMatrixLabels)))
    ax.set_xticklabels(cMatrixLabels)
    ax.set_yticks(np.arange(len(cMatrixLabels)))
    ax.set_yticklabels(cMatrixLabels)
    ax.set_xlabel('Severity Class (Actual)')
    ax.set_ylabel('Severity Class (Predicted)')
    ax.set_title(MdlName)
    for j in range(len(cMatrixLabels)):
        for k in range(len(cMatrixLabels)):
            ax.text(j-0.25,k,int(cMatrix[k,j]),color='blue',fontweight='semibold',fontsize=18)
    fig.colorbar(mappable=img,ax = ax, fraction=0.1)
    fig.tight_layout()
    return fig,ax

for i in range(len(Mdls['Mdl'])):
    Mdls['Mdl'][i] , \
    Mdls['Predictions'][i], \
    Mdls['Confusion Matrix'][i] = fitTestModel(Mdl=Mdls['Mdl'][i],MdlName=Mdls['MdlName'][i], 
                                                    XTrain=XTrain, YTrain=YTrain, XTest=XTest, YTest=YTest,
                                                    saveLocation='./Plots/report plots/mdlSelection/beta_{}.eps'.format(i))

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
def perceptronOnevsAll(XTrain,YTrain,XTest,YTest):
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

    return perceptronDict, classification, cMatrix, metrics(cMatrix)

## One vs One perceptron multiclass classifier
def perceptronOnevsOne(XTrain,YTrain,XTest,YTest):
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

    return perceptronDict, classification, cMatrix, metrics(cMatrix)


## Perceptrons
def perceptronsTrainTest(XTrain,YTrain,XTest,YTest):
    
    perceptrons = []
    print('One vs All')
    perceptronsDict = {'Binary Perceptrons': [],
                    'Classification': [],
                    'Confusion Matrix': [],
                    'Confusion Matrix Metrics': []}

    perceptronsDict['Binary Perceptrons'], \
    perceptronsDict['Classification'], \
    perceptronsDict['Confusion Matrix'],\
    perceptronsDict['Confusion Matrix Metrics'] = perceptronOnevsAll(XTrain,YTrain,XTest,YTest)

    perceptrons.append(perceptronsDict)

    print('One vs One')
    perceptronsDict = {'Binary Perceptrons': [],
                    'Classification': [],
                    'Confusion Matrix': [],
                    'Confusion Matrix Metrics': []}

    perceptronsDict['Binary Perceptrons'], \
    perceptronsDict['Classification'], \
    perceptronsDict['Confusion Matrix'],\
    perceptronsDict['Confusion Matrix Metrics'] = perceptronOnevsOne(XTrain,YTrain,XTest,YTest)

    perceptrons.append(perceptronsDict)
    return perceptrons

print('Perceptrons')
perceptrons = perceptronsTrainTest(XTrain,YTrain,XTest,YTest)
print('Perceptrons: Pre-Covid Data')
perceptronsPreCovid = perceptronsTrainTest(XTrainPreCovid,YTrainPreCovid,XTestPreCovid,YTestPreCovid)
print('Perceptrons: Post-Covid Data')
perceptronsPostCovid = perceptronsTrainTest(XTrainPostCovid,YTrainPostCovid,XTestPostCovid,YTestPostCovid)

# #############################################################################
# sys.exit("Just wanted correlation matrices lol")
# #############################################################################

from sklearn.svm import SVC
## One vs All SVC multi-class classifier
def svcOnevsAll(XTrain,YTrain,XTest,YTest):
    YTrain_dummies = pd.get_dummies(YTrain)
    binaryClassifiers = []
    for c,label in enumerate(YTrain_dummies.columns):
        clf = SVC(probability=True).fit(XTrain,YTrain_dummies[YTrain_dummies.columns[c]])
        binaryClassifiers.append(clf)

    predictions = []
    for clf in binaryClassifiers:
        predictions.append(clf.predict_proba(XTest))
    predProb = np.array(predictions).T

    classification = []
    for pred in predProb[1]:
        classification.append(YTrain_dummies.columns[np.where(pred==max(pred))[0][0]])
    classification = np.array(classification).reshape(-1)

    cMatrix = confusionMatrix(classificationTest = classification,
                              Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))
    
    svcDict = {'Binary Classifiers': binaryClassifiers,
               'Predictions': classification,
               'Confusion Matrix': cMatrix,
               'Confusion Matrix Metrics': metrics(cMatrix)}
    return svcDict

## One vs One SVC multi-class classifier
def svcOnevsOne(XTrain,YTrain,XTest,YTest):
    YTrain_dummies = pd.get_dummies(YTrain)
    YTest_dummies = pd.get_dummies(YTest)
    binaryClassifiers = np.empty((len(YTrain_dummies.columns),len(YTrain_dummies.columns)), dtype='object')
    for c1,label1 in enumerate(YTrain_dummies.columns):
        for c2,label2 in enumerate(YTrain_dummies.columns):
            if c1<c2:
                y1 = YTrain_dummies[YTrain_dummies.columns[c1]]
                y2 = YTrain_dummies[YTrain_dummies.columns[c2]]
                y = y1.iloc[ list(np.where( ((y1==1).astype(int) + (y2==1).astype(int))==1 )[0]) ]
                x = XTrain[list(y.index.astype(int))]
                clf = SVC(probability=False).fit(x,y)
                binaryClassifiers[c1][c2] = clf

    ## Predicitons from each model
    pred = pd.DataFrame(np.zeros(len(YTest_dummies)))
    for c1,label1 in enumerate(YTrain_dummies.columns):
        for c2,label2 in enumerate(YTrain_dummies.columns):
            if c1<c2:
                col = '{}_{}'.format(label1,label2)
                pred[col] = binaryClassifiers[c1][c2].predict(XTest)
    pred = pred.drop(pred.columns[0],axis=1)

    ## Assign labels to every model's prediction
    predLabels = pred.copy()
    for c1,label1 in enumerate(YTrain_dummies.columns):
        for c2,label2 in enumerate(YTrain_dummies.columns):
            if c1<c2:
                col = '{}_{}'.format(label1,label2)
                vector = pred[col]
                vector[vector==1] = label1
                vector[vector==0] = label2
                predLabels[col] = vector

    # Voting for classification
    classification = pd.DataFrame(np.zeros(len(predLabels)),columns=['CLS'])
    from scipy.stats import mode
    for i in range(len(predLabels)):
        classification.iloc[i] = ( mode(predLabels.iloc[i])[0].reshape(-1) )[0]
    
    cMatrix = confusionMatrix(classificationTest = classification,
                              Ytest = pd.Series(YTest))
    overallAccuracy, userAccuracy, producerAccuracy, kappaCoeff = metrics(cMatrix)
    print('Overall Accuracy: {}'.format(np.round(overallAccuracy,3)))
    print("User's Accuracy: {}".format(np.round(userAccuracy,3)))
    print("Producer's Accuracy: {}".format(np.round(producerAccuracy,3)))
    print('Kappa Coefficient: {}\n'.format(np.round(kappaCoeff,6)))
    
    svcDict = {'Binary Classifiers': binaryClassifiers,
               'Predictions': classification,
               'Confusion Matrix': cMatrix,
               'Confusion Matrix Metrics': metrics(cMatrix)}
    return svcDict

#################
print('SVM: One vs All')
randomNumbers = np.random.permutation(len(XTrain))
XTrainSVM = XTrain[randomNumbers[0:10000]]
YTrainSVM = YTrain[randomNumbers[0:10000]]
svcOnevsAllDict = svcOnevsAll(XTrainSVM,YTrainSVM,XTest,YTest)
print('SVM: One vs One')
svcOnevsOneDict = svcOnevsOne(XTrainSVM,YTrainSVM,XTest,YTest)

print('SVM: One vs All: Pre-Covid Data')
randomNumbers = np.random.permutation(len(XTrainPreCovid))
XTrainSVMPreCovid = XTrainPreCovid[randomNumbers[0:10000]]
YTrainSVMPreCovid = YTrainPreCovid[randomNumbers[0:10000]]
svcOnevsAllDictPreCovid = svcOnevsAll(XTrainSVMPreCovid,YTrainSVMPreCovid,XTestPreCovid,YTestPreCovid)
print('SVM: One vs One: Pre-Covid Data')
svcOnevsOneDictPreCovid = svcOnevsOne(XTrainSVMPreCovid,YTrainSVMPreCovid,XTestPreCovid,YTestPreCovid)

print('SVM: One vs All: Post-Covid Data')
randomNumbers = np.random.permutation(len(XTrainPostCovid))
XTrainSVMPostCovid = XTrainPostCovid[randomNumbers[0:10000]]
YTrainSVMPostCovid = YTrainPostCovid[randomNumbers[0:10000]]
svcOnevsAllDictPostCovid = svcOnevsAll(XTrainSVMPostCovid,YTrainSVMPostCovid,XTestPostCovid,YTestPostCovid)
print('SVM: One vs One: Post-Covid Data')
svcOnevsOneDictPostCovid = svcOnevsOne(XTrainSVMPostCovid,YTrainSVMPostCovid,XTestPostCovid,YTestPostCovid)

'''
Credible Intervals and Binomial Significance testing for sklearn models
Note: Just import libraries here: code is implemented in yuvraj.ipynb
'''
from regressionLib import credibleInterval, binomialSignificanceTest
# Implemented in notebook yuvraj.ipynb

'''
Cross Validation: random forest with AdaBoost
'''
# Implemented in notebook yuvraj.ipynb

'''
Neural Net: tensorflow
'''
# Implemented in notebook yuvraj.ipynb