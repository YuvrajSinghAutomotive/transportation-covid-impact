import sys

'''
System arguments:
    - sys.argv[0] : cleandata.py
    - sys.argv[1] : verbose = 0 [default] or 1
'''
verbose = sys.argv[1] if len(sys.argv)>1 else 0

'''
Download Dataset from Kaggle
'''
import kaggle
import zipfile
import os

# Download dataset if zip file not found
if 'us-accidents.zip' not in os.listdir():
    os.system('kaggle datasets download -d sobhanmoosavi/us-accidents')
else: print('Dataset already downloaded')
# extract data files
zipfile.ZipFile('us-accidents.zip').extractall()

'''
Clean Data: Import libraries
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

pd.set_option('display.max_columns', None)

dataDirectory = './'

def find_csv_files( path , suffix=".csv" ):
    files = os.listdir(path)
    return [ filename for filename in files if filename.endswith( suffix ) ]

filenames = find_csv_files( path=dataDirectory , suffix='.csv' )
print(filenames)

'''
Extract raw data from csv file(s)
'''
rawDataDf = []
for file in filenames:
    rawDataDf.append( pd.read_csv( dataDirectory + file ) )

print('Number of Raw Data: DataFrames extracted: {}'.format(len(rawDataDf)) )

## Preserve rawData in a DataFrame: use a DataFrame 'data' to make all changes
rawData = rawDataDf[0]
N = len(rawData)        ## number of observations in raw data

data = rawData.copy()

if verbose!=0:
    ## First look at dataset
    print(data.head())
    print(data.dtypes)

'''
Pre-Processing
Create feature: length of a traffic incident in seconds
This feature enables us to remove Start_Time and End_Time columns from the dataset
'''
startTime = []; invalidStartTimeCounter = 0
endTime = [];   invalidEndTimeCounter = 0
invalidTimeFormatIdx = []
    
for i in range(len(rawData)):
    startDateString = data['Start_Time'][i]
    endDateString = data['End_Time'][i]
    try:
        startTime.append(datetime.fromisoformat(startDateString))
        endTime.append(datetime.fromisoformat(endDateString))
    except ValueError:
        try: startTime.append(datetime.fromisoformat(startDateString))
        except: invalidStartTimeCounter += 1
        try: endTime.append(datetime.fromisoformat(endDateString))
        except: invalidEndTimeCounter += 1

def cleanData(verbose):
    import kaggle
    import zipfile
    import os

    '''
    Download Dataset from Kaggle
    '''
    # Download dataset if zip file not found
    if 'us-accidents.zip' not in os.listdir():
        os.system('kaggle datasets download -d sobhanmoosavi/us-accidents')
    else: print('Dataset already downloaded')
    # extract data files
    zipfile.ZipFile('us-accidents.zip').extractall()

    '''
    Clean Data: Import libraries
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime

    pd.set_option('display.max_columns', None)

    dataDirectory = './'

    def find_csv_files( path , suffix=".csv" ):
        files = os.listdir(path)
        return [ filename for filename in files if filename.endswith( suffix ) ]

    filenames = find_csv_files( path=dataDirectory , suffix='.csv' )
    print(filenames)

    '''
    Extract raw data from csv file(s)
    '''
    rawDataDf = []
    for file in filenames:
        rawDataDf.append( pd.read_csv( dataDirectory + file ) )

    print('Number of Raw Data: DataFrames extracted: {}'.format(len(rawDataDf)) )

    ## Preserve rawData in a DataFrame: use a DataFrame 'data' to make all changes
    rawData = rawDataDf[0]
    N = len(rawData)        ## number of observations in raw data

    data = rawData.copy()

    if verbose!=0:
        ## First look at dataset
        print(data.head())
        print(data.dtypes)

    '''
    Pre-Processing
    Create feature: length of a traffic incident in seconds
    This feature enables us to remove Start_Time and End_Time columns from the dataset
    '''
    startTime = []; invalidStartTimeCounter = 0
    endTime = [];   invalidEndTimeCounter = 0
    invalidTimeFormatIdx = []
        
    for i in range(len(rawData)):
        startDateString = data['Start_Time'][i]
        endDateString = data['End_Time'][i]
        try:
            startTime.append(datetime.fromisoformat(startDateString))
            endTime.append(datetime.fromisoformat(endDateString))
        except ValueError:
            try: startTime.append(datetime.fromisoformat(startDateString))
            except: invalidStartTimeCounter += 1
            try: endTime.append(datetime.fromisoformat(endDateString))
            except: invalidEndTimeCounter += 1
            
            invalidTimeFormatIdx.append(i)
            continue

    incidentTimeLength = np.array(endTime) - np.array(startTime)

    # Remove data points with invalid times
    data = data.drop(invalidTimeFormatIdx, axis=0)

    incidentTimeLength_seconds = []            ## compute time in seconds
    for t in list(incidentTimeLength):
        incidentTimeLength_seconds.append( t.total_seconds() )
    data['incidentTimeLength_seconds'] = incidentTimeLength_seconds

    data = data.drop(['Start_Time','End_Time'],axis=1)      ## date time columns no longer needed

    print('Deleted: {} values ({}% of raw data) because of invalid time format'.format( len(invalidTimeFormatIdx),100*len(invalidTimeFormatIdx)/N ) )

    '''
    Add column: is the incident happening pre-covid or post-covid?
    '''
    preCovid = []
    for i in range(len(data)):
        if startTime[i].year>=2020 and startTime[i].month>2:
            preCovid.append(False)
        else:
            preCovid.append(True)
    data['preCovid'] = preCovid

    '''
    Remove columns that are not numerically relevant
    Columns removed:
        - ID
        - Latitude and Longitude: not needed for now [not countable for computation]: we already have distance. Use them later for geo-location
        - Description: string with address of incident
        - Number: location on street where incident happened [not relevant for learning tasks]
        - Street: street name where incident happened [too many categoricals and irrelevant]
        - City and County: removed from now: we may change this to a smaller set of categorical values using some geographical techniques
        - Zipcode: not relevant for training since city and county captures this info without going into micro-level details. [Not a countable number]
        - Country: column's variance is 0 since the only country in the dataset is US
        - Airport_Code: too many categorical variables
        - Weather_Timestamp: [not fit for computation] we already have a weather condition categorical variable
    '''
    if verbose!=0:
        for col in data.columns:
            if data[col].dtype == object:
                print('column: {}. Number of possible values: {} \n\tValues = {}\n'.format(col, len(data[col].unique()), data[col].unique()) )

    data = data.drop(['ID','Description','Number','Street','Zipcode','Country','Airport_Code','Weather_Timestamp'],axis=1)

    ## Remove City and County for now
    data = data.drop(['City','County'],axis=1)

    ## remove latitude and Longitude for now: use them later for geographical summarizing of data
    data = data.drop(['Start_Lat','Start_Lng','End_Lat','End_Lng'],axis=1)
    if verbose!=0:
        print(data)
        print(data.dtypes)

    '''
    Drop NaN and missing values
    '''
    data = data.dropna()

    ## Print values in all categorical variables
    print('\n\n################################################################### \n')
    for col in data.columns:
        if data[col].dtype == object:
            print('column: {}. Number of possible values: {} \n\tValues = {}\n'.format(col, len(data[col].unique()), data[col].unique()) )

    print('\n\n################################################################### \n')

    print('Percentage of data retained after Pre-Processing: {} % \nNumber of observations retained: {}'.format( 100*len(data)/len(rawData) , len(data)) )

    print('\n\n################################################################### \n')

    ## Print Cleaned Data Variables (no dummies)
    print('\n All Variables (categoricals displayed without one-hot dummy conversion)\n{})'.format(data.dtypes) )

    '''
    Generate one-hot dummy variables
    '''
    for col in data.columns:
        if data[col].dtype == object:
            data = pd.get_dummies(data,columns=[col])
    for col in data.columns:
        if data[col].dtype == bool:
            data[col] = data[col].astype(int)     ## convert bool to int

    return data