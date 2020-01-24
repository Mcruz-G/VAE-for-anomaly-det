import pandas as pd 
import numpy as np

# fixedDim = 28
def get_data(dataName, test = False):
    ### This function extracts data from local file and returns a
    ### dataFrame free from nans

    #Create an index vector to get only each 50 timesteps 
    indexVec = []

    #Extract time series from csv
    if test:
        path = 'coladas_test/'+ dataName + '.csv'
    else:
        path = 'coladas/'+ dataName + '.csv'

    dataMat = pd.read_csv(path, sep = ';') 
    dataMat = dataMat.fillna(0).drop(['date','NAME_COLADA'], axis = 1)
    #Let's make a test for single variable time series
    dataMat = dataMat.iloc[:,2]

    for index in range(10000):
        if index % 40 == 0:
            indexVec.append(index)
    dataMat = dataMat.iloc[indexVec]
    # dataMat = dataMat.iloc[:fixedDim, :]
    dataMat = np.transpose(dataMat.values)
    # print(dataMat.shape)
    return dataMat


def normalize(data):
    ## Pretty obvious, but still:
    ## This function normalizes data to get values in -1, 1 range. 

    for colada in range(data.shape[0]):
        row = data[colada,:]
        maxVal, minVal = np.max(row), np.min(row)
        if abs(maxVal - minVal) != 0:
            data[colada,:] = (row - minVal )/(maxVal - minVal)
    return data