
import numpy as np
import pandas as pd
import datetime

dataBefore = "path to train-item-views.csv" #Path to Original Training Dataset
dataAfter = 'Preprocessed_data' #Path to Processed Dataset Folder
dayTime = 86400 #Validation Only one day = 86400 seconds

def removeShortSessions(data):
    #delete sessions of length < 1
    sessionLen = data.groupby('SessionID').size() #group by sessionID and get size of each session
    data = data[np.in1d(data.SessionID, sessionLen[sessionLen > 1].index)]
    return data

#Read Dataset in pandas Dataframe (Ignore Category Column)
train = pd.read_csv(dataBefore, sep=';', header=None, usecols=[0,1,2,3,4]) 
train.columns = ['SessionID', 'UserID', 'ItemID', 'Time', 'EventDate'] #Headers of dataframe
train = train[1:]

#remove sessions of less than 2 interactions
# train = removeShortSessions(train)
#delete records of items which appeared less than 5 times
itemLen = train.groupby('ItemID').size() #groupby itemID and get size of each item
train = train[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
#remove sessions of less than 2 interactions again
train = removeShortSessions(train)

train['Time'] = train['Time'].astype(int)
train['Time'] = train.swifter.apply(lambda r: (datetime.datetime.strptime(r['EventDate'], "%Y-%m-%d") + datetime.timedelta(milliseconds=r['Time'])).timestamp(), axis=1)
# test.to_csv(dataAfter + 'recSys15Test.txt', sep=',', index=False)
#Separate Training set into Train and Validation Splits
timeMax = train.Time.max()
sessionMaxTime = train.groupby('SessionID').Time.max()
sessionTrain = sessionMaxTime[sessionMaxTime < (timeMax - dayTime)].index #training split is all sessions that ended before the last 2nd day
sessionValid = sessionMaxTime[sessionMaxTime >= (timeMax - dayTime)].index #validation split is all sessions that ended during the last 2nd day
trainTR = train[np.in1d(train.SessionID, sessionTrain)]
trainVD = train[np.in1d(train.SessionID, sessionValid)]
#Delete records in train split where items are not in training split
trainVD = trainVD[np.in1d(trainVD.ItemID, trainTR.ItemID)]
#Delete Sessions in validation split which are less than 2
trainVD = removeShortSessions(trainVD)
#Convert To CSV
print('Training Set has', len(trainTR), 'Events, ', trainTR.SessionID.nunique(), 'Sessions, and', trainTR.ItemID.nunique(), 'Items\n\n')
trainTR.to_csv(dataAfter + 'recSys15TrainOnly.txt', sep=',', index=False)
print('Validation Set has', len(trainVD), 'Events, ', trainVD.SessionID.nunique(), 'Sessions, and', trainVD.ItemID.nunique(), 'Items\n\n')
trainVD.to_csv(dataAfter + 'recSys15Valid.txt', sep=',', index=False)
