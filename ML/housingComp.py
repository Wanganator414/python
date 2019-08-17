# %% [markdown]
#Setup Code 
import pandas as pd
import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
import warnings

# %% [markdown]
# **Train Model**
warnings.simplefilter(action='ignore', category=FutureWarning)
# file setup and cleaning
trainDataPath = "python\data\housePriceComp1\\train.csv"
trainData = pd.read_csv(trainDataPath)

testDataPath = "python\data\housePriceComp1\\test.csv"
testData = pd.read_csv(testDataPath)  

# Get list of non object type columns
# filteredDtypes = trainData.loc[:, trainData.dtypes != "object"]
# print(filteredDtypes.dtypes)
savePath = "python/data/housePriceComp1"
# filteredDtypes.dtypes.to_csv(os.path.join(savePath, r'filteredFeatures.csv'))

trainFeatures = ["OverallQual","OverallCond","LotArea", "Fireplaces", "PoolArea",
 "YearBuilt", "TotRmsAbvGrd", "YearRemodAdd", "TotalBsmtSF", "MasVnrArea",
 "1stFlrSF","2ndFlrSF","FullBath","HalfBath","BedroomAbvGr","GarageCars",
 "GarageArea","ScreenPorch", "MiscVal", "MoSold", "YrSold", "OpenPorchSF",
"GrLivArea","3SsnPorch","KitchenAbvGr","GarageYrBlt","WoodDeckSF"]

# filter data for null values and replace them with some value
dummyTrain = trainData.fillna(0, inplace=False)
# print(dummyTrain.isnull().sum())
# print(trainData.isnull().sum())

# orginal untouched X and y features from train data
X = dummyTrain[trainFeatures]
y = dummyTrain.SalePrice  # SalePrice only in train set

#Remember to remove SalePrice column once Y is set, since you dont need it to predict prices

# Split test data into train and val sets to test for best leaf node amount
# NOTE: train_size = 0.x and test_size=0.x are options to size the split sets
#Many more parameters to tune
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

initialModel = ExtraTreesRegressor(random_state=1)
initialModel.fit(train_X, train_y)

# Test optimal leaf max for model

def getMae(leafAmt, testX, testy, valX, valy):
    """
    Given maximum amount of desired nodes, returns avg error compared to validation data

    Max leaf amount can be of type iterable

    etc:
        >>>getMar(100,foo,bar,foor,boor)
        >>>> mae
    """
    sampleModel = ExtraTreesRegressor(max_leaf_nodes=leafAmt, random_state=1)
    sampleModel.fit(testX, testy)
    actualTest = sampleModel.predict(valX)
    mae = mean_absolute_error(valy, actualTest)
    return(mae)


# dict of test amt : mean absolute error of model
maeDict = {leafMax: getMae(leafMax, train_X, train_y, val_X, val_y)
           for leafMax in range(100, 500, 1)}

optimalLeaf = [k for k, v in maeDict.items() if v == min(maeDict.values())]
print(f"Optimal Max Leaf: {optimalLeaf}\nMae: {min(maeDict.values())}")


# Retrain on full training data set and prep for final test dataset
finalModel = ExtraTreesRegressor(max_leaf_nodes=optimalLeaf[0], random_state=1)
finalModel.fit(X, y)


# %% [markdown]
# **Test Model**
# Reassign X for test set
dummyTest = testData.fillna(0, inplace=False)
testFeatures = trainFeatures
X2 = dummyTest[testFeatures]

finalPredicts = finalModel.predict(X2)

#Make sure to use TESTING data id, NOT TRAINING data id, since test has less row than train
output = pd.DataFrame({'Id': dummyTest.Id,
                       'SalePrice': finalPredicts})
output.to_csv(os.path.join(savePath, r'finalOutput.csv'), index=False)



#%%
print(output.head)

#%%
