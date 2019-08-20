# %% [markdown]
# `Setup Code`
# Core libs
import pandas as pd
import os
import numpy as np

# Supporting libs for cleaning and organising data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Model imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Accuracy checking and warnings
from sklearn.metrics import mean_absolute_error
import warnings


# Ignore random future warning crap
warnings.simplefilter(action="ignore", category=FutureWarning)

# Import data from respective paths
trainDataPath = "ML\\data\housePriceComp1\\train.csv"
trainData = pd.read_csv(trainDataPath)

testDataPath = "ML\\data\housePriceComp1\\test.csv"
testData = pd.read_csv(testDataPath)

savePath = "python/data/housePriceComp1"
# filteredDtypes.dtypes.to_csv(os.path.join(savePath, r'filteredFeatures.csv'))

# %% [markdown]
# `Prep Data`

# Placeholder incase of polluting data
trainDataBeta = trainData

# trainDataBeta.SalePrice.dropna(axis=0,inplace=True);

# Get master Target and Prediction feature setup
y = trainDataBeta.SalePrice
# Get every feature except price for now...
X = trainDataBeta.drop(["SalePrice"], axis=1)

# SEPARATE training and validation data, with larger training size vs validation size
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=1
)


# SEPARATE numerical with categorical data

numCols = X_train_full.select_dtypes(exclude=["object"])
objCols = X_train_full.select_dtypes(include=["object"])

# (numCols.isnull().sum(axis=0))
# print(objCols.isnull().sum(axis=0))

# PRUNE string columns for cardinality, aka amount of unique values in said column
# High cardinality columns are pretty bad for One Hot encoding, may be candidates for label encoding though
cardThresh = (
    10
)  # random number for amount of unique values, may need better way to determine this
hCardCols = [col for col in objCols.columns if objCols[col].nunique() > cardThresh]
objCols = objCols.drop(list(hCardCols), axis=1, inplace=False)


# Keep selected columns only
totalCols = list(objCols + numCols)
X_train = X_train_full[totalCols].copy()
X_valid = X_valid_full[totalCols].copy()

# # CHECK FOR % NAN values in each column
# nanCols = trainDataBeta.columns.isnull()

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])

#Finishing pipelining tomorrow ....yawn Zzzzzz
#%%

# filter data for null values and replace them with some value
# dummyTrain = trainData.fillna(0, inplace=False)
# print(dummyTrain.isnull().sum())
# print(trainData.isnull().sum())

# orginal untouched X and y features from train data
X = dummyTrain[trainFeatures]
y = dummyTrain.SalePrice  # SalePrice only in train set

# Remember to remove SalePrice column once Y is set, since you dont need it to predict prices

# Split test data into train and val sets to test for best leaf node amount
# NOTE: train_size = 0.x and test_size=0.x are options to size the split sets
# Many more parameters to tune
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
    return mae


# dict of test amt : mean absolute error of model
maeDict = {
    leafMax: getMae(leafMax, train_X, train_y, val_X, val_y)
    for leafMax in range(100, 500, 1)
}

optimalLeaf = [k for k, v in maeDict.items() if v == min(maeDict.values())]
print(f"Optimal Max Leaf: {optimalLeaf}\nMae: {min(maeDict.values())}")


# Retrain on full training data set and prep for final test dataset
finalModel = ExtraTreesRegressor(max_leaf_nodes=optimalLeaf[0], random_state=1)
finalModel.fit(X, y)


# %% [markdown]
# `Test Model`
# Reassign X for test set
dummyTest = testData.fillna(0, inplace=False)
testFeatures = trainFeatures
X2 = dummyTest[testFeatures]

finalPredicts = finalModel.predict(X2)

# Make sure to use TESTING data id, NOT TRAINING data id, since test has less row than train
output = pd.DataFrame({"Id": dummyTest.Id, "SalePrice": finalPredicts})
output.to_csv(os.path.join(savePath, r"finalOutput.csv"), index=False)


#%%
print(output.head)

#%%
# Use imputation and encoding to better organize the info

