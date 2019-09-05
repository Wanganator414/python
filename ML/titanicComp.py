#%% [markdown]

### This kernel serves as practice documenting a supervised ML learning process without scrableing everything.
# ---


# %% [markdown]
# 1.Import required libraries here (might need to remove unused ones later)

# Core libs
import pandas as pd
import os
import numpy as np

# Supporting libs for cleaning and organising data
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Model imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Feature Engineering and visualization
import seaborn as sns
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Accuracy checking and warnings
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score
import warnings

# Ignore random future warning messages :)
warnings.simplefilter(action="ignore", category=FutureWarning)



#%% [markdown]
# 2.Import train and test datasets

#Import data from respective paths
trainDataPath = "ML\data\\titanic\\train.csv"
trainDataFull = pd.read_csv(trainDataPath)

testDataPath = "ML\data\\titanic\\test.csv"
testDataFull = pd.read_csv(testDataPath)

#Save path for later reference while saving to local file
savePath = "ML\data\\titanic"
# filteredDtypes.dtypes.to_csv(os.path.join(savePath, r'filteredFeatures.csv'))


#%% [markdown]
# 3.Separate X,y,and xTest

y = trainDataFull.Survived
X = trainDataFull.drop(["Survived"], axis=1)
xTest = testDataFull

#This combined dataframe is for holistic reference and feature engineering,make sure not to mix this up with actual predictions

#Temporarily combine train and test data for holistic analysis
all_data = pd.concat([X,xTest], ignore_index=False)
#ignore_index makes sure the actual indices do not go crazy when stacking dataframes



#%% [markdown]
# Method to check for NaN values in dataframes
def missingVals(df_In):
    missingCols = [col for col in df_In.columns.values if df_In[col].isnull().any()]
    missData = {
        "Index Col": missingCols,
        "Amount": [df_In[x].isnull().sum() for x in missingCols],
        "% Missing Values": [
            (df_In[x].isnull().sum() / len(df_In[x].index)) * 100 for x in missingCols
        ],
        "Type": [df_In[x].dtype for x in missingCols],
    }
    missingRates = pd.DataFrame(missData).set_index("Index Col")

    return missingRates.sort_values(by="% Missing Values", axis=0, ascending=False)

# trainDataFull.isnull().sum()
# Cabin and Age needs to be imputed intelligently, embarked can be imputed simply
# df.query() can be used similarly to grep

#%% [markdown]

#Check out the shape of both train and test data, as well as the features and data types

print(f"Train data shape and indices: {X.shape}")
print(f"Test data shape and indices: {xTest.shape}")

print("\nAvailable data types for each given feature: ")
X.describe()
X.dtypes

#%% [markdown]
# NaN values in Train
missingVals(X)

#%% [markdown]
# NaN values in Test
missingVals(xTest)

#%% [markdown]
# `Check for weird numerical values that don't make sense in each column`
# trainDataFull.describe()

# min fare is 0, which is odd, maybe use Avg cabin prices to gauge the fares
def editFare(df_In):
    """
    Takes df_In, type dataframe and edits the dataframe["Fare"] inplace based on passenger social class
    """
    # meanClassfare = df_In.pivot_table("Fare", index="Pclass", aggfunc="mean")
    # meanClassfare = list(meanClassfare["Fare"])
    # for x in range(len(df_In["Fare"])):
    # for x in df_In.index:
    #     if df_In.Fare[x] == 0:
    #         if df_In["Pclass"][x] == "1":
    #             df_In.Fare[x] = meanClassfare[0]
    #         if df_In["Pclass"][x] == "2":
    #             df_In.Fare[x] = meanClassfare[1]
    #         else:
    #             df_In.Fare[x] = meanClassfare[2]
    # # print(meanClassfare)

    for x in df_In.index:
        if df_In.Fare[x]==0:

    print("Replaced /$0 fares with mean fare of social class")
    # print(df_In["Fare"].apply(lambda x: df_In["Pclass"][x] in meanClassfare))


# editFare()
# X.describe()


#%% [markdown]
# #Pipelines start here
numCols = [col for col in X if X[col].dtype in ["int64", "float"]]

objCols = [col for col in X if X[col].dtype in ["object"]]

print(f"objCols: ${objCols} \nnumCols: ${numCols}")


numerical_transformer = Pipeline(steps=[("fillNA", SimpleImputer(strategy="mean"))])

# Preprocessing for categorical data
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, ["Age"]),
        ("cat", categorical_transformer, ["Cabin", "Embarked"]),
    ]
)


#%%
# Replace 2 NaN values based on Avg starting port for their class and gender
X.Embarked.fillna("C", inplace=True)

# Create special Cabin type for people without cabins
X.Cabin.fillna("N", inplace=True)
# Leave only cabin letter, extra stuff creates noise
X.Cabin = [i[0] for i in X.Cabin]

# Edit pricing for empty values based on class
editFare(X)
editFare(xTest)

model = XGBClassifier(random_state=1, learning_rate=0.05, n_estimators=100)
model2 = ExtraTreesClassifier(random_state=1, n_estimators=100)
# my_pipe = Pipeline(steps=[("preprocess", preprocessor)])
my_pipe.fit_transform(X, y)
fit = model.fit(X, y)
score = -1 * cross_val_score(model2, X, y, cv=6, scoring="neg_mean_absolute_error")
print(f"MAE: {score.mean()}")
#%% [markdown]
# `Testing out vanilla features without feature engineering`
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(score) + 1), rfecv.grid_scores_)
plt.show()

#%%
# %% [markdown] ==================================================================
# Fit test info and get final predictions
testData = testDataFull
editFare(testData)
finalPredicts = my_pipe.predict(testData)

output = pd.DataFrame({"PassengerId": testData.PassengerId, "Survived": finalPredicts})
output.to_csv(os.path.join(savePath, r"finalOutputXGClass.csv"), index=False)

#%%
print(classification_report(y, my_pipe.predict(X)))

#%%
