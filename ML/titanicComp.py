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

#There seems to be a **very large percentage of missing Cabin and Age data** in the training set. Percentages this high should be reconsidered and **not** deleted, since that would entail losing a lot fo potentially valuable data.
#<br/><br/>
#There also seems to be 2 people missing embarkation information, we will look at that later. We won't simply replace them with the most common occupance in their column, since the amount of samples are limited, arbitrarily altering 1 or 2 will have a noticeable effect on model quality.
#%% [markdown]
# NaN values in Test
missingVals(xTest)

#%% [markdown]

#Similar to the Training set, the Test set is also missing very high percentages of Cabin and Age data, we will address those collectively with the Training set. Since Cabin names could be shared by both sets.
#<br/><br/>
#There seems to be a single missing Fare values in the test set, we will look at that value later.


#%% [markdown]
#Now that we had a good understanding of what seems to be missing in both Train and Test sets, let us have a look at the Cabin column in the combined dataset.

#Looks at how many unique values there are
all_data.Cabin.nunique()

#%%[markdown]
#As seen above, there are over 186 unique Cabin values, that is quite a mess, let us have an even closer look.

#Gives occupance of each unique value
all_data.Cabin.value_counts().sort_values()

#%% [markdown]
#From the looks of it, aside from the NaN Cabin values, most Cabins start with an English letter and some numbers. There are even passengers who have assigned multiple cabin, which could be the result of traveling with family.
# <br/><br/>
# From a cardinality standpoint, this is simply too much noise in the data, we can simplify that by rewriting all the Cabins into simply their letter counterparts, ie. C76 --> C.
# <br/><br/>
# Regarding the NaN values, we can assume they were not assigned a Cabin due to pricing or social class, OR, they were not recorded correctly. We will have to check on this.

#Show passenger info for ones who have NaN as Cabin value, sorted by social class
all_data[all_data.Cabin.isnull()].sort_values(by="Pclass")

#%% [markdown]
#Now, according to what is shown above, passengers in NaN cabins have various Fare value and Pclass values. Thus we can safely eliminate the possibility that those without a cabin are of low social class.
#<br/><br/> 
# The likely possibility has to due to with inadequate documentation of the data.
#<br/><br/>
#For now we will go through with our rewriting of Cabin names and assign NaN values to Cabin "N".
all_data.Cabin.fillna("N",inplace=True)
#Assign first letter to Cabin name
all_data.Cabin = [i[0] for i in all_data.Cabin]

#%% [markdown]
#Before we can take care of the large amount of "N" cabins, we will consider how we will determine how each passenger will be put into which Cabin.
#<br/><br/>
#The most common method for doing so would be by ticket price, or in this case. The "Fare" feature, we can get the average Fare for each Cabin class and then allocate the passengers in cabin N to better assign passengers into their likely Cabin placements.
#<br/><br/>

#Avg fare price per Cabin type:

all_data.groupby("Cabin")['Fare'].mean().sort_values()


#%%[markdown]

#Based on the previous observation, we can create a function to help us group passengers into certain cabins based on price range.
def assignCabin(i):
    cabin = 0
    if i<16:
        cabin = "G"
    elif i>=16 and i<27:
        cabin = "F"
    elif i>=27 and i<38:
        cabin = "T"
    elif i>=38 and i<47:
        cabin = "A"
    elif i>= 47 and i<53:
        cabin = "E"
    elif i>= 53 and i<54:
        cabin = "D"
    elif i>=54 and i<116:
        cabin = 'C'
    else:
        cabin = "B"
    return cabin
    
#%% [markdown]
# But before we call the function, recall that there was a missing fare value in the Test dataset as well as a bunch of passengers having $0 as their "Fare" values, as seen here:
len(all_data.query("Fare==0"))
#Let's look at the missing Fare value in detail.
all_data[all_data.Fare.isnull()]

#%% [markdown]

#We see that Mr.Storey is a male class 3 citizen who embarked from port "S", we can take the average of all males in social class 3 who also boarded the Titanic at port "S" to subsitute for Mr.Storey's fare.
storneyFare = all_data[(all_data.Sex=="male") & (all_data.Embarked=="S") & (all_data.Pclass==3)].Fare.mean()
#Replace the NaN value
all_data.Fare.fillna(storneyFare,inplace=True)


#%% [markdown]
#Now we run the cabin assignment function on the N cabins

#separate cabin column into ones with N and ones without
cabins_N=all_data[all_data.Cabin=="N"]
cabins_NoN=all_data[all_data.Cabin!="N"]
#Apply function to each item in N column
cabins_N = cabins_N.Fare.apply(lambda x: assignCabin(x))
#Set cabin_NoN to series instead of dataframe, for later concatonation
cabins_NoN=cabins_NoN.Cabin

#Combine the refactored cabin data and replace original
all_data.Cabin = pd.concat([cabins_N,cabins_NoN],ignore_index=True)
#%%
#<br/><br/>
#Recall that we are missing 2 embarked values, let us fix that first.
#<br/><br/>

#Two NaN passengers are both class 1, in cabin B, and have Fare $80, so technically should be same embarkation port
all_data[all_data.Embarked.isna()]

#%%[markdown]
# The minimun value for Fare seems to be 0, which is weird. We will go check in detail why that is the case.

#%% [markdown]

def editFare(df_In):
    """
    Takes df_In, type dataframe and edits the dataframe["Fare"] in place based on passenger social class
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
