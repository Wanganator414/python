# %% [markdown]
# `Setup Code`

# Core libs
import pandas as pd
import os
import numpy as np

# Supporting libs for cleaning and organising data
from sklearn.model_selection import train_test_split, cross_val_score
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
from sklearn.metrics import mean_absolute_error
import warnings

# Ignore random future warning crap
warnings.simplefilter(action="ignore", category=FutureWarning)

# Import data from respective paths
trainDataPath = "ML\data\\titanic\\train.csv"
trainDataFull = pd.read_csv(trainDataPath)

testDataPath = "ML\data\\titanic\\test.csv"
testDataFull = pd.read_csv(testDataPath)

savePath = "ML\data\\titanic"

# filteredDtypes.dtypes.to_csv(os.path.join(savePath, r'filteredFeatures.csv'))

y = trainDataFull.Survived
X = trainDataFull.drop(["Survived"], axis=1)

trainX, validX, trainY, validY = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=1
)

#%% [markdown]
# `Checking for NaN values for Imputation`

missingCols = [col for col in trainX.columns.values if trainX[col].isnull().any()]
missData = {
    "Index Col": missingCols,
    "% Missing Values": [
        (trainX[x].isnull().sum() / len(trainX[x].index)) * 100 for x in missingCols
    ],
    "Type": [trainX[x].dtype for x in missingCols],
}
missingRates = pd.DataFrame(missData).set_index("Index Col")

missingRates.sort_values(by="% Missing Values", axis=0, ascending=False)
# trainDataFull.isnull().sum()
# Cabin and Age needs to be imputed intelligently, embarked can be imputed simply

# df.query() can be used similarly to grep

#%% [markdown]
# `Check for weird numerical values that don't make sense in each column`
# trainDataFull.describe()

# min fare is 0, which is odd, maybe impute it to be average of each social class?
def editFare(df_In):
    """
    Takes df_In, type dataframe and edits the dataframe["Fare"] inplace based on passenger social class
    """
    meanClassfare = df_In.pivot_table("Fare", index="Pclass", aggfunc="mean")
    meanClassfare = list(meanClassfare["Fare"])
    for x in range(len(df_In["Fare"])):
        if df_In["Fare"][x] == 0:
            if df_In["Pclass"][x] == "1":
                df_In["Fare"][x] = meanClassfare[0]
            if df_In["Pclass"][x] == "2":
                df_In["Fare"][x] = meanClassfare[1]
            else:
                df_In["Fare"][x] = meanClassfare[2]
    # print(meanClassfare)
    print("Replaced /$0 fares with mean fare of social class")
    # print(df_In["Fare"].apply(lambda x: df_In["Pclass"][x] in meanClassfare))


# editFare()
trainX.describe()


#%% [markdown]
# #Pipelines start here
numCols = [col for col in trainX if trainX[col].dtype in ["int64", "float"]]

objCols = [col for col in trainX if trainX[col].dtype in ["object"]]


numerical_transformer = Pipeline(
    steps=[("fillNA", SimpleImputer(missing_values=np.nan, strategy="mean"))]
)


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
        ("cat", categorical_transformer,["Cabin", "Embarked"]),
    ]
)
#%%
model = XGBClassifier(random_state=1, learning_rate=0.1, n_estimators=150)
my_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
model2 = my_pipe.fit(trainX, trainY)
my_pipe.fit(trainX, trainY)
preds=my_pipe.predict(validX)
score = -1 * cross_val_score(
    my_pipe, trainX, trainY, cv=5, scoring="neg_mean_absolute_error"
)
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
testData=testDataFull
finalPredicts = my_pipe.predict(testData)

output = pd.DataFrame({"PassengerId": testData.PassengerId, "Survived": finalPredicts})
output.to_csv(os.path.join(savePath, r"finalOutputXGClass.csv"), index=False)

#%%
