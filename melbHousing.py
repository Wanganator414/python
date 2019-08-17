#%% [markdown]
#SETUP IMPORTS HERE, RUN ONCE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

#%%
#path to data
filePath = "python\data\melb_data.csv"

housingData = pd.read_csv(filePath)
#drop ALL rows with missing values in them 

housingData=housingData.dropna()
#print(housingData.columns,  file=open('python\data\log.txt', 'w'))
#There is significant drop in samples after dropNA.....maybe try another way?
features=["Rooms","Bathroom","Car","Landsize","YearBuilt"]

#Prediction target variale (global unsliced)
y=housingData.Price
#Variables used to predict (global unsliced)
X=housingData[features]

#Split one data source into test and validation groups
testX,valX,testy,valy = train_test_split(X,y,random_state=1)

#utility func for testing node#
def getMae(leafAmt, testX, testy, valX, valy):
    """
    Given maximum amount of desired nodes, returns avg error compared to validation data

    Max leaf amount can be of type iterable

    etc:
        >>>getMar(100,foo,bar,foor,boor)
        >>>> mae
    """
    model = DecisionTreeRegressor(max_leaf_nodes=leafAmt,random_state=1)
    model.fit(testX,testy)
    actualTest = model.predict(valX)
    mae = mean_absolute_error(valy,actualTest)
    return(mae)

maeDict = {leafMax: getMae(leafMax,testX,testy,valX,valy) for leafMax in range(10,500,1)}

#ALL MAE HERE
#print(maeDict,"\n")

optimalNodesCount=min(maeDict,key=maeDict.get)

#Most Optimal Node #
print(f"Optimal Node: {optimalNodesCount}")


#Testing on larger dataset
bigModel = DecisionTreeRegressor(
    max_leaf_nodes=optimalNodesCount, random_state=1)
bigModel.fit(X, y)
bigPredict = bigModel.predict(X)
bigMae = mean_absolute_error(y, bigPredict)
print(f"MAE: {bigMae}")

#%% [markdown]
#*Snippet Testing, styling, saving to files etc*
file = open(r"python\data\log.txt", "w")
dat = {
"Original Stats":housingData["Price"],"Predicted":bigPredict
}
info = pd.DataFrame(dat,columns=["Original Stats","Predicted"])
print(info.dtypes)
#change add sci notations for better reading
#info = info.astype(object).style
info=info.astype(object)
#formatted & styled chart
# info.style.format('${0:,.2f}')
info.to_csv("python\data\info.csv")  # saved to pain old csv
#frees up memory form file
file.close()
#%% [markdown]
#**Overview of dtypes in relevant columns**
dtype=housingData[features].dtypes
print(dtype)











