#%%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import datasets
from numpy.random import seed
from numpy.random import randn
from numpy import std
from numpy import mean
from matplotlib import pyplot
#import plotly.plotly as py
import pandas as pd
import os
import numpy as np
import warnings
import scipy
import sys
import csv
import datetime as dt
import time as tm
import matplotlib.pyplot as plt
import seaborn as sns
print("DONE Imports")
#%%
#provide data paths
dataPath = "python\data\\train.csv"

houseData=pd.read_csv(dataPath)

# # uses  (all rows, :) Row|Col -> True/False Series to exclude all Object type columns
# filteredDtypes=houseData.loc[:, houseData.dtypes != "object"]
# #print(filteredDtypes.dtypes)
# savePath="python\data"
# filteredDtypes.dtypes.to_csv(os.path.join(savePath, r'featureList.csv'))

#print(houseData.SalePrice)


# # seed random number generator
# seed(1)
# # # prepare data
# data1 = 20 * randn(1000) + 100
# data2 = data1 + (10 * randn(1000) + 50)
# # summarize
# print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))
# print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# # plot
# pyplot.scatter(data1, data2)
# pyplot.show()

plt.figure(figsize=(16, 6))
ok=sns.barplot(x=houseData.head().index,y=houseData.head()["SalePrice"])
plt.show()
# print('scipy: {}'.format(scipy.__version__))
# print('pandas: {}'.format(pd.__version__))
# print('numpy: {}'.format(np.__version__))
# print('Python: {}'.format(sys.version))

# # warnings.filterwarnings('ignore')
# # %precision 2

# print(os.listdir("python\data\housePriceComp1"))
# tm.time()
# dt.datetime.now


#%%
