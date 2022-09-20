import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
color=sns.color_palette()
import sklearn.metrics as metrics
import warnings
warnings.filterwarnings("ignore")

default=pd.read_csv("default.csv")
default.head()
default.shape
default.describe() #gives out certain useful information
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(y=default["balance"])

plt.subplot(1,2,2)
sns.boxplot(y=default["income"])
plt.show()

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(default["student"])

plt.subplot(1,2,1)
sns.countplot(default["default"])
plt.show()

default["student"].value_counts()

default["student"].value_counts()(normalize=True)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(default["default"],default["balance"])

plt.subplot(1,2,2)
sns.countplot(default["default"],default["income"])
plt.show()

pd.crosstab(default["student"],default["default"],normalize="index").round()
sns.heatmap(default[["balance","income"]].corr(),annot=True)
plt.show()


default.isnull().sum()

q1,q3=default["balance"].quantile([.25,.75])
iqr=q3-q1
ll=q1-1.5*iqr
ul=q3-1.5*iqr

df=default[default["balance"]>ul]
print(df)
df["default"].count()

df["default"].value_counts(normalize=True)
df["default"].value_counts()

default["balance"]=np.where(default["balance"])>ul,ul,default["balance"]
sns.boxplot(y=default["balance"])
plt.show()
default=pd.get_dummies(default,drop_first=True)
default.head()
default.columns=["balance","income","default","student"]