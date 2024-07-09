#Importing libraries 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Perceptron

#Load the dataset
df=pd.read_csv("placement.csv")

print(df.shape)
df.head()

# plotting the dataset
sns.scatterplot(x=df['cgpa'],y=df['resume_score'],hue=df['placed'])

x=df.iloc[:,0:2]
y=df.iloc[:,-1]

p=Perceptron()

p.fit(x,y)

p.coef_

p.intercept_

pip install mlxtend

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(x.values, y.values, clf=p, legend=2)






