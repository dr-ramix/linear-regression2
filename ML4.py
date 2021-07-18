import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
diabet=load_diabetes()
x=diabet.data
y=diabet.target
model=LinearRegression()
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=1)
model.fit(xtrain,ytrain)
ypered=model.predict(xtest)
mse=mean_squared_error(ytest,ypered)
print(mse)
