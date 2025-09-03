# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. 1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Keshavarthini B
RegisterNumber: 212224040158

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)

print(df.head())
print(df.tail())

x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:

DATASET


<img width="195" height="521" alt="image" src="https://github.com/user-attachments/assets/0cd8884c-fe5d-4cb2-9edc-5f2a530009cf" />


HEAD VALUES


<img width="162" height="117" alt="image" src="https://github.com/user-attachments/assets/5037af34-1734-4dd3-9e46-70d04520e9cf" />


TAIL VALUES


<img width="160" height="117" alt="image" src="https://github.com/user-attachments/assets/3592ff05-b83c-4f97-8cf5-f325651ff6c3" />


X AND Y VALUES


<img width="610" height="517" alt="image" src="https://github.com/user-attachments/assets/2f8095ab-de09-4996-b2cc-db132c7d420e" />


PREDICTED VALUES OF X AND Y


<img width="666" height="67" alt="image" src="https://github.com/user-attachments/assets/c9c07ff2-03bb-44a4-8b4c-2ba9d77a50dd" />


TRAINING SET


<img width="728" height="518" alt="image" src="https://github.com/user-attachments/assets/0555e2c0-0c92-4198-bbde-0fabca4962ea" />


TESTING SET AND MSE,MAE and RMSE



<img width="727" height="612" alt="image" src="https://github.com/user-attachments/assets/516ca147-3524-4ba0-947a-eaead0f738cc" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
