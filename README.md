# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and read the Salary dataset
2. Check for any null values and preprocess the data
3. Split the data into Training dataset and Testing dataset
4. Import DecisionTreeRegressor from sklearn.tree and train the model
5. Test the model using the Testing dataset
6. Find the mean_squared_error and r2_score
7. Give input to the model and predict the results

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Sujithra D
RegisterNumber: 212222220052 
*/
import pandas as pd

data=pd.read_csv('/content/Salary (2).csv')
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
![image](https://github.com/user-attachments/assets/c3d2fad3-2dea-4c88-9673-ada508016bc1)
![image](https://github.com/user-attachments/assets/ccc79dbe-3f61-4871-b7d2-e94086352211)
![image](https://github.com/user-attachments/assets/8f2babe0-0b5b-4077-b361-701e20a9c919)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
