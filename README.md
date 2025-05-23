# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: JESPIAH SHIHANA P S
RegisterNumber:  212223040077
*/
```
```
import pandas as pd


data = pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position", "Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)
y_pred = df.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
mse

r2 = metrics.r2_score(y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:

![image](https://github.com/user-attachments/assets/015d99c8-ebbf-4040-9173-3730c35f3fcc)

![image](https://github.com/user-attachments/assets/4cca7468-bd4d-4148-b77b-21ea5eb13717)

![image](https://github.com/user-attachments/assets/915cedac-acb0-4946-8d44-9c1abc58832f)

![image](https://github.com/user-attachments/assets/e590e5c9-1968-4269-ade5-21c81593ee67)

![image](https://github.com/user-attachments/assets/25dca085-3539-47d6-94ab-7d6bb7e1e7c5)

![image](https://github.com/user-attachments/assets/6362e1cb-83e4-4e6a-bbd2-4a50d181137d)

![image](https://github.com/user-attachments/assets/63acd1e3-c2de-4171-83c8-b46525996120)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
