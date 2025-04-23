# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values. 

## Program:
```PYTHON
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by  : SANJAY M
RegisterNumber: 212223230187 
```
```PYTHON
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()

data.info()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
### data.head():
![image](https://github.com/user-attachments/assets/ddec5c92-8bd1-4de0-9786-2bcf94183ba6)
### data.info():
![image](https://github.com/user-attachments/assets/4b3cac95-ab30-4652-84f9-562059ded9fd)
### x.head():
![image](https://github.com/user-attachments/assets/9ed2e844-9b93-41e3-b33e-16c7b8bb7f69)
### Accuray:
![image](https://github.com/user-attachments/assets/b8681dcf-4dde-455b-b0bd-4f4c4e173ea9)
### Predict:
![image](https://github.com/user-attachments/assets/999c2837-a8e4-4dce-91e0-e092bd539a6b)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
