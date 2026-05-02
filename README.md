# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and split it into features (X) and target (y).
2. Normalize the feature values using standard scaling.
3.Split the dataset into training and testing sets, then train the SGD classifier.
4.Predict test results and evaluate performance using accuracy and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Suvetha k
RegisterNumber: 212225040444 
*/
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

df=load_iris()
x=df.data
y=df.target

scaler= StandardScaler()
x=scaler.fit_transform(x)
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model=SGDClassifier(max_iter=1000,random_state=42)
model.fit(x_train,y_train)
y_pred= model.predict(x_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n",classification_report(y_test,y_pred))
```

## Output:
<img width="884" height="364" alt="Screenshot 2026-05-02 194442" src="https://github.com/user-attachments/assets/855b8f61-b3c6-4246-8e4e-6f513cfd49a3" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
