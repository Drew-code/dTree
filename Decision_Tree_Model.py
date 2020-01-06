import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


data = pd.read_csv('Sample_data')
data = test_data.drop('unwanted variable',axis=1)
print(data.info())
print(data.head())




X = data.drop(['y_variable'],axis=1)
y = data['y_variable']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

dtree_predict = dtree.predict(X_test)

print(classification_report(y_test,dtree_predict))
print(confusion_matrix(y_test,dtree_predict))



