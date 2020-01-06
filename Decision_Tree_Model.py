import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


stock = pd.read_csv(r"D:\Datasets\Amazon_Stock.csv")
stock = stock.drop('time_stamp',axis=1)
print(stock.info())
print(stock.head())




X = stock.drop(['Change'],axis=1)
y = stock['Change']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

dtree_predict = dtree.predict(X_test)

dtree_predict = dtree.predict(X_test)
print("Decision Tree Classification report")
print(classification_report(y_test,dtree_predict))
print("Decision Tree Confusion Matrix")
print(confusion_matrix(y_test,dtree_predict))



