import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


train = pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.duplicated().sum() # yinelenen veri var mı?
train.describe().T # değişkenlerin matematiksel olarak sınıflandırılması


# boş verileri doldurma
train.isnull().sum() 
train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Embarked'].fillna('S',inplace=True) 
train.drop(columns=['Cabin'],inplace=True)

test.isnull().sum()
test.drop(columns=['Cabin'],inplace=True)
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test['Age'].fillna(test['Age'].mean(),inplace=True)


# gereksiz alanları sildik.
train.drop(columns=['Name','Ticket'],inplace=True)
test.drop(columns=['Name','Ticket'],inplace=True)
train['temp_Family']=train['SibSp'] + train['Parch']
test['temp_Family']=test['SibSp'] + test['Parch']
train.drop(columns=['SibSp','Parch'],inplace=True)
test.drop(columns=['SibSp','Parch'],inplace=True)


# yaşları gruplandırma
ages= [0,16,30,45,50]
labels = ['Childern','Young Adults','Middle-aged Adults','Old Adults']
train['AgeGroup'] = pd.cut(train['Age'], bins=ages, labels=labels, right=False)
test['AgeGroup'] = pd.cut(train['Age'], bins=ages, labels=labels, right=False)


# aileleri sınıflandırma
persons= [0,1,5,7,11]
labels = ['Alone','Dou','Small','Large']
train['Family'] = pd.cut(train['temp_Family'], bins=persons, labels=labels, right=False)
test['Family'] = pd.cut(train['temp_Family'], bins=persons, labels=labels, right=False)


#Final datasetlerini oluşturduk.
final_train=pd.get_dummies(train, columns=['Pclass','Sex','Embarked','AgeGroup','Family'], drop_first=True)
final_test=pd.get_dummies(test, columns=['Pclass','Sex','Embarked','AgeGroup','Family'], drop_first=True)


x_test=final_train.drop(columns="Survived")
y_test=final_train['Survived']
x_train, x_test, y_train, y_test=train_test_split(x_test,y_test,test_size=0.2,random_state=42)


lr=LogisticRegression(random_state = 72)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


clf=DecisionTreeClassifier(random_state = 72)
clf.fit(x_train,y_train)
y_pred_D=clf.predict(x_test)


knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train, y_train)
y_pred_knn = knn_classifier.predict(x_test)


print("Lojistik Regresyon:" ,accuracy_score(y_test,y_pred))
print("Karar Agaclari:" ,accuracy_score(y_test,y_pred_D))
print("KNN:" ,accuracy_score(y_test, y_pred_knn))

pred=clf.predict(final_test)
submission = pd.DataFrame({
    "PassengerId":test['PassengerId'],
    "Survived": pred
})

submission.to_csv('Submission_file.csv',index=False)