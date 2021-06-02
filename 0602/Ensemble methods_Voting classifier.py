#Ensemble methods_Voting classidier
#Importing libraries
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_Interactivity = 'all'

#Import the data
train=pd.read_csv('Train.csv')     #讀入資料檔
print('total train data: ' + str(train.shape))
print(train.info())
print(train.describe())
print(train.isnull().sum())    # No missing values found.

for i in train.columns:
    print( i , ':', train[i].nunique())
print(train.head())

#dropping id column as its of no use
train = train.drop(['id'],axis=1)
print(train.head())

# My dataset has 8 categorical columns which we need to encode. Here I am using one hot encoding.

train_encode = pd.get_dummies(train)
print(train_encode.shape)


train_ind = train_encode.drop(['netgain'],axis=1)
target = train_encode['netgain']
print( 'Train independent dataset shape:', train_ind.shape , 'and', 'Train target dataset shape:', target.shape)

#Splitting the dataset to train and test.
X_train, X_test, y_train, y_test = train_test_split(train_ind, target, test_size=0.30, random_state=42)
print('X train size: ', X_train.shape)
print('y train size: ', y_train.shape)
print('X test size: ', X_test.shape)
print('y test size: ', y_test.shape)


# Ensemble of several different types of models: voting classifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

rfClf = RandomForestClassifier(n_estimators=500, random_state=0) # 500 trees. 
svmClf = SVC(probability=True, random_state=0) # probability calculation
logClf = LogisticRegression(random_state=0)
nbclf = GaussianNB()

# constructing the ensemble classifier by mentioning the individual classifiers.
clf2 = VotingClassifier(estimators = [('rf',rfClf), ('svm',svmClf), ('log', logClf),('nb',nbclf)], voting='soft') 

# train the ensemble classifier
clf2.fit(X_train, y_train)

from sklearn.metrics import precision_score, accuracy_score
x_actual, x_pred = y_train, clf2.predict(X_train)
precision_score_VC_train = precision_score(x_actual, x_pred)
accuracy_score_VC_train = accuracy_score(x_actual, x_pred)
print('The precision score of Voting classifier on TRAIN is : ',round(precision_score_VC_train * 100,2), '%')
print('The accuracy score of Voting classifier on TRAIN is : ',round(accuracy_score_VC_train * 100,2), '%')

from sklearn.metrics import precision_score, accuracy_score
y_actual, y_pred = y_test, clf2.predict(X_test)
precision_score_VC_test = precision_score(y_actual, y_pred)
accuracy_score_VC_test = accuracy_score(y_actual, y_pred)
print('The precision score of Voting classifier on Test is : ',round(precision_score_VC_test * 100,2), '%')
print('The accuracy score of Voting classifier on Test is : ',round(accuracy_score_VC_test * 100,2), '%')

#Now let's plot the ROC curve and calculate AUC on the test set
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
adsu = clf2.predict_proba(X_test)[:,1]
plt.subplots(figsize=(8,6))
fpr, tpr, thresholds = roc_curve(y_test, adsu)
plt.plot(fpr, tpr)
x = np.linspace(0,1,num=50)
plt.plot(x,x,color='lightgrey',linestyle='--',marker='',lw=2,label='random guess')
plt.legend(fontsize = 14)
plt.xlabel('False positive rate', fontsize = 18)
plt.ylabel('True positive rate', fontsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

AUC_VC = auc(fpr,tpr)
print('VC AUC is: ', round(AUC_VC * 100,2), '%')

#getting the confusion matrix for the classification model
from sklearn.metrics import confusion_matrix # cofusion matrix / accuracy
print ('Confusion Matrix TRAIN:\n', confusion_matrix(y_train,x_pred))
print ('\nConfusion Matrix TEST:\n', confusion_matrix(y_test,y_pred))

# getting the classification report of the classification models
from sklearn.metrics import classification_report 
print ('Classification Report TRAIN:\n', classification_report(y_train,x_pred))
print ('\nClassification Report TEST:\n', classification_report(y_test,y_pred))