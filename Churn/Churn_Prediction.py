
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 20:08:09 2018

@author: Srinivas
"""
    # Churn Prediction 
    
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/Srinivas/Desktop/Machine Learning A-Z Template Folder/Part 11 - Model Practice/Churn/churn_data.csv')


dataset.isna().any()
dataset.isna().sum()

# removing NaN
dataset = dataset[pd.notnull(dataset['age'])]
dataset = dataset.drop(columns=['credit_score','rewards_earned'])

# Histograms

dataset2 = dataset.drop(columns = ['user','churn'])

fig = plt.figure(figsize=(15,12))
plt.suptitle('Histograms of numerical columns',fontsize=20)
for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,5,i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])
    
    vals = np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins=vals ,color = '#3F5D7D')
plt.tight_layout(rect=[0,0.03,1,0.95])

# Exploring un even features
dataset.drop(columns=['churn','user','housing','payment_type',
                      'zodiac_sign',]).corrwith(dataset.churn).plot.bar(
    figsize = (20,20),title = 'Correlation with Response Variable',fontsize=15,rot=45,grid=True )             

dataset = dataset.drop(columns=['app_web_user'])
dataset.to_csv('new_churn_data.csv',index=False)

dataset_new = pd.read_csv('C:/Users/Srinivas/Desktop/Machine Learning A-Z Template Folder/Part 11 - Model Practice/Churn/new_churn_data.csv')
user_identifier = dataset['user']
dataset_new = dataset_new.drop(columns=['user'])

# One Hot Encoding
dataset_new.rent_or_own.value_counts()
dataset_new = pd.get_dummies(dataset_new)
dataset_new.columns

dataset_new = dataset_new.drop(columns = ['zodiac_sign_na','payfreq_na','rent_or_own_na'])


# Splitting training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset_new.drop(columns='churn'),
                                                 dataset_new['churn'],
                                                 test_size=0.2,
                                                 random_state=0)     


# Balancing the Training Dataset
y_train.value_counts()

pos_index = y_train[y_train == 1].index
neg_index = y_train[y_train == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index
    
import random    
random.seed(0)
higher = np.random.choice(higher,size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower,higher))   

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values

X_train = X_train2
X_test = X_test2


# Fitting model to the training dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# Predicting test results
y_pred = classifier.predict(X_test)


# Evaluation Results
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score

cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)

df_cm = pd.DataFrame(cm,index=(0,1),columns=(0,1))
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,annot=True,fmt='g')
print('Test Data Accuracy: %0.4f' % accuracy_score(y_test,y_pred))

# Applying K-Fold  Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,
                             X = X_train,
                             y = y_train,
                             cv = 10)
    
accuracies.mean()


# Analyzing Coefficients
pd.concat([pd.DataFrame(X_train.columns,columns=["features"]),
           pd.DataFrame(np.transpose(classifier.coef_),columns=["coef"])],
axis =1)


# Will try to implement the solution with RFE model
#-----------------------------------------------
# Doing Feature Selection with RFE Model        #
#-----------------------------------------------

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
rfe = RFE(classifier,20) # We are selecting only 20 independent variables from 40
rfe = rfe.fit(X_train,y_train)

# Summarize the selection of Attributes
print(rfe.support_)
X_train.columns[rfe.support_]
rfe.ranking_


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train[X_train.columns[rfe.support_]],y_train)


# Predicting test results
y_pred = classifier.predict(X_test[X_test.columns[rfe.support_]])


# Evaluation Results
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score

cm = confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
precision_score(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)

df_cm = pd.DataFrame(cm,index=(0,1),columns=(0,1))
plt.figure(figsize=(10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm,annot=True,fmt='g')
print('Test Data Accuracy: %0.4f' % accuracy_score(y_test,y_pred))
 
# Analyzing Coefficients
    
pd.concat([pd.DataFrame(X_train.columns[rfe.support_],columns=["features"]),
           pd.DataFrame(np.transpose(classifier.coef_),columns=["coef"])],
axis =1)


# Formatting final results
final_results = pd.concat([y_test,userid],axis=1).dropna()
final_results['predicted_churn'] = y_pred
final_results = final_results[['user','churn','predicted_churn']].reset_index(drop=True)




