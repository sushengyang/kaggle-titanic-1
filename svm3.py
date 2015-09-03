# ================================
# Alexandre Caze
# Sept 3, 2015
# ================================

print ('')
print ('======================')
print ('Titanic Kaggle problem')
print ('  Feature selection ')
print ('======================')
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import svm

print ('')
print ('Importing train.csv ...')
print ('=======================')
train_df = pd.read_csv('train.csv',header=0) # Put the training set in a data frame

print ('')
print ('Cleaning data ...')
print ('=======================')

# Turns Sex (string) into Gender (1 or 0) 
most_common_sex = train_df.Sex.dropna().mode()
if len(train_df.Sex[ train_df.Sex.isnull() ]) > 0:
	print ('Sex is turned into the most common sex for ',len(train_df.Sex[ train_df.Sex.isnull() ]),' passengers (missing values).')
	train_df.loc[ (train_df.Sex.isnull()), 'Age'] = most_common_sex
else:
	print ('No missing value for the feature Sex.')

print ('Feature Sex (string) is turned into Gender (0 for male, 1 for female).')
train_df['Gender'] = train_df['Sex'].map({'male':0,'female':1}).astype(int)
train_df = train_df.drop(['Sex'],axis=1)

# Turns Age (float) into Adult (1 or 0)
# All the ages with no data -> make the median of all Ages
median_age = train_df.Age.dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
	print ('Age is turned into the median age for ',len(train_df.Age[ train_df.Age.isnull() ]),' passengers (missing values).')
	train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age
else:
	print ('No missing value for the feature Age.')

print ('Feature Age (float) is turned into Adult (0 if under 18, 1 if over 18).')
train_df['Adult'] = (train_df.Age>18).map({True:1,False:0}).astype(int)

print ('Feature MedianAge is defined as 0 if under the median age, 1 otherwise')
print ('Median Age : '+str(median_age))
train_df['MedianAge'] = (train_df.Age>median_age).map({True:1,False:0}).astype(int)

# Turns missing Pclass into most common
most_common_class = train_df.Pclass.mode().values
if len(train_df.Pclass[ train_df.Pclass.isnull() ]) > 0:
	print ('Pclass is turned into the most common class for ',len(train_df.Pclass[ train_df.Pclass.isnull() ]),' passengers (missing values).')
	train_df.loc[ (train_df.Age.isnull()), 'Pclass'] = most_common_class
else:
	print ('No missing value for the feature Pclass.')

print ('Pclass takes values in {1,2,3}.')

print ('All other features are dropped (Parch, PassengerId, Name, SibSp, Embarked, Fare, Ticket, Cabin).')
train_df = train_df.drop(['Parch','PassengerId','Name','SibSp','Embarked','Fare','Ticket','Cabin'],axis=1)

#print train_df

print ('')
print ('Importing test.csv ...')
print ('=======================')
test_df = pd.read_csv('test.csv',header=0) # Put the test set in a data frame

print ('')
print ('Cleaning data ...')
print ('=======================')

# Turns Sex (string) into Gender (1 or 0) 
most_common_sex = test_df.Sex.dropna().mode()
if len(test_df.Sex[ test_df.Sex.isnull() ]) > 0:
	print ('Sex is turned into the most common sex for ',len(test_df.Sex[ test_df.Sex.isnull() ]),' passengers (missing values).')
	test_df.loc[ (test_df.Sex.isnull()), 'Age'] = most_common_sex
else:
	print ('No missing value for the feature Sex.')

print ('Feature Sex (string) is turned into Gender (0 for male, 1 for female).')
test_df['Gender'] = test_df['Sex'].map({'male':0,'female':1}).astype(int)
test_df = test_df.drop(['Sex'],axis=1)

# Turns Age (float) into Adult (1 or 0)
# All the ages with no data -> make the median of all Ages
median_age = test_df.Age.dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
	print ('Age is turned into the median age for ',len(test_df.Age[ test_df.Age.isnull() ]),' passengers (missing values).')
	test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
else:
	print ('No missing value for the feature Age.')

print ('Feature Age (float) is turned into Adult (0 if under 18, 1 if over 18).')
test_df['Adult'] = (test_df.Age>18).map({True:1,False:0}).astype(int)

print ('Feature MedianAge is defined as 0 if under the median age, 1 otherwise')
print ('Median Age : '+str(median_age))
test_df['MedianAge'] = (test_df.Age>median_age).map({True:1,False:0}).astype(int)

# Turns missing Pclass into most common
most_common_class = test_df.Pclass.mode().values
if len(test_df.Pclass[ test_df.Pclass.isnull() ]) > 0:
	print ('Pclass is turned into the most common class for ',len(test_df.Pclass[ test_df.Pclass.isnull() ]),' passengers (missing values).')
	test_df.loc[ (test_df.Age.isnull()), 'Pclass'] = most_common_class
else:
	print ('No missing value for the feature Pclass.')

print ('Pclass takes values in {1,2,3}.')

print ('All other features are dropped (Parch, PassengerId, Name, SibSp, Embarked, Fare, Ticket, Cabin).')
test_df = test_df.drop(['Parch','PassengerId','Name','SibSp','Embarked','Fare','Ticket','Cabin'],axis=1)

#print test_df

print ('')
print ('Test Age features')
print ('=================')

print 'Age,Adult,MedianAge'
print train_df[['Age','Adult','MedianAge']].head(5)

print ('')
print ('Classifying using LinearSVC with Gender, Class and Age...')
print ('======================================================================')

print ('')
print ('Computing the SVM prediction.')
X = train_df[['Gender','Pclass','Age']].values
Y = train_df['Survived'].values
clf = svm.LinearSVC()
clf.fit(X,Y)

print ('')
print ('Writing Kaggle result file in GenderClassAge.csv')

Xtest = test_df[['Gender','Pclass','Age']].values
Ytest = clf.predict(Xtest)

f=open('GenderClassAge.csv', 'wb')
f.write('PassengerId,Survived\n')
i=892
for x in Ytest:
	f.write(str(i)+','+str(x)+'\n')
	i+=1

print ('Done.')

print ('Classifying using LinearSVC with Gender, Class and Adult...')
print ('======================================================================')

print ('')
print ('Computing the SVM prediction.')
X = train_df[['Gender','Pclass','Adult']].values
Y = train_df['Survived'].values
clf = svm.LinearSVC()
clf.fit(X,Y)

print ('')
print ('Writing Kaggle result file in GenderClassAdult.csv')

Xtest = test_df[['Gender','Pclass','Adult']].values
Ytest = clf.predict(Xtest)

f=open('GenderClassAdult.csv', 'wb')
f.write('PassengerId,Survived\n')
i=892
for x in Ytest:
	f.write(str(i)+','+str(x)+'\n')
	i+=1

print ('Done.')

print ('')
print ('Classifying using LinearSVC with Gender, Class and MedianAge...')
print ('======================================================================')

print ('')
print ('Computing the SVM prediction.')
X = train_df[['Gender','Pclass','MedianAge']].values
Y = train_df['Survived'].values
clf = svm.LinearSVC()
clf.fit(X,Y)

print ('')
print ('Writing Kaggle result file in GenderClassMedianAge.csv')

Xtest = test_df[['Gender','Pclass','MedianAge']].values
Ytest = clf.predict(Xtest)

f=open('GenderClassMedianAge.csv', 'wb')
f.write('PassengerId,Survived\n')
i=892
for x in Ytest:
	f.write(str(i)+','+str(x)+'\n')
	i+=1

print ('Done.')

print ('')
print ('=======')
print ('The End')
print ('=======')
