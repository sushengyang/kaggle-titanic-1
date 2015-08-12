# ================================
# Parse the data from train.csv 
# for the Titanic problem
#
# Alexandre Caze
# August 12, 2015
# ================================

print ('')
print ('==================================================')
print ('Logistic regression for the Titanic Kaggle problem')
print ('==================================================')
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

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
train_df = train_df.drop(['Age'],axis=1)

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
test_df = test_df.drop(['Age'],axis=1)

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
print ('Statistics ...')
print ('=================')

# Total number of data
passengers_nb = len(train_df)
print('')
print(str(passengers_nb)+' passengers in the training set.')

# Adult vs Child
adult_nb = len(train_df[train_df.Adult==1])
children_nb = len(train_df[train_df.Adult==0])
print ('{0:.3f} % adults and {1:.3f} % children.'.format(float(adult_nb)/passengers_nb, float(children_nb)/passengers_nb))

# Male vs Female
male_nb = len(train_df[train_df.Gender==0])
female_nb = len(train_df[train_df.Gender==1])
print ('{0:.3f} % male and {1:.3f} % female.'.format(float(male_nb)/passengers_nb, float(female_nb)/passengers_nb))

# 1st vs 2nd vs 3rd class
first_class_nb = len(train_df[train_df.Pclass==1])
second_class_nb = len(train_df[train_df.Pclass==2])
third_class_nb = len(train_df[train_df.Pclass==3])
print ('{0:.3f} % 1st class, {1:.3f} % 2nd class and {2:.3f} % 3rd class.'.format(float(first_class_nb)/passengers_nb, float(second_class_nb)/passengers_nb,float(third_class_nb)/passengers_nb))

# Total number of data in the test set
passengers_nb = len(test_df)
print('')
print(str(passengers_nb)+' passengers in the test set.')

# Adult vs Child
adult_nb = len(test_df[test_df.Adult==1])
children_nb = len(test_df[test_df.Adult==0])
print ('{0:.3f} % adults and {1:.3f} % children.'.format(float(adult_nb)/passengers_nb, float(children_nb)/passengers_nb))

# Male vs Female
male_nb = len(test_df[test_df.Gender==0])
female_nb = len(test_df[test_df.Gender==1])
print ('{0:.3f} % male and {1:.3f} % female.'.format(float(male_nb)/passengers_nb, float(female_nb)/passengers_nb))

# 1st vs 2nd vs 3rd class
first_class_nb = len(test_df[test_df.Pclass==1])
second_class_nb = len(test_df[test_df.Pclass==2])
third_class_nb = len(test_df[test_df.Pclass==3])
print ('{0:.3f} % 1st class, {1:.3f} % 2nd class and {2:.3f} % 3rd class.'.format(float(first_class_nb)/passengers_nb, float(second_class_nb)/passengers_nb,float(third_class_nb)/passengers_nb))

print('')
print('Survival rates:')
adults = train_df[train_df.Adult==1]
children = train_df[train_df.Adult==0]
male = train_df[train_df.Gender==0]
female = train_df[train_df.Gender==1]
firstclass = train_df[train_df.Pclass==1]
secondclass = train_df[train_df.Pclass==2]
thirdclass = train_df[train_df.Pclass==3]
print ('-------------------')
print ('Total     | {0:.3f} % |'.format(float(len(train_df[train_df.Survived==1]))/len(train_df)))
print ('-------------------')
print ('Adults    | {0:.3f} % |'.format(float(len(adults[adults.Survived==1]))/len(adults)))
print ('Children  | {0:.3f} % |'.format(float(len(children[children.Survived==1]))/len(children)))
print ('-------------------')
print ('Male      | {0:.3f} % |'.format(float(len(male[male.Survived==1]))/len(male)))
print ('Female    | {0:.3f} % |'.format(float(len(female[female.Survived==1]))/len(female)))
print ('-------------------')
print ('1st class | {0:.3f} % |'.format(float(len(firstclass[firstclass.Survived==1]))/len(firstclass)))
print ('2nd class | {0:.3f} % |'.format(float(len(secondclass[secondclass.Survived==1]))/len(secondclass)))
print ('3rd class | {0:.3f} % |'.format(float(len(thirdclass[thirdclass.Survived==1]))/len(thirdclass)))
print ('-------------------')

print('')
print('Interaction Gender / Age:')
adults_male = adults[adults.Gender==0]
adults_female = adults[adults.Gender==1]
children_male = children[children.Gender==0]
children_female = children[children.Gender==1]
print ('')
print ('          |   Male  |  Female |  Total  |')
print ('----------------------------------------|')
print ('Adults    | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(adults_male[adults_male.Survived==1]))/len(adults_male),float(len(adults_female[adults_female.Survived==1]))/len(adults_female),float(len(adults[adults.Survived==1]))/len(adults)))
print ('Children  | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(children_male[children_male.Survived==1]))/len(children_male),float(len(children_female[children_female.Survived==1]))/len(children_female),float(len(children[children.Survived==1]))/len(children)))
print ('----------------------------------------|')
print ('Total     | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(male[male.Survived==1]))/len(male),float(len(female[female.Survived==1]))/len(female),float(len(train_df[train_df.Survived==1]))/len(train_df)))
print ('----------------------------------------|')

print('')
print('Interaction Gender / Class:')
male_first = male[male.Pclass==1]
male_second = male[male.Pclass==2]
male_third = male[male.Pclass==3]
female_first = female[female.Pclass==1]
female_second = female[female.Pclass==2]
female_third = female[female.Pclass==3]
print ('')
print ('          |   Male  |  Female |  Total  |')
print ('----------------------------------------|')
print ('1st class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(male_first[male_first.Survived==1]))/len(male_first),float(len(female_first[female_first.Survived==1]))/len(female_first),float(len(firstclass[firstclass.Survived==1]))/len(firstclass)))
print ('2nd class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(male_second[male_second.Survived==1]))/len(male_second),float(len(female_second[female_second.Survived==1]))/len(female_second),float(len(secondclass[secondclass.Survived==1]))/len(secondclass)))
print ('3rd class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(male_third[male_third.Survived==1]))/len(male_third),float(len(female_third[female_third.Survived==1]))/len(female_third),float(len(thirdclass[thirdclass.Survived==1]))/len(thirdclass)))
print ('----------------------------------------|')
print ('Total     | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(male[male.Survived==1]))/len(male),float(len(female[female.Survived==1]))/len(female),float(len(train_df[train_df.Survived==1]))/len(train_df)))
print ('----------------------------------------|')

print('')
print('Interaction Age / Class:')
adults_first = adults[adults.Pclass==1]
adults_second = adults[adults.Pclass==2]
adults_third = adults[adults.Pclass==3]
children_first = children[children.Pclass==1]
children_second = children[children.Pclass==2]
children_third = children[children.Pclass==3]
print ('')
print ('          |  Adult  |  Child  |  Total  |')
print ('----------------------------------------|')
print ('1st class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(adults_first[adults_first.Survived==1]))/len(adults_first),float(len(children_first[children_first.Survived==1]))/len(children_first),float(len(firstclass[firstclass.Survived==1]))/len(firstclass)))
print ('2nd class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(adults_second[adults_second.Survived==1]))/len(adults_second),float(len(children_second[children_second.Survived==1]))/len(children_second),float(len(secondclass[secondclass.Survived==1]))/len(secondclass)))
print ('3rd class | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(adults_third[adults_third.Survived==1]))/len(adults_third),float(len(children_third[children_third.Survived==1]))/len(children_third),float(len(thirdclass[thirdclass.Survived==1]))/len(thirdclass)))
print ('----------------------------------------|')
print ('Total     | {0:.3f} % | {1:.3f} % | {2:.3f} % |'.format(float(len(adults[adults.Survived==1]))/len(adults),float(len(children[children.Survived==1]))/len(children),float(len(train_df[train_df.Survived==1]))/len(train_df)))
print ('----------------------------------------|')


print ('')
print ('=======')
print ('The End')
print ('=======')
