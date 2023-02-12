# Robin Donal
## TSF DATA SCIENCE INTERN
## TASK 6
### DECISION TREE

# Importing libraraies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#for encoding
from sklearn.preprocessing import LabelEncoder
#for train test splitting
from sklearn.model_selection import train_test_split
#for decision tree object
from sklearn.tree import DecisionTreeClassifier
#for checking testing results
from sklearn.metrics import classification_report, confusion_matrix
#for visualizing tree 
from sklearn.tree import plot_tree
# Reading the dataset
#reading the data
df=pd.read_csv(r"D:\adypu\INTERSHIP\Iris.csv")
df
# Summarising the dataset
#getting information of dataset
df.info()
# Check the null values
df.isnull().any()
df.shape
# EDA
# let's plot pair plot to visualise the attributes all at once
sns.pairplot(data=df, hue = 'Species')
# correlation matrix
sns.heatmap(df.corr())
# DECISION TREE ALGORITHM
target = df['Species']
df1 = df.copy()#creating a copy to avoid deleting the column from original data set
df1 = df1.drop('Species', axis =1)#dropping target variable
df1.shape
# Defining the attributes
X = df1
target
#label encoding
le = LabelEncoder()
target = le.fit_transform(target)#to convert categorical variables to numerical variable
target
y = target
# Splitting the data - 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.2, random_state = 42)

print("Training split input- ", X_train.shape)
print("Testing split input- ", X_test.shape)
# Defining the decision tree algorithm

dtree=DecisionTreeClassifier()
dtree.fit(X_train,y_train)

print('Decision Tree Classifer Created')
# Predicting the values of test data
y_pred = dtree.predict(X_test)
print("Classification report - \n", classification_report(y_test,y_pred))
cm =confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(dtree.score(X_test, y_test))
plt.title(all_sample_title, size = 15)

# Visualising the graph without the use of graphviz

plt.figure(figsize = (20,20))
dec_tree = plot_tree(decision_tree=dtree, feature_names = df1.columns, 
                     class_names =["Iris-setosa", "Iris-vercicolor", "Iris-verginica"] , filled = True , precision = 4, rounded = True)
