#!/usr/bin/env python
# coding: utf-8

# Machine learning project / Fall 2022
# 
# 
# ## Table of contents:
# * [Introduction](#first-bullet)
# * [Problem Formulation](#second-bullet)
# * [Method: Decision Trees](#dt)
# * [Dataset Consideration](#dsc)
# * [References](#references)
# 
# 
# ## Introduction <a class="anchor" id="first-bullet"></a>
# 
# To be added
# 
# Let us initalize the required packages and the dataset.

# In[46]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False  # enable code auto-completion')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('winequality-white.csv')


# ## Problem Formulation<a class="anchor" id="second-bullet"></a>
# 
# The application of this project is to teach a machine learning process to figure out if a white wine is average, above average or below average based on different attributes collected on the wines in the dataset. The dataset is provided by University of Minho in Portugal [1] and is based on Vinho Verde Region white wines. This dataset has 4898 tuples and a multitude of attributes of the wines is provided for each wine:
# 
# - fixed acidity
# - volatile acidity
# - citric acid
# - residual sugar
# - chlorides
# - free sulfur dioxide
# - total sulfur dioxide
# - density
# - pH
# - sulphates
# - alcohol
# - quality (score between 0 and 10)
# 
# Each tuple in the dataframe represents a single anonymous white wine type and each white wine has an value for each of the categories mentioned above. The first 11 noted values are based on physicochemical tests and have certain numerical values providing information regarding what is being measured. Quality is a numerical value determined based on sensory data and it can be thought of as an output variable. As such the base dataset could be considered to be a **multivariate categorical dataset** which notably does not have any missing values. Overall, quality will be used as a defining factor when it comes to describing the averageness of a wine. The machine learning algorithms will not rely on quality since quality is the deciding factor on wheter a wine is considered average or not.
# 
# Let us first plot this dataset based on quality score:

# In[47]:


sortedDF = df.sort_values(by='quality');
n = np.linspace(start=1,stop=df.shape[0],num=df.shape[0]);
plt.scatter(n,sortedDF['quality']);
plt.ylabel('Quality'); plt.xlabel('n');


# It is clear that there is a large amount of wines with rather average quality scores of 5 and 6. To avoid certain "dumb" machine learning results we derive the following simplistic model that assigns a score of averageness based on the given score considering quality. The averageness is determined by a classifier function as follows:
# 
#  - Score of **5** is considered as "Average" as it is the middle value in the 0-10 scale.
#  - Quality scores between **0 and 4** are treated as "Below Average"
#  - Finally anything between **6 and 10**, is treated as "Above Average".
#  
# These labels are given numerical values of -1, 0, and 1 respectively along with the name "averageness". This assignment of numerical parameters does not have any further implications. The goal of this assignment is to find a process that is able to determine if a wine could be considered average or possibly above or below average based on physiochemically measured attributes.
# 
# ## Method: Decision Trees<a class="anchor" id="dt"></a>
# 
# Since these attributes of the tuples can be thought of being the attributes that also define the wine, in this project **all of the variables represented above** will be considered as features for the machine learning process. The main method of machine learning that this assignment uses is **decision trees** [2]. This method is chosen because there is a multivariate set of features and a non-binary bin representation. Thus a more complex methods are required and a decision tree fits this purpose appropriately.
# 
# A loss function in this case for the hypothesis is Gini impurity and the loss function for the results is accuracy. The choice of Gini impurity is made as it allows the use of a ready-made library for decision tree. Choosing accuracy as the result loss function is the logical choice for a binning task as the classification between choosing correctly or incorrectly is the interesting question overall in such tasks.

# In[48]:


labels = [-1,0,1] # new labels to be assigned
cut_bins = [0,5,6,10] #cutting intervals/criteria [minvalue,0],(0,maxvalue]

averageness_val = pd.cut(df['quality'],bins=cut_bins,labels=labels,include_lowest=True).astype('int64')
df.insert(12,'averageness',averageness_val)

plt.scatter(n,df['averageness']); plt.xlabel("n");
plt.ylabel("averageness"); plt.yticks([-1,0,1]);
plt.title("averageness of wines");


# ## Dataset Consideration<a class="anchor" id="dsc"></a>
# 
# Before implementing the model, let us consider the correlations between attributes in the dataset:

# In[54]:


plt.figure(figsize=(10,10))
correlations = df[df.columns].corr(method='pearson')
sns.heatmap(correlations, annot = True)
plt.show()

plt.figure(figsize=(10,10))
correlations = df.drop(['density','quality'], axis=1).corr(method='pearson')
sns.heatmap(correlations, annot = True)
plt.show()


# In[ ]:





# From this correlation matrix, it is clear that there are some values with very high cross correlation levels (>0.7 or <-0.7). High amounts of cross correlation is not really wanted as these values can be considered to depend on one another instead of contributing towards a reasonable argument for the averageness value.
# 
# For the correlating values, one of the sets can and should be dropped from the dataframe without making significant changes to the model that will be used. As such both density-residual sugar and density-alcohol can be considered as highly correlating pairs of variables. Finally, quality should be dropped as an feature regardless as it is used directly to calculate the value of averageness.

# In[56]:


X = df.drop(['density','quality','averageness'], axis=1)
y = df['averageness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

dtClf = DecisionTreeClassifier(criterion="gini")
dtClf = dtClf.fit(X_train,y_train)

y_pred = dtClf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

ax= plt.subplot()

c_mat = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(c_mat, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=15)
ax.xaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
ax.yaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);


# # References<a class="anchor" id="references"></a> 
# 
# [1] Wine quality dataset
# 
# Retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality. Original from https://pcortez.dsi.uminho.pt/ by P. Cortez and original survey conducted by A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @ 2009
# 
# [2] Information about the machine learning models and loss functions (course book):
# 
# https://github.com/alexjungaalto/MachineLearningTheBasics/blob/master/
# MLBasicsBook.pdf
