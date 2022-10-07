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

# In[86]:


get_ipython().run_line_magic('config', 'Completer.use_jedi = False  # enable code auto-completion')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn import metrics

from sklearn.datasets import fetch_openml 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pydotplus

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

# In[2]:


sortedDF = df.sort_values(by='quality');
n = np.linspace(start=1,stop=df.shape[0],num=df.shape[0]);
plt.scatter(n,sortedDF['quality']);
plt.ylabel('Quality'); plt.xlabel('n');


# It is clear that there is a large amount of wines with rather average quality scores of 5 and 6. To avoid certain "dumb" machine learning results we derive the following simplistic model that assigns a score of averageness based on the given score considering quality. The averageness is determined by a classifier function as follows:
# 
#  - Score of **6** is considered as "Average" as it is the middle value in the 0-10 scale.
#  - Quality scores between **0 and 6** are treated as "Below Average"
#  - Finally anything between **7 and 10**, is treated as "Above Average".
#  
# These labels are given numerical values of -1, 0, and 1 respectively along with the name "averageness". This assignment of numerical parameters does not have any further implications. The goal of this assignment is to find a process that is able to determine if a wine could be considered average or possibly above or below average based on physiochemically measured attributes.
# 
# ## Method: Decision Trees<a class="anchor" id="dt"></a>
# 
# Since these attributes of the tuples can be thought of being the attributes that also define the wine, in this project **all of the variables represented above** will be considered as features for the machine learning process. The main method of machine learning that this assignment uses is **decision trees** [2]. This method is chosen because there is a multivariate set of features and a non-binary bin representation. Thus a more complex methods are required and a decision tree fits this purpose appropriately.
# 
# A loss function in this case for the hypothesis is Gini impurity and the loss function for the results is accuracy. The choice of Gini impurity is made as it allows the use of a ready-made library for decision tree. Choosing accuracy as the result loss function is the logical choice for a binning task as the classification between choosing correctly or incorrectly is the interesting question overall in such tasks.

# In[3]:


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

# In[4]:


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

# In[73]:


X = df.drop(['density','quality','averageness'], axis=1)
y = df['averageness']

X_train_FT, X_test, y_train_FT, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_FT, y_train_FT, test_size=0.2, random_state=1)


# In[74]:


dtClf = DecisionTreeClassifier(criterion="gini")
dtClf.fit(X_train,y_train)

y_pred_train = dtClf.predict(X_train)
y_pred = dtClf.predict(X_valid)

print("Training accuracy:",metrics.accuracy_score(y_train, y_pred_train))
print("Validation accuracy:",metrics.accuracy_score(y_valid, y_pred))

ax  = plt.subplot()

c_mat = metrics.confusion_matrix(y_valid, y_pred)
sns.heatmap(c_mat, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title('Confusion Matrix',fontsize=15)
ax.xaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
ax.yaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);


# In[75]:


## define a list of values for the number of hidden layers
num_layers = [1,2,4,6,8,12,16]    # number of hidden layers
num_neurons = 15  # number of neurons in each layer

mlp_tr_acc = []          
mlp_val_acc = []

for i, num in enumerate(num_layers):
    hidden_layer_sizes = tuple([num_neurons]*num)
    
    mlp_regr = MLPC(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
 
    mlp_regr.fit(X_train,y_train)
    
    y_pred_train = mlp_regr.predict(X_train)
    y_pred_val = mlp_regr.predict(X_valid)
    
    tr_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_valid, y_pred_val)
    
    mlp_tr_acc.append(tr_acc)
    mlp_val_acc.append(val_acc)
    
errors = {"num_hidden_layers":num_layers,
          "mlp_train_errors":mlp_tr_acc,
          "mlp_val_errors":mlp_val_acc,
         }
plt.plot(np.sort(y_pred_train))
pd.DataFrame(errors)


# In[82]:


# Testing and confusion matrices

# Decision tree

dtClf_Final = DecisionTreeClassifier(criterion="gini")
dtClf_Final.fit(X_train_FT,y_train_FT)

y_pred_DT = dtClf_Final.predict(X_test)
print("Decision tree test accuracy:",metrics.accuracy_score(y_test, y_pred_DT))

ax  = plt.subplot()

c_mat = metrics.confusion_matrix(y_test, y_pred_DT)
sns.heatmap(c_mat, annot=True, fmt='g', ax=ax);

ax.set_xlabel('Predicted labels',fontsize=15)
ax.set_ylabel('True labels',fontsize=15)
ax.set_title('Confusion Matrix: Decision tree',fontsize=15)
ax.xaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
ax.yaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
plt.show()

# MLP-C

num_layers = 2    # number of hidden layers
num_neurons = 15  # number of neurons in each layer

hidden_layer_sizes = tuple([num_neurons]*num_layers)
    
mlp_regr_Final = MLPC(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
    
mlp_regr_Final.fit(X_train_FT,y_train_FT)
    
y_test_MLPC = mlp_regr.predict(X_test)
    
print("MLP-Classifier test accuracy:",metrics.accuracy_score(y_test, y_test_MLPC))

ax2  = plt.subplot()

c_mat2 = metrics.confusion_matrix(y_test, y_pred_MLPC)
sns.heatmap(c_mat2, annot=True, fmt='g', ax=ax2);

ax2.set_xlabel('Predicted labels',fontsize=15)
ax2.set_ylabel('True labels',fontsize=15)
ax2.set_title('Confusion Matrix: MLPC (2 layers, 15 neurons)',fontsize=15)
ax2.xaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
ax2.yaxis.set_ticklabels(['below average', 'average', 'above average'],fontsize=9);
plt.show()


# In[87]:


d_tree = export_graphviz(dtClf_Final, feature_names = list(X.columns), filled = True, class_names=["below average","average", "above average"])
pydot_graph = pydotplus.graph_from_dot_data(d_tree)
pydot_graph.write_pdf('WineAverageness_Tree.pdf')


# # References<a class="anchor" id="references"></a> 
# 
# [1] Wine quality dataset
# 
# Retrieved from https://archive.ics.uci.edu/ml/datasets/wine+quality. Original from https://pcortez.dsi.uminho.pt/ by P. Cortez and original survey conducted by A. Cerdeira, F. Almeida, T. Matos and J. Reis, Viticulture Commission of the Vinho Verde Region(CVRVV), Porto, Portugal @ 2009
# 
# [2] Information about the machine learning models and loss functions (course book):
# 
# https://github.com/alexjungaalto/MachineLearningTheBasics/blob/master/
# 
# MLBasicsBook.pdf
