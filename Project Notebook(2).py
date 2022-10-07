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


sortedDF = df.sort_values(by='quality');
n = np.linspace(start=1,stop=df.shape[0],num=df.shape[0]);
plt.scatter(n,sortedDF['quality']);
plt.ylabel('Quality'); plt.xlabel('n');


labels = [-1,0,1] # new labels to be assigned
cut_bins = [0,5,6,10] #cutting intervals/criteria [minvalue,0],(0,maxvalue]

averageness_val = pd.cut(df['quality'],bins=cut_bins,labels=labels,include_lowest=True).astype('int64')
df.insert(12,'averageness',averageness_val)

plt.scatter(n,df['averageness']); plt.xlabel("n");
plt.ylabel("averageness"); plt.yticks([-1,0,1]);
plt.title("averageness of wines");


plt.figure(figsize=(10,10))
correlations = df[df.columns].corr(method='pearson')
sns.heatmap(correlations, annot = True)
plt.show()

plt.figure(figsize=(10,10))
correlations = df.drop(['density','quality'], axis=1).corr(method='pearson')
sns.heatmap(correlations, annot = True)
plt.show()



X = df.drop(['density','quality','averageness'], axis=1)
y = df['averageness']

X_train_FT, X_test, y_train_FT, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_FT, y_train_FT, test_size=0.2, random_state=1)


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

num_layers = [1,2,4,6,8,12,16]
num_neurons = 15

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


num_layers = 2
num_neurons = 15

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


d_tree = export_graphviz(dtClf_Final, feature_names = list(X.columns), filled = True, class_names=["below average","average", "above average"])
pydot_graph = pydotplus.graph_from_dot_data(d_tree)
pydot_graph.write_pdf('WineAverageness_Tree.pdf')
