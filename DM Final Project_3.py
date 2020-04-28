#Load libraries
import pandas as pd
from pydotplus import graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import webbrowser
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Unpickle clean dataframe
data = pd.read_pickle('data')

# Separate target and features
target='diagnosis'
y = data[target]
X = data.drop(columns=[target])
# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create the classifier object, ENTROPY as criterion
clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=100, max_depth=3, min_samples_leaf=5)
# Perform training with Entropy
clf_entropy.fit(X_train, y_train)
# Prediction on test using Entropy
y_pred_entropy = clf_entropy.predict(X_test)
# Calculate metrics Gini model; I choose Gini because the features values are continuous
print('Results of the Decision Tree using Entropy Criterion:')
print('Accuracy:', accuracy_score(y_test, y_pred_entropy))
print('Classification Report:')
print(classification_report(y_test, y_pred_entropy))
# Confusion matrix for Entropy model
conf_matrix_entropy = confusion_matrix(y_test, y_pred_entropy)
class_names = data['diagnosis'].unique()
df_cm_entropy = pd.DataFrame(conf_matrix_entropy, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm_entropy = sns.heatmap(df_cm_entropy, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_entropy.columns, xticklabels=df_cm_entropy.columns)
hm_entropy.yaxis.set_ticklabels(hm_entropy.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_entropy.xaxis.set_ticklabels(hm_entropy.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title('Confusion Matrix of Decision Tree using Entropy Criterion')
plt.tight_layout()
plt.show()

# Visualize Decision Tree
dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names=X.iloc[:,:].columns, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")
webbrowser.open_new(r'decision_tree_entropy.pdf')

# Create the classifier object, GINI as criterion
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
# Perform training with GiniIndex
clf_gini.fit(X_train, y_train)
# Prediction on test using Gini
y_pred_gini = clf_gini.predict(X_test)
# Calculate metrics Gini model; I choose Gini because the features values are continuous
print('Results Decision Tree using Gini Criterion:')
print('Accuracy:', accuracy_score(y_test, y_pred_gini))
print('Classification Report:')
print(classification_report(y_test, y_pred_gini))

# Confusion matrix for Gini model
conf_matrix_gini = confusion_matrix(y_test, y_pred_gini)
class_names = data['diagnosis'].unique()
df_cm_gini = pd.DataFrame(conf_matrix_gini, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm_gini = sns.heatmap(df_cm_gini, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_gini.columns, xticklabels=df_cm_gini.columns)
hm_gini.yaxis.set_ticklabels(hm_gini.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_gini.xaxis.set_ticklabels(hm_gini.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title('Confusion Matrix of Decision Tree using Gini Criterion')
plt.tight_layout()
plt.savefig('cmatrix_dt_gini_allf.png')
plt.show()

# Visualize Decision Tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=X.iloc[:,:].columns, out_file=None)
graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')

# Plot feature importances for GINI criterion
# Get feature importances
importances = clf_gini.feature_importances_
# Convert the importances into one-dimensional 1D array with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, 1:].columns)
# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)
# Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
plt.tight_layout()
plt.title("Feature Importance Decision Tree - Gini Criterion")
plt.savefig('feature_importance_dt.png')
plt.show()

