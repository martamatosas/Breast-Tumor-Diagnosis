#Load libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# Unpickle the clean dataframe
data = pd.read_pickle('data')

# Separate target and variables
target='diagnosis'
y = data[target]
X = data.drop(columns=[target])
# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create the random forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=100)
# Train the model
clf.fit(X_train, y_train)

# Plot feature importances
# Get feature importances
importances = clf.feature_importances_
# Convert the importances into one-dimensional 1D array with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, 1:].columns)
# Sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)
# Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)
plt.tight_layout()
plt.title("Feature Importance Randon Forest")
plt.savefig('feature_importance_rf.png')
plt.show()

# Prediction on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# Calculate metrics using all features
print("Results of Random Forest: \n")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC_AUC All Features: ", roc_auc_score(y_test, y_pred_score[:,1]) * 100)

# Confusion matrix RF all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['diagnosis'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix Random Forest - all features',fontsize=15)
plt.savefig('cmatrix_rf_all.png')
plt.show()

# Select features to perform training with random forest with k most important features
# Select the training dataset on k-features
f_importances_sorted = f_importances.sort_values(ascending=False)
features_ordered_list = list(f_importances_sorted.index)
#Beter results for k=5, k=15 (BEST), k=17, k=20, k=21, k=23, k=25, k=26, k=27, k=28,
k = 15
feature_k_selection = features_ordered_list[0:k]
newX_train = X_train[feature_k_selection]
# Select the testing dataset on k-features
newX_test = X_test[feature_k_selection]
# Perform training with random forest with k columns
# Create the random forest classifierr with k features
clf_k_features = RandomForestClassifier(n_estimators=100, random_state=100)
# Train the model
clf_k_features.fit(newX_train, y_train)
# Prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

# Calculate metrics using k most important features
print("Results Using k Features: \n")
print("Accuracy k features: ", accuracy_score(y_test, y_pred_k_features) * 100)
print("Classification Report k Features:")
print(classification_report(y_test, y_pred_k_features))
print("ROC_AUC k Features: ", roc_auc_score(y_test, y_pred_k_features_score[:,1]) * 100)

# Confusion matrix using k most important features
conf_matrix_k = confusion_matrix(y_test, y_pred_k_features)
class_names = data['diagnosis'].unique()
df_cm_k = pd.DataFrame(conf_matrix_k, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm_k = sns.heatmap(df_cm_k, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_k.columns, xticklabels=df_cm_k.columns)
hm.yaxis.set_ticklabels(hm_k.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm_k.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix Random Forest - k features', fontsize=15)
plt.savefig('cmatrix_rf_kfeatures.png')
plt.show()

# Select features to perform training with random forest with features from Decision Tree
# Select the training dataset on features from DT Gin9 max-depth=3, -> (0, 6)
#features_DT_list = ['concave points_worst', 'area_mean', 'area_worst', 'concave points_worst', 'texture_mean', 'concavity_mean']
#Variation substituting one concave points_worst by concave points_se -> (1, 5)
#features_DT_list = ['concave points_worst', 'area_mean', 'area_worst', 'texture_mean', 'concavity_mean', 'concave points_se']
#(1,3), adding 1 feature from DT Gini missing in features from the paper
features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']

#max_depth=3
#features_DT_list = ['concave points_worst', 'area_mean', 'area_worst', 'texture_mean', 'concavity_mean']
#features_DT_list = ['concave points_mean', 'area_se', 'perimeter_worst', 'texture_worst']
#max_depth=5
#features_DT_list = ['concave points_worst', 'area_mean', 'area_worst', 'texture_mean', 'concavity_mean']
#features DT entropy, malignant missclassified = 1, benign = 6
#features_DT_list = ['concave points_mean', 'area_se', 'area_worst', 'texture_worst', 'concavity_worst', 'texture_mean', 'texture_worst']

#featur"es paper, malignant missclassified = 2, benign missclassified = 3
#features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst']

#features_corr_list = ['texture_mean', 'symmetry_mean', 'fractal_dimension_mean', 'texture_worst', 'smoothness_worst', 'symmetry_worst']

#mi mix, (0, 9)
#features_DT_list = ['concave points_mean', 'perimeter_worst', 'concavity_mean', 'compactness_worst', 'smoothness_worst', 'texture_worst']

#mi mix, (0, 8)
#features_DT_list = ['concave points_mean', 'concavity_mean', 'perimeter_worst', 'compactness_worst', 'smoothness_worst', 'texture_worst']
#mi mix, (0, 8)
#features_DT_list = ['concave points_mean', 'concavity_mean', 'perimeter_worst', 'smoothness_worst', 'texture_worst']

#mi mix, (3, 12)
#features_DT_list = ['concave points_mean', 'concavity_mean', 'smoothness_worst', 'texture_worst']

newX2_train = X_train[features_DT_list]
# Select the testing dataset on k-features
newX2_test = X_test[features_DT_list]
# Create the random forest classifierr with the DT features
clf_DT_features = RandomForestClassifier(n_estimators=100, random_state=100)
# Train the model
clf_DT_features.fit(newX2_train, y_train)
# Prediction on test using k features
y_pred_DT_features = clf_DT_features.predict(newX2_test)
y_pred_DT_features_score = clf_DT_features.predict_proba(newX2_test)

# Calculate metrics using DT features
print("Results Using DT Features: \n")
print("Accuracy DT features: ", accuracy_score(y_test, y_pred_DT_features) * 100)
print("Classification Report DT Features:")
print(classification_report(y_test, y_pred_DT_features))
print("ROC_AUC DT Features: ", roc_auc_score(y_test, y_pred_DT_features_score[:,1]) * 100)

# Confusion matrix using DT features
conf_matrix_DT = confusion_matrix(y_test, y_pred_DT_features)
class_names = data['diagnosis'].unique()
df_cm_DT = pd.DataFrame(conf_matrix_DT, index=class_names, columns=class_names)
plt.figure(figsize=(5,5))
hm_DT = sns.heatmap(df_cm_DT, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm_DT.yaxis.set_ticklabels(hm_DT.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_DT.xaxis.set_ticklabels(hm_DT.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix RF - custom features list', fontsize=15)
plt.savefig('cmatrix_rf_custom.png')
plt.show()

# LIST OF FEATURES FROM THE PAPER
features_paper_list = ['texture_mean', 'area_worst', 'smoothness_worst']

newX3_train = X_train[features_paper_list]
# Select the testing dataset on k-features
newX3_test = X_test[features_paper_list]
# Create the random forest classifierr with the DT features
clf_paper_features = RandomForestClassifier(n_estimators=100, random_state=100)
# Train the model
clf_paper_features.fit(newX3_train, y_train)
# Prediction on test using k features
y_pred_paper_features = clf_paper_features.predict(newX3_test)
y_pred_paper_features_score = clf_paper_features.predict_proba(newX3_test)

# Calculate metrics using the features from the paper
print("Results Using Features Paper: \n")
print("Accuracy DT features paper: ", accuracy_score(y_test, y_pred_paper_features) * 100)
print("Classification Report Features Paper:")
print(classification_report(y_test, y_pred_paper_features))
print("ROC_AUC Features Paper: ", roc_auc_score(y_test, y_pred_paper_features_score[:,1]) * 100)

# Confusion matrix using list of features selected by the paper
conf_matrix_paper = confusion_matrix(y_test, y_pred_paper_features)
class_names = data['diagnosis'].unique()
df_cm_paper = pd.DataFrame(conf_matrix_paper, index=class_names, columns=class_names)
plt.figure(figsize=(5,5))
hm_paper = sns.heatmap(df_cm_paper, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm_paper.yaxis.set_ticklabels(hm_DT.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_paper.xaxis.set_ticklabels(hm_DT.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix RF - paper features list', fontsize=15)
plt.savefig('cmatrix_rf_paper.png')
plt.show()
