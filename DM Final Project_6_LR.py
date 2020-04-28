#Load libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Unpickle data
data = pd.read_pickle('data')

# Separate target and features
target='diagnosis'
y = data[target]
X = data.drop(columns=[target])
features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']
X = X[features_DT_list]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

clfs = {'lr': LogisticRegression(random_state=0)}

pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                ('clf', clf)])

param_grids = {}

# Parameter grid for Logistic Regression
C_range = [10]

param_grid = [{'clf__multi_class': ['ovr'],
               'clf__solver': ['newton-cg'],
               'clf__C': C_range}]

param_grids['lr'] = param_grid

gs = GridSearchCV(estimator=pipe_clfs['lr'],
                  param_grid=param_grids['lr'],
                  scoring='accuracy',
                  n_jobs=1,
                  iid=False,
                  cv=StratifiedKFold(n_splits=10,
                                     shuffle=True,
                                     random_state=0))

# Fit the pipeline
gs = gs.fit(X_train, y_train)

# Get prediction
y_pred = gs.predict(X_test)

conf_matrix_lr = confusion_matrix(y_test, y_pred)
class_names = data['diagnosis'].unique()
df_cm_lr = pd.DataFrame(conf_matrix_lr, index=class_names, columns=class_names)
plt.figure(figsize=(5,5))
hm_lr = sns.heatmap(df_cm_lr, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_lr.columns, xticklabels=df_cm_lr.columns)
hm_lr.yaxis.set_ticklabels(hm_lr.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_lr.xaxis.set_ticklabels(hm_lr.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix Logistic Regression - best parameters')
plt.savefig('cmatrix_lr_best.png')
plt.show()

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_pred = le.fit_transform(y_pred)

print('ROC_AUC Score', roc_auc_score(y_test, y_pred))