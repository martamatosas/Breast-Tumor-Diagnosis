#Load libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
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

clfs = {'rf': RandomForestClassifier(random_state=0)}

pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                ('clf', clf)])

param_grids = {}

# Parameter grid for Random Forest
param_grid = [{'clf__max_depth': [5],
                'clf__n_estimators': [100],
               'clf__min_samples_split': [2],
               'clf__min_samples_leaf': [5]}]

param_grids['rf'] = param_grid

gs = GridSearchCV(estimator=pipe_clfs['rf'],
                  param_grid=param_grids['rf'],
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

conf_matrix_rf = confusion_matrix(y_test, y_pred)
class_names = data['diagnosis'].unique()
df_cm_rf = pd.DataFrame(conf_matrix_rf, index=class_names, columns=class_names)
plt.figure(figsize=(5,5))
hm_rf = sns.heatmap(df_cm_rf, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_rf.columns, xticklabels=df_cm_rf.columns)
hm_rf.yaxis.set_ticklabels(hm_rf.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_rf.xaxis.set_ticklabels(hm_rf.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.title('Confusion matrix RF - best parameters')
plt.savefig('cmatrix_rf_best.png')
plt.show()

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_pred = le.fit_transform(y_pred)

print('ROC_AUC Score', roc_auc_score(y_test, y_pred))