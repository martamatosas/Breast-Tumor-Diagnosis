#Load libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
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

clfs = {'knn': KNeighborsClassifier()}

pipe_clfs = {}

for name, clf in clfs.items():
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                ('clf', clf)])

param_grids = {}

# Parameter grid for KNN

param_grid = [{'clf__n_neighbors':[11],
              'clf__leaf_size':[1],
              'clf__algorithm':['auto']}]

param_grids['knn'] = param_grid

gs = GridSearchCV(estimator=pipe_clfs['knn'],
                  param_grid=param_grids['knn'],
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

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))

le = LabelEncoder()
y_test = le.fit_transform(y_test)
y_pred = le.fit_transform(y_pred)

print('ROC_AUC Score', roc_auc_score(y_test, y_pred))

# CONFUSION MATRIX
# Plot non-normalized confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['B','M'])
cmd.plot(cmap='Blues')
cmd.ax_.set(xlabel='Predicted', ylabel='True')
plt.figure(figsize=(5,5))
plt.show()

tn, fp, fn, tp = cm.ravel()
print('Coefficients Confusion KNN - best parameters')
print('tn', tn, 'fp', fp, 'fn', fn, 'tp', tp)

# OUTPUT TO INPUT CAPSTONE PROJECT
# OUTPUT THE MODEL
import pickle
with open('clf.KNN', 'wb') as f:
    pickle.dump(gs, f)

# OUTPUT THE PREDICTIONS
from numpy import savetxt
savetxt('KNN_y_pred.csv', y_pred, delimiter=',')

# OUTPUT THE WRONG PREDICTIONS AND CORRESPONDING SAMPLE NUMBER
KNN_x_wrong = []
KNN_y_wrong = []

for i in range(len(y_pred)):
    y_t = y_test[i]
    y_p = y_pred[i]
    if y_t != y_p:
        KNN_x_wrong.append(i)
        KNN_y_wrong.append(y_p)
        print(i, 'predicted:', y_p, 'true:', y_t)

# save to csv file
savetxt('KNN_x_wrong.csv', KNN_x_wrong, delimiter=',')
savetxt('KNN_y_wrong.csv', KNN_y_wrong, delimiter=',')
