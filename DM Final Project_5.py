#Load libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")


# Unpickle the clean dataframe
data = pd.read_pickle('data')

# Separate target and features
target='diagnosis'
y = data[target]
X = data.drop(columns=[target])

# Encode the target
le = LabelEncoder()
y = le.fit_transform(y)

# Subset X by the 5 selected features
features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']
X = X[features_DT_list]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# Build the param grids, the pipline and the GridSeacch to find the best hyperparameters for each model
# Include the StandardScaler in the pipeline and the 10 fold cross-validation in the GridSearch

clfs = {'lr': LogisticRegression(random_state=0),
        'dt': DecisionTreeClassifier(random_state=0),
        'rf': RandomForestClassifier(random_state=0),
        'xgb': XGBClassifier(seed=0),
        'svc': SVC(random_state= 0),
        'knn': KNeighborsClassifier()}

pipe_clfs = {}

for name, clf in clfs.items():
    # Implement me
    pipe_clfs[name] = Pipeline([('StandardScaler', StandardScaler()),
                                ('clf', clf)])

param_grids = {}

# Parameter grid for Logistic Regression
C_range = [10 ** i for i in range(-4, 5)]

param_grid = [{'clf__multi_class': ['ovr'],
               'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
               'clf__C': C_range},

              {'clf__multi_class': ['multinomial'],
               'clf__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
               'clf__C': C_range}]

param_grids['lr'] = param_grid

# Parameter grid for Decision Tree
param_grid = [{'clf__min_samples_split': [1, 2, 3, 4, 5],
               'clf__min_samples_leaf': [1,2, 3, 4, 5, 6, 7, 8, 10]}]

param_grids['dt'] = param_grid

# Parameter grid for Random Forest
param_grid = [{'clf__max_depth': [3, 5, 10, 20, 30, 40, None],
                'clf__n_estimators': [10, 100, 150],
               'clf__min_samples_split': [2, 5, 10, 30],
               'clf__min_samples_leaf': [1, 2, 5, 10]}]

param_grids['rf'] = param_grid

# Parameter grid for xgboost
param_grid = [{'clf__eta': [10 ** i for i in range(-4, 1)],
               'clf__gamma': [0, 1, 2, 3, 4, 5, 10],
               'clf__lambda': [10 ** i for i in range(-4, 5)]}]

param_grids['xgb'] = param_grid

# Parameter grid for SVC
param_grid = [{'clf__C': [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
               'clf__gamma': [1, 0.1, 0.01, 0.001],
               'clf__kernel':  ['linear', 'poly', 'rbf', 'sigmoid']}]

param_grids['svc'] = param_grid

# Parameter grid for KNN

param_grid = [{'clf__n_neighbors':[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
              'clf__leaf_size':[1,2, 3, 4, 5],
              'clf__algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]

param_grids['knn'] = param_grid


# HYPERPARAMETER TUNNING
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# The list of [best_score_, best_params_, best_estimator_]
best_score_param_estimators = []

# For each classifier
for name in pipe_clfs.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipe_clfs[name],
                      param_grid=param_grids[name],
                      scoring='accuracy',
                      n_jobs=1,
                      iid=False,
                      cv=StratifiedKFold(n_splits=10,
                                         shuffle=True,
                                         random_state=0))
    # Fit the pipeline
    gs = gs.fit(X_train, y_train)

    # Update best_score_param_estimators
    best_score_param_estimators.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

# SELECTION OF BEST PARAMETERS
# Sort best_score_param_estimators in descending order of the best_score_
best_score_param_estimators = sorted(best_score_param_estimators, key=lambda x : x[0], reverse=True)

# For each [best_score_, best_params_, best_estimator_]
for best_score_param_estimator in best_score_param_estimators:
    # Print out [best_score_, best_params_, best_estimator_], where best_estimator_ is a pipeline
    # Since we only print out the type of classifier of the pipeline
    print([best_score_param_estimator[0], best_score_param_estimator[1], type(best_score_param_estimator[2].named_steps['clf'])], end='\n\n')
