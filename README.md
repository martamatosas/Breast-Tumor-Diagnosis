# Breast Tumor Diagnosis

The objective is to predict the diagnosis of a Fine Needle Aspirate biopsy from a breast tumor. When a fine needle aspirate biopsy is performed, a small drop of fluid is obtained and the sample is evaluated by a pathologist. In this case, the sample is expressed onto a glass slide and stained. The image for digital analysis is generated by a color video camera mounted on top of a microscope. To successfully analyze the digital image, a user selects a set of nuclei and specifies each location of each cell nucleus boundary through a graphical user interface. The system then analyzes and computes features relative to the size, shape and texture of those nuclei. Specifically, it calculates the mean value, the largest value and the standard error of 10 features over the range of cells selected.
The 10 extracted features are: radius (mean of distances from center to points on the perimeter), texture (standard deviation of gray-scale values), perimeter, area, smoothness (local variation in radius lengths), compactness (perimeter^2 / area - 1.0), concavity (severity of concave portions of the contour), concave points (number of concave portions of the contour), symmetry andfractal dimension ("coastline approximation").

Feature selection (5 out of 30) was conducted using decision trees.

Hyper parameter tunning was performed on the following classifiers: logistic regression, decision tree, random forest, KNN, XGBoost and SVC.

**Key results:**

Best performing classifier is logistc regression: achieved accuracy of 98.24%, and precision and recall of 100% and 96%, respectively, for malignant diagnosis. 

**Information and instructions regarding the files included in this repository.**

To run the project, please locate in your working directory the data.csv file together with all 11 files listed below as Data Mining project Python files.

Original data zip file: "beast-cancer-wisconsin-data" zip file, as downloaded from Kaggle, included only for reference.

Dataset: "data" csv file.

Accompanying paper from the University of Wisconsin: "Nuclear Feature Extraction for Breast Tumor Diagnosis" pdf file.

**Python files:**

Files "DM Final Project_n", where n ranges from 1 to 5 should be run un order:

"DM Final Project_1" reads, inspects the dataset and cleans the dataset; output: a clean data frame (pickle command)

"DM Final Project_2" reads the clean dataframe (unpickle), include all the Exploratory Data Analysis

"DM Final Project_3" reads the clean dataframe (unpickle), includes feature importance with Decision Tree and criterion Gini and   Entropy. Output is feature importance plot and pdf files of the 2 decision trees.

"DM Final Project_4" reads the clean dataframe (unpickle), includes feature importance with Random Forest, fitting of Random Forest with the "k" most important features according to Random Forest feature importance (k defined as a variable), the list of features selected by the University of Wisconsin, and several custom combinations, including the final list of 5 features labeled features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']. Output is feature importance plot, classification report, ROC_AUC score and confusion matrix for all.

"DM Final Project_5": reads the clean dataframe (unpickle), includes subseting the feature space according to the above labeled as features_DT_list, standardization, and hyper parameter tuning with GridSearchCV and StratifiedKFold. Output: accuracy and best parameters for each of the classifiers. 

Files named "DM Final Project_6_aaa" where where aaa indicates the acronym of the classifier can be run in any order. "aaa" acronyms: 'lr' = Logistisc Regression, 'xgb' = XGBoost, 'SVC' = Support Vector Machine, 'RF' = Random Forest, 'DT' = Decision Tree and 'KNN' = K=Nearest Neighbors.

"DM Final Project_6_aaa": reads the clean dataframe (unpickle), includes subseting the feature space, standardization, fits the classifier with best hyper parameters as per output of program "DM Final Project_5". Output is ROC-AUC score, classification report and confusion matrix.
