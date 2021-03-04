# Breast Tumor Diagnosis
Prediciton of breast tumor doagnosis.

Feature selection (5 out of 30) was conducted using decision trees.

Hyper parameter tunning was performed on the following classifiers: logistic regression, decision tree, random forest, KNN, XGBoost and SVC,

Best performing classifier is logistc regression: achieved accuracy of 98.24%, and precision and recall of 100% and 96%, respectively, for malignant diagnosis. 

Information and instructions regarding the files included in this repository.

**To run the project, please locate in your working directory the data.csv file together with all 11 files listed below as Data Mining project Python files.**

Original data zip file: "beast-cancer-wisconsin-data" zip file, as downloaded from Kaggle, included only for reference.

Dataset: "data" csv file

Accompanying paper from the University of Wisconsin: "Nuclear Feature Extraction for Breast Tumor Diagnosis" pdf file

**Python files:**

Files "DM Final Project_n", where n ranges from 1 to 5 should be run un order:

"DM Final Project_1" reads, inspects the dataset and cleans the dataset; output: a clean data frame (pickle command)

"DM Final Project_2" reads the clean dataframe (unpickle), include all the Exploratory Data Analysis

"DM Final Project_3" reads the clean dataframe (unpickle), includes feature importance with Decision Tree and criterion Gini and   Entropy. Output is feature importance plot and pdf files of the 2 decision trees.

"DM Final Project_4" reads the clean dataframe (unpickle), includes feature importance with Random Forest, fitting of Random Forest with the "k" most important features according to Random Forest feature importance (k defined as a variable), the list of features selected by the University of Wisconsin, and several custom combinations, including the final list of 5 features labeled features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']. Output is feature importance plot, classification report, ROC_AUC score and confusion matrix for all.

"DM Final Project_5": reads the clean dataframe (unpickle), includes subseting the feature space according to the above labeled as features_DT_list, standardization, and hyper parameter tuning with GridSearchCV and StratifiedKFold. Output: accuracy and best parameters for each of the classifiers. 

Files named "DM Final Project_6_aaa" where where aaa indicates the acronym of the classifier can be run in any order. "aaa" acronyms: 'lr' = Logistisc Regression, 'xgb' = XGBoost, 'SVC' = Support Vector Machine, 'RF' = Random Forest, 'DT' = Decision Tree and 'KNN' = K=Nearest Neighbors.

"DM Final Project_6_aaa": reads the clean dataframe (unpickle), includes subseting the feature space, standardization, fits the classifier with best hyper parameters as per output of program "DM Final Project_5". Output is ROC-AUC score, classification report and confusion matrix.

**To run the Demo of the project, please locate in your working directory the "GUI_final" python file together with all 10 images listed below under the Demo section.**Acronym **Demo:**

GUI File: "GUI_Final" python file.

Image files associated: "best_parameters", "best_results", "boxplots_worst", "corr_worst", "countplot_target", "feature_importance1", "feature_selection", "Front", "histograms_worst" and "polar_worst". 
 
**Final Project Report:**
"DATS6103 _Group 2_Project Final Report.pdf"

**Final Project Presentation:**
"DM Final Project Presentation_MMF_vFinal".
