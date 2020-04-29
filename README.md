# Group-2
Group 2 Data Mining Final Project repository.
Please read information and instructions regarding the files included in this repository.

Dataset: "beast-cancer-wisconsin-data" zip file
Accompanying paper from the University of Wisconsin: "Nuclear Feature Extraction for Breast Tumor Diagnosis" pdf file

Data Mining project Python files: should be located in your working directory together with the dataset 
    all named "DM Final Project_n_aaa" where n indicates the order in which they should be run
    there are six "DM_final Project_6_aaa" files, where aaa indicates the acronym of the classifier. These final files can be run in any order.
    "DM Final Project_1" reads, inspects the dataset and cleans the dataset; output: a clean data frame (pickle command)
    "DM Final Project_2" reads the clean dataframe (unpickle), include all the Ecploratory Data Analysis
    "DM Final Project_3" reads the clean dataframe (unpickle), includes feature importance with Decision Tree and criterion Gini and   Entropy
    "DM Final Project_4" reads the clean dataframe (unpickle), includes feature importance with Random Forest, fitting of Random Forest with fk most important features according to Random Forest feature importance, the list of features selected by the Univeristy of Wisconsin, and several custom combinations, includig the final list of 5 features labeled features_DT_list = ['texture_mean', 'area_worst', 'smoothness_worst', 'area_mean', 'concavity_mean']
    "DM Final Project_5": reads the clean dataframe (unpickle), includes subseting the feature space according to the above labeled as features_DT_list, standardization, and hyper parameter tuning with GridSearchCV and StratifiedKFold. Output: accuracy and best parameters for each of the classifiers. 
    "DM Final Project_6_aaa": reads the clean dataframe (unpickle), includes subseting the feature space, standardization, fits the classifier with best hyper parameters as per output of program "DM Final Project_5". Output is ROC-AUC score, classification report and confusion matrix. As mentioned, aaa stands for the acronym of each classifier as follows: 'lr' = Logistisc Regression, 'xgb' = XGBoost, 'SVC' = Support Vector Machine, 'RF' = Random Forest, 'DT' = Decision Tree and 'KNN' = KNEarest Neighbors.
    
GUI:
File: "GUI_Final" python file.
Image files associated, must be place in the same directory as the "GUI_Final".
    
   
