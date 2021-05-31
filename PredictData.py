
import os
import sys

import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
import statistics
warnings.filterwarnings("ignore")

class PredictData:

    def DataCleaningInTrainDataSet(self, trainOrginalData):

        # **************  Missing value imputation *********************
        missingDataInVariable = trainOrginalData.isnull().sum()
        # 'Gender' , 'Married' , 'Dependents' , 'Self_Employed', 'Credit_History' 'LoanAmount', 'Loan_Amount_Term' having missing values
        trainOrginalData['Gender'].fillna(trainOrginalData['Gender'].mode()[0], inplace=True)
        trainOrginalData['Married'].fillna(trainOrginalData['Married'].mode()[0], inplace=True)
        trainOrginalData['Dependents'].fillna(trainOrginalData['Dependents'].mode()[0], inplace=True)
        trainOrginalData['Self_Employed'].fillna(trainOrginalData['Self_Employed'].mode()[0], inplace=True)
        trainOrginalData['Loan_Amount_Term'].fillna(trainOrginalData['Loan_Amount_Term'].mode()[0], inplace=True)
        trainOrginalData['Credit_History'].fillna(trainOrginalData['Credit_History'].mode()[0], inplace=True)
        # As per analysis we see 'LoanAmount' has many outliers , so we are using median
        trainOrginalData['LoanAmount'].fillna(trainOrginalData['LoanAmount'].median(), inplace=True)

        # **************  Outlier Treatment *********************

        # as per our analysis we found LoanAmount contains outliers so we have to treat them as the presence of outliers affects
        # the distribution of the data

        # remove the outliers with  " log transformation " in train data
        trainOrginalData['LoanAmount_log'] = np.log(trainOrginalData['LoanAmount'])
        trainOrginalData['LoanAmount_log'].hist(bins=20)
        trainOrginalData['LoanAmount_log'] = np.log(trainOrginalData['LoanAmount'])

        # Replace '3+' value with 3 in Dependents
        trainOrginalData['Dependents'].replace('3+', 3, inplace=True)

        # Replace Loan_Status with binary value
        trainOrginalData['Loan_Status'].replace('Y', 1, inplace=True)
        trainOrginalData['Loan_Status'].replace('N', 0, inplace=True)

        # Lets drop Loan_ID variables as it do not have any effect on the loan status
        cleanDatainTrainOrginalData = trainOrginalData.drop('Loan_ID', axis=1)



        return cleanDatainTrainOrginalData

    def DataCleaningInTestDataSet(self, testOrginalData):

        testOrginalData['Dependents'].replace('3+', 3, inplace=True)
        # **************  Missing value imputation *********************
        missingDataInVariable = testOrginalData.isnull().sum()
        # 'Gender','Dependents' , 'Self_Employed', 'Credit_History' 'LoanAmount', 'Loan_Amount_Term' having missing values
        testOrginalData['Gender'].fillna(testOrginalData['Gender'].mode()[0], inplace=True)
        testOrginalData['Dependents'].fillna(testOrginalData['Dependents'].mode()[0], inplace=True)
        testOrginalData['Self_Employed'].fillna(testOrginalData['Self_Employed'].mode()[0], inplace=True)
        testOrginalData['Loan_Amount_Term'].fillna(testOrginalData['Loan_Amount_Term'].mode()[0], inplace=True)
        testOrginalData['Credit_History'].fillna(testOrginalData['Credit_History'].mode()[0], inplace=True)
        # As per analysis we see 'LoanAmount' has many outliers , so we are using median
        testOrginalData['LoanAmount'].fillna(testOrginalData['LoanAmount'].median(), inplace=True)

        # **************  Outlier Treatment *********************
        # remove the outliers with  " log transformation " in test data and created new feature "LoanAmount_log"
        testOrginalData['LoanAmount_log'] = np.log(testOrginalData['LoanAmount'])

        return testOrginalData

    def ConstructModel(self, train, test, submissionData, dataToBeSaved):

        try:

            # Will start with " logistic Regression "
            predictionScore = []
            X = train.drop('Loan_Status', 1)
            y = train.Loan_Status
            X = pd.get_dummies(X)
            train = pd.get_dummies(X)
            test1 = test
            # Lets drop Loan_ID variables as it do not have any effect on the loan status
            test1 = test1.drop('Loan_ID', axis=1)
            test1 = pd.get_dummies(test1)
            i = 1
            kf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                #print('\n{} of kfold {}'.format(i, kf.n_splits))
                x_train, x_test = X.loc[train_index], X.loc[test_index]

                y_train, y_test = y[train_index], y[test_index]

                model = LogisticRegression()
                clf = model.fit(x_train, y_train)
                LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1,max_iter=100, multi_class='ovr', n_jobs=1, penalty='12', random_state=1, solver='liblinear',
                                   tol=0.0001, verbose=0, warm_start=False)

                pred_csv = model.predict(x_test)
                score = accuracy_score(y_test, pred_csv)
                #print("Prediction Score for : ", score)
                predictionScore.append(score)

                #print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(model.score(x_test, y_test)))
                confusionmatrix = confusion_matrix(y_test, pred_csv)
                #print(confusionmatrix)
                #print(classification_report(y_test, pred_csv))
                plot_confusion_matrix(clf, x_train, y_train)
                #plt.show()
                i += 1
                pred_test = model.predict(test1)
                pred = model.predict_proba(x_test)[:, 1]


            print("Accuracy: %0.2f (+/- %0.2f)" % (statistics.mean(predictionScore) , statistics.stdev(predictionScore) * 2))

            submission = pd.read_csv(submissionData)
            submission['Loan_Status'] = pred_test

            submission['Loan_ID'] = test['Loan_ID']
            submission['Loan_Status'].replace(0, 'N', inplace=True)
            submission['Loan_Status'].replace(1, 'y', inplace=True)

            pd.DataFrame(submission, columns=['Loan_ID', 'Loan_Status']).to_csv(dataToBeSaved)
        except ValueError:
            print("Error in Value")



