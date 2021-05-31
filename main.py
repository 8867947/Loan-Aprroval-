
import sys
import pandas as pd
from PredictData import PredictData

if __name__ == '__main__':
    executablepath = sys.executable
    executablepath = executablepath.replace("\\venv\\Scripts\\python.exe", "")
    trainOrginalData = pd.read_csv(executablepath + "\\" + "LoanApplicantData.csv")
    testOrignalData = pd.read_csv(executablepath + "\\" + "test_data.csv")
    submissionData = executablepath + "\\" + "sample_submission.csv"
    dataToBeSaved = executablepath + "\\" + "Output.csv"
    predictdata = PredictData()
    train = predictdata.DataCleaningInTrainDataSet(trainOrginalData)
    test = predictdata.DataCleaningInTestDataSet(testOrignalData)
    predictdata.ConstructModel(train, test, submissionData, dataToBeSaved)




