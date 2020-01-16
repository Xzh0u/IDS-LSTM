import pandas as pd
import pyreadr as py  # Library to read .Rdata files in python


def load_RData():
    # Reading train data in .R format
    train_data_0 = py.read_r("data/RData/TEP_FaultFree_Training.RData")
    train_data_1 = py.read_r("data/RData/TEP_Faulty_Training.RData")
    # Reading test data in .R format
    test_data_0 = py.read_r("data/RData/TEP_FaultFree_Testing.RData")
    test_data_1 = py.read_r("data/RData/TEP_Faulty_Testing.RData")
    print("Finish reading data.")

    # Concatinating the train and the test dataset
    tr = [train_data_0['fault_free_training'], train_data_1['faulty_training']]
    train = pd.concat(tr)  # Train dataframe
    ts = [test_data_0['fault_free_testing'], test_data_1['faulty_testing']]
    test = pd.concat(ts)  # Test dataframe

    # Save the datasets into csv file
    train.to_csv("data/train.csv")
    test.to_csv("data/test.csv")


# Read csv data
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
print('finish reading csv files!')

print('Shape of the Train dataset:', train_data.shape)
print("Shape of the Test dataset:", test_data.shape)

Sampled_train = pd.DataFrame()  # dataframe to store the train dataset
Sampled_test = pd.DataFrame()  # dataframe to store test
Sampled_cv = pd.DataFrame()  # dataframe to store cv data

# Program to construct the sample train data
frame = []
for i in set(train_data['faultNumber']):
    b_i = pd.DataFrame()
    if i == 0:
        b_i = train_data[train_data['faultNumber'] == i][0:20000]
        frame.append(b_i)
    else:
        fr = []
        b = train_data[train_data['faultNumber'] == i]
        for x in range(1, 25):
            b_x = b[b['simulationRun'] == x][20:500]
            fr.append(b_x)

        b_i = pd.concat(fr)

    frame.append(b_i)
Sampled_train = pd.concat(frame)

# Program to construct the sample CV Data
frame = []
for i in set(train_data['faultNumber']):
    b_i = pd.DataFrame()
    if i == 0:
        b_i = train_data[train_data['faultNumber'] == i][20000:30000]
        frame.append(b_i)
    else:
        fr = []
        b = train_data[train_data['faultNumber'] == i]
        for x in range(26, 35):
            b_x = b[b['simulationRun'] == x][20:500]
            fr.append(b_x)

        b_i = pd.concat(fr)

    frame.append(b_i)
Sampled_cv = pd.concat(frame)

# Program to construct Sampled Test data
frame = []
for i in set(test_data['faultNumber']):
    b_i = pd.DataFrame()
    if i == 0:
        b_i = test_data[test_data['faultNumber'] == i][0:2000]
        frame.append(b_i)
    else:
        fr = []
        b = test_data[test_data['faultNumber'] == i]
        for x in range(1, 11):
            b_x = b[b['simulationRun'] == x][160:660]
            fr.append(b_x)

        b_i = pd.concat(fr)

    frame.append(b_i)
Sampled_test = pd.concat(frame)
print("Finish sample data!")

# Storing the Train, Test and CV dataset into csv file for further use.
Sampled_train.to_csv("data/sampled/train.csv")
Sampled_test.to_csv("data/sampled/test.csv")
Sampled_cv.to_csv("data/sampled/cv.csv")
