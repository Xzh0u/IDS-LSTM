import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 52
hidden_size = 128
num_layers = 2
num_classes = 21
batch_size = 256
num_epochs = 2
learning_rate = 0.01

# Reading the data in the form of csv
train = pd.read_csv('data/sampled/train.csv')
test = pd.read_csv("data/sampled/test.csv")
cv = pd.read_csv("data/sampled/cv.csv")

print("Shape of the sampled train data:", train.shape)
print("Shape of the sampled test data:", test.shape)
print("Shape of the sampled CV data:", cv.shape)

# Sorting the Datasets w.r.t. to the simulation runs
train.sort_values(by=['simulationRun', 'faultNumber'], inplace=True)
test.sort_values(by=['simulationRun', 'faultNumber'], inplace=True)
cv.sort_values(by=['simulationRun', 'faultNumber'], inplace=True)

# Removing faults 3,9 and 15
ts = test.drop(test[(test.faultNumber == 3) | (test.faultNumber == 9) | (
    test.faultNumber == 15)].index).reset_index()


y_test = ts['faultNumber']

# Removing unnecessary features from train, test and cv data.
ts.drop(['faultNumber', 'Unnamed: 0', 'Unnamed: 0.1',
         'simulationRun', 'sample', 'index'], axis=1, inplace=True)

x_test = np.resize(ts, (88832, 1, 52))

test_loader = torch.utils.data.DataLoader(dataset=x_test,
                                          batch_size=batch_size,
                                          shuffle=False)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
    #   self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # input: tensor of shape (seq_len, batch, input_size)
        # Set initial hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('saved/model.pkl'))
model.eval()  # change into test mode

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    model.float()
    for i, data in enumerate(test_loader):
        labels = torch.Tensor(list(y_test.values))[
            i * batch_size: (i + 1) * batch_size].long()
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on test examples: {} %'.format(
        100 * correct / total))
