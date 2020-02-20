import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import to_categorical

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 52
hidden_size = 128
num_layers = 2
num_classes = 21
batch_size = 256
num_epochs = 100
learning_rate = 0.01

# Reading the data in the form of csv
train = pd.read_csv("data/sampled/train.csv")
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
tr = train.drop(train[(train.faultNumber == 3) | (
    train.faultNumber == 9) | (train.faultNumber == 15)].index).reset_index()
ts = test.drop(test[(test.faultNumber == 3) | (test.faultNumber == 9) | (
    test.faultNumber == 15)].index).reset_index()
cv_ = cv.drop(cv[(cv.faultNumber == 3) | (cv.faultNumber == 9)
                 | (cv.faultNumber == 15)].index).reset_index()
print(tr)

y_train = tr['faultNumber']
y_test = ts['faultNumber']
y_cv = cv_['faultNumber']

# y_train = to_categorical(tr['faultNumber'], num_classes=21)
# y_test = to_categorical(ts['faultNumber'], num_classes=21)
# y_cv = to_categorical(cv_['faultNumber'], num_classes=21)

# Removing unnecessary features from train, test and cv data.
tr.drop(['faultNumber', 'Unnamed: 0', 'Unnamed: 0.1',
         'simulationRun', 'sample', 'index'], axis=1, inplace=True)
ts.drop(['faultNumber', 'Unnamed: 0', 'Unnamed: 0.1',
         'simulationRun', 'sample', 'index'], axis=1, inplace=True)
cv_.drop(['faultNumber', 'Unnamed: 0', 'Unnamed: 0.1',
          'simulationRun', 'sample', 'index'], axis=1, inplace=True)
print(tr)

# Data normalization
train_normalized = (tr - np.mean(tr)) / np.std(tr)
test_normalized = (ts - np.mean(ts)) / np.std(ts)
cv_normalized = (cv_ - np.mean(cv_)) / np.std(cv_)
print(train_normalized)

# print('Shape of the Train dataset:', train_normalized.shape)
# print("Shape of the Test dataset:", test_normalized.shape)
# print("Shape of the CV dataset:", cv_normalized.shape)

# Resizing the train, test and cv data.
x_train = np.resize(train_normalized, (230400, 1, 52))
x_test = np.resize(test_normalized, (88832, 1, 52))
x_cv = np.resize(cv_normalized, (93440, 1, 52))

# data_loader
train_loader = torch.utils.data.DataLoader(dataset=x_train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=x_test,
                                          batch_size=batch_size,
                                          shuffle=False)

cv_loader = torch.utils.data.DataLoader(dataset=x_cv,
                                        batch_size=batch_size,
                                        shuffle=True)
print("DataLoader prepared!")

# build the network


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
    #    self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)
    #    self.softmax = nn.Softmax(dim=1)

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

    #    out = self.softmax(out)
        return out


model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model.float()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
print(total_step)
print(len(test_loader))
for epoch in range(num_epochs):
    for i, datas in enumerate(train_loader):
        # Forward pass
        outputs = model(datas.float())
        loss = criterion(outputs, torch.Tensor(list(y_train.values))[
                         i * batch_size: (i + 1) * batch_size].long())
        # loss = criterion(outputs,
        #                  torch.Tensor(y_train[i * batch_size: (i + 1) * batch_size, :]))

        # Backward and optimize
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    model.float()
    for i, data in enumerate(test_loader):
        labels = torch.Tensor(list(y_test.values))[
            i * batch_size: (i + 1) * batch_size]
        outputs = model(data.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on test examples: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.pkl')