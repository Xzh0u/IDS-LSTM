import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch


# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 52
hidden_size = 128
num_layers = 2
num_classes = 21
batch_size = 256
num_epochs = 2
learning_rate = 0.01

# preprocess test data (temproary)
# test = pd.read_csv("data/sampled/test.csv")
# test.sort_values(by=['simulationRun', 'faultNumber'], inplace=True)

# ts = test.drop(test[(test.faultNumber == 3) | (test.faultNumber == 9) | (
#     test.faultNumber == 15)].index).reset_index()

# y_test = ts['faultNumber']
# ts.drop(['faultNumber', 'Unnamed: 0', 'Unnamed: 0.1',
#          'simulationRun', 'sample', 'index'], axis=1, inplace=True)

# test_normalized = (ts - np.mean(ts)) / np.std(ts)

# x_test = np.resize(test_normalized, (88832, 1, 52))

# test_loader = torch.utils.data.DataLoader(dataset=x_test,
#                                           batch_size=batch_size,
#                                           shuffle=False)


class LSTM(nn.Module):
    # build the LSTM model
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
    #   self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):  # input: tensor of shape (seq_len, batch_size, input_size)
        # Set initial hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size)  # .to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size)  # .to(device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(input_size, hidden_size, num_layers, num_classes)  # .to(device)
model.load_state_dict(torch.load('saved/model.pkl'))
# model.eval()  # change into test mode

# # Test the model
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for i, data in enumerate(test_loader):
#         labels = torch.Tensor(list(y_test.values))[
#             i * batch_size: (i + 1) * batch_size].long()
#         outputs = model(data.float())
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Test Accuracy of the model on test examples: {} %'.format(
#         100 * correct / total))


def predict(data):  # data is a list
    data = torch.from_numpy(np.array(data)).view(1, 1, 52)
    model = LSTM(input_size, hidden_size, num_layers,
                 num_classes)  # .to(device)
    model.load_state_dict(torch.load('saved/model.pkl'))

    outputs = model(data.float())
    _, predicted = torch.max(outputs.data, 1)
    result = predicted.item()

    if 16.3 < torch.max(outputs.data).item() < 30:
        loss = (torch.max(outputs.data).item()) / 15
    elif 30 < torch.max(outputs.data).item() < 50:
        loss = (torch.max(outputs.data).item() - 15) / 15
    elif torch.max(outputs.data).item() > 50:
        loss = (torch.max(outputs.data).item() - 40) / 7
    else:
        loss = random.uniform(0, 0.1)
        result = 0

    return result, loss


a = [-0.075299, 0.229284, -0.327069, 0.374839, 0.664286, 0.308727, -0.241830, -0.095738, 0.127902, -0.121672, 0.253674, 0.692829, -0.224319, -0.484372, 0.658943, -0.253818, 1.802471, -0.155913, -0.230774, 0.104397, 0.172798, 0.100519, 0.144300, 0.055570, -0.221357, 0.065689,
     0.066313, 0.246596, 0.140299, 0.082253, -0.218322, 0.053356, 0.066365, 0.260274, 0.197073, 0.194764, 0.029705, -0.052890, 0.108787, -0.016712, 0.072082, -0.185158, 0.032485, -0.315530, -0.476972, -0.089156, 0.015487, 0.692601, 0.658959, -0.211038, 0.059118, -0.578951]
print(predict(a))
