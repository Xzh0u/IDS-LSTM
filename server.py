import glob
import sys
import os
from datetime import datetime
from py.predict import Predictor
from py.predict import ttypes
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer

import numpy as np
import pandas as pd
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

input_size = 52
hidden_size = 128
num_layers = 2
num_classes = 21
batch_size = 256
num_epochs = 2
learning_rate = 0.01


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


class PredictionHandler:
    def ping(self):
        print('ping()')

    def pong(self, data):
        print(data)
        data.append(1.5)
        return data

    def predict(self, data, timestamp):  # data is a list
        data = torch.from_numpy(np.array(data)).view(1, 1, 52)
        print(datetime.now(), " Receive data successfully.")
        model = LSTM(input_size, hidden_size, num_layers,
                     num_classes)  # .to(device)
        script_dir = os.path.dirname(__file__)
        model.load_state_dict(torch.load(
            os.path.join(script_dir, 'saved/model.pkl')))

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
            # loss = random.uniform(0, 0.1)
            loss = torch.max(outputs.data).item() / 100
            result = 0

        pred = ttypes.pred()
        pred.type = result
        pred.loss = loss
        pred.timestamp = timestamp
        print(pred)
        return pred


if __name__ == '__main__':
    model = LSTM(input_size, hidden_size, num_layers,
                 num_classes)  # .to(device)
    script_dir = os.path.dirname(__file__)
    model.load_state_dict(torch.load(
        os.path.join(script_dir, 'saved/model.pkl')))
    handler = PredictionHandler()
    processor = Predictor.Processor(handler)
    transport = TSocket.TServerSocket(host='127.0.0.1', port=9090)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    # server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
    # You could do one of these for a multithreaded server
    server = TServer.TThreadedServer(
        processor, transport, tfactory, pfactory)
    server.serve()
