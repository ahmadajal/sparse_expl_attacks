import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
import time

class model_credit(nn.Module):
    def __init__(self, input_size, first_hidden_layer_dim, num_layers, dropout):
        super(model_credit, self).__init__()
        """
        dropout: dropout rate between fully connected layers;
        """
        self.dropout = dropout
        MLP_modules = []
        for i in range(num_layers):
            MLP_modules.append(nn.Linear(input_size, first_hidden_layer_dim//(2**i)))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=self.dropout))
            input_size = first_hidden_layer_dim//(2**i)
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(input_size, 1)

        self._init_weight_()

    def _init_weight_(self):
        """ We leave the weights initialization here. """

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        output_MLP = self.MLP_layers(x)
        prediction = self.predict_layer(output_MLP)
        return prediction.view(-1)
