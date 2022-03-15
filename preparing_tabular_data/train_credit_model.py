import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
import time
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.0001, help='lr')
argparser.add_argument('--epochs', type=int, default=5)
argparser.add_argument('--num_layers', type=int, default=4)
argparser.add_argument('--first_hidden_size', type=int, default=256)
argparser.add_argument('--dropout', type=float, default=0.0, help='dropout')

args = argparser.parse_args()

df_normalized = pd.read_csv("data/normalized_credit_default.csv")
y_values = pd.read_csv("data/y_values_credit_default.csv")["default payment next month"]
print(df_normalized.shape)
x_tr, x_te, y_tr, y_te = train_test_split(df_normalized, y_values, test_size=0.2, stratify=y_values, random_state=0)
features_tensor_tr = torch.tensor(np.array(x_tr), dtype=torch.float)
target_tensor_tr = torch.tensor(y_tr.values)
###
features_tensor_te = torch.tensor(np.array(x_te), dtype=torch.float)
target_tensor_te = torch.tensor(y_te.values)
train_dataset = data_utils.TensorDataset(features_tensor_tr, target_tensor_tr)
test_dataset = data_utils.TensorDataset(features_tensor_te, target_tensor_te)
train_loader = data_utils.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = data_utils.DataLoader(test_dataset, batch_size=128, shuffle=False)
##
def accuracy(predicted_logits, reference):
    """Compute the ratio of correctly predicted labels"""
    pred_labels = [1 if i else 0 for i in F.sigmoid(predicted_logits) > 0.5]
    correct_predictions = pred_labels==reference.detach().cpu().numpy()
    return correct_predictions.sum() / len(correct_predictions)
##
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

###
### training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_credit(df_normalized.shape[1], args.first_hidden_size, args.num_layers, args.dropout)
model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    model.train() # Enable dropout (if have).
    start_time = time.time()
    for features, target in train_loader:
        features = features.to(device)
        target = target.float().to(device)

        model.zero_grad()
        prediction = model(features)
        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()

    model.eval()
    test_accs = []
    weights_for_avg = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x)
            acc = accuracy(prediction, batch_y)
            test_accs.append(acc)
            weights_for_avg.append(len(batch_x))
    elapsed_time = time.time() - start_time
    print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
            time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
    print("test acc: {:.3f}".format(np.mean(test_accs)))
