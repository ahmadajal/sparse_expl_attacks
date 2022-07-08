import os

import optuna
from optuna.trial import TrialState
import os
import numpy as np
import scipy
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import rtdl
from catboost.datasets import epsilon
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import zero
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms

# set seed:
zero.improve_reproducibility(123456)

with open("data/Learning to Rank Challenge/set1.train.txt", "r") as f:
    lines = f.readlines()
train1_dict = []
# we do not need query id (qid) for training
for l in lines:
    train1_dict.append({int(e.split(":")[0]):float(e.split(":")[1]) for e in l.strip().split(" ")[2:]})
    train1_dict[-1]["class"] = int(l.strip().split(" ")[0])

with open("data/Learning to Rank Challenge/set1.valid.txt", "r") as f:
    lines = f.readlines()
val1_dict = []
for l in lines:
    val1_dict.append({int(e.split(":")[0]):float(e.split(":")[1]) for e in l.strip().split(" ")[2:]})
    val1_dict[-1]["class"] = int(l.strip().split(" ")[0])

with open("data/Learning to Rank Challenge/set1.test.txt", "r") as f:
    lines = f.readlines()
test1_dict = []
for l in lines:
    test1_dict.append({int(e.split(":")[0]):float(e.split(":")[1]) for e in l.strip().split(" ")[2:]})
    test1_dict[-1]["class"] = int(l.strip().split(" ")[0])

train1 = pd.DataFrame(train1_dict)
val1 = pd.DataFrame(val1_dict)
test1 = pd.DataFrame(test1_dict)
###
train1_labels = train1["class"]
val1_labels = val1["class"]
test1_labels = test1["class"]
###
train1.drop("class", axis=1, inplace=True)
val1.drop("class", axis=1, inplace=True)
test1.drop("class", axis=1, inplace=True)
###
train1.fillna(0, inplace=True)
val1.fillna(0, inplace=True)
test1.fillna(0, inplace=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data
task_type = "multiclass"

if task_type != 'regression':
    train1_labels = LabelEncoder().fit_transform(train1_labels).astype('int64')
    val1_labels = LabelEncoder().fit_transform(val1_labels).astype('int64')
    test1_labels = LabelEncoder().fit_transform(test1_labels).astype('int64')
n_classes = int(max(train1_labels)) + 1 if task_type == 'multiclass' else None
print(n_classes)

X = {}
y = {}

X['test'] = test1
y['test'] = test1_labels

X["train"] = train1
y["train"] = train1_labels

X["val"] = val1
y["val"] = val1_labels


X = {
    k: torch.tensor(v.to_numpy(), device=device).float()
    for k, v in X.items()
}
y = {k: torch.tensor(v, device=device) for k, v in y.items()}

# DEVICE = torch.device("cpu")
# BATCHSIZE = 128
# CLASSES = 10
# DIR = os.getcwd()
# EPOCHS = 10
# LOG_INTERVAL = 10
# N_TRAIN_EXAMPLES = BATCHSIZE * 30
# N_VALID_EXAMPLES = BATCHSIZE * 10
# for hpo of the MLP model
hidden_layer_sizes = [[1024], [512], [256], [128], [64],
[1024, 512], [512, 256], [256, 128], [128, 64],
[1024, 512, 256], [512, 256, 128], [256, 128, 64]]

def define_model(trial):
    # We optimize the number of layers, hidden units and dropout ratio in each layer.
    d_out = n_classes or 1

    # model = rtdl.ResNet.make_baseline(
    #     d_in=train1.shape[1],
    #     d_main=128,
    #     d_hidden=trial.suggest_int("hidden_size", 64, 1024),
    #     dropout_first=trial.suggest_float("dropout_hidden", 0.0, 0.5),
    #     dropout_second=0.0,
    #     n_blocks=trial.suggest_int("n_blocks", 1, 16),
    #     d_out=d_out,
    # )
    model = rtdl.MLP.make_baseline(
    d_in=train1.shape[1],
    d_layers=trial.suggest_categorical("d_layers", hidden_layer_sizes),
    dropout=trial.suggest_float("dropout", 0.0, 0.5),
    d_out=d_out,
    )

    return model

def apply_model(model, x_num, x_cat=None):
    if isinstance(model, rtdl.FTTransformer):
        return model(x_num, x_cat)
    elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
        assert x_cat is None
        return model(x_num)
    else:
        raise NotImplementedError(
            f'Looks like you are using a custom model: {type(model)}.'
            ' Then you have to implement this branch first.'
        )


@torch.no_grad()
def evaluate(model, part):
    model.eval()
    prediction = []
    for batch in zero.iter_batches(X[part], 1024):
        prediction.append(apply_model(model, batch))
    prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
    target = y[part].cpu().numpy()

    if task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    else:
        assert task_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std
    return score

batch_size = 256

def objective(trial):

    # Generate the model.
    model = define_model(trial).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    optimizer = (
    model.make_default_optimizer()
    if isinstance(model, rtdl.FTTransformer)
    else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay))

    loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
    )
    #load data
    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
    # Training of the model.
    n_epochs = 10
    for epoch in range(1, n_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            loss = loss_fn(apply_model(model, x_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()

        val_score = evaluate(model, 'val')

        trial.report(val_score, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_score


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=5000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
