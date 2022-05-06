import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data_utils
import rtdl
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import zero
from captum.attr import Saliency, InputXGradient, IntegratedGradients, DeepLift
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.01, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_topk_tabular_data/one_time/',
                        help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['deep_lift', 'saliency', 'integrated_grad',
                            'input_times_grad'],
                       default='integrated_grad')
argparser.add_argument('--topk', type=int, default=10)
argparser.add_argument('--max_num_pixels', type=int, default=10)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not",
                        default=False)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
argparser.add_argument('--out_loss_coeff', type=float, default=1.0)
argparser.add_argument('--dataset', type=str, default="epsilon")
argparser.add_argument('--task_type', type=str, default="binclass")
args = argparser.parse_args()

def convert_relu_to_softplus(model, beta):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)
    return model

def topk_coord(array, used_inds, k=1):
    array_abs = torch.abs(array)
    if len(used_inds)>0:
        array_abs[np.array(used_inds).T.tolist()] = 0.0
    inds = torch.topk(array_abs.view((array.size()[0],-1)), k=k, dim=1)[1].detach().cpu().numpy()
    for k_ind in range(k):
        used_inds = used_inds + [[i, j[k_ind]] for i,j in enumerate(inds)]
    ###
    new_inds = np.array([[i, j[0]] for i,j in enumerate(inds)])
    for k_ind in range(1,k):
        new_inds = np.concatenate((new_inds, np.array([[i, j[k_ind]] for i,j in enumerate(inds)])))
    ###
    new_inds = new_inds.T.tolist()
    ###
    new_array = torch.zeros_like(array)
    new_array[new_inds] = array[new_inds]
    ###
    return new_array, used_inds

def normalize_expl(expl):
    expl = torch.abs(expl)
    expl = expl / torch.sum(expl , dim=1, keepdim=True)
    return expl

# dictionary to convert the method name to the proper class
expl_methods={
            "saliency": Saliency,
            "integrated_grad": IntegratedGradients,
            "input_times_grad": InputXGradient,
            "deep_lift": DeepLift
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zero.improve_reproducibility(72+args.add_to_seed)

if args.dataset == "epsilon":
    eps_test = pd.read_csv("../preparing_tabular_data/data/epsilon/eps_test.csv")
    print(eps_test.shape)
    # data
    args.task_type = "binclass"

    X_te = eps_test.iloc[:, 1:].astype('float32')
    y_te = eps_test.iloc[:, 0].astype('float32' if args.task_type == 'regression' else 'int64')
    if args.task_type != 'regression':
        y_te = LabelEncoder().fit_transform(y_te).astype('int64')
    n_classes = int(max(y_te)) + 1 if args.task_type == 'multiclass' else None
    X = {}
    y = {}

    X['test'] = X_te
    y['test'] = y_te
elif args.dataset == "yahoo":
    test1 = pd.read_csv("../preparing_tabular_data/data/Learning to Rank Challenge/test_data1.csv")
    test1_labels = test1["class"]
    test1.drop("class", axis=1, inplace=True)
    args.task_type = "multiclass"
    if args.task_type != 'regression':
        test1_labels = LabelEncoder().fit_transform(test1_labels).astype('int64')
    n_classes = int(max(test1_labels)) + 1 if args.task_type == 'multiclass' else None
    X = {}
    y = {}

    X['test'] = test1
    y['test'] = test1_labels
else:
    raise ValueError("invalid dataset name: {}".format(args.dataset))



X = {
    k: torch.tensor(v.to_numpy(), device=device, dtype=torch.float32)
    for k, v in X.items()
}
if args.task_type=="binclass":
    y = {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in y.items()}
else:
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

###
BATCH_SIZE = 64
indices = np.random.randint(len(y["test"]), size=BATCH_SIZE)
#### load data #####
examples = X["test"][indices].to(device)
labels = y["test"][indices].to(device)
###
if args.dataset == "epsilon":
    model = torch.load("../preparing_tabular_data/models/rtdl_resnet_epsilon.pth")
elif args.dataset == "yahoo":
    model = torch.load("../preparing_tabular_data/models/rtdl_resnet_yahoo.pth")
model = convert_relu_to_softplus(model, beta=10)
model = model.eval().to(device)
# keep only data points for which the model predicts correctly
with torch.no_grad():
    if args.task_type=="binclass":
        preds = torch.tensor([1 if torch.sigmoid(l)>0.5 else 0 for l in model(examples).detach()]).to(device)
    else:
        preds = model(examples).argmax(dim=1).detach()
    samples_to_pick = (preds==labels)
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
###
examples = examples.requires_grad_(True)
###
print(examples.size())
BATCH_SIZE = examples.size()[0]

####
# sigma for the smooth grad and uniform grad methods
# sigma = tuple((torch.max(examples[i]) -
#         torch.min(examples[i])).item() * 0.1 for i in range(examples.size()[0]))
# if len(sigma)==1:
#     sigma=sigma[0]

####
# creating the mask for top-k attack
explanation = expl_methods[args.method](model)
if args.task_type=="binclass":
    org_expl = explanation.attribute(examples)
else:
    org_expl = explanation.attribute(examples, target=labels)
org_expl = normalize_expl(org_expl)

org_expl = org_expl.detach()
if BATCH_SIZE == 1:
    mask = torch.zeros_like(org_expl).flatten()
    mask[torch.argsort(org_expl.view(-1))[-args.topk:]]=1
    mask = mask.view(org_expl.size())
else:
    mask = torch.zeros_like(org_expl).view(BATCH_SIZE, -1)
    topk_perbatch = torch.argsort(org_expl.view(BATCH_SIZE, -1))[:, -args.topk:]
    for _ in range(mask.size()[0]):
        mask[_][topk_perbatch[_]] = 1
    mask = mask.view(org_expl.size())
# orginal logits for the output loss
if args.task_type == "binclass":
    org_logits = torch.sigmoid(model(examples))
else:
    org_logits = F.softmax(model(examples), dim=1)
org_logits = org_logits.detach()
###
x_adv = copy.deepcopy(examples)
optimizer = optim.Adam([x_adv], lr=args.lr)

# ub and lb for valid range:
if args.dataset == "yahoo":
    lb = 0.0
    ub = 1.0
else:
    lb = torch.min(X["test"], dim=0)[0].detach()
    ub = torch.max(X["test"], dim=0)[0].detach()


######
optimizer.zero_grad()
if args.task_type == "binclass":
    adv_expl = explanation.attribute(x_adv)
else:
    adv_expl = explanation.attribute(x_adv, target=labels)
adv_expl = normalize_expl(adv_expl)
expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=1, dtype=torch.float))
if args.task_type == "binclass":
    out_loss = F.mse_loss(torch.sigmoid(model(x_adv)), org_logits)
else:
    out_loss = F.mse_loss(F.softmax(model(x_adv), dim=1), org_logits)
loss = expl_loss + args.out_loss_coeff * out_loss
loss.backward()
# normalize gradient
x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=1, keepdim=True))
# pick the top k coordinates of the update
x_adv.grad, used_indices = topk_coord(x_adv.grad, [], k=args.topk)
###
optimizer.step()
# update step
x_adv.data = examples.data + torch.clip((x_adv-examples).data, (lb-examples).data, (ub-examples).data)

# after finishing the iterations
# load the relu model again:
if args.dataset == "epsilon":
    model = torch.load("../preparing_tabular_data/models/rtdl_resnet_epsilon.pth")
elif args.dataset == "yahoo":
    model = torch.load("../preparing_tabular_data/models/rtdl_resnet_yahoo.pth")
####
explanation = expl_methods[args.method](model)
####
if args.task_type == "binclass":
    adv_expl = explanation.attribute(x_adv)
else:
    adv_expl = explanation.attribute(x_adv, target=labels)
adv_expl = normalize_expl(adv_expl)
topk_ints = []
for i in range(mask.size()[0]):
    _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
    _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
    topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                        topk_adv_ind.cpu().detach().numpy())))/args.topk)
############################
if args.task_type == "binclass":
    preds_org = (torch.sigmoid(model(examples)) > 0.5).squeeze()
else:
    preds_org = model(examples).argmax(dim=1)
print("org acc: ", (labels==preds_org).sum()/BATCH_SIZE)
if args.task_type == "binclass":
    preds = (torch.sigmoid(model(x_adv)) > 0.5).squeeze()
else:
    preds = model(x_adv).argmax(dim=1)
print("adv acc: ", (labels==preds).sum()/BATCH_SIZE)
############################
print(torch.max(x_adv))
print("all top-k intersection: ", topk_ints)
print("mean top-k intersection and std: ", np.mean(topk_ints), np.std(topk_ints))
n_pixels = torch.sum(torch.abs(x_adv-examples) > 1e-10, dim=1)
print("total pixels changed: ", n_pixels)
torch.save(x_adv, f"{args.output_dir}x_{args.method}.pth")
