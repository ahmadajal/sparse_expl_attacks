import numpy as np
import pandas as pd
import re
import copy
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import matplotlib
import sys
sys.path.append("../preparing_tabular_data/")
from tabular_models import model_credit
# for distances between expls
from scipy.stats import spearmanr as spr
import scipy.spatial as spatial
sys.path.append("../attacks/")
import mister_ed.utils.pytorch_utils as utils
from utils import load_image, torch_to_image, get_expl, convert_relu_to_softplus, plot_overview, UniGrad
from captum.attr import Saliency
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
argparser.add_argument('--lr', type=float, default=0.5, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_expl_topk_pgd0_tabular/',
                        help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=20)
argparser.add_argument('--max_num_pixels', type=int, default=20)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not",
                        default=False)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
argparser.add_argument('--additive_lp_bound', type=float, default=0.03, help='l_p bound for additive attack')
args = argparser.parse_args()

# project to l0 box
def project_L0_box_torch(y, k, lb, ub):
    x = torch.clone(y)
    p1 = torch.sum(x**2, dim=-1)
    p2 = torch.minimum(torch.minimum(ub - x, x - lb), torch.zeros_like(x))
    p2 = torch.sum(p2**2, dim=-1)
    p3 = torch.sort(torch.reshape(p1-p2, (p2.size()[0],-1)))[0][:,-k]
    x = x*(torch.logical_and(lb <=x, x <= ub)) + lb*(lb > x) + ub*(x > ub)
    x = x * torch.unsqueeze((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)

    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(72+args.add_to_seed)
torch.manual_seed(72+args.add_to_seed)
torch.cuda.manual_seed(72+args.add_to_seed)
df_normalized = pd.read_csv("../preparing_tabular_data/data/normalized_credit_default.csv")
y_values = pd.read_csv("../preparing_tabular_data/data/y_values_credit_default.csv")["default payment next month"]
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
BATCH_SIZE = 32
indices = np.random.randint(128, size=BATCH_SIZE)
#### load images #####
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices].to(device)
labels = labels_batch[indices].to(device)
###
model = torch.load("../preparing_tabular_data/models/first_credit_model.pth")
model = convert_relu_to_softplus(model, beta=100)
model = model.eval().to(device)
# keep only data points for which the model predicts correctly
with torch.no_grad():
    preds = torch.tensor([1 if F.sigmoid(l)>0.5 else 0 for l in model(examples).detach()]).to(device)
    samples_to_pick = (preds==labels)
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
###
examples = examples.requires_grad_(True)
###
print(examples.size())
BATCH_SIZE = examples.size()[0]
####
####
# sigma for the smooth grad and uniform grad methods
# sigma = tuple((torch.max(examples[i]) -
#         torch.min(examples[i])).item() * 0.1 for i in range(examples.size()[0]))
# if len(sigma)==1:
#     sigma=sigma[0]

# creating the mask for top-k attack
sal = Saliency(model)
org_expl = sal.attribute(examples)
# org_expl = get_expl(model, normalizer.forward(examples), args.method, desired_index=labels, smooth=args.smooth,
#                     sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
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

# delta = nn.Parameter(torch.zeros_like(examples))
x_adv = copy.deepcopy(examples)
optimizer = optim.Adam([x_adv], lr=args.lr)
######
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_iter//2], gamma=0.4)
######

for iter_no in range(args.num_iter):
    optimizer.zero_grad()
    adv_expl = sal.attribute(x_adv)
    # adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
    #                     sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
    expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=1, dtype=torch.float))
    # print(expl_loss)
    # print(l1_norm)
    # loss = expl_loss + args.smooth_loss_c * torch.mean(l1_norm)
    expl_loss.backward()
    optimizer.step()
    ##
    scheduler.step(iter_no)
    ##
    # x_adv.data = x_adv.data + (torch.rand_like(x_adv.data)-0.5)*1e-12
    ##
    # project delta
    x_adv.data = examples.data + project_L0_box_torch((x_adv-examples).data.unsqueeze(dim=2).unsqueeze(dim=3),
        args.max_num_pixels, -examples.data.unsqueeze(dim=2).unsqueeze(dim=3),
        (1.0-examples.data).unsqueeze(dim=2).unsqueeze(dim=3)).squeeze(dim=3).squeeze(dim=2).data
    ###
    print(torch.where(torch.abs(x_adv-examples)[5]>1e-10))
    ###
    ## return to valid value range
    # x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
    topk_ints = []
    for i in range(mask.size()[0]):
        _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
        _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
        topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                            topk_adv_ind.cpu().detach().numpy())))/args.topk)
    if args.verbose:
        print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))
# after finishing the iterations
# adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
#                     sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
adv_expl = sal.attribute(x_adv)
topk_ints = []
for i in range(mask.size()[0]):
    _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
    _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
    topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                        topk_adv_ind.cpu().detach().numpy())))/args.topk)
############################
print(torch.max(x_adv))
print("mean top-k intersection: ", np.mean(topk_ints))
n_pixels = torch.sum(torch.abs(x_adv-examples) > 1e-10, dim=1)
print("total pixels changed: ", n_pixels)
torch.save(x_adv, f"{args.output_dir}x_{args.method}.pth")
