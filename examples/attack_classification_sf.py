import numpy as np
import re
import copy
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import sys
# for distances between expls
from scipy.stats import spearmanr as spr
import scipy.spatial as spatial
sys.path.append("../attacks/")
import mister_ed.utils.pytorch_utils as utils
from utils import load_image, torch_to_image, get_expl, convert_relu_to_softplus, plot_overview, UniGrad
sys.path.append("../../pytorch-cifar/models/")
from resnet_softplus import ResNet18, ResNet50
sys.path.append("../SparseFool/")
import torchvision.transforms as transforms
import torchvision.models as torch_models
import os
from utils_sparsefool import nnz_pixels, inv_tf, inv_tf_pert, get_label
from sparsefool import sparsefool
from utils_sparsefool import valid_bounds

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=50, help='number of iterations')
argparser.add_argument('--lr', type=float, default=1.0, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_expl_topk_class_sf/',
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
args = argparser.parse_args()

class Modified_topk_loss:
    def __init__(self, model, mask, max_intersection, m2, labels, sigma):
        self.model = model
        self.mask = mask
        self.max_intersection = max_intersection
        self.m2 = m2
        self.labels = labels
        self.sigma = sigma

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        expl_x = get_expl(self.model, normalizer.forward(x), args.method, desired_index=self.labels, smooth=args.smooth,
                            sigma=self.sigma, normalize=True, multiply_by_input=args.multiply_by_input)
        expl_loss = torch.sum(expl_x*self.mask, dim=(1,2), dtype=torch.float)
        logits = torch.cat(((expl_loss/self.max_intersection)-(self.m2/self.max_intersection),
                            -1*(expl_loss/self.max_intersection)+(self.m2/self.max_intersection))).T
        if len(logits.size())==1:
            logits = logits.unsqueeze(0)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(72+args.add_to_seed)
torch.manual_seed(72+args.add_to_seed)
torch.cuda.manual_seed(72+args.add_to_seed)
data_mean = np.array([0.4914, 0.4822, 0.4465])
data_std = np.array([0.2023, 0.1994, 0.2010])
normalizer = utils.DifferentiableNormalize(mean=data_mean, std=data_std)
cifar_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(
        cifar_test,
        batch_size=128,
        shuffle=True
    )
BATCH_SIZE = 64
indices = np.random.randint(128, size=BATCH_SIZE)
#### load images #####
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices].to(device)
labels = labels_batch[indices].to(device)
###
examples = examples.requires_grad_(True)
###
model = ResNet18()
model.load_state_dict(torch.load("../../Robust_Explanations/notebooks/models/RN18_standard.pth")["net"])
model = model.eval().to(device)
####
normalizer = utils.DifferentiableNormalize(mean=data_mean,
                                               std=data_std)

####
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(examples[i]) -
        torch.min(examples[i])).item() * 0.1 for i in range(examples.size()[0]))
if len(sigma)==1:
    sigma=sigma[0]
# creating the mask for top-k attack
org_expl = get_expl(model, normalizer.forward(examples), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
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

topk_ints = []
num_pixels = []
adv = torch.zeros_like(examples)
for b in range(BATCH_SIZE):
    x_org = examples[b:b+1].detach()
    max_intersection = (torch.sum(org_expl[b]*mask[b], dim=(1,2), dtype=torch.float))
    ###
    # m2 = torch.topk(org_expl[b].flatten(), k=200)[0][-args.topk:].sum()
    m2 = max_intersection*0.2
    m_topk_loss = Modified_topk_loss(model, mask[b], max_intersection, m2, labels[b:b+1], sigma[b])
    # print(m_topk_loss(x_org))
    lb = torch.zeros_like(x_org)
    ub = torch.ones_like(x_org)
    ######################
    # Execute SparseFool #
    ######################
    # Params
    lambda_ = 3.

    x_adv, r, pred_label, fool_label, loops = sparsefool(x_org, m_topk_loss, lb, ub, lambda_, args.num_iter,
                                                     num_classes=2, device=device)
    adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels[b:b+1], smooth=args.smooth,
                        sigma=sigma[b], normalize=True, multiply_by_input=args.multiply_by_input)
    _, topk_mask_ind = torch.topk(mask[b].flatten(), k=args.topk)
    _, topk_adv_ind = torch.topk(adv_expl.flatten(), k=args.topk)
    topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                        topk_adv_ind.cpu().detach().numpy())))/args.topk)
    num_pixels.append(np.sum(np.amax(np.abs((x_adv-x_org).cpu().detach().numpy()) > 1e-10, axis=1), axis=(1,2))[0])
    adv[b:b+1] = copy.deepcopy(x_adv.data)

############################
preds_org = model(normalizer.forward(examples)).argmax(dim=1)
print("org acc: ", (labels==preds_org).sum()/BATCH_SIZE)
preds = model(normalizer.forward(adv)).argmax(dim=1)
print("adv acc: ", (labels==preds).sum()/BATCH_SIZE)
############################
print("topk_ints: ", topk_ints)
print("mean loss: ", np.mean(topk_ints))
print("num_pixels: ", num_pixels)
print("mean num pixels: ", np.mean(num_pixels))
torch.save(adv, f"{args.output_dir}x_{args.method}.pth")
adv_expl = get_expl(model, normalizer.forward(adv), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
print("avg cosd from org expl", np.mean([spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])]))
