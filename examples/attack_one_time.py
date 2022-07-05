import numpy as np
import copy
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
from resnet_softplus_10 import ResNet18, ResNet50
from resnet import ResNet18 as ResNet18_ReLu

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--lr', type=float, default=0.05, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_topk_cifar-10/onetime/',
                        help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad', 'deep_lift'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=20)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not",
                        default=False)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
args = argparser.parse_args()

def topk_coord(array, used_inds, k=1):
    array_norm = torch.norm(array, p=1, dim=1)
    if len(used_inds)>0:
        array_norm[np.array(used_inds).T.tolist()] = 0.0
    inds = torch.topk(array_norm.view((array.size()[0],-1)), k=k, dim=1)[1]
    new_inds = [np.unravel_index(i, shape=(array.size()[2], array.size()[3])) for i in inds.cpu().numpy()]
    for k_ind in range(k):
        used_inds = used_inds + [[i, j[0][k_ind], j[1][k_ind]] for i,j in enumerate(new_inds)]
    ###
    new_inds0 = np.array([[i, 0, j[0][0], j[1][0]] for i,j in enumerate(new_inds)])
    new_inds1 = np.array([[i, 1, j[0][0], j[1][0]] for i,j in enumerate(new_inds)])
    new_inds2 = np.array([[i, 2, j[0][0], j[1][0]] for i,j in enumerate(new_inds)])
    for k_ind in range(1,k):
        new_inds0 = np.concatenate((new_inds0, np.array([[i, 0, j[0][k_ind], j[1][k_ind]] for i,j in enumerate(new_inds)])))
        new_inds1 = np.concatenate((new_inds1, np.array([[i, 1, j[0][k_ind], j[1][k_ind]] for i,j in enumerate(new_inds)])))
        new_inds2 = np.concatenate((new_inds2, np.array([[i, 2, j[0][k_ind], j[1][k_ind]] for i,j in enumerate(new_inds)])))
    ###
    new_inds0 = new_inds0.T.tolist()
    new_inds1 = new_inds1.T.tolist()
    new_inds2 = new_inds2.T.tolist()
    ###
    new_array = torch.zeros_like(array)
    new_array[new_inds0] = array[new_inds0]
    new_array[new_inds1] = array[new_inds1]
    new_array[new_inds2] = array[new_inds2]
    ###
    return new_array, used_inds

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
model = ResNet18()
model.load_state_dict(torch.load("../../Robust_Explanations/notebooks/models/RN18_standard.pth")["net"])
model = model.eval().to(device)
####
model_relu = ResNet18_ReLu()
model_relu.load_state_dict(torch.load("../../Robust_Explanations/notebooks/models/RN18_standard.pth")["net"])
model_relu = model_relu.eval().to(device)
####
# keep only data points for which the model predicts correctly
with torch.no_grad():
    preds = model_relu(normalizer.forward(examples)).argmax(dim=1).detach()
    samples_to_pick = (preds==labels)
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
####
examples = examples.requires_grad_(True)
###
print(examples.size())
BATCH_SIZE = examples.size()[0]
####
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(examples[i]) -
        torch.min(examples[i])).item() * 0.1 for i in range(examples.size()[0]))
if len(sigma)==1:
    sigma=sigma[0]
####
# creating the mask for top-k attack
org_expl = get_expl(model, normalizer.forward(examples), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
org_expl = org_expl.detach()
org_logits = F.softmax(model(normalizer.forward(examples)), dim=1)
org_logits = org_logits.detach()
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
###
x_adv = copy.deepcopy(examples)
optimizer = optim.Adam([x_adv], lr=args.lr)
######
optimizer.zero_grad()
adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
###
preds_adv = model(normalizer.forward(x_adv)).argmax(dim=1).detach().cpu()
loss = expl_loss - torch.mean(model(normalizer.forward(x_adv))[torch.arange(BATCH_SIZE), preds_adv], dtype=torch.float)
# loss = expl_loss + 1e1 * F.mse_loss(F.softmax(model(normalizer.forward(x_adv)), dim=1), org_logits)
loss.backward()
# normalize gradient
x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1,2,3), keepdim=True))
# pick the top k coordinates of the update
x_adv.grad, used_indices = topk_coord(x_adv.grad, [], k=args.topk)
# x_adv.grad = x_adv.grad * mask
###
optimizer.step()
# update step
x_adv.data = examples.data + torch.clip((x_adv-examples).data, -examples.data, (1.0-examples.data))

# after finishing the attack, compute the org and adv explanation for the
# ReLu model
org_expl = get_expl(model_relu, normalizer.forward(examples), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
adv_expl = get_expl(model_relu, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
                    sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
topk_ints = []
for i in range(mask.size()[0]):
    _, topk_mask_ind = torch.topk(org_expl[i].flatten(), k=args.topk)
    _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
    topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                        topk_adv_ind.cpu().detach().numpy())))/args.topk)
############################
preds_org = model_relu(normalizer.forward(examples)).argmax(dim=1)
print("org acc: ", (labels==preds_org).sum()/BATCH_SIZE)
preds = model_relu(normalizer.forward(x_adv)).argmax(dim=1)
print("adv acc: ", (labels==preds).sum()/BATCH_SIZE)
############################
print(torch.max(x_adv), torch.min(x_adv))
print("mean top-k intersection: ", np.mean(topk_ints))
print("all top-k intersection: ", topk_ints)
n_pixels = np.sum(np.amax(np.abs((x_adv-examples).cpu().detach().numpy()) > 1e-10, axis=1), axis=(1,2))
print("total pixels changed: ", n_pixels)
print("avg pixels changed: ", np.mean(n_pixels))
torch.save(x_adv, f"{args.output_dir}x_{args.method}.pth")
print("avg cosd from org expl", np.mean([spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])]))
