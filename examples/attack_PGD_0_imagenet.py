from collections import OrderedDict
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
from vgg_model_imagenet import VGG16
from vgg_model_imagenet_softplus import VGG16_softplus

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
argparser.add_argument('--lr', type=float, default=0.2, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_topk_imagenet/pgd0/',
                        help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad', 'deep_lift'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=1000)
argparser.add_argument('--max_num_pixels', type=int, default=1000)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not",
                        default=False)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
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
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
normalizer = utils.DifferentiableNormalize(mean=data_mean, std=data_std)
trasform_imagenet = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                        torchvision.transforms.CenterCrop(224),
                                        torchvision.transforms.ToTensor()])

imagenet_val = torchvision.datasets.ImageNet(root="../../Robust_Explanations/notebooks/data/", split="val",
                                            transform=trasform_imagenet)

test_loader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=512,
        shuffle=True
    )
BATCH_SIZE = 16
indices = np.random.randint(512, size=BATCH_SIZE)
#### load images #####
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices].to(device)
labels = labels_batch[indices].to(device)
###
model = torchvision.models.vgg16(pretrained=True)
s_dict = model.state_dict()
if args.method == "deep_lift":
    model = VGG16_softplus()
    change_key = {k:v for k,v in zip(s_dict.keys(), model.state_dict().keys())}
    new_s_dict = OrderedDict([(change_key[k], v) for k,v in s_dict.items()])
    model.load_state_dict(new_s_dict)
# we need to substitute the ReLus with softplus to avoid zero second derivative
if args.method != "deep_lift":
    model = convert_relu_to_softplus(model, beta=10)
model = model.eval().to(device)
###
model_relu = torchvision.models.vgg16(pretrained=True)
if args.method == "deep_lift":
    model_relu = VGG16()
    model_relu.load_state_dict(new_s_dict)
model_relu = model_relu.eval().to(device)
###
# keep only data points for which the model predicts correctly
with torch.no_grad():
    preds = model_relu(normalizer.forward(examples)).argmax(dim=1).detach()
    samples_to_pick = (preds==labels)
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
##### for Integreated Grad
if args.method == "integrated_grad":
    examples = examples[1:2]
    labels = labels[1:2]
####
examples = examples.requires_grad_(True)
###
print(examples.size())
BATCH_SIZE = examples.size()[0]
####
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(normalizer.forward(examples)[i]) -
        torch.min(normalizer.forward(examples)[i])).item() * 0.2 for i in range(examples.size()[0]))
if len(sigma)==1:
    sigma=sigma[0]

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
    mask = mask.unsqueeze(dim=0)
else:
    mask = torch.zeros_like(org_expl).view(BATCH_SIZE, -1)
    topk_perbatch = torch.argsort(org_expl.view(BATCH_SIZE, -1))[:, -args.topk:]
    for _ in range(mask.size()[0]):
        mask[_][topk_perbatch[_]] = 1
    mask = mask.view(org_expl.size())

x_adv = copy.deepcopy(examples)
optimizer = optim.Adam([x_adv], lr=args.lr)

for iter_no in range(args.num_iter):
    optimizer.zero_grad()
    adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
                        sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
    if BATCH_SIZE == 1:
        adv_expl = adv_expl.unsqueeze(dim=0)
    expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
    loss = expl_loss + 1e5 * F.mse_loss(F.softmax(model(normalizer.forward(x_adv)), dim=1), org_logits)
    loss.backward()
    # normalize gradient
    x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1,2,3), keepdim=True))
    ###
    optimizer.step()
    ##
    # project delta
    x_adv.data = examples.data + project_L0_box_torch((x_adv-examples).data.permute(0, 2, 3, 1),
        args.max_num_pixels, -examples.data.permute(0, 2, 3, 1),
        (1.0-examples.data).permute(0, 2, 3, 1)).permute(0, 3, 1, 2).data
    ###
    topk_ints = []
    for i in range(mask.size()[0]):
        _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
        _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
        topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                            topk_adv_ind.cpu().detach().numpy())))/args.topk)
    if args.verbose:
        print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))
# after finishing the iterations, compute the org and adv explanation for the
# ReLu model
###
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
print(torch.max(x_adv))
print("all top-k intersection: ", topk_ints)
print("mean top-k intersection: ", np.mean(topk_ints))
n_pixels = np.sum(np.amax(np.abs((x_adv-examples).cpu().detach().numpy()) > 1e-10, axis=1), axis=(1,2))
print("total pixels changed: ", n_pixels)
torch.save(x_adv, f"{args.output_dir}x_{args.method}.pth")
print("avg cosd from org expl", np.mean([spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])]))
