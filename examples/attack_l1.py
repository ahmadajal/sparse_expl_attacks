import numpy as np
import re
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

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
argparser.add_argument('--lr', type=float, default=0.0005, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_expl_topk_relu_l1/',
                        help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=1000)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not",
                        default=False)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--additive_lp_bound', type=float, default=0.03, help='l_p bound for additive attack')
argparser.add_argument('--smooth_loss_c', type=float, default=0.0001,
                    help='coefficient of the smooth loss term')
args = argparser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(72)
torch.manual_seed(72)
torch.cuda.manual_seed(72)
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
BATCH_SIZE = 4
indices = np.random.randint(512, size=BATCH_SIZE)
#### load images #####
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices].to(device)
labels = labels_batch[indices].to(device)
###
examples = examples.requires_grad_(True)
###
model = torchvision.models.vgg16(pretrained=True)
# we need to substitute the ReLus with softplus to avoid zero second derivative
model = convert_relu_to_softplus(model, beta=100)
model = model.eval().to(device)
####
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(examples[i]) -
        torch.min(examples[i])).item() * 0.2 for i in range(examples.size()[0]))
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

delta = nn.Parameter(torch.zeros_like(examples))
x_adv = examples + delta
optimizer = optim.Adam([delta], lr=args.lr)
######
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.num_iter//2], gamma=0.4)
######

for iter_no in range(args.num_iter):
    optimizer.zero_grad()
    adv_expl = get_expl(model, normalizer.forward(x_adv), args.method, desired_index=labels, smooth=args.smooth,
                        sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
    expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
    l1_norm = torch.norm(delta, p=1, dim=(1,2,3))
    # print(expl_loss)
    # print(l1_norm)
    loss = expl_loss + args.smooth_loss_c * torch.mean(l1_norm)
    loss.backward()
    optimizer.step()
    ##
    scheduler.step(iter_no)
    ##
    x_adv = examples + delta
    topk_ints = []
    for i in range(mask.size()[0]):
        _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
        _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
        topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                            topk_adv_ind.cpu().detach().numpy())))/args.topk)
    if args.verbose:
        print("{}: expl loss: {}".format(iter_no, topk_ints))
