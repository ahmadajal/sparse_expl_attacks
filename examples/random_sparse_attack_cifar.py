# EXTERNAL LIBRARIES
import numpy as np
import re
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import sys
import itertools
# for distances between expls
from scipy.stats import spearmanr as spr
import scipy.spatial as spatial
sys.path.append("../attacks/")
# mister_ed
import mister_ed.loss_functions as lf
import mister_ed.utils.pytorch_utils as utils
import mister_ed.utils.image_utils as img_utils
import mister_ed.cifar10.cifar_loader as cifar_loader
import mister_ed.cifar10.cifar_resnets as cifar_resnets
import mister_ed.adversarial_training as advtrain
import mister_ed.utils.checkpoints as checkpoints
import mister_ed.adversarial_perturbations as ap
import mister_ed.adversarial_attacks as aa
import mister_ed.spatial_transformers as st
import mister_ed.config as config
# ReColorAdv
import recoloradv.perturbations as pt
import recoloradv.color_transformers as ct
import recoloradv.color_spaces as cs
from recoloradv import norms
from utils import load_image, torch_to_image, get_expl, convert_relu_to_softplus, plot_overview, UniGrad
# explanation
import sys
# sys.path.append("../Spatial_transform/ST_ADV_exp_imagenet/")
sys.path.append("../../PerceptualSimilarity/") # for LPIPS similarity
sys.path.append("../../Perc-Adversarial/") # for perceptual color distance regulariation - https://github.com/ZhengyuZhao/PerC-Adversarial
import lpips
from differential_color_functions import rgb2lab_diff, ciede2000_diff
sys.path.append("../../pytorch-cifar/models/")
from resnet import ResNet18, ResNet50
data_mean=np.array([0.4914, 0.4822, 0.4465])
data_std=np.array([0.2023, 0.1994, 0.2010])
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--output_dir', type=str, default='output_expl_topk_relu_random_pert_cifar/', help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=20)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--indices', nargs=2, default=[0, 1024], type=int,
                        help="to only take part of the batch (to avoide memory blow-up)")
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
args = argparser.parse_args()
from PIL import Image
# set seed to get the same images every time
np.random.seed(72+args.add_to_seed)
torch.manual_seed(72+args.add_to_seed)
torch.cuda.manual_seed(72+args.add_to_seed)
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
if len(images_batch.size()) == 3:
    examples = images_batch[indices].unsqueeze(dim=0)
else:
    examples = images_batch[indices]
labels = labels_batch[indices]
######################
model = ResNet18()
model.load_state_dict(torch.load("../../Robust_Explanations/notebooks/models/RN18_standard.pth")["net"])
model = model.eval()
####
normalizer = utils.DifferentiableNormalize(mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010])

if utils.use_gpu():
    examples = examples.cuda()
    labels = labels.cuda()
    model.cuda()
####
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(normalizer.forward(examples)[i]) -
        torch.min(normalizer.forward(examples)[i])).item() * 0.1 for i in range(examples.size()[0]))
if len(sigma)==1:
    sigma=sigma[0]

org_expl = get_expl(model, normalizer.forward(examples), args.method, desired_index=labels,
                       smooth=args.smooth, sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
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

inds = []
for i in range(32):
    for j in range(32):
        inds.append([i,j])
inds = np.array(inds)
all_corners = [list(i) for i in itertools.product([0, 1], repeat=3)]
x_adv = copy.deepcopy(examples)
finished = np.array([0]*BATCH_SIZE)
no_print = np.array([0]*BATCH_SIZE)
for num_tries in range(200):
    unfinished_batches = np.where(finished==0)[0]
    to_pick = np.random.choice(range(len(inds)), size=BATCH_SIZE)
    pixels = inds[to_pick]
    for b in unfinished_batches:
        min_topk = 1.0
        for ind, c in enumerate(all_corners):
            x_adv[b, :, pixels[b, 0], pixels[b, 1]] = torch.tensor(c).cuda()
            adv_expl_batch = get_expl(model, normalizer.forward(x_adv[b].unsqueeze(dim=0)), args.method,
                desired_index=labels[b], smooth=args.smooth, sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
            _, topk_mask_ind = torch.topk(mask[b].flatten(), k=args.topk)
            _, topk_adv_ind = torch.topk(adv_expl_batch.flatten(), k=args.topk)
            topk_c = float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                topk_adv_ind.cpu().detach().numpy())))/args.topk
            if topk_c <= min_topk:
                min_topk = topk_c
                best_ind = ind
        x_adv[b, :, pixels[b, 0], pixels[b, 1]] = torch.tensor(all_corners[best_ind]).cuda()

    adv_expl = get_expl(model, normalizer.forward(x_adv), args.method,
            desired_index=labels, smooth=args.smooth, sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
    if BATCH_SIZE==1:
        _, topk_mask_ind = torch.topk(mask.flatten(), k=args.topk)
        _, topk_adv_ind = torch.topk(adv_expl.flatten(), k=args.topk)
        topk_ints = float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                            topk_adv_ind.cpu().detach().numpy())))/args.topk
        if topk_ints == 1:
            break
    else:
        topk_ints = []
        for i in range(mask.size()[0]):
            _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
            _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
            topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                topk_adv_ind.cpu().detach().numpy())))/args.topk)
        finished[np.where(np.array(topk_ints)==0.0)[0]]=1
    # for printing the resutls
    if sum(finished - no_print) > 0:
        just_finished = np.where((finished-no_print)==1)[0]
        print("batch {} finished".format(just_finished))
        print(num_tries)
        no_print[just_finished]=1
    ###
    if sum(finished) == BATCH_SIZE:
        break

print("Final topk intersection: ", topk_ints)
print("Final cosd from org expl", [spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])])
loss_lpips = lpips.LPIPS(net='vgg')
print("LPIPS: ", [loss_lpips.forward(examples.cpu()[i:i+1],
                x_adv.cpu()[i:i+1]).item() for i in range(BATCH_SIZE)])
torch.save(x_adv, f"{args.output_dir}x_{args.method}.pth")
