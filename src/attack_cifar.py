import argparse
import os
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
from scipy.stats import spearmanr as spr
import scipy.spatial as spatial

from sparse_expl_attacks.sparse_attack import SparseAttack
from sparse_expl_attacks.utils import DifferentiableNormalize, get_expl, topk_intersect
from sparse_expl_attacks import image_datasets_dir, output_dir, models_weights_dir
from models.resnet_softplus_10 import ResNet18
from models.resnet import ResNet18 as ResNet18_ReLu

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
argparser.add_argument('--lr', type=float, default=0.1, help='learning rate')
argparser.add_argument('--attack_type', type=str, default='greedy',
                        help='type of the sparse explanation attack')
argparser.add_argument('--expl_method', help='explanation method',
                       choices=["saliency", "lrp", "guided_backprop", "integrated_grad", 
                                "input_times_grad", "smooth_grad", "deep_lift"],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=20)
argparser.add_argument('--max_num_features', type=int, default=20)
argparser.add_argument('--verbose',type=bool, help="verbose", default=False)
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="Update the seed (for running several experiments)")
args = argparser.parse_args()

SEED = 72
# Set the seed.
np.random.seed(SEED+args.add_to_seed)
torch.manual_seed(SEED+args.add_to_seed)
torch.cuda.manual_seed(SEED+args.add_to_seed)
# Set the device (cpu or gpu).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Defining the normalizer for CIFAR-10 images.
data_mean = np.array([0.4914, 0.4822, 0.4465])
data_std = np.array([0.2023, 0.1994, 0.2010])
normalizer = DifferentiableNormalize(mean=data_mean, std=data_std)
# Loading the CIFAR-10 test dataset.
cifar_test = torchvision.datasets.CIFAR10(root=os.path.join(image_datasets_dir, "cifar-10"), 
                                          train=False, download=True,
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
# Softplus model
model = ResNet18()
model.load_state_dict(torch.load(os.path.join(models_weights_dir, "RN18_standard.pth"))["net"])
model = model.eval().to(device)
# ReLu model
model_relu = ResNet18_ReLu()
model_relu.load_state_dict(torch.load(os.path.join(models_weights_dir, "RN18_standard.pth"))["net"])
model_relu = model_relu.eval().to(device)
# Keep only instances for which the model prediction is correct.
with torch.no_grad():
    preds = model_relu(normalizer.forward(examples)).argmax(dim=1).detach()
    samples_to_pick = (preds==labels)
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
# Compute sigma for the smooth gradient method.
if args.expl_method == "smooth_grad":
    sigma = tuple((torch.max(examples[i]) -
        torch.min(examples[i])).item() * 0.1 for i in range(examples.size()[0]))
else:
    sigma = None
# Perform the attack.
sparse_attack = SparseAttack(
    model=model,
    model_relu=model_relu,
    expl_method=args.expl_method,
    num_iter=args.num_iter,
    lr=args.lr,
    topk=args.topk,
    max_num_features=args.max_num_features, 
    normalizer=normalizer
)
x_adv = sparse_attack.attack(
    attack_type=args.attack_type,
    x_input=examples,
    y_input=labels,
    sigma=sigma
)
# after finishing the iterations, compute the org and adversarial explanation 
# for the ReLU model
org_expl = get_expl(model=model_relu, 
                    x=normalizer.forward(examples),
                    expl_method=args.expl_method,
                    device=device,
                    true_label=labels,
                    sigma=sigma,
                    normalize=True)
adv_expl = get_expl(model=model_relu, 
                    x=normalizer.forward(x_adv),
                    expl_method=args.expl_method,
                    device=device,
                    true_label=labels,
                    sigma=sigma,
                    normalize=True)
topk_ints = []
for i in range(examples.size()[0]):
    topk_ints.append(topk_intersect(org_expl[i], adv_expl[i], args.topk))
    
preds_org = model_relu(normalizer.forward(examples)).argmax(dim=1)
print("org acc: ", (labels==preds_org).sum()/BATCH_SIZE)
preds = model_relu(normalizer.forward(x_adv)).argmax(dim=1)
print(" acc: ", (labels==preds).sum()/BATCH_SIZE)

print(torch.max(x_adv))
print("mean top-k intersection: ", np.mean(topk_ints))
print("all top-k intersection: ", topk_ints)
n_pixels = np.sum(np.amax(np.abs((x_adv-examples).cpu().detach().numpy()) > 1e-10, axis=1), axis=(1,2))
print("total pixels changed: ", n_pixels)
print("avg pixels changed: ", np.mean(n_pixels))
save_dir = os.path.join(output_dir, "cifar-10", args.attack_type)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
torch.save(x_adv, os.path.join(save_dir, f"x_{args.expl_method}.pth"))
print("avg cosd from org expl", np.mean([spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])]))