import argparse
import os
from collections import OrderedDict

import numpy as np
import scipy.spatial as spatial
import torch
import torchvision

from models.vgg_imagenet import VGG16, VGG16_softplus
from sparse_expl_attacks import image_datasets_dir, output_dir
from sparse_expl_attacks.sparse_attack import SparseAttack
from sparse_expl_attacks.utils import (
    DifferentiableNormalize,
    convert_relu_to_softplus,
    get_expl,
    topk_intersect,
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_iter", type=int, default=1000, help="number of iterations")
argparser.add_argument("--lr", type=float, default=0.2, help="learning rate")
argparser.add_argument(
    "--attack_type",
    type=str,
    default="greedy",
    help="type of the sparse explanation attack",
)
argparser.add_argument(
    "--expl_method",
    help="explanation method",
    choices=[
        "saliency",
        "lrp",
        "guided_backprop",
        "integrated_grad",
        "input_times_grad",
        "smooth_grad",
        "deep_lift",
    ],
    default="saliency",
)
argparser.add_argument("--topk", type=int, default=1000)
argparser.add_argument("--max_num_features", type=int, default=1000)
argparser.add_argument("--gamma", type=float, default=1e5)
argparser.add_argument(
    "--perturbation_per_iter",
    type=int,
    default=1,
    help="number of features to perturb in the greedy iterations.",
)
argparser.add_argument("--verbose", type=bool, help="verbose", default=False)
argparser.add_argument(
    "--add_to_seed",
    default=0,
    type=int,
    help="Update the seed (for running several experiments)",
)
args = argparser.parse_args()

SEED = 72
# Set the seed.
np.random.seed(SEED + args.add_to_seed)
torch.manual_seed(SEED + args.add_to_seed)
torch.cuda.manual_seed(SEED + args.add_to_seed)
# Set the device (cpu or gpu).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Defining the normalizer for Imagenet images.
data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
normalizer = DifferentiableNormalize(mean=data_mean, std=data_std)
# Transform function to transform Imagnet images to the proper size.
trasform_imagenet = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]
)
# Loading the Imagenet validation dataset.
imagenet_val = torchvision.datasets.ImageNet(
    root=os.path.join(image_datasets_dir, "imagenet"), split="val", transform=trasform_imagenet
)

test_loader = torch.utils.data.DataLoader(imagenet_val, batch_size=512, shuffle=True)
BATCH_SIZE = 16
indices = np.random.randint(512, size=BATCH_SIZE)
# Load images.
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices].to(device)
labels = labels_batch[indices].to(device)
# Softplus model
model = torchvision.models.vgg16(pretrained=True)
s_dict = model.state_dict()
if args.expl_method == "deep_lift":
    model = VGG16_softplus()
    change_key = {k: v for k, v in zip(s_dict.keys(), model.state_dict().keys())}
    new_s_dict = OrderedDict([(change_key[k], v) for k, v in s_dict.items()])
    model.load_state_dict(new_s_dict)
else:
    model = convert_relu_to_softplus(model, beta=10)
model = model.eval().to(device)
# ReLu model
model_relu = torchvision.models.vgg16(pretrained=True)
if args.expl_method == "deep_lift":
    model_relu = VGG16()
    model_relu.load_state_dict(new_s_dict)
model_relu = model_relu.eval().to(device)
# Keep only instances for which the model prediction is correct.
with torch.no_grad():
    preds = model_relu(normalizer.forward(examples)).argmax(dim=1).detach()
    samples_to_pick = preds == labels
    examples = examples[samples_to_pick]
    labels = labels[samples_to_pick]
# For integrated gradient we have to pick only one sample
# to avoid GPU memory issue.
if args.expl_method == "integrated_grad":
    ind = np.random.choice(len(examples))
    examples = examples[ind : ind + 1]  # noqa E203
    labels = labels[ind : ind + 1]  # noqa E203
BATCH_SIZE = examples.size()[0]
print(f"number of samples: {examples.size()[0]}")
# Compute sigma for the smooth gradient method.
if args.expl_method == "smooth_grad":
    sigma = tuple(
        (torch.max(examples[i]) - torch.min(examples[i])).item() * 0.2
        for i in range(examples.size()[0])
    )
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
    gamma=args.gamma,
    normalizer=normalizer,
)
x_adv = sparse_attack.attack(
    attack_type=args.attack_type,
    x_input=examples,
    y_input=labels,
    sigma=sigma,
    perturbation_per_iter=args.perturbation_per_iter,
)
# after finishing the iterations, compute the org and adversarial explanation
# for the ReLU model
org_expl = get_expl(
    model=model_relu,
    x=normalizer.forward(examples),
    expl_method=args.expl_method,
    device=device,
    true_label=labels,
    sigma=sigma,
    normalize=True,
)
adv_expl = get_expl(
    model=model_relu,
    x=normalizer.forward(x_adv),
    expl_method=args.expl_method,
    device=device,
    true_label=labels,
    sigma=sigma,
    normalize=True,
)
topk_ints = []
for i in range(examples.size()[0]):
    topk_ints.append(topk_intersect(org_expl[i], adv_expl[i], args.topk))

preds_org = model_relu(normalizer.forward(examples)).argmax(dim=1)
print("org acc: ", (labels == preds_org).sum() / BATCH_SIZE)
preds = model_relu(normalizer.forward(x_adv)).argmax(dim=1)
print("adv acc: ", (labels == preds).sum() / BATCH_SIZE)

print(torch.max(x_adv))
print("mean top-k intersection: ", np.mean(topk_ints))
print("all top-k intersection: ", topk_ints)
n_pixels = np.sum(
    np.amax(np.abs((x_adv - examples).cpu().detach().numpy()) > 1e-10, axis=1),
    axis=(1, 2),
)
print("total pixels changed: ", n_pixels)
print("avg pixels changed: ", np.mean(n_pixels))
save_dir = os.path.join(output_dir, "imagenet", args.attack_type)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(x_adv, os.path.join(save_dir, f"x_{args.expl_method}.pth"))
print(
    "avg cosd from org expl",
    np.mean(
        [
            spatial.distance.cosine(
                adv_expl[i].detach().cpu().flatten(),
                org_expl[i].detach().cpu().flatten(),
            )
            for i in range(adv_expl.size()[0])
        ]
    ),
)
