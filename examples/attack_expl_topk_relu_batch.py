# EXTERNAL LIBRARIES
import numpy as np
import re

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
sys.path.append("../PerceptualSimilarity/") # for LPIPS similarity
sys.path.append("../Perc-Adversarial/") # for perceptual color distance regulariation - https://github.com/ZhengyuZhao/PerC-Adversarial
import lpips
from differential_color_functions import rgb2lab_diff, ciede2000_diff

data_mean = np.array([0.485, 0.456, 0.406])
data_std = np.array([0.229, 0.224, 0.225])
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_iter', type=int, default=100, help='number of iterations')
argparser.add_argument('--img_idx', type=int, default=657)
argparser.add_argument('--target_img_idx', type=int, default=80)
argparser.add_argument('--lr', type=float, default=0.001, help='lr')
argparser.add_argument('--output_dir', type=str, default='output_expl_topk_relu/', help='directory to save results to')
argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'saliency', 'integrated_grad',
                            'input_times_grad', 'uniform_grad'],
                       default='saliency')
argparser.add_argument('--topk', type=int, default=1000)
argparser.add_argument('--smooth',type=bool, help="whether to use the smooth explanation or not", default=False)
argparser.add_argument('--multiply_by_input',type=bool,
                        help="whether to multiply the explanation by input or not", default=False)
argparser.add_argument('--additive_lp_bound', type=float, default=0.03, help='l_p bound for additive attack')
argparser.add_argument('--stadv_lp_bound', type=float, default=0.05, help='l_p bound for spatial transformation')
argparser.add_argument('--norm_weights', nargs=3, default=[1.0, 1.0, 1.0], type=float,
                        help='norm weights for combining smooth loss for each attack')
argparser.add_argument('--lp_reg', type=int, default=2, help='lp perturbation norm regularizer')
argparser.add_argument('--ciede2000_reg', type=float, default=0.0, help='ciede2000 distance regularizer coeff. if 0 then there will be no such regularization')
argparser.add_argument('--attack_type', nargs=3, default=[0, 0, 1], type=int,
                       help='type of the attack. only 0 and 1 values are accepted.\
                       the order is [ReColorAdv, STadv, additive]')
argparser.add_argument('--early_stop_for', type=str, default=None, help='eraly stop for part of the loss')
argparser.add_argument('--early_stop_value', type=float, default=10.0, help='stop the optimization if \
                        (part of) the loss dropped below a certain level.')
argparser.add_argument('--indices', nargs=2, default=[0, 1024], type=int,
                        help="to only take part of the batch (to avoide memory blow-up)")
argparser.add_argument('--add_to_seed', default=0, type=int,
                        help="to be added to the seed")
args = argparser.parse_args()
print(args.smooth)
a = np.array(args.attack_type)
if ((a != 1) & (a != 0)).any():
    raise ValueError("only 0 or 1 values are accepted for attack type combination")
from PIL import Image
# im = Image.open("../sample_imagenet/sample_0.jpg")
# examples = torchvision.transforms.ToTensor()(
#         torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im)))
# examples = examples.unsqueeze(0)
# labels = torch.tensor([17])
# set seed to get the same images every time
np.random.seed(72+args.add_to_seed)
torch.manual_seed(72+args.add_to_seed)
torch.cuda.manual_seed(72+args.add_to_seed)
trasform_imagenet = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                        torchvision.transforms.CenterCrop(224),
                                        torchvision.transforms.ToTensor()])

imagenet_val = torchvision.datasets.ImageNet(root="../../Robust_Explanations/notebooks/data/", split="val", transform=trasform_imagenet)

test_loader = torch.utils.data.DataLoader(
        imagenet_val,
        batch_size=1024,
        shuffle=True
    )
BATCH_SIZE = 4
indices = np.random.randint(1024, size=BATCH_SIZE)
#### load images #####
dataiter = iter(test_loader)
images_batch, labels_batch = next(dataiter)
examples = images_batch[indices][args.indices[0]: args.indices[1]]
labels = labels_batch[indices][args.indices[0]: args.indices[1]]
######################
model = torchvision.models.vgg16(pretrained=True)
# model = torchvision.models.resnet18(pretrained=True)
# we need to substitute the ReLus with softplus to avoid zero second derivative
model = convert_relu_to_softplus(model, beta=100)
####
model = model.eval()
####
normalizer = utils.DifferentiableNormalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

if utils.use_gpu():
    examples = examples.cuda()
    labels = labels.cuda()
    model.cuda()
###
examples = examples.requires_grad_(True)
###
######
print(torch.cuda.memory_allocated())
######
# sigma for the smooth grad and uniform grad methods
sigma = tuple((torch.max(normalizer.forward(examples)[i]) -
        torch.min(normalizer.forward(examples)[i])).item() * 0.2 for i in range(examples.size()[0]))
if len(sigma)==1:
    sigma=sigma[0]
print("sigma: ", sigma)
## expl loss
class EXPL_Loss_topk(lf.PartialLoss):
    def __init__(self, classifier, mask, topk, method, normalizer=None, smooth=False, multiply_by_input=False):
        super(EXPL_Loss_topk, self).__init__()
        self.classifier = classifier
        self.mask = mask
        self.topk = topk
        self.method = method
        self.normalizer = normalizer
        self.smooth = smooth
        self.multiply_by_input = multiply_by_input
        self.nets.append(self.classifier)


    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get adversarial expl
        adv_expl = get_expl(self.classifier, classifier_in, self.method, desired_index=labels,
                            smooth=self.smooth, sigma=sigma, normalize=True, multiply_by_input=self.multiply_by_input)
        ####temp#####
        # adv_expl, all_adv_expl = UniGrad(self.classifier, classifier_in, target=labels,
        #                                 num_steps=5, sigma=sigma, return_noisy_expls=True)
        # loss_expl = 0
        # for expl in all_adv_expl:
        #     expl = torch.sum(expl.squeeze(), 0, True)
        #     expl = expl / torch.sum(expl)
        #     loss_expl = loss_expl + F.mse_loss(expl, self.target_expl)
        # loss_expl = loss_expl / len(all_adv_expl)
        #############
        if BATCH_SIZE==1:
            loss_expl = torch.sum(adv_expl*self.mask)
            _, topk_mask_ind = torch.topk(self.mask.flatten(), k=self.topk)
            _, topk_adv_ind = torch.topk(adv_expl.flatten(), k=self.topk)
            topk_ints = float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                topk_adv_ind.cpu().detach().numpy())))/self.topk
            print("expl loss:", topk_ints)
        else:
            loss_expl = torch.mean(torch.sum(adv_expl*self.mask, dim=(1,2,3), dtype=torch.float))
            topk_ints = []
            for i in range(self.mask.size()[0]):
                _, topk_mask_ind = torch.topk(self.mask[i].flatten(), k=self.topk)
                _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=self.topk)
                topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                    topk_adv_ind.cpu().detach().numpy())))/self.topk)
            print("expl loss:", topk_ints)
        return loss_expl

## output loss
class OUTPUT_Loss_mse(lf.PartialLoss):
    def __init__(self, classifier, org_logits, normalizer=None):
        super(OUTPUT_Loss_mse, self).__init__()
        self.classifier = classifier
        self.org_logits = org_logits
        self.normalizer = normalizer
        self.nets.append(self.classifier)


    def forward(self, examples, labels, *args, **kwargs):
        classifier_in = self.normalizer.forward(examples)
        classifier_out = self.classifier.forward(classifier_in)

        # get target expl
        loss_output = F.mse_loss(F.softmax(self.classifier(classifier_in), dim=1), self.org_logits)
        print("output loss:", loss_output.item())
        return loss_output

# target expl
# im = Image.open("../sample_imagenet/sample_0_target.jpg")
# target_examples = torchvision.transforms.ToTensor()(
#         torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im)))
# target_examples = target_examples.unsqueeze(0)
# if utils.use_gpu():
#     target_examples = target_examples.cuda()
# target_label = model(normalizer.forward(target_examples)).argmax()
# target explanation map from the same method being attacked
# target_expl = get_expl(model, normalizer.forward(target_examples), args.method, desired_index=target_labels,
#                        smooth=args.smooth, sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
# target_expl = target_expl.detach()
# # original logits
# org_logits = F.softmax(model(normalizer.forward(examples)), dim=1)
# org_logits = org_logits.detach()

# creating the mask for top-k attack
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
print(mask.size())

# This threat model defines the regularization parameters of the attack.
recoloradv_threat = ap.ThreatModel(pt.ReColorAdv, {
    'xform_class': ct.FullSpatial,
    'cspace': cs.CIELUVColorSpace(), # controls the color space used
    'lp_style': 'inf',
    'lp_bound': [0.06, 0.06, 0.06],  # [epsilon_1, epsilon_2, epsilon_3]
    'xform_params': {
      'resolution_x': 16,            # R_1
      'resolution_y': 32,            # R_2
      'resolution_z': 32,            # R_3
    },
    'use_smooth_loss': True,
})

# we define the additive threat model.
additive_threat = ap.ThreatModel(ap.DeltaAddition, {
   'lp_style': 1, #'inf',
   'lp_bound': args.additive_lp_bound,
})

# spatial transformation attack
stadv_threat = ap.ThreatModel(
                ap.ParameterizedXformAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=args.stadv_lp_bound,
                    xform_class=st.FullSpatial,
                    use_stadv=True,
                ))

all_threats = [recoloradv_threat, stadv_threat, additive_threat]
threat_comb = []
norm_weights = []
for ind, threat in enumerate(all_threats):
    if args.attack_type[ind] == 1:
        threat_comb.append(all_threats[ind])
        norm_weights.append(args.norm_weights[ind])
print("attack: ", threat_comb)
# Combine all the threat models.
combined_threat = ap.ThreatModel(
    ap.SequentialPerturbation,
    threat_comb,
    ap.PerturbationParameters(norm_weights=norm_weights),
)

# Again, define the optimization terms.
# explanation loss
expl_loss = EXPL_Loss_topk(model, mask, args.topk, args.method, normalizer, smooth=args.smooth, multiply_by_input=args.multiply_by_input)

# # output loss
# output_loss = OUTPUT_Loss_mse(model, org_logits, normalizer)
#
# # temp: LPIPS loss to ensure visual similarity
# lpips_loss = lf.LpipsRegularization(examples)
#
# # temp: ciede2000 loss regularization to penalize perceptual color distance
# ciede2000_loss = lf.ciede2000Regularization(examples)

# adv_loss = lf.CWLossF6(model, normalizer)
smooth_loss = lf.PerturbationNormLoss(lp=args.lp_reg)
attack_loss = lf.RegularizedLoss({'expl': expl_loss, 'smooth': smooth_loss},
                                 {'expl': 1.0 , 'smooth': 10.0},  # lambda = 0.05
                                 negate=True) # Need this true for PGD type attacks
print("attack loss", attack_loss.losses)

# Setup and run PGD over both perturbations at once.
pgd_attack_obj = aa.PGD(model, normalizer, combined_threat, attack_loss)
perturbation = pgd_attack_obj.attack(examples, labels, num_iterations=args.num_iter, signed=False,
                                     optimizer=optim.Adam, optimizer_kwargs={'lr': args.lr}, step_size=args.lr, # for signed grad
                                     verbose=True, early_stop_for=args.early_stop_for,
                                     early_stop_value=args.early_stop_value)

# computing adv explanation again with normalize to print the final topk intersect
adv_expl = get_expl(model, normalizer.forward(perturbation.adversarial_tensors()), args.method,
                    desired_index=labels, smooth=args.smooth, sigma=sigma, normalize=True, multiply_by_input=args.multiply_by_input)
### release the memory ###
torch.cuda.empty_cache()
##########################
if BATCH_SIZE==1:
    _, topk_mask_ind = torch.topk(mask.flatten(), k=args.topk)
    _, topk_adv_ind = torch.topk(adv_expl.flatten(), k=args.topk)
    topk_ints = float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                        topk_adv_ind.cpu().detach().numpy())))/args.topk
else:
    topk_ints = []
    for i in range(mask.size()[0]):
        _, topk_mask_ind = torch.topk(mask[i].flatten(), k=args.topk)
        _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=args.topk)
        topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                            topk_adv_ind.cpu().detach().numpy())))/args.topk)
print("Final topk intersection: ", topk_ints)
# print("Final spr", spr(adv_expl.detach().cpu().flatten(), target_expl.detach().cpu().flatten()))
print("Final cosd from org expl", [spatial.distance.cosine(adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten()) for i in range(adv_expl.size()[0])])
# ########
# get the explanations again for the ReLU model
# first, delete the previous explanation tensors to release the memory!
del adv_expl
del org_expl
### release the memory ###
torch.cuda.empty_cache()
##########################
model = torchvision.models.vgg16(pretrained=True)
model = model.eval()
if utils.use_gpu():
    model.cuda()
# original explanation and logits - need not to be normalized as it is not part of the objective func
org_expl = get_expl(model, normalizer.forward(examples), args.method, desired_index=labels,
                    smooth=args.smooth, sigma=sigma, multiply_by_input=args.multiply_by_input).detach().cpu()
### release the memory ###
torch.cuda.empty_cache()
##########################
# adv explanation
adv_expl = get_expl(model, normalizer.forward(perturbation.adversarial_tensors()), args.method,
                    desired_index=labels, smooth=args.smooth, sigma=sigma, multiply_by_input=args.multiply_by_input).detach().cpu()
### release the memory ###
torch.cuda.empty_cache()
##########################
s = "_smooth" if args.smooth else ""
##########
if args.indices[1] - args.indices[0] < BATCH_SIZE:
    BATCH_SIZE = args.indices[1] - args.indices[0]
for i in range(BATCH_SIZE):
    plot_overview([normalizer.forward(examples[i:i+1]),
                   normalizer.forward(examples[i:i+1]),
                   normalizer.forward(perturbation.adversarial_tensors()[i:i+1])], \
                   [org_expl[i:i+1], org_expl[i:i+1], adv_expl[i:i+1]], \
                   data_mean, data_std, filename=f"{args.output_dir}overview_{args.method}{s}_{i}.png")
torch.save(perturbation.adversarial_tensors(), f"{args.output_dir}x_{args.method}{s}.pth") # this is the unnormalized tensor

print("attack type:", args.attack_type)
if sum(args.attack_type) <= 3:
    print(perturbation.layer_00.perturbation_norm(perturbation.adversarial_tensors(), 1))
if sum(args.attack_type) >= 2:
    print(perturbation.layer_01.perturbation_norm(perturbation.adversarial_tensors(), 2))
if sum(args.attack_type) == 3:
    print(perturbation.layer_02.perturbation_norm(perturbation.adversarial_tensors(), 2))
loss_lpips = lpips.LPIPS(net='vgg')
print("LPIPS: ", [loss_lpips.forward(examples.cpu()[i:i+1],
                perturbation.adversarial_tensors().cpu()[i:i+1]).item() for i in range(BATCH_SIZE)])
