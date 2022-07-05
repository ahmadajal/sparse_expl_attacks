import re
import numpy as np
import torch
from torch import nn
from torch import optim
import torchvision
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt

from mister_ed.cifar10 import cifar_resnets
from mister_ed.utils.pytorch_utils import DifferentiableNormalize
from mister_ed import config
from mister_ed import adversarial_perturbations as ap
from mister_ed import adversarial_attacks as aa
from mister_ed import spatial_transformers as st
from mister_ed import loss_functions as lf
from mister_ed import adversarial_training as advtrain

from recoloradv import perturbations as pt
from recoloradv import color_transformers as ct
from recoloradv import color_spaces as cs

# For generating the explanations we use the captum package (https://github.com/pytorch/captum)
from captum.attr import (
            Saliency,
            IntegratedGradients,
            LRP,
            GuidedBackprop,
            NoiseTunnel,
            GuidedGradCam,
            InputXGradient,
            DeepLift)
from captum.attr import visualization as viz

# dictionary to convert the method name to the proper class
expl_methods={
            "saliency": Saliency,
            "integrated_grad": IntegratedGradients,
            "lrp": LRP,
            "guided_backprop": GuidedBackprop,
            "guided_gradcam": GuidedGradCam,
            "input_times_grad": InputXGradient,
            "deep_lift": DeepLift
}

def load_pretrained_cifar10_model(
  path: str, resnet_size: int = 32,
) -> Tuple[nn.Module, DifferentiableNormalize]:
    """
    Loads a pretrained CIFAR-10 ResNet from the given path along with its
    associated normalizer.
    """

    model: nn.Module = getattr(cifar_resnets, f'resnet{resnet_size}')()
    model_state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict({re.sub(r'^module\.', '', k): v for k, v in
                           model_state['state_dict'].items()})

    normalizer = DifferentiableNormalize(
      mean=config.CIFAR10_MEANS,
      std=config.CIFAR10_STDS,
    )

    return model, normalizer


def get_attack_from_name(
    name: str,
    classifier: nn.Module,
    normalizer: DifferentiableNormalize,
    verbose: bool = False,
) -> advtrain.AdversarialAttackParameters:
    """
    Builds an attack from a name like "recoloradv" or "stadv+delta" or
    "recoloradv+stadv+delta".
    """

    threats = []
    norm_weights = []

    for attack_part in name.split('+'):
        if attack_part == 'delta':
            threats.append(ap.ThreatModel(
                ap.DeltaAddition,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=8.0 / 255,
                ),
            ))
            norm_weights.append(0.0)
        elif attack_part == 'stadv':
            threats.append(ap.ThreatModel(
                ap.ParameterizedXformAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=0.05,
                    xform_class=st.FullSpatial,
                    use_stadv=True,
                ),
            ))
            norm_weights.append(1.0)
        elif attack_part == 'recoloradv':
            threats.append(ap.ThreatModel(
                pt.ReColorAdv,
                ap.PerturbationParameters(
                    lp_style='inf',
                    lp_bound=[0.06, 0.06, 0.06],
                    xform_params={
                        'resolution_x': 16,
                        'resolution_y': 32,
                        'resolution_z': 32,
                    },
                    xform_class=ct.FullSpatial,
                    use_smooth_loss=True,
                    cspace=cs.CIELUVColorSpace(),
                ),
            ))
            norm_weights.append(1.0)
        else:
            raise ValueError(f'Invalid attack "{attack_part}"')

    sequence_threat = ap.ThreatModel(
        ap.SequentialPerturbation,
        threats,
        ap.PerturbationParameters(norm_weights=norm_weights),
    )

    # use PGD attack
    adv_loss = lf.CWLossF6(classifier, normalizer, kappa=float('inf'))
    st_loss = lf.PerturbationNormLoss(lp=2)
    loss_fxn = lf.RegularizedLoss({'adv': adv_loss, 'pert': st_loss},
                                  {'adv': 1.0,      'pert': 0.05},
                                  negate=True)

    pgd_attack = aa.PGD(classifier, normalizer, sequence_threat, loss_fxn)
    return advtrain.AdversarialAttackParameters(
        pgd_attack,
        1.0,
        attack_specific_params={'attack_kwargs': {
            'num_iterations': 100,
            'optimizer': optim.Adam,
            'optimizer_kwargs': {'lr': 0.001},
            'signed': False,
            'verbose': verbose,
        }},
    )

def UniGrad(model, x, target, num_steps, sigma, abs_value=True, return_noisy_expls=False):
    def multi_sigma_uniform_noise(x, sigma):
        noise = [x[i].data.new(x[i].size()).uniform_(-sigma[i], sigma[i]) for i in range(x.size()[0])]
        noise = torch.stack(noise)
        return noise
    def add_noise(x, sigma):
        if isinstance(sigma, tuple):
            assert len(sigma) == len(x), (
                "The number of input tensors "
                "in {} must be equal to the number of stdevs values {}".format(
                    len(x), len(sigma)
                )
            )
            noisy_x = [x+multi_sigma_uniform_noise(x,sigma) for _ in range(num_steps)]
        else:
            assert isinstance(
                sigma, float
            ), "stdevs must be type float. " "Given: {}".format(type(sigma))
            noisy_x = [x+x.data.new(x.size()).uniform_(-sigma, sigma) for i in range(num_steps)]
        return noisy_x
    sm = Saliency(model)
    # noisy_x = [x+x.data.new(x.size()).uniform_(-sigma, sigma) for i in range(num_steps)]
    noisy_x = add_noise(x, sigma)
    all_expl = [sm.attribute(img, target=target, abs=abs_value) for img in noisy_x]
    expl = torch.stack(all_expl)
    expl = torch.mean(expl, dim=0)
    if return_noisy_expls:
        return expl, all_expl
    else:
        return expl

def get_expl(model, x, method, desired_index=None, smooth=False, sigma=1.0, abs_value=True, layer=None, normalize=False, multiply_by_input=False):
    """
    helper function to compute the explanation heatmap
    layer: (only used if method = guided_gradcam) Layer for which GradCAM attributions are computed.
    Currently, only layers with a single tensor output are supported.
    """
    if desired_index is None:
        desired_index = model(x).argmax()
    if method == "uniform_grad": # for uniform gradient method
        heatmap = UniGrad(model, x, target=desired_index, num_steps=10, sigma=sigma, abs_value=abs_value)
    else: # for rest of the methods
        try:
            explantion = expl_methods[method] # this is the explanation class to call
        except KeyError:
            raise ValueError("invalid explnation method: {}".format(method))
        #####
        if method == "guided_gradcam":
            if layer is None:
                raise ValueError("for Guided GradCAM method the layer attribute cannot be None")
            explantion_method = explantion(model, layer=layer) # an instance of the "explanation" class
        else:
            explantion_method = explantion(model) # an instance of the "explanation" class
        #####
        if smooth and method == "saliency":
            smooth_explanation_method = NoiseTunnel(explantion_method)
            heatmap = smooth_explanation_method.attribute(x, nt_samples=10, stdevs=sigma, nt_type='smoothgrad', target=desired_index, abs=abs_value)
        else:
            if method == "saliency":
                heatmap = explantion_method.attribute(x, target=desired_index, abs=abs_value)
            else:
                heatmap = explantion_method.attribute(x, target=desired_index)

    ###################
    if multiply_by_input:
        heatmap = x * heatmap
    ###################
    # Update: added abs value for normalize
    if normalize and heatmap.size()[0]==1:
        if abs_value:
            heatmap = torch.sum(torch.abs(heatmap.squeeze()), 0, True)
            heatmap = heatmap / torch.sum(heatmap)
        else:
            heatmap = torch.sum(heatmap.squeeze(), 0, True)
            heatmap = heatmap / torch.sum(heatmap)
    if normalize and heatmap.size()[0]>1:
        if abs_value:
            heatmap = torch.sum(torch.abs(heatmap), 1, True)
            heatmap = heatmap / torch.sum(heatmap, (1,2,3), True)
        else:
            heatmap = torch.sum(heatmap, 1, True)
            heatmap = heatmap / torch.sum(heatmap, (1,2,3), True)
    ### release the memory ###
    torch.cuda.empty_cache()
    ##########################
    return heatmap

def load_image(data_mean, data_std, device, image_name):
    """
    Helper method to load an image into a torch tensor. Includes preprocessing.
    """
    im = Image.open(image_name)
    x = torchvision.transforms.Normalize(mean=data_mean, std=data_std)(
        torchvision.transforms.ToTensor()(
            torchvision.transforms.CenterCrop(224)(torchvision.transforms.Resize(256)(im))))
    x = x.unsqueeze(0).to(device)
    return x

def clamp(x, mean, std):
    """
    Helper method for clamping the adversarial example in order to ensure that it is a valid image
    """
    upper = torch.from_numpy(np.array((1.0 - mean) / std)).to(x.device)
    lower = torch.from_numpy(np.array((0.0 - mean) / std)).to(x.device)

    if x.shape[1] == 3:  # 3-channel image
        for i in [0, 1, 2]:
            x[0][i] = torch.clamp(x[0][i], min=lower[i], max=upper[i])
    else:
        x = torch.clamp(x, min=lower[0], max=upper[0])
    return x

def torch_to_image(tensor, mean=0, std=1):
    """
    Helper function to convert torch tensor containing input data into image.
    """
    if len(tensor.shape) == 4:
        img = tensor.permute(0, 2, 3, 1)

    img = img.contiguous().squeeze().detach().cpu().numpy()

    img = img * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)
    return np.clip(img, 0, 1)


def heatmap_to_image(heatmap):
    """
    Helper image to convert torch tensor containing a heatmap into image.
    """
    if len(heatmap.shape) == 4:
        heatmap = heatmap.permute(0, 2, 3, 1)

    img = heatmap.squeeze().data.cpu().numpy()

    img = img / np.max(np.abs(img))  # divide by maximum
    img = np.maximum(-1, img)
    img = np.minimum(1, img) * 0.5  # clamp to -1 and divide by two -> range [-0.5, 0.5]
    img = img + 0.5

    return img


def convert_relu_to_softplus(model, beta):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)
    return model

def change_beta_softplus(model, beta):
    """
    This function changes the beta parameter of a softplus network.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.Softplus):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            change_beta_softplus(child, beta)
    return model


def plot_overview(images, heatmaps, mean, std,
                captions=['Target Image', 'Original Image', 'Manipulated Image', 'Target Explanation', 'Original Explanation', 'Manipulated Explanation'],
                filename="overview.png", images_per_row=3):
    fig, ax = plt.subplots(2, images_per_row, figsize=(12, 8))
    for i, im in enumerate(images):
        ax[0, i].imshow(torch_to_image(im, mean, std))
        ax[0, i].set_title(captions[i])
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
    for j, expl in enumerate(heatmaps):
        viz.visualize_image_attr(np.transpose(expl.squeeze().cpu().detach().numpy(), (1,2,0)),
                             np.transpose(images[j].squeeze().cpu().detach().numpy(), (1,2,0)),
                             method='heat_map',
                             cmap="Reds",
                             show_colorbar=False,
                             outlier_perc=2,
                             fig_size=(4,4), plt_fig_axis=(fig, ax[1, j]), use_pyplot=False)
        ax[1, j].set_title(captions[images_per_row+j])
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
