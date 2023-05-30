import re
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Function, Variable
import torchvision.transforms as transforms
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
from typing import List

# For generating the explanations we use the captum package (https://github.com/pytorch/captum)
from captum.attr import (
            Saliency,
            IntegratedGradients,
            LRP,
            GuidedBackprop,
            NoiseTunnel,
            InputXGradient,
            DeepLift)
from captum.attr import visualization as viz

# dictionary to convert the method name to the proper class
expl_methods={
            "saliency": Saliency,
            "smooth_grad": Saliency, 
            "integrated_grad": IntegratedGradients,
            "lrp": LRP,
            "guided_backprop": GuidedBackprop,
            "input_times_grad": InputXGradient,
            "deep_lift": DeepLift
}


class DifferentiableNormalize(Function):

    def __init__(self, mean, std):
        super(DifferentiableNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.differentiable = True
        self.nondiff_normer = transforms.Normalize(mean, std)


    def __call__(self, var):
        if self.differentiable:
            return self.forward(var)
        else:
            return self.nondiff_normer(var)


    def _setter(self, c, mean, std):
        """ Modifies params going forward """
        if mean is not None:
            self.mean = mean
        assert len(self.mean) == c

        if std is not None:
            self.std = std
        assert len(self.std) == c

        if mean is not None or std is not None:
            self.nondiff_normer = transforms.Normalize(self.mean, self.std)


    def differentiable_call(self):
        """ Sets the __call__ method to be the differentiable version """
        self.differentiable = True


    def nondifferentiable_call(self):
        """ Sets the __call__ method to be the torchvision.transforms version"""
        self.differentiable = False


    def forward(self, var, mean=None, std=None):
        """ Normalizes var by subtracting the mean of each channel and then
            dividing each channel by standard dev
        ARGS:
            self - stores mean and std for later
            var - Variable of shape NxCxHxW
            mean - if not None is a list of length C for channel-means
            std - if not None is a list of length C for channel-stds
        RETURNS:
            variable of normalized var
        """
        c = var.shape[1]
        self._setter(c, mean, std)

        mean_var = Variable(var.data.new(self.mean).view(1, c, 1, 1))
        std_var = Variable(var.data.new(self.std).view(1, c, 1, 1))
        return (var - mean_var) / std_var


def next_topk_coord(array: torch.Tensor, used_inds: List, available_batches: List, k: int = 1) -> Tuple[torch.Tensor, List]:
    """Pick the next top (lagest) coordinate in the array. The largest coordinate is
    the coordinate with the largest average channel values.

    Args:
        array: Input array.
        used_inds: Indices which were picked before and cannot be used anymore.
        available_batches: The batch indices that could still be updated.
        k: Number of top coordinates to pick.

    Returns:
        The new array which only has the sofar picked top coordinates of the input array,
        The indices that were chosen in this step.
    """
    array_norm = torch.norm(array, p=1, dim=1)
    if len(used_inds)>0:
        array_norm[np.array(used_inds).T.tolist()] = 0.0
    inds = torch.topk(array_norm.view((array.size()[0],-1)), k=k, dim=1)[1]
    new_inds = [np.unravel_index(i, shape=(array.size()[2], array.size()[3])) for i in inds.cpu().numpy()]
    chosen_inds = []
    for k_ind in range(k):
        chosen_inds = chosen_inds +  [[i, j[0][k_ind], j[1][k_ind]] for i,j in enumerate(new_inds)]
    chosen_inds = np.array(chosen_inds)[available_batches].tolist()
    #
    new_inds0 = np.array([[e[0], 0, e[1], e[2]] for e in chosen_inds]).T.tolist()
    new_inds1 = np.array([[e[0], 1, e[1], e[2]] for e in chosen_inds]).T.tolist()
    new_inds2 = np.array([[e[0], 2, e[1], e[2]] for e in chosen_inds]).T.tolist()
    ### using the previous inds doesn't add to the sparsity
    ### and as we clamp later doesn't also violate the valid range
    new_inds_used0 = np.array([[e[0], 0, e[1], e[2]] for e in used_inds]).T.tolist()
    new_inds_used1 = np.array([[e[0], 1, e[1], e[2]] for e in used_inds]).T.tolist()
    new_inds_used2 = np.array([[e[0], 2, e[1], e[2]] for e in used_inds]).T.tolist()
    ###
    new_array = torch.zeros_like(array)
    new_array[new_inds0] = array[new_inds0]
    new_array[new_inds_used0] = array[new_inds_used0]
    new_array[new_inds1] = array[new_inds1]
    new_array[new_inds_used1] = array[new_inds_used1]
    new_array[new_inds2] = array[new_inds2]
    new_array[new_inds_used2] = array[new_inds_used2]
    ###
    return new_array, chosen_inds


def project_L0_box_torch(x_input: torch.Tensor, 
                         k: int, 
                         lb: float, 
                         ub: float) -> torch.Tensor:
    """Project the input to the L_0 ball wehre ||x||_0 = k, i.e, finding the 
    closest point to x with only k non-zero elements.

    Args:
        x_input: Input tensor.
        k: Number of non-zero elements.
        lb: Minimum of the valid range of the data.
        ub: Maximum of the valid range of the data.

    Returns:
        Closest point to x with only k non-zero elements.
    """
    x_output = torch.clone(x_input)
    p1 = torch.sum(x_output**2, dim=-1)
    p2 = torch.minimum(torch.minimum(ub - x_output, x_output - lb), torch.zeros_like(x_output))
    p2 = torch.sum(p2**2, dim=-1)
    p3 = torch.sort(torch.reshape(p1-p2, (p2.size()[0],-1)))[0][:,-k]
    x_output = x_output * (torch.logical_and(lb <= x_output, x_output <= ub)) + lb * (lb > x_output) + ub * (x_output > ub)
    x_output = x_output * torch.unsqueeze((p1 - p2) >= p3.reshape([-1, 1, 1]), -1)
    return x_output


def get_expl(model: nn.Module, 
             x: torch.Tensor, 
             expl_method: str,
             device: torch.device, 
             true_label: int = None, 
             sigma: Tuple = None, 
             abs_value: bool = True,  
             normalize: bool = False, 
             multiply_by_input: bool = False) -> torch.Tensor:
    """This function computes the explanation of the model for all instance in the input.

    Args:
        model: The (trained) model.
        x: Input instance fo which we want to compute the explanation. 
        (could be a batch of inputs)
        expl_method: Name of the explanation method. Possible options are: "saliency", 
        "lrp", "guided_backprop", "integrated_grad", "input_times_grad", "smooth_grad", 
        "deep_lift".
        device: The torch device used for the computation.
        true_label: Ground truth label(s) for the input. Defaults to None. If set to None, 
        the model prediction will be used as ground truth labels.
        sigma: Standard deviation of the noise for the smooth gradient method (One value 
        for each instance in the batch). Defaults to None.
        abs_value: Whether to output the absolute value of the explanation or not. 
        Defaults to True.
        normalize: Whether to normalize the explanation so that the sum of the attributions 
        is one. Defaults to False.
        multiply_by_input: Whether to multiply the explanation by the input or not. Defaults 
        to False.

    Returns:
        heatmap: The explanation tensor. If normalize is False, then the size of this tensor is 
        (B, C, H, W) and otherwise the size is (B, 1, H, W). For batch size equal to one, the 
        size of the heatmap tensor is either (1, H, W) or (C, H, W) depending on normalize.
    """
    # Check if the explanation method name is supported.
    assert expl_method in expl_methods.keys(), f"explanation method {expl_method} is not supported."
    # If the ground truth label(s) are not given as input.
    if true_label is None:
        true_label = model(x).argmax()
    # For batch size = 1.
    if len(sigma) == 1:
        sigma = sigma[0]

    # Defining the explanation class to call.
    explantion = expl_methods[expl_method]
    # An instance of the "explanation" class.
    explantion_method = explantion(model)
    if expl_method == "smooth_grad":
        smooth_explanation_method = NoiseTunnel(explantion_method)
        heatmap = smooth_explanation_method.attribute(x, nt_samples=10, stdevs=sigma, nt_type='smoothgrad', target=true_label, abs=abs_value)
    elif expl_method == "saliency":
        heatmap = explantion_method.attribute(x, target=true_label, abs=abs_value)
    else:
        heatmap = explantion_method.attribute(x, target=true_label)

    # Multiplying the explanation by the input.
    if multiply_by_input:
        heatmap = x * heatmap
    # Normalizing the explanation such that sum of all attributions is one.
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
    # Release the memory when using GPU!
    if device.type.startswith("cuda"):
        torch.cuda.empty_cache()
    return heatmap


def convert_relu_to_softplus(model: nn.Module, beta: float) -> nn.Module:
    """Replaces the ReLU activations in a model with softplus.

    Args:
        model: The model.
        beta : The $\beta$ parameter of the softplus function.

    Returns:
        model with softplus activations.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            convert_relu_to_softplus(child, beta)
    return model


def change_beta_softplus(model: nn.Module, beta: float) -> nn.Module:
    """This function changes the beta parameter of a neural network with 
    softplus activations.

    Args:
        model: The model with softplus activation.
        beta: New $\beta$ parameter for the sftplus activations.

    Returns:
        model with the updated softplus activations.
    """
    for child_name, child in model.named_children():
        if isinstance(child, nn.Softplus):
            setattr(model, child_name, nn.Softplus(beta=beta))
        else:
            change_beta_softplus(child, beta)
    return model


def topk_intersect(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Top-k intersection between two tensors.

    Args:
        x: First tensor. shape = (C, H, W).
        y: Second tensor. shape = (C, H, W).
        k: top-k value.

    Returns:
        The topk intersection value, i.e, how many coordinates are shared
        among the topk values of tensors x and y.
    """
    _, topk_x_ind = torch.topk(x.flatten(), k=k)
    _, topk_y_ind = torch.topk(y.flatten(), k=k)
    intersect = float(len(np.intersect1d(topk_x_ind.cpu().detach().numpy(),
                        topk_y_ind.cpu().detach().numpy())))/k
    return intersect


def weighted_topk_intersect(x: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Weighted top-k intersections between two tensors. The weights are based on 
    the rank of the coordinates.

    Args:
        x: First tensor. shape = (C, H, W).
        y: Second tensor. shape = (C, H, W).
        k: top-k value.

    Returns:
        The weighted intersection value.
    """
    _, topk_x_ind = torch.topk(x.flatten(), k=k)
    _, topk_y_ind = torch.topk(y.flatten(), k=k)
    assert len(topk_x_ind)==len(topk_y_ind)
    ints, i_x, _ = np.intersect1d(topk_x_ind, topk_y_ind, return_indices=True)
    if len(ints)==0:
        return 0
    else:
        return sum(k-np.sort(i_x)) / (k*(k+1)/2)


# def plot_overview(images, heatmaps, mean, std,
#                 captions=['Target Image', 'Original Image', 'Manipulated Image', 'Target Explanation', 'Original Explanation', 'Manipulated Explanation'],
#                 filename="overview.png", images_per_row=3):
#     fig, ax = plt.subplots(2, images_per_row, figsize=(12, 8))
#     for i, im in enumerate(images):
#         ax[0, i].imshow(torch_to_image(im, mean, std))
#         ax[0, i].set_title(captions[i])
#         ax[0, i].set_xticks([])
#         ax[0, i].set_yticks([])
#     for j, expl in enumerate(heatmaps):
#         viz.visualize_image_attr(np.transpose(expl.squeeze().cpu().detach().numpy(), (1,2,0)),
#                              np.transpose(images[j].squeeze().cpu().detach().numpy(), (1,2,0)),
#                              method='heat_map',
#                              cmap="Reds",
#                              show_colorbar=False,
#                              outlier_perc=2,
#                              fig_size=(4,4), plt_fig_axis=(fig, ax[1, j]), use_pyplot=False)
#         ax[1, j].set_title(captions[images_per_row+j])
#     plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)