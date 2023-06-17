import copy
from typing import Tuple

import numpy as np
import scipy.spatial as spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sparse_expl_attacks.utils import (
    DifferentiableNormalize,
    get_expl,
    next_topk_coord,
    project_L0_box_torch,
    topk_intersect,
)


class SparseAttack:
    def __init__(
        self,
        model: nn.Module,
        model_relu: nn.Module,
        expl_method: str,
        num_iter: int,
        lr: float,
        topk: int,
        max_num_features: int,
        gamma: float,
        normalizer: DifferentiableNormalize,
        manual_device: str = None,
    ) -> None:
        """This attack finds a sparse perturbation to manipulate the explanation
        of a deep neural network (DNN) classifier. This attack uses the topk loss
        to ensure the manipulation of the explanation.

        Args:
            model: The (trained) model against which the explanation attack will be
            performed. This model should have softmax non-linearities.
            model_relu: Same model with ReLU non-linearities.
            expl_method: Name of the explanation method. Possible options are:
            "saliency", "lrp", "guided_backprop", "integrated_grad",
            "input_times_grad", "smooth_grad", "deep_lift".
            num_iter: Number of iterations to optimize the adversarial input.
            lr: Learning rate.
            topk: The number of top k attributes in the topk loss.
            max_num_features: Maximum number of input features permitted to be perturbed.
            gamma: The output loss term coefficient in the attack loss.
            normalizer: normalizer object to normalize the input.
            manual_device: Manually set the device on which the atack will be performed.
            If equal to None then gpu will be selected if available.
        """
        self.model = model
        self.model_relu = model_relu
        self.expl_method = expl_method
        self.num_iter = num_iter
        self.lr = lr
        self.topk = topk
        self.max_num_features = max_num_features
        self.gamma = gamma
        self.normalizer = normalizer
        if manual_device:
            self.device = manual_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(
        self,
        attack_type: str,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Perform the sparse explanation attack against the explanations of instances
        in x_input.

        Args:
            attack_type: Type of the sparse explanation attack. Possible values
            are "greedy", "pgd0", and "single_step".
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            sigma: Standard deviation of the noise for smooth gradient method.
            verbose: Print the result of the attack after each iteration.
        """
        # Check if the attack type is valid.
        assert attack_type in [
            "increase_decrease",
            "greedy",
            "pgd0_decrease",
            "pgd0",
            "single_step",
        ]
        # Input should have 4 dimensions.
        assert len(x_input.size()) == 4
        # sigma should not be None for smooth and uniform gradient.
        if self.expl_method == "smooth_grad":
            assert sigma

        x_input = x_input.requires_grad_(True)
        BATCH_SIZE = x_input.size()[0]

        # Creating the mask for the top-k attack.
        org_expl = get_expl(
            model=self.model,
            x=self.normalizer.forward(x_input),
            expl_method=self.expl_method,
            device=self.device,
            true_label=y_input,
            sigma=sigma,
            normalize=True,
        )
        org_expl = org_expl.detach()
        if BATCH_SIZE == 1:
            mask = torch.zeros_like(org_expl).flatten()
            mask[torch.argsort(org_expl.view(-1))[-self.topk :]] = 1  # noqa: E203
            mask = mask.view(org_expl.size())
        else:
            mask = torch.zeros_like(org_expl).view(BATCH_SIZE, -1)
            topk_perbatch = torch.argsort(org_expl.view(BATCH_SIZE, -1))[
                :, -self.topk :  # noqa: E203
            ]
            for _ in range(mask.size()[0]):
                mask[_][topk_perbatch[_]] = 1
            mask = mask.view(org_expl.size())

        # How many features to perturb in each iteration of the greedy attack.
        perturbation_per_iter = kwargs.get("perturbation_per_iter", 1)
        # Perform the attack
        x_adv = self.__getattribute__(f"{attack_type}_iterations")(
            x_input,
            y_input,
            mask,
            BATCH_SIZE,
            sigma,
            verbose,
            perturbation_per_iter=perturbation_per_iter,
        )
        return x_adv

    def greedy_iterations(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Iterations of the Greedy attack

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input.
            sigma: Standard deviation of the noise for smooth gradient method.
            verbose: Print the result of the attack after each iteration.
        """
        # Logits of the model for the non_perturbed input.
        org_logits = F.softmax(self.model(self.normalizer.forward(x_input)), dim=1)
        org_logits = org_logits.detach()
        x_adv = copy.deepcopy(x_input)
        optimizer = optim.Adam([x_adv], lr=self.lr)
        # we need to stop perturbing more features if the topk is zero for an instance
        unfinished_batches = set(np.arange(batch_size))
        max_not_reached = set(np.arange(batch_size))
        # when we do an update along a dimension, we keep that in a list so to avoid
        # choosing the same dimension in the next iterations
        used_indices = []
        # How many features to perturb in each iteration.
        perturbation_per_iter = kwargs.get("perturbation_per_iter", 1)
        for iter_no in range(self.num_iter):
            optimizer.zero_grad()
            adv_expl = get_expl(
                model=self.model,
                x=self.normalizer.forward(x_adv),
                expl_method=self.expl_method,
                device=self.device,
                true_label=y_input,
                sigma=sigma,
                normalize=True,
            )
            # topk explanation loss.
            expl_loss = torch.mean(
                torch.sum(adv_expl * mask, dim=(1, 2, 3), dtype=torch.float)
            )
            # Logits of the model for the adversarial input.
            adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
            loss = expl_loss + self.gamma * F.mse_loss(adv_logits, org_logits)
            loss.backward()
            # Normalize gradient
            x_adv.grad = x_adv.grad / (
                1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1, 2, 3), keepdim=True)
            )
            # Find all the batches that are not finished yet!
            available_batches = sorted(unfinished_batches.intersection(max_not_reached))
            # Pick the next top coordinate.
            x_adv.grad, chosen_indices = next_topk_coord(
                x_adv.grad, used_indices, available_batches, k=perturbation_per_iter
            )
            optimizer.step()
            # Update the used indices:
            used_indices = used_indices + chosen_indices
            # Update step, we have to clip the update to make sure x_adv is
            # in the valid range.
            x_adv.data = x_input.data + torch.clip(
                (x_adv - x_input).data, -x_input.data, (1.0 - x_input.data)
            )
            # Compute the topk intersection in this iteration.
            topk_ints = []
            for i in range(mask.size()[0]):
                topk_ints.append(topk_intersect(mask[i], adv_expl[i], self.topk))
            unfinished_batches = unfinished_batches - set(
                np.where(np.array(topk_ints) == 0.0)[0]
            )
            n_pixels = np.sum(
                np.amax(np.abs((x_adv - x_input).cpu().detach().numpy()) > 1e-10, axis=1),
                axis=(1, 2),
            )
            max_not_reached = set(np.where(n_pixels < self.max_num_features)[0])
            if verbose:
                print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))
                print("num remaining: ", len(unfinished_batches))
            if len(unfinished_batches) == 0:
                break

        return x_adv

    def decrease_iterations(
        self,
        x_input: torch.Tensor,
        x_adv: torch.Tensor,
        y_input: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
    ) -> torch.Tensor:
        """Reduce iterations to be applied after the greedy increase of the perturbed
        features. This method should only be used inside the increase_decrease
        method.

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            x_adv: Adversarial input computed from the greedy iterations.
            y_input: The ground truth label corresponding to x_input.
            batch_size: Batch size of the input.
            sigma: Standard deviation of the noise for smooth gradient method.
            Defaults to None.

        Returns:
            An adversarial input with same or less perturbed features and with the
            same or less topk intersection loss.
        """
        # The topk intersection losses resulted from the greedy iterations.
        org_expl = get_expl(
            model=self.model_relu,
            x=self.normalizer.forward(x_input),
            expl_method=self.expl_method,
            device=self.device,
            true_label=y_input,
            sigma=sigma,
            normalize=True,
        )
        adv_expl = get_expl(
            model=self.model_relu,
            x=self.normalizer.forward(x_adv),
            expl_method=self.expl_method,
            device=self.device,
            true_label=y_input,
            sigma=sigma,
            normalize=True,
        )
        topk_ints = []
        for i in range(x_input.size()[0]):
            topk_ints.append(topk_intersect(org_expl[i], adv_expl[i], self.topk))
        # The cosine distances resulting from the greedy iterations.
        cos_dists = [
            spatial.distance.cosine(
                adv_expl[i].detach().cpu().flatten(),
                org_expl[i].detach().cpu().flatten(),
            )
            for i in range(x_input.size()[0])
        ]
        # The adversarial noise tensor.
        r_adv = x_adv - x_input
        # Perturbation mask to keep track of the perturbed features that have been
        # checked already.
        perturb_mask = torch.amax(torch.abs(r_adv) > 1e-10, dim=1, keepdim=True).int()
        perturb_mask = perturb_mask.repeat(1, 3, 1, 1)
        features_checked = 0
        while features_checked < self.max_num_features and perturb_mask.nonzero().size()[0]:
            features_checked += 1
            # Find the minimum perturbation coordinate in all batches. The minimum coordinate
            # is the coordinate with the minumum l_1 norm along the channel axis.
            non_zero_r_adv = torch.sum(
                torch.abs(r_adv * perturb_mask) + (1 - perturb_mask) * 1e5, dim=1, keepdim=True
            )
            min_indices = torch.argmin(non_zero_r_adv.view(batch_size, -1), dim=1)
            # Create a temporary noise tensor without the selected indices.
            temp_r_adv = copy.deepcopy(r_adv.view(batch_size, 3, -1).data)
            temp_r_adv[list(range(batch_size)), :, min_indices.cpu().numpy()] = 0.0
            temp_r_adv = temp_r_adv.view(x_input.size())
            # Create a temporary adversarial input made from the temporary noise.
            temp_x_adv = copy.deepcopy(x_adv)
            temp_x_adv.data = x_input.data + temp_r_adv.data
            # Check if the temporary adversarial input can improve the top-k loss in any batch.
            temp_adv_expl = get_expl(
                model=self.model_relu,
                x=self.normalizer.forward(temp_x_adv),
                expl_method=self.expl_method,
                device=self.device,
                true_label=y_input,
                sigma=sigma,
                normalize=True,
            )
            new_topk_ints = []
            for i in range(x_input.size()[0]):
                new_topk_ints.append(topk_intersect(org_expl[i], temp_adv_expl[i], self.topk))
            # Check if the temporary adversarial input can improve the cosine distance
            # in any batch.
            new_cos_dists = [
                spatial.distance.cosine(
                    temp_adv_expl[i].detach().cpu().flatten(),
                    org_expl[i].detach().cpu().flatten(),
                )
                for i in range(x_input.size()[0])
            ]
            improved_topk_int = (np.array(new_topk_ints) - np.array(topk_ints)) <= 0
            imporved_cosd = (np.array(new_cos_dists) - np.array(cos_dists)) >= -1e-3
            successful_batches = np.logical_and(improved_topk_int, imporved_cosd)
            # Update the adversarial input and noise.
            x_adv.data[successful_batches] = temp_x_adv.data[successful_batches]
            r_adv.data[successful_batches] = temp_r_adv.data[successful_batches]
            # We do not need to check these indices anymore, regardless of the above result!
            perturb_mask = perturb_mask.view(batch_size, 3, -1)
            perturb_mask[list(range(batch_size)), :, min_indices.cpu().numpy()] = 0.0
            perturb_mask = perturb_mask.view(x_input.size())

        return x_adv

    def increase_decrease_iterations(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Greedy increase and reduce attack.

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input.
            sigma: Standard deviation of the noise for smooth gradient method.
            Defaults to None.
            verbose: Print the result of the attack after each iteration. Defaults to False.

        Returns:
            The adversarial input that manipulates the explanation but keeps the prediction
            unchanged.
        """
        perturbation_per_iter = kwargs.get("perturbation_per_iter", 1)
        x_adv_increase = self.greedy_iterations(
            x_input,
            y_input,
            mask,
            batch_size,
            sigma,
            perturbation_per_iter=perturbation_per_iter,
        )

        x_adv = self.decrease_iterations(x_input, x_adv_increase, y_input, batch_size, sigma)

        return x_adv

    def pgd0_iterations(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Iterations of the PGD_0 attack.

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input.
            sigma: Standard deviation of the noise for smooth gradient method.
            verbose: Print the result of the attack after each iteration.
        """
        # Logits of the model for the non_perturbed input.
        org_logits = F.softmax(self.model(self.normalizer.forward(x_input)), dim=1)
        org_logits = org_logits.detach()
        x_adv = copy.deepcopy(x_input)
        optimizer = optim.Adam([x_adv], lr=self.lr)

        for iter_no in range(self.num_iter):
            optimizer.zero_grad()
            adv_expl = get_expl(
                model=self.model,
                x=self.normalizer.forward(x_adv),
                expl_method=self.expl_method,
                device=self.device,
                true_label=y_input,
                sigma=sigma,
                normalize=True,
            )
            # topk explanation loss.
            expl_loss = torch.mean(
                torch.sum(adv_expl * mask, dim=(1, 2, 3), dtype=torch.float)
            )
            # Logits of the model for the adversarial input.
            adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
            loss = expl_loss + self.gamma * F.mse_loss(adv_logits, org_logits)
            loss.backward()
            # Normalize gradient
            x_adv.grad = x_adv.grad / (
                1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1, 2, 3), keepdim=True)
            )
            optimizer.step()
            # Project delta on the L_0 ball.
            x_adv.data = (
                x_input.data
                + project_L0_box_torch(
                    (x_adv - x_input).data.permute(0, 2, 3, 1),
                    self.max_num_features,
                    -x_input.data.permute(0, 2, 3, 1),
                    (1.0 - x_input.data).permute(0, 2, 3, 1),
                )
                .permute(0, 3, 1, 2)
                .data
            )
            # Compute the topk intersection in this iteration.
            topk_ints = []
            for i in range(mask.size()[0]):
                topk_ints.append(topk_intersect(mask[i], adv_expl[i], self.topk))
            if verbose:
                print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))

        return x_adv

    def pgd0_decrease_iterations(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Find perturbation with pgd0 and reduce the perturbation.

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input.
            sigma: Standard deviation of the noise for smooth gradient method.
            Defaults to None.
            verbose: Print the result of the attack after each iteration. Defaults to False.

        Returns:
            The adversarial input that manipulates the explanation but keeps the prediction
            unchanged.
        """
        x_adv_increase = self.pgd0_iterations(x_input, y_input, mask, batch_size, sigma)

        x_adv = self.decrease_iterations(x_input, x_adv_increase, y_input, batch_size, sigma)

        return x_adv

    def single_step_iterations(
        self,
        x_input: torch.Tensor,
        y_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        sigma: Tuple = None,
        verbose: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """The single step attack

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input
            sigma: Standard deviation of the noise for smooth gradient method.
            verbose: Print the result of the attack after each iteration.
        """
        org_logits = F.softmax(self.model(self.normalizer.forward(x_input)), dim=1)
        org_logits = org_logits.detach()
        x_adv = copy.deepcopy(x_input)
        optimizer = optim.Adam([x_adv], lr=self.lr)
        optimizer.zero_grad()
        adv_expl = get_expl(
            model=self.model,
            x=self.normalizer.forward(x_adv),
            expl_method=self.expl_method,
            device=self.device,
            true_label=y_input,
            sigma=sigma,
            normalize=True,
        )
        # topk explanation loss.
        expl_loss = torch.mean(torch.sum(adv_expl * mask, dim=(1, 2, 3), dtype=torch.float))
        # Logits of the model for the adversarial input.
        adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
        loss = expl_loss + self.gamma * F.mse_loss(adv_logits, org_logits)
        loss.backward()
        # Normalize gradient
        x_adv.grad = x_adv.grad / (
            1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1, 2, 3), keepdim=True)
        )
        # Pick the top k coordinates of the update.
        x_adv.grad, chosen_indices = next_topk_coord(
            x_adv.grad, [], np.arange(batch_size), k=self.topk
        )
        optimizer.step()
        # Update step
        x_adv.data = x_input.data + torch.clip(
            (x_adv - x_input).data, -x_input.data, (1.0 - x_input.data)
        )
        return x_adv
