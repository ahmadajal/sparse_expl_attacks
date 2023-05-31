from typing import Tuple
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sparse_expl_attacks.utils import get_expl, DifferentiableNormalize, next_topk_coord, project_L0_box_torch

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
            expl_method: Name of the explanation method. Possible options are: "saliency", 
            "lrp", "guided_backprop", "integrated_grad", "input_times_grad", "smooth_grad", 
            "deep_lift".
            num_iter: Number of iterations to optimize the adversarial input.
            lr: Learning rate.
            topk: The number of top k attributes in the topk loss.
            max_num_features: Maximum number of input features permitted to be perturbed.
            normalizer: normalizer object to normalize the input.
            manual_device: Manually set the device on which the atack will be performed. If 
            equal to None then gpu will be selected if available.
        """
        self.model = model
        self.model_relu = model_relu
        self.expl_method = expl_method
        self.num_iter = num_iter
        self.lr = lr
        self.topk = topk
        self.max_num_features = max_num_features
        self.normalizer = normalizer
        if manual_device:
            self.device = manual_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def attack(self,
               attack_type: str, 
               x_input: torch.Tensor, 
               y_input: torch.Tensor, 
               sigma: Tuple = None,
               verbose: bool = False) -> torch.Tensor:
        """Perform the sparse explanation attack against the explanations of instances
        in x_input.

        Args:
            attack_type: Type of the sparse explanation attack. Possible values are "greedy", 
            "pgd0", and "single_step".
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            sigma: Standard deviation of the noise for smooth and uniform gradient methods.
            verbose: Print the result of the attack after each iteration.
        """
        # Check if the attack type is valid.
        assert attack_type in ["greedy", "pgd0", "single_step"]
        # Input should have 4 dimensions.
        assert len(x_input.size()) == 4
        # sigma should not be None for smooth and uniform gradient.
        if self.expl_method == "smooth_grad":
            assert sigma
        
        x_input = x_input.requires_grad_(True)
        BATCH_SIZE = x_input.size()[0]

        # Creating the mask for the top-k attack.
        org_expl = get_expl(model=self.model, 
                            x=self.normalizer.forward(x_input), 
                            expl_method=self.expl_method, 
                            device=self.device,
                            true_label=y_input, 
                            sigma=sigma, 
                            normalize=True)
        org_expl = org_expl.detach()
        if BATCH_SIZE == 1:
            mask = torch.zeros_like(org_expl).flatten()
            mask[torch.argsort(org_expl.view(-1))[-self.topk:]]=1
            mask = mask.view(org_expl.size())
        else:
            mask = torch.zeros_like(org_expl).view(BATCH_SIZE, -1)
            topk_perbatch = torch.argsort(org_expl.view(BATCH_SIZE, -1))[:, -self.topk:]
            for _ in range(mask.size()[0]):
                mask[_][topk_perbatch[_]] = 1
            mask = mask.view(org_expl.size())
        
        # Perform the attack
        x_adv = self.__getattribute__(f"{attack_type}_iterations")(
            x_input,
            y_input,
            mask,
            BATCH_SIZE,
            sigma,
            verbose
        )
        return x_adv

    def greedy_iterations(self, 
                          x_input: torch.Tensor, 
                          y_input: torch.Tensor, 
                          mask: torch.Tensor, 
                          batch_size: int, 
                          sigma: Tuple = None,
                          verbose: bool = False) -> torch.Tensor:
        """Iterations of the Greedy attack

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input
            sigma: Standard deviation of the noise for smooth and uniform gradient methods.
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
        #
        for iter_no in range(self.num_iter):
            optimizer.zero_grad()
            adv_expl = get_expl(model=self.model, 
                                x=self.normalizer.forward(x_adv), 
                                expl_method=self.expl_method, 
                                device=self.device,
                                true_label=y_input, 
                                sigma=sigma, 
                                normalize=True)
            # topk explanation loss.
            expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
            # Logits of the model for the adversarial input.
            adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
            loss = expl_loss + 1e1 * F.mse_loss(adv_logits, org_logits)
            loss.backward()
            # Normalize gradient
            x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1,2,3), keepdim=True))
            # Find all the batches that are not finished yet!
            available_batches = sorted(unfinished_batches.intersection(max_not_reached))
            # Pick the next top coordinate.
            x_adv.grad, chosen_indices = next_topk_coord(x_adv.grad, used_indices, available_batches)
            optimizer.step()
            # Update the used indices:
            used_indices = used_indices + chosen_indices
            # Update step, we have to clip the update to make sure x_adv is in the valid range.
            x_adv.data = x_input.data + torch.clip((x_adv-x_input).data, -x_input.data, (1.0-x_input.data))
            # Compute the topk intersection in this iteration.
            topk_ints = []
            for i in range(mask.size()[0]):
                _, topk_mask_ind = torch.topk(mask[i].flatten(), k=self.topk)
                _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=self.topk)
                topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                    topk_adv_ind.cpu().detach().numpy())))/self.topk)
            unfinished_batches = unfinished_batches - set(np.where(np.array(topk_ints)==0.0)[0])
            n_pixels = np.sum(np.amax(np.abs((x_adv-x_input).cpu().detach().numpy()) > 1e-10, axis=1), axis=(1,2))
            max_not_reached = set(np.where(n_pixels < self.max_num_features)[0])
            if verbose:
                print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))
                print("num remaining: ", len(unfinished_batches))
            if len(unfinished_batches) == 0:
                break

        return x_adv
    


    def pgd0_iterations(self, 
                        x_input: torch.Tensor, 
                        y_input: torch.Tensor, 
                        mask: torch.Tensor,  
                        batch_size: int, 
                        sigma: Tuple = None,
                        verbose: bool = False) -> torch.Tensor:
        """Iterations of the PGD_0 attack

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input
            sigma: Standard deviation of the noise for smooth and uniform gradient methods.
            verbose: Print the result of the attack after each iteration.
        """
        # Logits of the model for the non_perturbed input.
        org_logits = F.softmax(self.model(self.normalizer.forward(x_input)), dim=1)
        org_logits = org_logits.detach()
        x_adv = copy.deepcopy(x_input)
        optimizer = optim.Adam([x_adv], lr=self.lr)

        for iter_no in range(self.num_iter):
            optimizer.zero_grad()
            adv_expl = get_expl(model=self.model, 
                                x=self.normalizer.forward(x_adv), 
                                expl_method=self.expl_method, 
                                device=self.device,
                                true_label=y_input, 
                                sigma=sigma, 
                                normalize=True)
            # topk explanation loss.
            expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
            # Logits of the model for the adversarial input.
            adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
            loss = expl_loss + 1e1 * F.mse_loss(adv_logits, org_logits)
            loss.backward()
            # Normalize gradient
            x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1,2,3), keepdim=True))
            optimizer.step()
            # Project delta on the L_0 ball.
            x_adv.data = x_input.data + project_L0_box_torch(
                (x_adv-x_input).data.permute(0, 2, 3, 1),
                self.max_num_features, -x_input.data.permute(0, 2, 3, 1),
                (1.0-x_input.data).permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2).data
            # Compute the topk intersection in this iteration.
            topk_ints = []
            for i in range(mask.size()[0]):
                _, topk_mask_ind = torch.topk(mask[i].flatten(), k=self.topk)
                _, topk_adv_ind = torch.topk(adv_expl[i].flatten(), k=self.topk)
                topk_ints.append(float(len(np.intersect1d(topk_mask_ind.cpu().detach().numpy(),
                                    topk_adv_ind.cpu().detach().numpy())))/self.topk)
            if verbose:
                print("{}: mean expl loss: {}".format(iter_no, np.mean(topk_ints)))

        return x_adv
    

    def single_step_iterations(self, 
                               x_input: torch.Tensor, 
                               y_input: torch.Tensor, 
                               mask: torch.Tensor,  
                               batch_size: int, 
                               sigma: Tuple = None,
                               verbose: bool = False) -> torch.Tensor:
        """The single step attack

        Args:
            x_input: Input to the model, size = (B, C, H, W).
            y_input: The ground truth label corresponding to x_input.
            mask: Mask tensor for the topk attack.
            batch_size: Batch size of the input
            sigma: Standard deviation of the noise for smooth and uniform gradient methods.
            verbose: Print the result of the attack after each iteration.
        """
        org_logits = F.softmax(self.model(self.normalizer.forward(x_input)), dim=1)
        org_logits = org_logits.detach()
        x_adv = copy.deepcopy(x_input)
        optimizer = optim.Adam([x_adv], lr=self.lr)
        optimizer.zero_grad()
        adv_expl = get_expl(model=self.model, 
                            x=self.normalizer.forward(x_adv), 
                            expl_method=self.expl_method, 
                            device=self.device,
                            true_label=y_input, 
                            sigma=sigma, 
                            normalize=True)
        # topk explanation loss.
        expl_loss = torch.mean(torch.sum(adv_expl*mask, dim=(1,2,3), dtype=torch.float))
        # Logits of the model for the adversarial input.
        adv_logits = F.softmax(self.model(self.normalizer.forward(x_adv)), dim=1)
        loss = expl_loss + 1e1 * F.mse_loss(adv_logits, org_logits)
        loss.backward()
        # Normalize gradient
        x_adv.grad = x_adv.grad / (1e-10 + torch.sum(torch.abs(x_adv.grad), dim=(1,2,3), keepdim=True))
        # Pick the top k coordinates of the update.
        x_adv.grad, chosen_indices = next_topk_coord(x_adv.grad, [], np.arange(batch_size), k=self.topk)
        optimizer.step()
        # Update step
        x_adv.data = x_input.data + torch.clip((x_adv-x_input).data, -x_input.data, (1.0-x_input.data))
        return x_adv
    
    

    
        
