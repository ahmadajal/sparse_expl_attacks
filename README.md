# Sparse Explanation Attacks

This repository contains the code and results for the paper __"Sparse Explanation Attacks: How to Change the Explanation of Neural Classifiers with Minimal Effort"__ submitted to ICDE 2023.

The files in this repository are organized as follows:
- `attacks` folder: contains some helper scripts for calculating explanations of different explanation methods and also for performing the attacks.
- `examples` folder: contains the scripts for the explanation attacks introduced in the paper.
- `preparing_tabular_data` folder: contains some helper notebooks/scripts which were used to preprocess the tabular datasets, train a neual network on them and optimize the hyper-parameters of the neural networks trained on tabular datasets.
- `sample_attack_results` folder: contains the results of the sparse explanation attacks against the explanations of a few Imagenet samples.

For instance, if you want to run the greedy attack against the explanations of a random batch of CIFAR-10 images, simply run:
```
python attack_greedy.py --method <expl_method> --lr <step_size> --topk <topk> --max_num_pixels <max_pixels_to_perturb>
```
You can set the step size, explnation method and other parameters using the argument parser in the script.
