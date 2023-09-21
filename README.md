# Sparse Explanation Attacks

This repository contains the code and results for the paper __Sparse Attacks for Manipulating Explanations in Deep Neural Network Models__ accepted at ICDM 2023.


If you want to run the GrID attack against the gradient explanations of a random batch of CIFAR-10 images, simply run:
```
python attack_cifar.py --expl_method saliency --attack_type increase_decrease
```
