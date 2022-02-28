#!/bin/bash
for lr in 0.0001 0.0005 0.001 0.005 0.01 0.05
do
  for smooth_loss_c in 0.000001 0.00001 0.0001 0.001 0.01 0.1
  do
    echo lr=$lr, smooth L coeff=$smooth_loss_c | tee -a output_expl_topk_relu_l1/output_hpo.log
    python attack_expl_topk_relu_batch.py --norm_weights 0.0 0.0 1.0 --lp_reg 1 --lr $lr \
    --output_dir output_expl_topk_relu_l1/ --smooth_loss_c $smooth_loss_c \
    --additive_lp_bound 1000.0 | tee -a output_expl_topk_relu_l1/output_hpo.log
    echo "\n" | tee -a output_expl_topk_relu_l1/output_hpo.log
  done
done
