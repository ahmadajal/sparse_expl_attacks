#!/bin/bash
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
do
  # python attack_one_time.py --add_to_seed $s --method "saliency" | tee -a output_topk_cifar-10/onetime/output.log
  # python attack_PGD_0.py --add_to_seed $s --method "saliency" | tee -a output_topk_cifar-10/pgd0/output.log
  python attack_coordinate.py --add_to_seed $s --method "input_times_grad" --lr 0.2 | tee -a output_topk_cifar-10/coordinate/output_gradxinput.log
done
####
