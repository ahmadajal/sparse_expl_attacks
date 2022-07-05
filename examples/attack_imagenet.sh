#!/bin/bash
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
do
  python attack_one_time_imagenet.py --add_to_seed $s --method "deep_lift" | tee -a output_topk_imagenet/onetime/output_dl.log
  python attack_PGD_0_imagenet.py --add_to_seed $s --method "deep_lift" | tee -a output_topk_imagenet/pgd0/output_dl.log
  python attack_coordinate_imagenet.py --add_to_seed $s --method "deep_lift" | tee -a output_topk_imagenet/coordinate/output_dl.log
done
####
