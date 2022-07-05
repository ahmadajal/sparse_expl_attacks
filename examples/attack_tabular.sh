#!/bin/bash
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
do
  python attack_one_time_tabular_data.py --topk 20 --max_num_pixels 20 --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/onetime/output_ig_mlp.log
  python attack_PGD_0_tabular_data.py --max_num_pixels 20 --topk 20 --num_iter 40 --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/pgd0/output_ig_mlp.log
  python attack_coordinate_tabular_data.py --topk 20 --num_iter 100 --max_num_pixels 20 --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/coordinate/output_ig_mlp.log
  ####
  python attack_one_time_tabular_data.py --topk 20 --max_num_pixels 20 --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/onetime/output_gradxinput_mlp.log
  python attack_PGD_0_tabular_data.py --max_num_pixels 20 --topk 20 --num_iter 40 --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/pgd0/output_gradxinput_mlp.log
  python attack_coordinate_tabular_data.py --topk 20 --num_iter 100 --max_num_pixels 20 --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/coordinate/output_gradxinput_mlp.log
  ####
  python attack_one_time_tabular_data.py --topk 20 --max_num_pixels 20 --dataset "yahoo" --lr 0.05 --out_loss_coeff 1e2 --task_type "multi_class" --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/onetime/output_ig_yahoo_mlp.log
  python attack_PGD_0_tabular_data.py --max_num_pixels 20 --topk 20 --dataset "yahoo" --task_type "multiclass" --lr 0.1 --out_loss_coeff 1e2 --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/pgd0/output_ig_yahoo_mlp.log
  python attack_coordinate_tabular_data.py --max_num_pixels 20 --topk 20 --num_iter 100 --dataset "yahoo" --task_type "multiclass" --lr 0.1 --out_loss_coeff 1e2 --add_to_seed $s --model_type "mlp" | tee -a output_topk_tabular_data/coordinate/output_ig_yahoo_mlp.log
  ###
  python attack_one_time_tabular_data.py --topk 20 --max_num_pixels 20 --dataset "yahoo" --lr 0.05 --out_loss_coeff 1e2 --task_type "multi_class" --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/onetime/output_gradxinput_yahoo_mlp.log
  python attack_PGD_0_tabular_data.py --max_num_pixels 20 --topk 20 --dataset "yahoo" --task_type "multiclass" --lr 0.1 --out_loss_coeff 1e2 --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/pgd0/output_gradxinput_yahoo_mlp.log
  python attack_coordinate_tabular_data.py --max_num_pixels 20 --topk 20 --num_iter 100 --dataset "yahoo" --task_type "multiclass" --lr 0.1 --out_loss_coeff 1e2 --add_to_seed $s --method "input_times_grad" --model_type "mlp" | tee -a output_topk_tabular_data/coordinate/output_gradxinput_yahoo_mlp.log
done
####
