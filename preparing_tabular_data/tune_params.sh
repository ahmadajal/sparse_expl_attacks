#!/bin/bash
for lr in 0.0001 0.0005 0.001
do
  for first_hidden_size in 512 256 128
  do
    for num_layers in 4 3 2
    do
      for dropout in 0.0 0.2
      do
        echo lr=$lr, first_hidden_size=$first_hidden_size, num_layers=$num_layers, dropout=$dropout | tee -a output.log
        python train_credit_model.py --lr $lr --first_hidden_size $first_hidden_size \
        --num_layers $num_layers --dropout $dropout | tee -a output.log
        echo "\n" | tee -a output.log
      done
    done
  done
done
