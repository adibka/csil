#!/bin/bash

for demos in 3 10
do
    for seed in 0 1 2 3 4
    do
        python run_csil.py \
          --env_name=Humanoid-v2 \
          --expert_path=experts/Humanoid-v2_25.pkl \
          --entropy_coefficient=0.01 \
          --policy_pretrain_steps=500 \
          --policy_pretrain_lr=0.001 \
          --num_demonstrations=$demos \
          --seed=$seed \
          --num_steps=310000 \
          --eval_every=10000 \
          --evaluation_episodes=10
    done
done
