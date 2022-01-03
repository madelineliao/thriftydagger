#!/bin/bash
ENVIRONMENT=Reach2D
N=1000
POLICY_MODE=right_up
SAMPLE_MODE=oracle_mix
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname oracle_mix.pkl \
    --sample_mode $SAMPLE_MODE \
    --policy_mode $POLICY_MODE
