#!/bin/bash
ENVIRONMENT=Reach2D
N=1000
POLICY_MODE=up_right
SAMPLE_MODE=oracle
SEED=0

python ./src/generate_data.py \
    --save_dir ./data/$ENVIRONMENT \
    --environment $ENVIRONMENT \
    --N_trajectories $N \
    --seed $SEED \
    --save_fname $SAMPLE_MODE\_$POLICY_MODE.pkl \
    --sample_mode $SAMPLE_MODE \
    --policy_mode $POLICY_MODE 