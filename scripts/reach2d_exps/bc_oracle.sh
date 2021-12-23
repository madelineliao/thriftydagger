#!/bin/bash
python main.py \
	--exp_name dec22_bc_oracle_reach2d_linear \
    --data_path ./data/scripted_oracle_reach2d.pkl \
    --environment Reach2D \
    --method BC \
    --arch LinearModel \
    --num_models 1 \
    --seed 4 \
	--overwrite
