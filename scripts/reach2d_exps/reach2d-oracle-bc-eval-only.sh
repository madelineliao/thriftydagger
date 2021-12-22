#!/bin/bash

python run_thriftydagger.py \
        --input_file ./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl \
        --environment Reach2D \
        --bc_only \
        --num_bc_episodes 1000 \
		--num_test_episodes 100 \
        --no_render \
        --algo_sup \
		--seed 4 \
        dec20_reach2d_data_1k_bc_only_eval_linear_model_testing_cleaned
