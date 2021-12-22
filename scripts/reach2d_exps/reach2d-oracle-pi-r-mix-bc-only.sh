#!/bin/bash

python run_thriftydagger.py \
        --input_file ./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl \
		--input_file_2 ./data/dec18_gen_pi_r_reach2d_data_1k_fixed/dec18_gen_pi_r_reach2d_data_1k_fixed_s4/reach2d_pi_r-1000.pkl \
		--reach_sampling_type pi_r \
        --environment Reach2D \
        --bc_only \
        --num_bc_episodes 1000 \
		--num_test_episodes 100 \
        --no_render \
        --algo_sup \
		--seed 4 \
		--gen_data \
		--gen_data_output pi_r_mix_pick_place \
        dec19_reach2d_oracle_pi_r_mix_1000_bc_only_test_seed_reproducibility
