#!/bin/bash

python run_thriftydagger.py \
        --input_file ./data/dec14_gen_oracle_reach2d_data_1k/dec14_gen_oracle_reach2d_data_1k_s4/pick-place-data-1000.pkl \
		--input_file_2 ./data/dec15_gen_pi_r_noisy_reach2d_data_1k/dec15_gen_pi_r_noisy_reach2d_data_1k_s4/reach2d_pi_r-1000.pkl \
		--gen_data_output reach2d_oracle_pi_r_noisy_mix \
		--reach_sampling_type pi_r_noisy \
        --environment Reach2D \
        --bc_only \
        --num_bc_episodes 1000 \
		--num_test_episodes 100 \
        --no_render \
        --algo_sup \
		--gen_data \
        dec15_reach2d_oracle_pi_r_noisy_mix_100_bc_only_hidden20_eval
