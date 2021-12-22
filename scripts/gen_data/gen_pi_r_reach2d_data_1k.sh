#!/bin/bash

python run_thriftydagger.py \
        --gen_data \
		--gen_data_output reach2d_pi_r \
		--reach_sampling_type pi_r \
		--eval ./data/dec15_reach2d_data_1k_bc_only_hidden20_eval/dec15_reach2d_data_1k_bc_only_hidden20_eval_s4/pyt_save/model.pt\
        --input_file ./data/dec18_gen_pi_r_reach2d_data_1k_debugging/dec18_gen_pi_r_reach2d_data_1k_debugging/reach2d_pi_r-1000.pkl \
        --environment Reach2D \
        --bc_only \
        --num_bc_episodes 1000 \
		--num_test_episodes 0 \
        --no_render \
        --algo_sup \
        dec18_gen_pi_r_reach2d_data_1k_fixed
