#!/bin/bash
python src/main.py \
	--exp_name dec25/ensemble_oracle_reach2d_mlp_test_refactor_eval \
	--eval_only \
	--model_path ./out/dec25/ensemble_oracle_reach2d_mlp_test_refactor/model_4.pt\
    --environment Reach2D \
    --method BC \
    --arch MLP \
	--hidden_size 20 \
    --num_models 2 \
    --seed 4 \
    --overwrite
