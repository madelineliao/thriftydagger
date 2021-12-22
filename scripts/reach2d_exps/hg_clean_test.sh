#!/bin/bash

python main.py \
		--data_path ./data/nov18-gen-data-pick-place/nov18-gen-data-pick-place_s4/pick-place-data-30.pkl \
		--environment PickPlace \
		--method HGDagger \
		--arch MLP \
		--num_models 5 \
		--trajectories_per_rollout 1 \
		--policy_train_epochs 1 \
		--epochs 1 \
		--robosuite
