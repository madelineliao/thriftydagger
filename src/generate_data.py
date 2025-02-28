import argparse
import numpy as np
import os
import pickle
import random
import torch

from constants import (
    NUT_ASSEMBLY_MAX_TRAJ_LEN,
    PICKPLACE_MAX_TRAJ_LEN,
    REACH2D_ACT_MAGNITUDE,
    REACH2D_MAX_TRAJ_LEN,
    REACH2D_PILLAR_MAX_TRAJ_LEN,
    REACH2D_RANGE_X,
    REACH2D_RANGE_Y,
)
from envs import Reach2D, Reach2DPillar
from envs.grid import Grid
from policies import NutAssemblyPolicy, Reach2DPillarPolicy, Reach2DPolicy
from util import get_model_type_and_kwargs, init_model, setup_robosuite


def parse_args():
    parser = argparse.ArgumentParser()

    # Sampling parameters
    parser.add_argument("--environment", type=str, default="Reach2D", help="Name of environment for data sampling.")
    parser.add_argument(
        "--robosuite", action="store_true", help="Whether or not the environment is a Robosuite environment"
    )
    parser.add_argument(
        "--N_trajectories", type=int, default=1000, help="Number of trajectories (demonstrations) to sample."
    )
    parser.add_argument("--add_noise", action="store_true", help="If true, noise is added to sampled states.")
    parser.add_argument("--noise_mean", type=float, default=1.0, help="Mean to use for Gaussian noise.")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Std to use for Gaussian noise.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")

    # Saving
    parser.add_argument("--save_dir", default="./data", type=str, help="Directory to save the data in.")
    parser.add_argument("--save_fname", type=str, help="File name for the saved data.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If true and the save file exists already, it will be overwritten with newly-generated data.",
    )

    # Arguments specific to the Reach2D environment
    parser.add_argument("--policy", type=str, default="straight", help="Specifies which policy to use.")
    parser.add_argument(
        "--policy2", type=str, default=None, help="Specifies second policy to use if sampling from a mixture."
    )
    parser.add_argument(
        "--random_start_state", action="store_true", help="Start at a random point instead of the origin."
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Model path to use as pi_r when args.sample_mode samples from pi_r."
    )
    parser.add_argument("--arch", type=str, default="LinearModel", help="Model architecture to use.")
    parser.add_argument(
        "--num_models", type=int, default=1, help="Number of models in the ensemble; if 1, a non-ensemble model is used"
    )
    parser.add_argument(
        "--perc",
        type=float,
        default=0.5,
        help=(
            "For use with args.sample_mode == 'oracle_pi_r_mix' only. Percentage of oracle trajectories to use (vs. "
            "policy-sampled trajectories)."
        ),
    )

    # Robosuite
    parser.add_argument("--no_render", action="store_true", help="If provided, Robosuite rendering is skipped.")

    args = parser.parse_args()
    return args


def sample(env, policy, N_trajectories, interactive_robosuite=False):
    demos = []
    if interactive_robosuite:
        print("Press 'Z' to reset (and ignore) the current demonstration")
        print("Press Ctrl-C to quit + save all demos recorded so far.\n")
    try:
        action_lim = env.action_space.high[0]
        for _ in range(N_trajectories):
            curr_obs = env.reset()
            obs = []
            act = []
            done = False
            success = False
            
            while not done and not success:
                action = policy.act(curr_obs)
                if interactive_robosuite:
                    # Use 'Z' press to indicate 'reset'
                    if action[3] != 0:
                        curr_obs = env.reset()
                        obs = []
                        act = []
                        done = False
                        success = False
                        continue
                action = torch.clamp(action, min=-action_lim, max=action_lim)
                obs.append(torch.tensor(curr_obs).float())
                act.append(action.float())
                
                curr_obs, success, done, _ = env.step(action)
            
            demos.append({"obs": obs, "act": act, "success": success})
    except KeyboardInterrupt:
        return demos
    
    return demos


def get_mixture_dataset(file1, file2, perc, N_trajectories):
    demos1 = np.array(pickle.load(open(file1, "rb")))
    demos2 = np.array(pickle.load(open(file2, "rb")))
    
    num_demos1 = int(perc * N_trajectories)
    num_demos2 = N_trajectories - num_demos1
    
    idxs1 = torch.randperm(len(demos1))
    idxs2 = torch.randperm(len(demos2))
    
    demos1 = demos1[idxs1[:num_demos1]]
    demos2 = demos2[idxs2[:num_demos2]]
    
    demos = list(demos1) + list(demos2)

    return demos

def load_model(args, device):
    # TODO: add other envs
    env = Reach2D(device, max_ep_len=REACH2D_MAX_TRAJ_LEN, random_start_state=args.random_start_state)
    model_type, model_kwargs = get_model_type_and_kwargs(args, obs_dim=env.obs_dim, act_dim=env.act_dim)
    model = init_model(model_type, model_kwargs, device=device, num_models=args.num_models)

    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    return model


def get_env_and_policies(args, device):
    if args.robosuite:
        if args.environment == "PickPlace":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=PICKPLACE_MAX_TRAJ_LEN)
        elif args.environment == "NutAssembly":
            env, robosuite_cfg = setup_robosuite(args, max_traj_len=NUT_ASSEMBLY_MAX_TRAJ_LEN)
        else:
            raise NotImplementedError(f"Environment {args.environment} has not been implemented yet!")

        policy = NutAssemblyPolicy(args.policy, env, robosuite_cfg)
        policy2 = None

    elif args.environment == "Reach2D":
        model = None if args.policy != "model" else load_model(args, device)
        policy = Reach2DPolicy(args.policy, model=model)
        grid = None

        if policy.uses_grid:
            x_ticks = int(REACH2D_RANGE_X / REACH2D_ACT_MAGNITUDE) + 1
            y_ticks = int(REACH2D_RANGE_Y / REACH2D_ACT_MAGNITUDE) + 1
            grid = Grid(
                x1=0, x2=REACH2D_RANGE_X, y1=0, y2=REACH2D_RANGE_Y, x_ticks=x_ticks, y_ticks=y_ticks, omitted_shape=None
            )

        env = Reach2D(
            device,
            max_ep_len=REACH2D_MAX_TRAJ_LEN,
            grid=grid,
            random_start_state=args.random_start_state,
            range_x=REACH2D_RANGE_X,
            range_y=REACH2D_RANGE_Y,
        )

        model = None if args.policy2 != "model" else load_model(args, device)
        policy2 = Reach2DPolicy(args.policy2, model=model) if args.policy2 is not None else None

    elif args.environment == "Reach2DPillar":
        x_ticks = int(REACH2D_RANGE_X / REACH2D_ACT_MAGNITUDE) + 1
        y_ticks = int(REACH2D_RANGE_Y / REACH2D_ACT_MAGNITUDE) + 1
        grid = Grid(
            x1=0, x2=REACH2D_RANGE_X, y1=0, y2=REACH2D_RANGE_Y, x_ticks=x_ticks, y_ticks=y_ticks, omitted_shape=None
        )

        env = Reach2DPillar(
            device,
            max_ep_len=REACH2D_PILLAR_MAX_TRAJ_LEN,
            grid=grid,
            random_start_state=args.random_start_state,
            range_x=REACH2D_RANGE_X,
            range_y=REACH2D_RANGE_Y,
        )

        policy = Reach2DPillarPolicy(args.policy, env.pillar)
        policy2 = Reach2DPillarPolicy(args.policy2, env.pillar) if args.policy2 is not None else None
    else:
        raise NotImplementedError(
            f"Data generation for the environment '{args.environment}' has not been implemented yet!"
        )

    return env, policy, policy2


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_path = os.path.join(args.save_dir, args.save_fname)
    if not args.overwrite and os.path.isfile(save_path):
        raise FileExistsError(
            f"The file {save_path} already exists. If you want to overwrite it, rerun with the argument --overwrite."
        )
    os.makedirs(args.save_dir, exist_ok=True)

    print("Generating data...")

    if args.policy2 is not None:
        # Generate mixture dataset
        if args.robosuite:
            file1 = os.path.join(args.save_dir, f"{args.policy}.pkl")
            file2 = os.path.join(args.save_dir, f"{args.policy2}.pkl")
            
            if os.path.isfile(file1) and os.path.isfile(file2):
                demos = get_mixture_dataset(file1, file2, args.perc, args.N_trajectories)
            else:
                raise FileNotFoundError(f"To generate a mixture of Robosuite policies, two files with saved data for the two policies must exist at {file1} and {file2}!")
        else:
            env, policy1, policy2 = get_env_and_policies(args, device)
            interactive_robosuite1 = args.robosuite and args.policy == "user"
            interactive_robosuite2 = args.robosuite and args.policy2 == "user"
            num_demos1= int(args.perc * args.N_trajectories)
            num_demos2 = args.N_trajectories - num_demos1
            demos1 = sample(env, policy1, num_demos1, interactive_robosuite=interactive_robosuite1)
            demos2 = sample(env, policy2, num_demos2, interactive_robosuite=interactive_robosuite2)
            demos = demos1 + demos2
    else:
        env, policy1, policy2 = get_env_and_policies(args, device)
        interactive_robosuite = args.robosuite and args.policy == "user"
        demos = sample(env, policy1, args.N_trajectories, interactive_robosuite=interactive_robosuite)
        
    print("Data generated! Saving data...")

    with open(save_path, "wb") as f:
        pickle.dump(demos, f)
    print(f"Data saved to {save_path}!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
