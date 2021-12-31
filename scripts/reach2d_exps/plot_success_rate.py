import argparse
import matplotlib.pyplot as plt
import os
import pickle
DATA_SOURCES = ['oracle', 'pi_r', 'oracle_pi_r_mix']
NS=[50, 100, 200, 300, 400, 500, 750, 1000]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='LinearModel')
    parser.add_argument('--ckpt_file', type=str, default='model_4.pt')
    parser.add_argument('--date', type=str, default='dec28')
    parser.add_argument('--environment', type=str, default='Reach2D')
    parser.add_argument('--method', type=str, default='BC')
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('--seed', type=int, default=4)

    args= parser.parse_args()
    return args


def main(args):
    exp_name_arch = args.arch if args.num_models == 1 else 'Ensemble' + args.arch
    for data_source in DATA_SOURCES:
        successes = []
        for N in NS:
            exp_name = f'{args.date}/{args.environment}/{args.method}/{exp_name_arch}/eval/{data_source}_N{N}_seed{args.seed}' 
            exp_dir = os.path.join('./out', exp_name)
            data_file = os.path.join(exp_dir, 'eval_auto_data.pkl')

            with open(data_file, 'rb') as f:
                data = pickle.load(f)
            num_successes = sum([traj['success'] for traj in data])
            successes.append(num_successes)
        plt.plot(NS, successes, marker='o', label=data_source)
    plt.xlabel('N')
    plt.ylabel('# successes')
    plt.ylim(0, 105)
    plt.title(f'{args.method} w/ {exp_name_arch} Auto-Only Rollout')
    plt.legend()
    save_path = f'./out/{args.date}/{args.environment}/{args.method}/{exp_name_arch}/eval/successes_vs_N.png'
    plt.savefig(save_path)
    print(f'Plot saved to {save_path}')

if __name__ == '__main__':
    args = parse_args()
    main(args)
