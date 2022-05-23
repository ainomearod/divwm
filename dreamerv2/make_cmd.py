import argparse
import json
import os


def generate_train_cmds(params, num_trials=1, start_index=0, newlines=False, xpid_generator=None, xpid_prefix='', include_wandb_group=False):
    separator = ' \\\n' if newlines else ' '

    cmds = []

    if xpid_generator:
        params['xpid'] = xpid_generator(params)
        if include_wandb_group:
            params['wandb_group'] = params['xpid']

    start_seed = params['seed']

    for t in range(num_trials):
        params['seed'] = start_seed + t + start_index

        cmd = [f'python -m main']

        trial_idx = t + start_index
        for k, v in params.items():
            if k == 'xpid':
                v = f'{v}_{trial_idx}'
            if k == 'prefix':
                continue

            cmd.append(f'--{k}={v}')

        cmd = separator.join(cmd)

        cmds.append(cmd)

    return cmds


def generate_all_params_for_grid(grid, defaults={}):
    def update_params_with_choices(prev_params, param, choices):
        updated_params = []
        for v in choices:
            for p in prev_params:
                updated = p.copy()
                updated[param] = v
                updated_params.append(updated)

        return updated_params

    all_params = [{}]
    for param, choices in grid.items():
        all_params = update_params_with_choices(all_params, param, choices)

    full_params = []
    for p in all_params:
        d = defaults.copy()
        d.update(p)
        full_params.append(d)

    return full_params


def parse_args():
    parser = argparse.ArgumentParser(description='Make commands')

    parser.add_argument('--json', type=str, default=None, help='Name of .json config for hyperparameter search-grid')
    parser.add_argument('--dir', type=str, default="dreamerv2/train_configs", help='Loc of json config')

    parser.add_argument('--num_trials', type=int, default=1, help='Name of .json config for hyperparameter search-grid')

    parser.add_argument('--start_index', default=0, type=int, help='Starting trial index of xpid runs')

    parser.add_argument('--count', action='store_true', help='Print number of generated commands at the end of output.')

    parser.add_argument("--wandb_base_url", type=str, default=None, help='wandb base url')
    parser.add_argument("--wandb_api_key", type=str, default=None, help='wandb api key')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandb project name')

    parser.add_argument('--include_wandb_group', action="store_true", help='Whether to include wandb group in cmds.')

    return parser.parse_args()


def xpid_from_params(p):
    task = p["task"]
    method = p["method"]
    num_agents = p["num_agents"]
    expl_train_steps = p["explorer_train_steps"]
    expl_train_steps = p["explorer_train_steps"]
    model_train_steps = p["offline_model_train_steps"]
    batch_size = p["train_every"]
    eval_type = p["eval_type"]
    if eval_type == "labels":
        eval_type += str(p["task_train_steps"])
    reinit = "reinit" + p["explorer_reinit"]
    cascade_params = {"cascade_alpha": "alpha", "cascade_k": "k", "cascade_metric": "met", "cascade_proj": "proj", "cascade_states": "states", "cascade_feat": "feat", "cascade_average": "average", "cascade_scale": "scale", "cascade_sample": "sample"}
    if p["cascade_alpha"] > 0:
        cascade = ""
        for k,v in cascade_params.items():
            if k in p:
                cascade += f'_{v}{p[k]}'
    else:
        cascade = ""

    return f'{p["prefix"]}_{task}_{method}_{num_agents}_batchsize{batch_size}_explsteps{expl_train_steps}{reinit}_modelsteps{model_train_steps}_eval{eval_type}{cascade}'

if __name__ == '__main__':
    args = parse_args()

    # Default parameters
    params = {}

    json_filename = args.json
    if not json_filename.endswith('.json'):
        json_filename += '.json'

    grid_path = os.path.join(os.path.expandvars(os.path.expanduser(args.dir)), json_filename)
    config = json.load(open(grid_path))
    grid = config['grid']

    if args.wandb_base_url:
        params['wandb_base_url'] = args.wandb_base_url
    if args.wandb_api_key:
        params['wandb_api_key'] = args.wandb_api_key

    # Generate all parameter combinations within grid, using defaults for fixed params
    all_params = generate_all_params_for_grid(grid, defaults=params)

    if 'prefix' not in params:
        params['prefix'] = 'divwm'
    # Print all commands
    count = 0
    for p in all_params:
        cmds = generate_train_cmds(p, num_trials=args.num_trials, start_index=args.start_index, newlines=True, xpid_generator=xpid_from_params, xpid_prefix='', include_wandb_group=args.include_wandb_group)

        for c in cmds:
            print(c + '\n')
            count += 1

    if args.count:
        print(f'Generated {count} commands.')