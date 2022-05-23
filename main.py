
import gym
import crafter
import dreamerv2.api as dv2
import argparse
from utils import str2bool
from dreamerv2 import common
from dreamerv2.train import run

## TODO
# wandb
# offline
# lexa
# make sure no reward used
# make sure dep efficient setting

def main(args):

    ## get defaults
    config = dv2.defaults
    if args.task: 
        if 'crafter' in args.task:
            config = config.update(dv2.configs['crafter'])
        elif 'minigrid' in args.task:
            config = config.update(dv2.configs['minigrid'])
        elif 'atari' in args.task:
            config = config.update(dv2.configs['atari'])
        elif 'dmc' in args.task:
            config = config.update(dv2.configs['dmc_vision'])

    params = vars(args)
    config = config.update(params)

    ## this will likely always be true
    config = config.update({
        # 'expl_behavior': 'Plan2Explore',
        'pred_discount': False,
        'grad_heads': ['decoder'], # this means we dont learn the reward head
        'expl_intr_scale': 1.0,
        'expl_extr_scale': 0.0,
        'discount': 0.99,
    })

    run(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')

    # Main Arguments.
    parser.add_argument('--xpid', type=str, default=None, help='experiment id')
    parser.add_argument('--steps', type=int, default=1e6, help='number of environment steps to train')
    parser.add_argument('--train_every', type=int, default=1e5, help='number of environment steps to train')
    parser.add_argument('--offline_model_train_steps', type=int, default=25001,  help='=250 * train_every (in thousands) + 1. Default assumes 100k.')
    parser.add_argument('--offline_model_save_every', type=int, default=1000000,  help='=250 * train_every (in thousands) + 1. Default assumes 100k.')
    parser.add_argument('--explorer_train_steps', type=int, default=1e5,  help='Inside the WM')
    parser.add_argument('--explorer_reinit', type=str2bool, nargs='?', const=True, default=False, help='whether to re-initialize the explorers.')
    parser.add_argument('--task', type=str, default='crafter_noreward', help='environment to train on')
    parser.add_argument('--logdir', default='~/wm_logs/', help='directory to save agent logs')
    parser.add_argument('--checkpoint', type=str2bool, nargs='?', const=True, default=False, help='whether to checkpoint.')
    parser.add_argument('--method', type=str, default='single_disag', choices=['single_disag', 'multihead_disag', 'pop_div_disag', 'pop_div'], help='Which exploration method.')
    parser.add_argument('--num_agents', type=int, default=1,  help='Exploration Population size.')
    parser.add_argument('--fix_seed', type=str2bool, nargs='?', const=True, default=False, help='whether to reset env with fixed seed.')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--envs', type=int, default=1,  help='Number of parallel envs.')
    parser.add_argument('--envs_parallel', type=str, default="none",  help='How to parallelize.')
    parser.add_argument('--eval_envs', type=int, default=1,  help='Number of parallel eval envs.')
    parser.add_argument('--eval_eps', type=int, default=100,  help='Number of eval eps.')
    parser.add_argument('--eval_type', type=str, default='coincidental',  help='How to evaluate the model.')
    parser.add_argument('--task_train_steps', type=int, default=1e5,  help='Inside the WM, if task=="labels"')
    parser.add_argument('--expl_behavior', type=str, default='Plan2Explore',  help='Explore behavior.')
    parser.add_argument('--load_pretrained', type=str, default='none', help='name of pretrained model')
    parser.add_argument('--offline_dir', type=str, default='none', help='directory to load offline dataset')
    parser.add_argument('--replay_dir', type=str, default='none', help='directory to load train/eval episodes.')
    parser.add_argument('--load_wm', type=str, default='none', help='path to load pretrained wm.')
    parser.add_argument('--offline_lmbd', type=int, default=10,  help='Lmbd.')

    # CASCADE
    parser.add_argument('--cascade_alpha', type=float, default=0,  help='Cascade weight.')
    parser.add_argument('--cascade_metric', type=str, default="euclidean",  help='Cascade metric.')
    parser.add_argument('--cascade_feat', type=str, default="deter",  help='Cascade features if state based.')
    parser.add_argument('--cascade_proj', type=int, default=0,  help='Cascade projection, default is zero which means none.')
    parser.add_argument('--cascade_states', type=str, default="all",  help='Cascade states, we can either use the entire trajectory or just final state etc.')
    parser.add_argument('--cascade_k', type=int, default=5,  help='number of nearest neighbors to use in the mean dist.')
    parser.add_argument('--cascade_average', type=str2bool, nargs='?', const=True, default=False, help='Average cascade rewards over num agents before.')
    parser.add_argument('--cascade_scale', type=float, default=1.0,  help='Cascade scale.')
    parser.add_argument('--cascade_sample', type=int, default=10,  help='max number of cascade states')


    # Logging arguments.
    parser.add_argument("--wandb_silent", type=str2bool, nargs='?', const=True, default=False, help="Disable wandb logging")
    parser.add_argument("--wandb_base_url", type=str, default='https://api.wandb.ai', help='wandb base url')
    parser.add_argument("--wandb_api_key", type=str, default=None, help='wandb api key')
    parser.add_argument("--wandb_entity", type=str, default='divwm', help='Team name')
    parser.add_argument("--wandb_project", type=str, default='dmc', help='wandb project name for logging')
    parser.add_argument("--wandb_group", type=str, default=None, help='wandb group name for logging')

    args = parser.parse_args()
    main(args)
