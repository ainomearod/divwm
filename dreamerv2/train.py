import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings
import pickle
import wandb
import time
from collections import defaultdict
from scipy.stats import gmean
import math
from PIL import Image
import matplotlib
# Use Agg backend to run matplotlib headlessly
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DMC_TASK_IDS = {
    'dmc_walker_all': ['stand', 'walk', 'run', 'flip'],
    'dmc_cheetah_all': ['run-fwd', 'run-bwd', 'flip-fwd', 'flip-bwd'],
}

try:
  import rich.traceback
  rich.traceback.install()
except ImportError:
  pass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import agent
import common

def run(config):

  if config.wandb_base_url:
    os.environ["WANDB_BASE_URL"] = config.wandb_base_url
  if config.wandb_api_key:
    os.environ["WANDB_API_KEY"] = config.wandb_api_key

  while True:
    try:
      wandb.init(project=config.wandb_project,
                 entity="divwm",
                 config=config,
                 name=config.xpid)
      break
    except:
      os.environ["WANDB_START_METHOD"] = "thread"
      print("\nRetrying wandb")
      time.sleep(10)

  logdir = pathlib.Path(config.logdir + config.xpid).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  MAX_EVAL_ENVS = 100

  ## setting up the GPU
  import tensorflow as tf
  tf.config.experimental_run_functions_eagerly(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print(message)
  else:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  ## load the stats that we keep track of
  if (logdir / 'stats.pkl').exists():
    stats = pickle.load(open(f"{logdir}/stats.pkl", "rb"))
    print("\nLoaded stats: \n")
    print(stats, flush=True)
  else:
    stats = {
      'num_deployments': 0,
      'num_wm_trainings': 0,
      'num_expl_trainings': 0,
      'num_evals': 0
    }
    
    stats['should_train_reward_heads'] = 1
    stats['should_train_behaviors'] = {}
    tasks = ['']
    if config.task in DMC_TASK_IDS:
      tasks = DMC_TASK_IDS[config.task]
    for behave in tasks:
      stats['should_train_behaviors'][behave] = 1
    pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))

  multi_reward = config.task in ['dmc_walker_all', 'dmc_cheetah_all']
  replay_dir = logdir / 'train_episodes'
  if config.replay_dir != 'none':
    replay_dir = pathlib.Path(config.replay_dir).expanduser()
  
  ## load dataset - even if using offline we dont want to load offline *again8 if we have already deployed
  if config.offline_dir == 'none' or stats['num_deployments'] > 0:
    train_replay = common.Replay(replay_dir, offline_init=False,
      multi_reward=multi_reward, **config.replay)
  else:
    train_replay = common.Replay(replay_dir, offline_init=True,
      multi_reward=multi_reward, offline_directory=config.offline_dir, **config.replay)

  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length,
      multi_reward=multi_reward))
  step = common.Counter(train_replay.stats['total_steps'])
  outputs = [
      common.TerminalOutput(),
      common.JSONLOutput(logdir),
      common.TensorBoardOutput(logdir),
  ]
  logger = common.Logger(step, outputs, multiplier=config.action_repeat)
  metrics = collections.defaultdict(list)

  def make_env(mode, seed=1):
    if '_' in config.task:
      suite, task = config.task.split('_', 1)
    else:
      suite, task = config.task, ''
    if suite == 'dmc':
      env = common.DMC(
          task, config.action_repeat, config.render_size, config.dmc_camera)
      env = common.NormalizeAction(env)
    elif suite == 'atari':
      env = common.Atari(
          task, config.action_repeat, config.render_size,
          config.atari_grayscale, life_done=False) # do not terminate on life loss
      env = common.OneHotAction(env)
    elif suite == 'crafter':
      assert config.action_repeat == 1
      outdir = logdir / 'crafter' if mode == 'train' else None
      reward = bool(['noreward', 'reward'].index(task)) or mode == 'eval'
      env = common.Crafter(outdir, reward)
      env = common.OneHotAction(env)
    elif suite == 'minigrid':
      # fixed seed applies to eval envs only
      if mode == 'eval':
        env = common.make_minigrid_env(task, fix_seed=True, seed=seed, num_agents=config.num_agents)
      else:
        env = common.make_minigrid_env(task, fix_seed=False, seed=None, num_agents=config.num_agents)
    elif suite == 'kitchen':
      env = common.KitchenEnv(config.action_repeat, use_goal_idx=False, log_per_goal=False)
      env = common.NormalizeAction(env)
    else:
      raise NotImplementedError(suite)
    env = common.TimeLimit(env, config.time_limit)
    return env

  def per_episode(ep, mode, task='none'):
    length = len(ep['reward']) - 1
    if task in DMC_TASK_IDS:
      scores = {
        key: np.sum([val[idx] for val in ep['reward'][1:]])
        for idx, key in enumerate(DMC_TASK_IDS[task])}
      print_rews = f'{mode.title()} episode has {length} steps and returns '
      print_rews += ''.join([f"{key}:{np.round(val,1)} " for key,val in scores.items()])
      print(print_rews)
      for key,val in scores.items():
        logger.scalar(f'{mode}_return_{key}', val)
    else:
      score = float(ep['reward'].astype(np.float64).sum())
      print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
      logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    replay = dict(train=train_replay, eval=eval_replay)[mode]
    logger.add(replay.stats, prefix=mode)
    logger.write()
  
  def get_stats_at_idx(driver, task, idx):
    prefix = "eval_"
    eps = driver._eps[idx]
    env = driver._envs[idx]
    eval_data = defaultdict(list)
    if task in ['crafter_noreward']:
      for ep in eps:
        for key, val in ep.items():
          if 'log_achievement_' in key:
            eval_data[prefix + 'rew_'+key.split('log_achievement_')[1]].append(val.item())
            eval_data[prefix + 'sr_'+key.split('log_achievement_')[1]].append(1 if val.item() > 0 else 0)

        eval_data['reward'].append(ep['log_reward'].item())
      eval_data = {key: np.mean(val) for key, val in eval_data.items()}
      eval_data[prefix + 'crafter_score'] = gmean([val for key, val in eval_data.items() if 'eval_sr' in key])

    elif 'minigrid' in task:
      rewards = [ep['reward'].item() for ep in eps]
      eval_data[prefix + 'mean_reward'] = np.mean(rewards)
      eval_data[prefix + 'coverage'] = np.sum([1 if r > 0 else 0 for r in rewards])/len(rewards)
      
      ## State_count and episodic_state_count are dict of form: (x, y): count, e.g. (1,2): 5
      ## total_state_count is int
      state_count = common.get_minigrid_attr(env, 'state_count')
      total_state_count = common.get_minigrid_attr(env, 'total_state_count')
      episodic_state_count = common.get_minigrid_attr(env, 'episodic_state_count')
      eval_data[prefix + 'state_count'] = state_count
      eval_data[prefix + 'total_state_count'] = total_state_count
      eval_data[prefix + 'episodic_state_count'] = episodic_state_count

      state_count_agent = common.get_minigrid_attr(env, 'state_count_agent')
      episodic_state_count_agent = common.get_minigrid_attr(env, 'episodic_state_count_agent')
      for j in range(config.num_agents):
        eval_data[prefix + f'state_count_agent{j}'] = state_count_agent[j]
        eval_data[prefix + f'total_state_count_agent{j}'] = total_state_count
        eval_data[prefix + f'episodic_state_count_agent{j}'] = episodic_state_count_agent[j]

      state_count_per_room = common.get_minigrid_attr(env, 'state_count_per_room')
      total_state_count_per_room = common.get_minigrid_attr(env, 'total_state_count_per_room')
      episodic_state_count_per_room = common.get_minigrid_attr(env, 'episodic_state_count_per_room')
      for j in range(len(state_count_per_room)):
        state_count = state_count_per_room[j]
        episodic_state_count = episodic_state_count_per_room[j]
        eval_data[prefix + f"state_count_room{j}"] = state_count
        eval_data[prefix + f"episodic_state_count_room{j}"] = episodic_state_count
        eval_data[prefix + f"total_state_count_room{j}"] = total_state_count_per_room[j]
      
      if idx < 10:
        visitation = common.get_minigrid_attr(env, 'visitation')
        mask = np.where(visitation < 1, True, False)
        img = sns.heatmap(visitation, alpha = 1.0, mask=mask, zorder = 2, xticklabels=False, yticklabels=False, linecolor='white', cmap='Reds')
        background = common.get_minigrid_attr(env, 'img')
        img.imshow(background, aspect = img.get_aspect(),
            extent = img.get_xlim() + img.get_ylim(),
            zorder = 1)
        img_save_path = logdir / "heatmaps"
        img_save_path.mkdir(parents=True, exist_ok=True)
        seed = common.get_minigrid_attr(env, 'seed')
        img.get_figure().savefig(f"{img_save_path}/eval_visitation_heatmap_seed{seed}_step{step.value}.png")
        img.get_figure().clf()

        ## Save visitation counts
        stat_save_path = logdir / "visitation_counts"
        stat_save_path.mkdir(parents=True, exist_ok=True)
        np.save(f"{stat_save_path}/eval_visitation_counts_seed{seed}_step{step.value}.npy", visitation)
    elif task in DMC_TASK_IDS:
      rewards = [ep['reward'] for ep in eps[1:]]
      for idx, goal in enumerate(DMC_TASK_IDS[task]):
        eval_data[prefix + 'reward_' + goal] = np.sum([r[idx] for r in rewards])

    return eval_data

  def get_stats(driver, task):
    num_envs = len(driver._eps)
    per_env_data = defaultdict(list)
    for i in range(num_envs):
      stat = get_stats_at_idx(driver, task, i)
      for k, v in stat.items():
        # per_env_data[f"env{i}_"+k] = v
        per_env_data[k].append(v)

    data = {}
    for k, v in per_env_data.items():
      if "state_count" in k:
        # list of dict / list of int (for total state count)
        data[k] = v
      else:
        data[k] = np.mean(v)
    return data

  def eval(driver):
    ## reward for the exploration agents
    mets = {}
    mean_pop = {}
    for idx in range(config.num_agents):
      policy = lambda *args: agnt.policy(*args, policy_idx=idx, mode='explore')
      driver(policy, episodes=config.eval_eps, policy_idx=idx, save_img=False) # whether to save a snapshot of the image obs
      data = get_stats(driver, task=config.task)
    
      ## Log per agent stats except for state counts
      # wandb_data.update({f'agent{idx}_'+key: np.mean(val) for key,val in data.items() if "state_count" not in key})
      
      if idx == 0:
        for key, val in data.items():
          if "state_count" in key:
            # val is a list of dict
            mean_pop[key] = val
          # elif "state_entropy" in key or "state_coverage" in key:
          #   continue
          else:
            mean_pop[key] = np.mean(val)
      else:
        for key,val in data.items():
          if "total_state_count" in key:
            # the total state count for each env remains unchanged
            continue
          elif "state_count" in key:
            # merge list of dict
            for idx, cnt in enumerate(val):
              for k, v in cnt.items():
                mean_pop[key][idx][k] += v
          else:
            mean_pop[key] += np.mean(val)

    for k, v in mean_pop.items():
      if "total_state_count" in k:
        continue
      if "state_count" in k:
        # calculate state coverage and entropy
        # v is a list of state count dict (one for each env)
        state_count = [vv.values() for vv in v]
        normalized_state_count = [[float(i)/sum(sc) for i in sc] for sc in state_count]
        entropy_metric_name = k.replace("state_count", "state_entropy")
        mets.update({entropy_metric_name: np.mean([sum([-i*math.log(i, 2) for i in nsc]) for nsc in normalized_state_count])})
        coverage_metric_name = k.replace("state_count", "state_coverage")
        total_state_count_metric_name = k.replace("state_count", "total_state_count")
        total_state_count_metric_name = total_state_count_metric_name.replace("episodic_", "")
        # total state count is a list of int, one for each env
        total_state_count = mean_pop[total_state_count_metric_name]
        mets.update({coverage_metric_name: np.mean([len(state_count[i])*1.0 / total_state_count[i] for i in range(len(total_state_count))])})

    mets.update({key: np.mean(val) for key, val in mean_pop.items() if "state_count" not in key})
    return mets

  print('Create envs.')
  # For now let's limit the number of eval envs to 100
  num_eval_envs = min(config.eval_envs, MAX_EVAL_ENVS)
  if config.envs_parallel == 'none':
    train_envs = [make_env('train') for _ in range(config.envs)]
    if 'minigrid' in config.task and config.eval_type != 'labels':
      eval_envs = [make_env('eval', seed=s) for s in range(1, num_eval_envs+1)]
      if (logdir / 'eval_envs_stats.pkl').exists():
        common.load_minigrid_stats(eval_envs, logdir)
        print("\n Loaded eval env stats.")
    else:
      eval_envs = [make_env('eval') for _ in range(num_eval_envs)]
  else:
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode), config.envs_parallel)
    train_envs = [make_async_env('train') for _ in range(config.envs)]
    eval_envs = [make_async_env('eval') for _ in range(num_eval_envs)]
    obs_space = train_envs[0].access('obs_space')()
  act_space = train_envs[0].act_space
  obs_space = train_envs[0].obs_space
  train_driver = common.Driver(train_envs)
  train_driver.on_episode(lambda ep: per_episode(ep, mode='train', task=config.task))
  train_driver.on_step(lambda tran, worker: step.increment())
  train_driver.on_step(train_replay.add_step)
  train_driver.on_reset(train_replay.add_step)
  train_driver.reset()
  eval_driver = common.Driver(eval_envs)
  eval_driver.on_episode(eval_replay.add_episode)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval', task=config.task))
  eval_driver.reset()

  if stats['num_deployments'] == 0:
    if config.offline_dir == 'none':
      prefill = max(0, config.train_every - train_replay.stats['total_steps'])
      if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        ##begin with random agent - we may want to change this to a URL agent.
        random_agent = common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1, policy_idx=-1)
        train_driver.reset()

        eval_driver(random_agent, episodes=1, policy_idx=-1)
        eval_data = get_stats(eval_driver, task=config.task)
        wandb_data= {}
        wandb_data.update({key: np.mean(val) for key, val in eval_data.items() if "state_count" not in key})
        wandb_data.update({'steps': 0})
        wandb.log(wandb_data)
        eval_driver.reset()
    # either we already had an offline dataset or we just collected random data.. so 1 deployment!
    stats['num_deployments'] += 1
  train_dataset = iter(train_replay.dataset(**config.offline_model_dataset))
  if config.offline_split_val:
      val_dataset = iter(train_replay.dataset(validation=True, **config.offline_model_dataset))

  print('Create agent.')
  agnt = agent.Agent(config, obs_space, act_space, step)
  load_pretrained = config.load_pretrained
  if config.load_pretrained != "none":
    try:
        train_agent = common.CarryOverState(agnt.train)
        train_agent(next(train_dataset))
        
        agnt._expl_behavior.intr_rewnorm_cascade = None

        print("\nLoading pretrained model...")
        path = pathlib.Path('~/divwm/dreamerv2/').expanduser()
        name = config.load_pretrained + str(config.seed) + '.pkl'
        agnt.load(path / name)
        agnt._expl_behavior.intr_rewnorm_cascade = [common.StreamNorm(**config.expl_reward_norm) for _ in range(config.num_agents)]

        ## Assume we've done 1 full cycle 
        stats = {
          'num_deployments': 1,
          'num_wm_trainings': 1,
          'num_expl_trainings': 1,
          'num_evals': 1
        }
        if config.task in DMC_TASK_IDS:
          stats['should_train_reward_heads'] = 1
          stats['should_train_behaviors'] = {}
          for behave in DMC_TASK_IDS[config.task]:
            stats['should_train_behaviors'][behave] = 1
        print("\nSuccessfully loaded pretrained model.")
    except:
        load_pretrained = "none"
        print("\nUnable to load pretrained model.")

  if load_pretrained == "none":
    if config.explorer_reinit:
      print("\nReinit explorers...")
      mets = agnt.model_train(next(train_dataset)) # init model
      mets = agnt.agent_train_eval(next(train_dataset)) # init behavior policy
      if (logdir / 'variables.pkl').exists():
        try:
          agnt.load(logdir / 'variables.pkl')
        except:
          mets = agnt.agent_train_expl(next(train_dataset)) # init explorers
          agnt.load(logdir / 'variables.pkl')
    else:
      train_agent = common.CarryOverState(agnt.train)
      train_agent(next(train_dataset))
      if (logdir / 'variables.pkl').exists():
        print("\n\nStart loading model checkpoint...")
        agnt.load(logdir / 'variables.pkl')
      print("\nFinished init agent.")
  
  if config.load_wm != "none":
    try:
      print("\n\nLoading pretrained wm...")
      path = pathlib.Path(config.load_wm).expanduser()
      agnt.wm.load(path)
      print("\nSuccessfully loaded pretrained wm.")
      stats = {
          'num_deployments': 1,
          'num_wm_trainings': 1,
          'num_expl_trainings': 0,
          'num_evals': 0
      }
    except:
      print("\nUnable to load pretrained wm.")
  

  """
  ## each loop we do one of the following:
  1. deploy explorers to collect data
  2. train WM
  3. train explorers WM  
  4. evaluate WM
  """
  while step < config.steps:

    #this is for debugging the explore step only
    #stats = {'num_deployments': 1, 'num_wm_trainings': 1, 'num_expl_trainings': 0, 'num_evals': 0}

    wandb_data = {}
    should_train_explorers = stats['num_expl_trainings'] < stats['num_wm_trainings']
    should_deploy = stats['num_deployments'] <= stats['num_evals']
    should_train_wm = stats['num_wm_trainings'] < stats['num_deployments']
    should_eval = stats['num_evals'] < stats['num_expl_trainings']

    assert should_train_explorers + should_deploy + should_train_wm + should_eval == 1

    if should_deploy:
      print("\n\nStart collecting data...", flush=True)
      ## collect a batch of steps with the expl policy
      ## need to increment steps here
     
      num_steps = int(config.train_every / config.num_agents)
      for idx in range(config.num_agents):
        expl_policy = lambda *args: agnt.policy(*args, policy_idx=idx, mode='explore')
        train_driver(expl_policy, steps=num_steps, episodes=1, policy_idx=idx)
   
      train_driver.reset()
      wandb_data.update({'steps': step.value, 'running_score': train_replay.stats['running_score'], 'solved_levels': train_replay.stats['solved_levels'], 'max_scores': train_replay.stats['max_scores'], 'mean_scores': train_replay.stats['mean_scores']})
      print(f"\nsteps: {step.value}, running_score: {train_replay.stats['running_score']}, solved_levels: {train_replay.stats['solved_levels']}, max_scores: {train_replay.stats['max_scores']},  mean_scores: {train_replay.stats['mean_scores']}\n",flush=True)
      if config.task == "crafter_noreward":
        for key, val in train_replay.achievements.items():
          print(f"*{key}: {np.mean(val)}")

      stats['num_deployments'] += 1

    elif should_eval:
      ## train eval agents and then eval them
      print('\n\nStart evaluation...', flush=True)
      if config.eval_type == 'coincidental':
        eval_logger = common.Logger(0, outputs)
        mets = eval(eval_driver)
        wandb_data.update(mets)
        wandb_data.update({'steps': step.value})
        for name, values in mets.items():
          eval_logger.scalar(name, np.array(values, np.float64).mean())
        eval_logger.write()

      elif config.eval_type == 'labels':
        ## Here we do a three step process:
        ## 1. Train reward heads, keeping all other layers frozen
        ## 2. Train a task specific policy per head
        ## 3. Evaluate this policy on the downstream task, zero shot
        if 'should_train_reward_heads' not in stats or stats['should_train_reward_heads'] == 1:
          print("\n\nStart Training Reward Heads..", flush=True)
          model_train_data = defaultdict(list)
          model_step = common.Counter(0)
          model_logger = common.Logger(model_step, outputs)
          while model_step < config.offline_model_train_steps:
            for it in range(20):
              model_step.increment()
              mets = agnt.model_reward_train(next(train_dataset))

              for key, value in mets.items():
                model_train_data[key].append(value)

            for name, values in mets.items():
              model_logger.scalar(name, np.array(values, np.float64).mean())
              if model_step <= 20:
                model_train_data['begin_' + name] = np.array(values, np.float64).mean()
              else:
                model_train_data['end_' + name] = np.array(values, np.float64).mean()
              # mets[name].clear()
            model_logger.write()
          wandb_data.update({key: np.mean(val) for key, val in model_train_data.items()})
          wandb_data.update({'steps': step.value})

          # saves models
          stats['should_train_reward_heads'] = 0
          pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))
          wandb.log(wandb_data)
          agnt.save(logdir / 'variables.pkl')

        print("\n\nTrained Reward Heads.", flush=True)

        print("\n\nStart Training Behavior Policies..", flush=True)
        behav_train_data = {}
        tasks = ['']
        if config.task in DMC_TASK_IDS:
          tasks = DMC_TASK_IDS[config.task]
        for idx, task in enumerate(tasks):
          if 'should_train_behaviors' not in stats or task not in stats['should_train_behaviors'] or stats['should_train_behaviors'][task] == 1:
            print("\nStart Training "+ task)
            train_dataset = iter(train_replay.dataset(**config.offline_train_dataset))
            behav_step = common.Counter(0)
            behav_logger = common.Logger(behav_step, outputs)
            while behav_step < config.task_train_steps:
              for it in range(20):
                behav_step.increment()
              
                mets = agnt.agent_train_eval(next(train_dataset), goal=task)
                for key, value in mets.items():
                  metrics[key].append(value)

              for name, values in mets.items():
                behav_logger.scalar(name, np.array(values, np.float64).mean())
              behav_logger.write()

            print("\n\nStart Evaluating " + task, flush=True)
            eval_policy = lambda *args: agnt.policy(*args, mode='eval', goal=task)
            eval_driver(eval_policy, episodes=config.eval_eps)
            eval_data = get_stats(eval_driver, task=config.task)
            rew = eval_data["eval_reward_" + task] if task != '' else eval_data["eval_reward"]
            behav_train_data['eval_reward_' + task] = rew
            wandb_data.update({"eval_reward_" + task: np.mean(rew)})
            wandb_data.update({'steps': step.value})
            #behav_train_data['wm_perf'] =  ** would be good to add this

            if 'should_train_behaviors' not in stats:
              stats['should_train_behaviors'] = {}
            stats['should_train_behaviors'][task] = 0
            # saves models
            # Skip this for the last behavior because we're doing it at the end of the main loop
            if idx < len(tasks) - 1:
              pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))
              wandb.log(wandb_data)
              agnt.save(logdir / 'variables.pkl')
          print("\nTrained " + task)

        
        # reset stats
        stats['should_train_reward_heads'] = 1
        for behave in tasks:
          stats['should_train_behaviors'][behave] = 1

      elif config.eval_type == 'none':
        pass
      else:
        eval_policy = lambda *args: agnt.policy(*args, mode='eval')
        eval_driver(eval_policy, episodes=config.eval_eps)
      eval_driver.reset()
      stats['num_evals'] += 1

      # saves extra copy of the first fully trained model
      if stats['num_evals'] == 1:
        agnt.save(logdir / 'pretrained_variables.pkl')

    elif should_train_wm:
      print('\n\nStart model training...', flush=True)
      model_train_data = {}
      model_step = common.Counter(0)
      model_logger = common.Logger(model_step, outputs)
      should_pretrain = (stats['num_wm_trainings'] == 0 and config.offline_dir != "none")
      if should_pretrain:
        # if retrain, use all offline data
        batch_size = config.offline_model_dataset["batch"] * config.offline_model_dataset["length"]
        model_train_steps = train_replay._loaded_steps // batch_size - 1
      else:
        model_train_steps = config.offline_model_train_steps
      while model_step < model_train_steps:
        if config.offline_split_val:
          # Compute model validation loss as average over chunks of data
          val_losses = []
          for _ in range(10):
            val_loss, _, _, val_mets = agnt.wm.loss(next(val_dataset))
            val_losses.append(val_loss)
          model_logger.scalar(f'validation_model_loss', np.array(val_losses, np.float64).mean())

        for it in range(20):
          model_step.increment()
          mets = agnt.model_train(next(train_dataset))

          for key, value in mets.items():
            metrics[key].append(value)

        for name, values in metrics.items():
          model_logger.scalar(name, np.array(values, np.float64).mean())
          if model_step <= 20:
            model_train_data['begin_' + name] = np.array(values, np.float64).mean()
          else:
            model_train_data['end_' + name] = np.array(values, np.float64).mean()
          metrics[name].clear()
        model_logger.write()
      wandb_data.update({key: np.mean(val) for key, val in model_train_data.items()})
      wandb_data.update({'steps': step.value})
      stats['num_wm_trainings'] += 1
      if stats['num_deployments'] * config.train_every % config.offline_model_save_every == 0:
        print('Saving model')
        agnt.wm.save(logdir / f'model_{stats["num_deployments"] * config.train_every}.pkl')

    elif should_train_explorers:
      print("\n\nStart training Explorers..", flush=True)
      ## this trains the exploration policies
      if config.explorer_reinit:
        agnt.re_init_explorers()
      expl_train_data = {}
      train_dataset = iter(train_replay.dataset(**config.offline_train_dataset))
      # eval_dataset = iter(eval_replay.dataset(**config.offline_train_dataset))
      expl_step = common.Counter(0)
      expl_logger = common.Logger(expl_step, outputs)
      while expl_step < config.explorer_train_steps:

        for it in range(20):
          expl_step.increment()
          mets = agnt.agent_train_expl(next(train_dataset))

          for key, value in mets.items():
            metrics[key].append(value)
        for name, values in metrics.items():
          expl_logger.scalar(name, np.array(values, np.float64).mean())
          if expl_step <= 20:
            expl_train_data['begin_' + name] = np.array(values, np.float64).mean()
          else:
            expl_train_data['end_' + name] = np.array(values, np.float64).mean()
          metrics[name].clear()
        expl_logger.write()
      wandb_data.update({key: np.mean(val) for key, val in expl_train_data.items()})
      wandb_data.update({'steps': step.value})
      stats['num_expl_trainings'] += 1

    # saves WM + eval + expl policies
    pickle.dump(stats, open(f"{logdir}/stats.pkl", "wb"))
    wandb.log(wandb_data)
    agnt.save(logdir / 'variables.pkl')

    # saves eval envs
    if 'minigrid' in config.task:
      common.save_minigrid_stats(eval_envs, logdir)
   


  for env in train_envs + eval_envs:
    try:
      env.close()
    except Exception:
      pass
