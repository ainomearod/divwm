import atexit
import os
import sys
import threading
import traceback
import pickle

import cloudpickle
import gym
import numpy as np

from collections import defaultdict

from .cdmc import make_dmc_all
from .rendering import render_minigrid
from .recorder import Recorder

class Room:
  def __init__(self, top, size):
    self.top = top
    self.size = size
    self.doors = [None]

class StateCountStats:
  def __init__(self, env_height, env_width, num_rooms, num_agents):
      self.visitation = np.zeros((env_height, env_width))
      self.episodic_visitation = np.zeros((env_height, env_width))
      self.state_count = defaultdict(int)
      self.episodic_state_count = defaultdict(int)
      self.total_state_count = 0
      self.total_state_count_per_room = [0 for _ in range(num_rooms)]
      # self._set_total_state_count()
      self.state_count_per_room = [defaultdict(int) for _ in range(num_rooms)]
      self.episodic_state_count_per_room = [defaultdict(int) for _ in range(num_rooms)]

      ## Agent Stats
      self.visitation_agent = [np.zeros((env_height, env_width)) for _ in range(num_agents)]
      self.episodic_visitation_agent = [np.zeros((env_height, env_width)) for _ in range(num_agents)]
      self.state_count_agent = [defaultdict(int) for _ in range(num_agents)]
      self.episodic_state_count_agent = [defaultdict(int) for _ in range(num_agents)]


class GymWrapper:

  def __init__(self, env, obs_key='image', act_key='action'):
    self._env = env
    self._obs_is_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_is_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    if self._obs_is_dict:
      spaces = self._env.observation_space.spaces.copy()
    else:
      spaces = {self._obs_key: self._env.observation_space}
    return {
        **spaces,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    if self._act_is_dict:
      return self._env.action_space.spaces.copy()
    else:
      return {self._act_key: self._env.action_space}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    if not self._act_is_dict:
      action = action[self._act_key]
    obs, reward, done, info = self._env.step(action, policy_idx)
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = float(reward)
    obs['is_first'] = False
    obs['is_last'] = done
    obs['is_terminal'] = info.get('is_terminal', done)
    return obs

  def reset(self):
    obs = self._env.reset()
    if not self._obs_is_dict:
      obs = {self._obs_key: obs}
    obs['reward'] = 0.0
    obs['is_first'] = True
    obs['is_last'] = False
    obs['is_terminal'] = False
    return obs

def make_minigrid_env(task, fix_seed, seed, num_agents=1):
  import gym_minigrid
  env = gym.make("MiniGrid-"+task)
  env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
  # env = gym_minigrid.wrappers.RGBImgObsWrapper(env)

  minigrid_env_type = 'default'
  if 'MultiRoom' in task:
    minigrid_env_type = 'multiroom'
  elif 'FourRooms' in task:
    minigrid_env_type = 'fourroom'
  elif 'DoorKey' in task or 'KeyCorridor' in task or 'ObstructedMaze' in task:
    minigrid_env_type = 'roomgrid'

  if fix_seed:
    env = gym_minigrid.wrappers.ReseedWrapper(env, seeds=[seed])
    
  env = StateCountWrapper(env, minigrid_env_type, num_agents)

  env = GymWrapper(env)
  env = ResizeImage(env)
  if hasattr(env.act_space['action'], 'n'):
    env = OneHotAction(env)
  else:
    env = NormalizeAction(env)
  return env

def save_minigrid_stats(envs, logdir):
  stats_array = [env._env._env._env.stats for env in envs]
  pickle.dump(stats_array, open(f"{logdir}/eval_envs_stats.pkl", "wb"))

def load_minigrid_stats(envs, logdir):
   if (logdir / 'eval_envs_stats.pkl').exists():
     stats_array = pickle.load(open(f"{logdir}/eval_envs_stats.pkl", "rb"))
     for i, env in enumerate(envs):
       env._env._env._env.stats = stats_array[i]

def get_minigrid_attr(env, attr):
  wrapped = env._env._env._env.stats
  if attr in ["img", "seed", "goal", "orig_img"]:
    wrapped = env._env._env._env
  assert hasattr(wrapped, attr)
  return getattr(wrapped, attr)


class StateCountWrapper(gym.core.Wrapper):

  def __init__(self, env, minigrid_env_type='default', num_agents=1):
    super().__init__(env)

    try:
      self.seed = env.seeds[0]
    except:
      self.seed = 9999
    
    self.num_agents = num_agents
    
    """
    Room:
      top: (topX, topY)
      size: (sizeX, sizeY)
      entryDoorPos: (x, y)
      exitDoorPos: (x, y)
    """
    self.minigrid_env_type = minigrid_env_type
    if minigrid_env_type == 'multiroom':
      # multiroom envs 
      self.rooms = self.env.rooms
    # TODO: fill these
    elif minigrid_env_type == 'fourroom':
      # four room envs
      room_width = self.env.width // 2
      room_height = self.env.height // 2
      self.rooms = [
        Room((0,0), (room_width, room_height)),
        Room((room_width, 0), (room_width, room_height)),
        Room((0, room_height), (room_width, room_height)),
        Room((room_width, room_height), (room_width, room_height))
      ]
    elif minigrid_env_type == 'roomgrid':
      # room grid envs
      self.rooms = np.array(self.env.room_grid).flatten()
    else:
      # 1 room envs
      self.rooms = [Room((0,0), (self.env.width, self.env.height))]

    ## Global Stats
    self.stats = StateCountStats(self.env.height, self.env.width, len(self.rooms), self.num_agents)
    # self.visitation = np.zeros((self.env.height, self.env.width))
    # self.episodic_visitation = np.zeros((self.env.height, self.env.width))
    # self.state_count = defaultdict(int)
    # self.episodic_state_count = defaultdict(int)
    # self.total_state_count = 0
    # self.total_state_count_per_room = [0 for _ in range(len(self.rooms))]
    # self._set_total_state_count()
    # self.state_count_per_room = [defaultdict(int) for _ in range(len(self.rooms))]
    # self.episodic_state_count_per_room = [defaultdict(int) for _ in range(len(self.rooms))]

    # ## Agent Stats
    # self.visitation_agent = [np.zeros((self.env.height, self.env.width)) for _ in range(self.num_agents)]
    # self.episodic_visitation_agent = [np.zeros((self.env.height, self.env.width)) for _ in range(self.num_agents)]
    # self.state_count_agent = [defaultdict(int) for _ in range(self.num_agents)]
    # self.episodic_state_count_agent = [defaultdict(int) for _ in range(self.num_agents)]
    self._set_total_state_count()
  
  def step(self, action, policy_idx=0, save_img=False, step=0):
    obs, reward, done, info = self.env.step(action)

    # Tuple based on which we index the counts
    # We use the position after an update
    env = self.unwrapped
    tup = (tuple(env.agent_pos))

    # Get the count for this key
    pre_count = 0
    if tup in self.stats.state_count:
        pre_count = self.stats.state_count[tup]

    # Update the count for this key
    new_count = pre_count + 1
    self.stats.state_count[tup] = new_count
    self.stats.episodic_state_count[tup] = new_count
    self.stats.visitation[tup[1]][tup[0]] += 1
    self.stats.episodic_visitation[tup[1]][tup[0]] += 1

    room = self._which_room(tup[0], tup[1])
    self.stats.state_count_per_room[room][tup] = new_count
    self.stats.episodic_state_count_per_room[room][tup] = new_count

    pre_count = 0
    if tup in self.stats.state_count_agent[policy_idx]:
        pre_count = self.stats.state_count_agent[policy_idx][tup]

    # Update the count for this key
    new_count = pre_count + 1
    self.stats.state_count_agent[policy_idx][tup] = new_count
    self.stats.episodic_state_count_agent[policy_idx][tup] = new_count
    self.stats.visitation_agent[policy_idx][tup[1]][tup[0]] += 1
    self.stats.episodic_visitation_agent[policy_idx][tup[1]][tup[0]] += 1

    return obs, reward, done, info
  
  def _which_room(self, x, y):
    room_idx = 0
    for i in range(len(self.rooms)):
      room = self.rooms[i]
      topX, topY, sizeX, sizeY = room.top[0], room.top[1], room.size[0], room.size[1]
      if x > topX and x < topX + sizeX - 1 and y > topY and y < topY + sizeY - 1:
        room_idx = i
        break
    return room_idx
  
  def _get_room_size(self, room):
    sizeX, sizeY = room.size[0], room.size[1]
    return sizeX * sizeY

  def _set_total_state_count(self):
    num_door = 0
    for i, room in enumerate(self.rooms):
      count = self._get_room_size(room)
      self.stats.total_state_count += count
      self.stats.total_state_count_per_room[i] = count
      if self.minigrid_env_type in ['fourroom', 'roomgrid']:
        for d in room.doors:
          if d is not None:
            num_door += 1
    self.stats.total_state_count += num_door // 2
        

  def reset(self, **kwargs):
    ret = self.env.reset(**kwargs)

    # clear episodic state counts
    self.stats.episodic_state_count = defaultdict(int)
    self.stats.episodic_state_count_per_room = [defaultdict(int) for _ in range(len(self.rooms))]
    self.stats.episodic_visitation = np.zeros((self.env.width, self.env.height))

    # A compact numpy encoding of the grid
    self.encoded_grid = self.env.grid.encode()
    # An image of the intial state of the world
    # self.img = self.env.render('rgb_array', highlight=False)
    self.img, self.goal, self.orig_img = render_minigrid(self.env)

    # reset the total state counts
    self.stats.total_state_count = 0
    self.stats.total_state_count_per_room = [0 for _ in range(len(self.rooms))]
    self._set_total_state_count()

    return ret

class DMC:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None):
    os.environ['MUJOCO_GL'] = 'egl'
    domain, task = name.split('_', 1)
    if task == 'all':
        self._dict_reward = True
    else:
        self._dict_reward = False
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if domain == 'manip':
      from dm_control import manipulation
      self._env = manipulation.load(task + '_vision')
    elif domain == 'locom':
      from dm_control.locomotion.examples import basic_rodent_2020
      self._env = getattr(basic_rodent_2020, task)()
    elif task == 'all':
        import time
        seed = int(str(int((time.time()*10000)))[-6:]) # random seed generator
        self._env = make_dmc_all(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=False)
    else:
      from dm_control import suite
      self._env = suite.load(domain, task)
    self._action_repeat = action_repeat
    self._size = size
    if camera in (-1, None):
      camera = dict(
          quadruped_walk=2, quadruped_run=2, quadruped_escape=2,
          quadruped_fetch=2, locom_rodent_maze_forage=1,
          locom_rodent_two_touch=1,
      ).get(name, 0)
    self._camera = camera
    self._ignored_keys = []
    for key, value in self._env.observation_spec().items():
      if value.shape == (0,):
        print(f"Ignoring empty observation key '{key}'.")
        self._ignored_keys.append(key)

  @property
  def obs_space(self):
    spaces = {
        'image': gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }
    for key, value in self._env.observation_spec().items():
      if key in self._ignored_keys:
        continue
      if value.dtype == np.float64:
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, value.shape, np.float32)
      elif value.dtype == np.uint8:
        spaces[key] = gym.spaces.Box(0, 255, value.shape, np.uint8)
      else:
        raise NotImplementedError(value.dtype)
    return spaces

  @property
  def act_space(self):
    spec = self._env.action_spec()
    action = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)
    return {'action': action}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    assert np.isfinite(action['action']).all(), action['action']
    if self._dict_reward:
        reward = []
    else:
        reward = 0.0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action['action'])
      if self._dict_reward:
        curr_reward = []
        for key, val in time_step.reward.items():
          curr_reward.append(val)
        if len(reward) == 0:
          reward = curr_reward
        else:
          reward = [sum(x) for x in zip(reward, curr_reward)]
      else:
          reward += time_step.reward or 0.0
      if time_step.last():
        break
    assert time_step.discount in (0, 1)
    obs = {
        'reward': reward,
        'is_first': False,
        'is_last': time_step.last(),
        'is_terminal': time_step.discount == 0,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs

  def reset(self):
    time_step = self._env.reset()
    obs = {
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'image': self._env.physics.render(*self._size, camera_id=self._camera),
    }
    obs.update({
        k: v for k, v in dict(time_step.observation).items()
        if k not in self._ignored_keys})
    return obs


class Atari:

  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky=True, all_actions=False):
    assert size[0] == size[1]
    import gym.wrappers
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    with self.LOCK:
      env = gym.envs.atari.AtariEnv(
          game=name, obs_type='image', frameskip=1,
          repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=all_actions)
    # Avoid unnecessary rendering in inner env.
    env._get_obs = lambda: None
    # Tell wrapper that the inner env has no action repeat.
    env.spec = gym.envs.registration.EnvSpec('NoFrameskip-v0')
    self._env = gym.wrappers.AtariPreprocessing(
        env, noops, action_repeat, size[0], life_done, grayscale)
    self._size = size
    self._grayscale = grayscale

  @property
  def obs_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    return {
        'image': gym.spaces.Box(0, 255, shape, np.uint8),
        'ram': gym.spaces.Box(0, 255, (128,), np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    image, reward, done, info = self._env.step(action['action'])
    # info = self.info(info)
    if save_img:
      from PIL import Image
      import pathlib
      ## NOTE: define save path
      save_path = ""
      save_path = pathlib.Path(save_path).expanduser()
      save_path.mkdir(parents=True, exist_ok=True)
      orig_img = self._env.render(mode="rgb_array")
      im = Image.fromarray(orig_img)
      im.save(save_path / f"policy_{policy_idx}_step_{step}.png")
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': done,
    }

  def reset(self):
    with self.LOCK:
      image = self._env.reset()
    if self._grayscale:
      image = image[..., None]
    return {
        'image': image,
        'ram': self._env.env._get_ram(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

  def close(self):
    return self._env.close()


class Crafter:

  def __init__(self, outdir=None, reward=True, seed=None):
    import crafter
    self._env = crafter.Env(reward=reward, seed=seed)
    self._env = Recorder(
        self._env, outdir,
        save_stats=True,
        save_video=False,
        save_episode=False,
    )
    self._achievements = crafter.constants.achievements.copy()

  @property
  def obs_space(self):
    spaces = {
        'image': self._env.observation_space,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'log_reward': gym.spaces.Box(-np.inf, np.inf, (), np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': gym.spaces.Box(0, 2 ** 31 - 1, (), np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    image, reward, done, info = self._env.step(action['action'], policy_idx)
    obs = {
        'image': image,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': info['discount'] == 0,
        'log_reward': info['reward'],
    }
    obs.update({
        f'log_achievement_{k}': v
        for k, v in info['achievements'].items()})
    return obs

  def reset(self):
    obs = {
        'image': self._env.reset(),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
        'log_reward': 0.0,
    }
    obs.update({
        f'log_achievement_{k}': 0
        for k in self._achievements})
    return obs


class Dummy:

  def __init__(self):
    pass

  @property
  def obs_space(self):
    return {
        'image': gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),
    }

  @property
  def act_space(self):
    return {'action': gym.spaces.Box(-1, 1, (6,), dtype=np.float32)}

  def step(self, action):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': False,
        'is_last': False,
        'is_terminal': False,
    }

  def reset(self):
    return {
        'image': np.zeros((64, 64, 3)),
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }


class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  def step(self, action, policy_idx=0, save_img=False, step=0):
    assert self._step is not None, 'Must reset environment.'
    obs = self._env.step(action, policy_idx, save_img, step)
    self._step += 1
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
      self._step = None
    return obs

  def reset(self):
    self._step = 0
    return self._env.reset()


class NormalizeAction:

  def __init__(self, env, key='action'):
    self._env = env
    self._key = key
    space = env.act_space[key]
    self._mask = np.isfinite(space.low) & np.isfinite(space.high)
    self._low = np.where(self._mask, space.low, -1)
    self._high = np.where(self._mask, space.high, 1)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = gym.spaces.Box(low, high, dtype=np.float32)
    return {**self._env.act_space, self._key: space}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self._env.step({**action, self._key: orig}, policy_idx)


class OneHotAction:

  def __init__(self, env, key='action'):

    assert hasattr(env.act_space[key], 'n')
    self._env = env
    self._key = key
    self._random = np.random.RandomState()

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def act_space(self):
    shape = (self._env.act_space[self._key].n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    space.n = shape[0]
    return {**self._env.act_space, self._key: space}

  def step(self, action, policy_idx=0, save_img=False, step=0):
    index = np.argmax(action[self._key]).astype(int)
    reference = np.zeros_like(action[self._key])
    reference[index] = 1
    if not np.allclose(reference, action[self._key]):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step({**action, self._key: index}, policy_idx, save_img, step)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.act_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference


class ResizeImage:

  def __init__(self, env, size=(64, 64)):
    self._env = env
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
    return spaces

  def step(self, action, policy_idx=0, save_img=False, step=0):
    obs = self._env.step(action, policy_idx)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def reset(self):
    obs = self._env.reset()
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


class RenderImage:

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._shape = self._env.render().shape

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self._env, name)
    except AttributeError:
      raise ValueError(name)

  @property
  def obs_space(self):
    spaces = self._env.obs_space
    spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
    return spaces

  def step(self, action, save_img=False, step=0):
    obs = self._env.step(action)
    obs[self._key] = self._env.render('rgb_array')
    return obs

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._env.render('rgb_array')
    return obs


class Async:

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _CLOSE = 4
  _EXCEPTION = 5

  def __init__(self, constructor, strategy='thread'):
    self._pickled_ctor = cloudpickle.dumps(constructor)
    if strategy == 'process':
      import multiprocessing as mp
      context = mp.get_context('spawn')
    elif strategy == 'thread':
      import multiprocessing.dummy as context
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = context.Pipe()
    self._process = context.Process(target=self._worker, args=(conn,))
    atexit.register(self.close)
    self._process.start()
    self._receive()  # Ready.
    self._obs_space = None
    self._act_space = None

  def access(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      pass  # The connection was already closed.
    self._process.join(5)

  @property
  def obs_space(self):
    if not self._obs_space:
      self._obs_space = self.access('obs_space')()
    return self._obs_space

  @property
  def act_space(self):
    if not self._act_space:
      self._act_space = self.access('act_space')()
    return self._act_space

  def step(self, action, blocking=False):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=False):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, conn):
    try:
      ctor = cloudpickle.loads(self._pickled_ctor)
      env = ctor()
      conn.send((self._RESULT, None))  # Ready.
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      conn.send((self._EXCEPTION, stacktrace))
    finally:
      try:
        conn.close()
      except IOError:
        pass  # The connection was already closed.
