import threading
import os
from itertools import combinations

import gym
import numpy as np

import metaworld.envs.mujoco.sawyer_xyz.v1 as sawyer


class BenchEnv:

  LOCK = threading.Lock()

  def __init__(self, action_repeat, width=64):
    os.environ['MUJOCO_GL'] = 'egl'
    self._action_repeat = action_repeat
    self._width = width
    self._size = (self._width, self._width)

  @property
  def obs_space(self):
    shape = self._size + (3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return {'image': space,
        'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
        'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool),
        'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool),}

  @property
  def act_space(self):
    return {'action': self._env.action_space}

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      state = self._env.reset()
    return self._get_obs(state)

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode, self._width, self._width)

  def render_offscreen(self):
    from d4rl.kitchen.adept_envs.simulation.sim_robot import RenderMode
    img = self.renderer.render_offscreen(
                self._width, self._width, mode=RenderMode.RGB, camera_id=-1)
    return np.flipud(np.fliplr(img))

  def _get_obs(self, state):
    return {'image': self.render_offscreen(), 'state': state}

class KitchenEnv(BenchEnv):

  def __init__(self, action_repeat=1, use_goal_idx=False, log_per_goal=False,  control_mode='end_effector', width=64):
    from d4rl.kitchen.kitchen_envs import KitchenMicrowaveKettleLightTopLeftBurnerV0 
    super().__init__(action_repeat, width)
    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    with self.LOCK:
      self._env =  KitchenMicrowaveKettleLightTopLeftBurnerV0(frame_skip=16, control_mode = control_mode, imwidth=width, imheight=width)

      self._env.sim_robot.renderer._camera_settings = dict(
        distance=1.86, lookat=[-0.3, .5, 2.], azimuth=90, elevation=-60)

    self.rendered_goal = False
    self._env.reset()
    self.init_qpos = self._env.sim.data.qpos.copy()
    self.goal_idx = 0
    self.obs_element_goals, self.obs_element_indices, self.goal_configs = get_kitchen_benchmark_goals()
    self.goals = list(range(len(self.obs_element_goals)))

  def set_goal_idx(self, idx):
    self.goal_idx = idx

  def get_goal_idx(self):
    return self.goal_idx

  def get_goals(self):
    return self.goals

  def _get_obs(self, state):
    image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}
    if self.log_per_goal:
      for i, goal_idx in enumerate(self.goals):
        # add rewards for all goals
        task_rel_success, all_obj_success = self.compute_success(goal_idx)
        obs['metric_success_task_relevant/goal_'+str(goal_idx)] = task_rel_success
        obs['metric_success_all_objects/goal_'+str(goal_idx)]   = all_obj_success
    if self.use_goal_idx:
      task_rel_success, all_obj_success = self.compute_success(self.goal_idx)
      obs['metric_success_task_relevant/goal_'+str(self.goal_idx)] = task_rel_success
      obs['metric_success_all_objects/goal_'+str(self.goal_idx)]   = all_obj_success

    return obs

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action['action'])
      reward = self.compute_reward()
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return {
        **obs,
        'reward': reward,
        'is_first': False,
        'is_last': done,
        'is_terminal': done,
    }

  def compute_reward(self, goal=None):
    if goal is None:
      goal = self.goal
    qpos = self._env.sim.data.qpos.copy()

    if len(self.obs_element_indices[goal]) > 9 :
        return  -np.linalg.norm(qpos[self.obs_element_indices[goal]][9:] - self.obs_element_goals[goal][9:])
    else:
        return -np.linalg.norm(qpos[self.obs_element_indices[goal]] - self.obs_element_goals[goal])

  def compute_success(self, goal = None):

    if goal is None:
      goal = self.goal
    qpos = self._env.sim.data.qpos.copy()

    goal_qpos = self.init_qpos.copy()
    goal_qpos[self.obs_element_indices[goal]] = self.obs_element_goals[goal]

    per_obj_success = {
    'bottom_burner' : ((qpos[9]<-0.38) and (goal_qpos[9]<-0.38)) or ((qpos[9]>-0.38) and (goal_qpos[9]>-0.38)),
    'top_burner':    ((qpos[13]<-0.38) and (goal_qpos[13]<-0.38)) or ((qpos[13]>-0.38) and (goal_qpos[13]>-0.38)),
    'light_switch':  ((qpos[17]<-0.25) and (goal_qpos[17]<-0.25)) or ((qpos[17]>-0.25) and (goal_qpos[17]>-0.25)),
    'slide_cabinet' :  abs(qpos[19] - goal_qpos[19])<0.1,
    'hinge_cabinet' :  abs(qpos[21] - goal_qpos[21])<0.2,
    'microwave' :      abs(qpos[22] - goal_qpos[22])<0.2,
    'kettle' : np.linalg.norm(qpos[23:25] - goal_qpos[23:25]) < 0.2
    }
    task_objects = self.goal_configs[goal]

    task_rel_success = 1
    for _obj in task_objects:
      task_rel_success *= per_obj_success[_obj]

    all_obj_success = 1
    for _obj in per_obj_success:
      all_obj_success *= per_obj_success[_obj]

    return int(task_rel_success), int(all_obj_success)

  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj

    # random.sample(list(obs_element_goals), 1)[0]
    backup_qpos = self._env.sim.data.qpos.copy()
    backup_qvel = self._env.sim.data.qvel.copy()

    qpos = self.init_qpos.copy()
    qpos[self.obs_element_indices[self.goal]] = self.obs_element_goals[self.goal]

    self._env.set_state(qpos, np.zeros(len(self._env.init_qvel)))

    goal_obs = self._env.render('rgb_array', width=self._env.imwidth, height=self._env.imheight)

    self._env.set_state(backup_qpos, backup_qvel)

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  def reset(self):

    with self.LOCK:
      state = self._env.reset()
    if not self.use_goal_idx:
      self.goal_idx = np.random.randint(len(self.goals))
    self.goal = self.goals[self.goal_idx]
    self.rendered_goal = False
    obs = self._get_obs(state)
    return {
        **obs,
        'reward': 0.0,
        'is_first': True,
        'is_last': False,
        'is_terminal': False,
    }

def get_kitchen_benchmark_goals():

    object_goal_vals = {'bottom_burner' :  [-0.88, -0.01],
                          'light_switch' :  [ -0.69, -0.05],
                          'slide_cabinet':  [0.37],
                          'hinge_cabinet':   [0., 0.5],
                          'microwave'    :   [-0.5],
                          'kettle'       :   [-0.23, 0.75, 1.62]}

    object_goal_idxs = {'bottom_burner' :  [9, 10],
                    'light_switch' :  [17, 18],
                    'slide_cabinet':  [19],
                    'hinge_cabinet':  [20, 21],
                    'microwave'    :  [22],
                    'kettle'       :  [23, 24, 25]}

    base_task_names = [ 'bottom_burner', 'light_switch', 'slide_cabinet', 
                        'hinge_cabinet', 'microwave', 'kettle' ]

    
    goal_configs = []
    #single task
    for i in range(6):
      goal_configs.append( [base_task_names[i]])

    #two tasks
    for i,j  in combinations([1,2,3,5], 2) :
      goal_configs.append( [base_task_names[i], base_task_names[j]] )
    
    obs_element_goals = [] ; obs_element_indices = []
    for objects in goal_configs:
        _goal = np.concatenate([object_goal_vals[obj] for obj in objects])
        _goal_idxs = np.concatenate([object_goal_idxs[obj] for obj in objects])

        obs_element_goals.append(_goal)
        obs_element_indices.append(_goal_idxs)
  
    return obs_element_goals, obs_element_indices, goal_configs


class RoboBinEnv(BenchEnv):
  def __init__(self, action_repeat, use_goal_idx=False, log_per_goal=False, 
                image_width=64, metric_rew_cap=100000):
    from d4rl.kitchen.adept_envs.simulation.renderer import DMRenderer
    super().__init__(action_repeat)

    self._env = sawyer.SawyerTwoBlockBinEnv()
    self._env.random_init = False

    #workspace limits
    self._env.mocap_low = (-0.5, 0.40, 0.07)
    self._env.mocap_high = (0.5, 0.8, 0.5)
    self._env.goals = get_robobin_benchmark_goals()

    self._action_repeat = action_repeat
    self._width = image_width
    self.metric_rew_cap = metric_rew_cap
    self._size = (self._width, self._width)

    #camera parameters
    self.renderer = DMRenderer(self._env.sim, camera_settings=dict(
          distance=0.6, lookat=[0, 0.65, 0], azimuth=90, elevation=41+180))

    self.use_goal_idx = use_goal_idx
    self.log_per_goal = log_per_goal
    self.rendered_goal = False

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(action)
      total_reward += min(reward, self.metric_rew_cap)
      if done:
        break
    obs = self._get_obs(state)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v
    return obs, total_reward, done, info

  def reset(self):
    self.rendered_goal = False
    if self.use_goal_idx:
      self._env.goal = self.get_goals()[self.get_goal_idx()]
    return super().reset()

  def _get_obs(self, state):
    obs = super()._get_obs(state)
    obs['image_goal'] = self.render_goal()
    obs['goal'] = self._env.goals[self._env.goal_idx]
    if self.log_per_goal:
      obs = self._env.add_pertask_success(obs)
    elif self.use_goal_idx:
      obs = self._env.add_pertask_success(obs, self._env.goal_idx)

    return obs

  def set_goal_idx(self, idx):
    self._env.goal_idx = idx

  def get_goal_idx(self):
    return self._env.goal_idx

  def get_goals(self):
    return self._env.goals


  def render_goal(self):
    if self.rendered_goal:
      return self.rendered_goal_obj
    # TODO use self.render_state

    obj_init_pos_temp = self._env.init_config['obj_init_pos'].copy()
    goal = self._env.goals[self._env.goal_idx]

    self._env.init_config['obj_init_pos'] = goal[3:]
    self._env.obj_init_pos = goal[3:]
    self._env.hand_init_pos = goal[:3]
    self._env.reset_model()
    action = np.zeros(self._env.action_space.low.shape)
    state, reward, done, info = self._env.step(action)

    goal_obs = self.render_offscreen()
    self._env.hand_init_pos = self._env.init_config['hand_init_pos']
    self._env.init_config['obj_init_pos'] = obj_init_pos_temp
    self._env.obj_init_pos = self._env.init_config['obj_init_pos']
    self._env.reset()

    self.rendered_goal = True
    self.rendered_goal_obj = goal_obs
    return goal_obs

  def render_state(self, state):
    assert (len(state.shape) == 1)
    # Save init configs
    hand_init_pos = self._env.hand_init_pos
    obj_init_pos = self._env.init_config['obj_init_pos']
    # Render state
    hand_pos, obj_pos, hand_to_goal = np.split(state, 3)
    self._env.hand_init_pos = hand_pos
    self._env.init_config['obj_init_pos'] = obj_pos
    self._env.reset_model()
    obs = self._get_obs(state)
    # Revert environment
    self._env.hand_init_pos = hand_init_pos
    self._env.init_config['obj_init_pos'] = obj_init_pos
    self._env.reset()
    return obs['image']

  def render_states(self, states):
    assert (len(states.shape) == 2)
    imgs = []
    for s in states:
      img = self.render_state(s)
      imgs.append(img)
    return np.array(imgs)

def get_robobin_benchmark_goals():
  pos1 = np.array([-0.1, 0.7, 0.04])
  pos2 = np.array([ 0.1, 0.7, 0.04])
  delta = np.array([0, 0.15, 0])
  v_delta = np.array([0,0,0.06])
  hand = np.array([0, 0.65, 0.2])

  goaldictlist = [

    #reaching
    {'obj1': pos1, 'obj2': pos2, 'hand': hand + np.array([0.12,0.1, -0.1])},
    {'obj1': pos1, 'obj2': pos2, 'hand': hand + np.array([-0.1,0.2, -0.1])},

    #pushing
    {'obj1': pos1, 'obj2': pos2 + delta, 'hand': hand},
    {'obj1': pos1 - delta, 'obj2': pos2, 'hand': hand},

  #push both
    {'obj1': pos1+delta, 'obj2': pos2 + delta, 'hand': hand},
    {'obj1': pos1-delta, 'obj2': pos2 - delta, 'hand': hand},

    #pickplace
    {'obj1': pos2 + delta, 'obj2': pos2, 'hand': hand},

    #pickplace both
    {'obj1': pos2+delta, 'obj2': pos1+delta, 'hand': hand}]

  return [np.concatenate([_dict['hand'], _dict['obj1'], _dict['obj2']])
                          for _dict in goaldictlist]