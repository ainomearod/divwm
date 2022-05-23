import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
import sys

import agent
import common


class Random(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.act_space = act_space
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
        self.config = self.config.update({
            'actor.dist': 'onehot' if discrete else 'trunc_normal'})

  def actor(self, feat):
    shape = feat.shape[:-1] + self.act_space.shape
    if self.config.actor.dist == 'onehot':
      return common.OneHotDist(tf.zeros(shape))
    else:
      dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
      return tfd.Independent(dist, 1)

  def train(self, start, context, data):
    return None, {}


class Plan2Explore(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    self.reward = reward
    self.wm = wm
    self._init_actors()

    stoch_size = config.rssm.stoch
    if config.rssm.discrete:
      stoch_size *= config.rssm.discrete
    size = {
        'embed': 32 * config.encoder.cnn_depth,
        'stoch': stoch_size,
        'deter': config.rssm.deter,
        'feat': config.rssm.stoch + config.rssm.deter,
    }[self.config.disag_target]
    self._networks = [
        common.MLP(size, **config.expl_head)
        for _ in range(config.disag_models)]
    self.opt = common.Optimizer('expl', **config.expl_opt)
    self.extr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)

  def _init_actors(self):
    # TODO: implement multihead AC
    self.intr_rewnorm = common.StreamNorm(**self.config.expl_reward_norm)
    self.ac = [agent.ActorCritic(self.config, self.act_space, self.tfstep) for _ in range(self.config.num_agents)]
    if self.config.cascade_alpha > 0:
        self.intr_rewnorm_cascade = [common.StreamNorm(**self.config.expl_reward_norm) for _ in range(self.config.num_agents)]
    self.actor = [ac.actor for ac in self.ac]
    # if self.config.num_agents == 1:
    #   self.ac = agent.ActorCritic(self.config, self.act_space, self.tfstep)
    #   if self.config.cascade_alpha > 0:
    #     self.intr_rewnorm_cascade = common.StreamNorm(**self.config.expl_reward_norm)
    #   self.actor = self.ac.actor
    # elif self.config.method == "single_disag":
    #   self.ac = [agent.ActorCritic(self.config, self.act_space, self.tfstep) for _ in range(self.config.num_agents)]
    #   # self.ac = agent.PopulationActorCritic(self.config, self.act_space, self.tfstep)
    #   if self.config.cascade_alpha > 0:
    #     self.intr_rewnorm_cascade = [common.StreamNorm(**self.config.expl_reward_norm) for _ in range(self.config.num_agents)]
    #   self.actor = [ac.actor for ac in self.ac]
    # elif self.config.method == "multihead_disag":
    #   self.ac = agent.MultiHeadActorCritic(self.config, self.act_space, self.tfstep)
    #   if self.config.cascade_alpha > 0:
    #     self.intr_rewnorm_cascade = [common.StreamNorm(**self.config.expl_reward_norm) for _ in range(self.config.num_agents)]
    #   self.actor = self.ac.actor

  def train(self, start, context, data):
    metrics = {}
    stoch = start['stoch']
    if self.config.rssm.discrete:
      stoch = tf.reshape(
          stoch, stoch.shape[:-2] + (stoch.shape[-2] * stoch.shape[-1]))
    target = {
        'embed': context['embed'],
        'stoch': stoch,
        'deter': start['deter'],
        'feat': context['feat'],
    }[self.config.disag_target]
    inputs = context['feat']
    if self.config.disag_action_cond:
      action = tf.cast(data['action'], inputs.dtype)
      inputs = tf.concat([inputs, action], -1)
    metrics.update(self._train_ensemble(inputs, target))
    gpu = tf.config.list_physical_devices('GPU')
    if gpu:
      tf.config.experimental.set_memory_growth(gpu[0], True)
      print(f"Before: {tf.config.experimental.get_memory_usage('GPU:0')}", flush=True)
    self.cascade = []
    if self.config.method == "single_disag":
      reward_func = self._intr_reward_incr
    elif self.config.method == "multihead_disag":
      ## Note: this method doesn't work now
      reward_func = self._intr_reward_pop
    else:
      reward_func = self._intr_reward
    print("training explorers", flush=True)
    [metrics.update(ac.train(self.wm, start, data['is_terminal'], reward_func)) for ac in self.ac]
    self.cascade = []
    print("finished training explorers", flush=True)
    return None, metrics

  def _intr_reward(self, seq, rtn_meta=True):
    disags = []
    for i in range(int(seq['feat'].shape[1]/100)):
      inputs = seq['feat'][:, i*100:(i+1)*100, :]
      if self.config.disag_action_cond:
        action = tf.cast(seq['action'][:, i*100:(i+1)*100, :], inputs.dtype)
        inputs = tf.concat([inputs, action], -1)
      preds = [head(inputs).mode() for head in self._networks]
      disags.append(tf.cast(tf.tensor(preds).std(0).mean(-1), tf.float16))
    disag = tf.concat(disags, axis=1)
    if self.config.disag_log:
      disag = tf.math.log(disag)
    reward = self.config.expl_intr_scale * self.intr_rewnorm(disag)[0]
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.extr_rewnorm(
          self.reward(seq))[0]
    if rtn_meta:
      return reward, {'Disagreement': [disag.mean()]}
    else:
      return reward

  @tf.function
  def get_cascade_actions(self, seqs):
    ## compute cascade rewards
    cascade_rew = []
    for i, seq in enumerate(seqs):
      if i == 0:
        cascade_rew += [tf.cast(tf.zeros([seq['deter'].shape[0], seq['deter'].shape[1]]), tf.float16)]
      else:
        if self.config.method == "single_disag":
          policy = self.actor[len(seqs)-1](tf.stop_gradient(seq['feat'][:-2]))
        elif self.config.method == "multihead_disag":
          policy = self.actor(tf.stop_gradient(seq['feat'][:-2]), len(seqs)-1)
        log_pi_i = policy.log_prob(seq['action'][1:-1])
        rew = tf.zeros_like(log_pi_i)
        for idx in range(i):
          if self.config.method == "single_disag":
            policy = self.actor[idx](tf.stop_gradient(seq['feat'][:-2]))
          elif self.config.method == "multihead_disag":
            policy = self.actor(tf.stop_gradient(seq['feat'][:-2]), idx)
          log_pi_j = policy.log_prob(seq['action'][1:-1])
          rew += tf.stop_gradient(log_pi_i) - tf.stop_gradient(log_pi_j) # i think this is right
        cascade_rew += [tf.concat([tf.cast(tf.zeros([1, rew.shape[1]]), tf.float16), rew, tf.cast(tf.zeros([1, rew.shape[1]]), tf.float16)], 0)]
    return cascade_rew

  @tf.function
  def get_dists(self, obs, cascade):
    ### zzz way to do this
    out = []
    for idx in range(obs.shape[1]):
      cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
      ob = tf.reshape(obs[:, idx, :], [obs.shape[0], 1, obs.shape[-1]])
      dists = tf.math.sqrt(tf.einsum('ijk, ijk->ij', cascade - ob, cascade - ob))
      topk_mean = tf.negative(tf.math.top_k(tf.negative(dists), k=self.config.cascade_k)[0])
      out += [tf.reshape(tf.math.reduce_mean(topk_mean, axis=-1), (1, -1))]
    return tf.concat(out, axis=1)

  @tf.function
  def get_cascade_states_all(self, seqs):
    ## compute cascade rewards
    idxs = tf.range(tf.shape(seqs[0][self.config.cascade_feat])[1])

    # hyperparam, number of samples to use
    if len(seqs) > 10:
      ridxs = tf.random.shuffle(idxs)[:10]
    else:
      ridxs = tf.random.shuffle(idxs)[:20]

    cascade_rew = []
    for i, seq in enumerate(seqs):
      if i == 0:
        cascade_rew += [tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0], seq[self.config.cascade_feat].shape[1]]), tf.float16)]
      elif self.config.cascade_states == "all":
        cascade = tf.concat([tf.gather(seq[self.config.cascade_feat], ridxs, axis=1) for seq in seqs[:i]], axis=0)
        cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
        obs = seq[self.config.cascade_feat]
        obs = tf.reshape(obs, [-1, 1, obs.shape[-1]])
        if self.config.cascade_proj > 0:
          # project to low dim
          pass
        dists = self.get_dists(obs, cascade)
        cascade_rew += [tf.cast(tf.reshape(dists, (seq[self.config.cascade_feat].shape[0], seq[self.config.cascade_feat].shape[1])), tf.float16)]
      elif self.config.cascade_states == "final":
        cascade = tf.concat([tf.gather(seq[self.config.cascade_feat], ridxs, axis=1)[-1] for seq in seqs[:i]], axis=0)
        cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
        obs = seq[self.config.cascade_feat][-1]
        obs = tf.reshape(obs, [-1, 1, obs.shape[-1]])
        dists = self.get_dists(obs, cascade)
        cascade_rew += [tf.concat([tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0] - 1, seq[self.config.cascade_feat].shape[1]]), tf.float16), tf.cast(dists, tf.float16)], axis=0)]
    return cascade_rew

  @tf.function
  def _intr_reward_pop(self, seqs):
    ## disagreement
    rewards = []
    for seq in seqs:
      reward, mets = self._intr_reward(seq)
      rewards.append(reward)
    # CASCADE
    if self.config.cascade_alpha > 0:
      ## reward = (1 - \alpha) * disagreement + \alpha * diversity
      if self.config.cascade_metric in ["euclidean"]:
        cascade_rewards = self.get_cascade_states(seqs)
      elif self.config.cascade_metric in ["kl"]:
        cascade_rewards = self.get_cascade_actions(seqs)
      cascade_rewards = [self.intr_rewnorm_cascade[i](rew)[0] for i, rew in enumerate(cascade_rewards)]
      rewards = [rew * (1 - self.config.cascade_alpha) + self.config.cascade_alpha * c_rew for rew, c_rew in zip(rewards, cascade_rewards)]
    return rewards, mets

  def get_cascade_states_indiv(self, seq):
    if len(self.cascade) == 0:
      return tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0], seq[self.config.cascade_feat].shape[1]]), tf.float16)

    if self.config.cascade_states == "all":
      cascade = tf.concat(self.cascade, axis=0)
      cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
      obs = seq[self.config.cascade_feat]
      obs = tf.reshape(obs, [-1, 1, obs.shape[-1]])
      if self.config.cascade_proj > 0:
        # project to low dim
        pass
      dists = self.get_dists(obs, cascade)
      return tf.cast(tf.reshape(dists, (seq[self.config.cascade_feat].shape[0], seq[self.config.cascade_feat].shape[1])), tf.float16)
    elif self.config.cascade_states == "final":
      cascade = tf.concat([cas[-1] for cas in self.cascade], axis=0)
      cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
      obs = seq[self.config.cascade_feat][-1]
      obs = tf.reshape(obs, [-1, 1, obs.shape[-1]])
      dists = self.get_dists(obs, cascade)
      return tf.concat([tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0] - 1, seq[self.config.cascade_feat].shape[1]]), tf.float16), tf.cast(dists, tf.float16)], axis=0)

  def get_cascade_actions_indiv(self, seq):
    agent_idx = len(self.cascade)
    # if agent_idx == 0:
    #   return tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0], seq[self.config.cascade_feat].shape[1]]), tf.float16)

    if self.config.method == "single_disag":
      policy = self.actor[agent_idx](tf.stop_gradient(seq['feat'][:-2]))
    elif self.config.method == "multihead_disag":
      policy = self.actor[agent_idx](tf.stop_gradient(seq['feat'][:-2]))
    log_pi_i = policy.log_prob(seq['action'][1:-1])
    rew = tf.zeros_like(log_pi_i)

    if self.config.cascade_metric == "kl_full":
      for idx in range(agent_idx):
        prev_policy = self.actor[idx](tf.stop_gradient(seq['feat'][:-2]))
        rew += policy.kl_divergence(prev_policy)
    else:
      for idx in range(agent_idx):
        if self.config.method == "single_disag":
          policy = self.actor[idx](tf.stop_gradient(seq['feat'][:-2]))
        elif self.config.method == "multihead_disag":
          policy = self.actor[idx](tf.stop_gradient(seq['feat'][:-2]))
        log_pi_j = tf.stop_gradient(policy.log_prob(seq['action'][1:-1]))
        if self.config.cascade_metric == "kl":
          rew += tf.abs(log_pi_i - log_pi_j) # i think this is right
        elif self.config.cascade_metric == "kl_weighted":
          rew += tf.exp(log_pi_i) * tf.abs(log_pi_i - log_pi_j)
    if self.config.cascade_average:
      rew /= (agent_idx + 1.0)
    return tf.cast(rew, tf.float16)
  
  def get_cascade_entropy(self):
    if self.config.cascade_states == "final":
      cascade = tf.concat(self.cascade, axis=0)
      cascade = tf.reshape(cascade, [-1, cascade.shape[-1]])
      entropy = tf.math.reduce_variance(cascade, axis=-1).mean()
      if self.config.cascade_metric == "entropy_gain":
        reward = entropy - self.entropy
      else:
        reward = entropy
      self.entropy = entropy
      return reward

  def _intr_reward_incr(self, seq):
    agent_idx = len(self.cascade)
    ## disagreement
    reward, meta = self._intr_reward(seq)
    # CASCADE
    if self.config.cascade_alpha > 0:
      ## reward = (1 - \alpha) * disagreement + \alpha * diversity
      if self.config.cascade_metric in ["euclidean"]:
        cascade_reward = self.get_cascade_states_indiv(seq)
        if len(self.cascade) == 0:
          idxs = tf.range(tf.shape(seq[self.config.cascade_feat])[1])
          self.ridxs = tf.random.shuffle(idxs)[:self.config.cascade_sample]
        self.cascade.append(tf.gather(seq[self.config.cascade_feat], self.ridxs, axis=1))
      elif self.config.cascade_metric in ["kl", "kl_full", "kl_weighted"]:
        cascade_reward = self.get_cascade_actions_indiv(seq)
        zero_padding = tf.zeros([2, cascade_reward.shape[1]], dtype=cascade_reward.dtype)
        cascade_reward = tf.concat([cascade_reward, zero_padding], 0)
        self.cascade.append([]) ## track agent idx
      elif self.config.cascade_metric in ["entropy", "entropy_gain"]:
        if len(self.cascade) == 0:
          idxs = tf.range(tf.shape(seq[self.config.cascade_feat])[1])
          size = min(seq[self.config.cascade_feat].shape[1], self.config.cascade_sample)
          self.ridxs = tf.random.shuffle(idxs)[:size]
          self.dist = None
          self.entropy = 0
        if self.config.cascade_states == "final":
          self.cascade.append(tf.gather(seq[self.config.cascade_feat][-1], self.ridxs, axis=1))
        else:
          self.cascade.append(tf.gather(seq[self.config.cascade_feat], self.ridxs, axis=1))
        cascade_reward = self.get_cascade_entropy()
        cascade_reward = tf.concat([tf.cast(tf.zeros([seq[self.config.cascade_feat].shape[0] - 1, seq[self.config.cascade_feat].shape[1]]), tf.float16), tf.cast(tf.broadcast_to(cascade_reward, shape=(1, seq[self.config.cascade_feat].shape[1])), tf.float16)], axis=0)
      cascade_reward = self.intr_rewnorm_cascade[agent_idx](cascade_reward)[0]
      meta.update({'Diversity': [cascade_reward.mean()]})
      #print(f"Reg rew size: {reward[0].shape}. Cascade size: {cascade_reward[0].shape}", flush=True)
      reward = reward * (1 - self.config.cascade_alpha) + self.config.cascade_alpha * cascade_reward * self.config.cascade_scale
    return reward, meta
    

  def _train_ensemble(self, inputs, targets):
    if self.config.disag_offset:
      targets = targets[:, self.config.disag_offset:]
      inputs = inputs[:, :-self.config.disag_offset]
    targets = tf.stop_gradient(targets)
    inputs = tf.stop_gradient(inputs)
    with tf.GradientTape() as tape:
      preds = [head(inputs) for head in self._networks]
      loss = -sum([pred.log_prob(targets).mean() for pred in preds])
    metrics = self.opt(tape, loss, self._networks)
    return metrics

class ModelLoss(common.Module):

  def __init__(self, config, act_space, wm, tfstep, reward):
    self.config = config
    self.reward = reward
    self.wm = wm
    self.ac = agent.ActorCritic(config, act_space, tfstep)
    self.actor = self.ac.actor
    self.head = common.MLP([], **self.config.expl_head)
    self.opt = common.Optimizer('expl', **self.config.expl_opt)

  def train(self, start, context, data):
    metrics = {}
    target = tf.cast(context[self.config.expl_model_loss], tf.float16)
    with tf.GradientTape() as tape:
      loss = -self.head(context['feat']).log_prob(target).mean()
    metrics.update(self.opt(tape, loss, self.head))
    metrics.update(self.ac.train(
        self.wm, start, data['is_terminal'], self._intr_reward))
    return None, metrics

  def _intr_reward(self, seq):
    reward = self.config.expl_intr_scale * self.head(seq['feat']).mode()
    if self.config.expl_extr_scale:
      reward += self.config.expl_extr_scale * self.reward(seq)
    return reward
