import numpy as np
# try:
#   import tensorflow.compat.v1 as tf
# except Exception:
#   import tensorflow as tf
import torch
import torch.nn as nn
import copy

# from _pybullet_env.deep_mimic.learning.tf_agent import TFAgent
from _pybullet_env.learning.torch_agent import TorchAgent
# from _pybullet_env.deep_mimic.learning.solvers.mpi_solver import MPISolver
# import _pybullet_env.deep_mimic.learning.tf_util as TFUtil
import _pybullet_env.learning.nets.net_builder as NetBuilder
import _pybullet_env.learning.nets.pgactor as pgactor
import _pybullet_env.learning.nets.pgcritic as pgcritic
import _pybullet_env.learning.torch_util as PTUtil
# from _pybullet_env.learning.tf_normalizer import TFNormalizer
from _pybullet_env.learning.torch_normalizer import TorchNormalizer
import _pybullet_env.learning.rl_util as RLUtil
from _pybullet_utils.logger import Logger
import _pybullet_utils.mpi_util as MPIUtil
import _pybullet_utils.math_util as MathUtil
from _pybullet_env.action_space import ActionSpace
from _pybullet_env.env import Env
'''
Policy Gradient Agent
'''


class PGAgent(TorchAgent):
  NAME = 'PG'

  ACTOR_NET_KEY = 'ActorNet'
  ACTOR_STEPSIZE_KEY = 'ActorStepsize'
  ACTOR_MOMENTUM_KEY = 'ActorMomentum'
  ACTOR_WEIGHT_DECAY_KEY = 'ActorWeightDecay'
  ACTOR_INIT_OUTPUT_SCALE_KEY = 'ActorInitOutputScale'

  CRITIC_NET_KEY = 'CriticNet'
  CRITIC_STEPSIZE_KEY = 'CriticStepsize'
  CRITIC_MOMENTUM_KEY = 'CriticMomentum'
  CRITIC_WEIGHT_DECAY_KEY = 'CriticWeightDecay'

  EXP_ACTION_FLAG = 1 << 0

  def __init__(self, world, id, json_data):
    print("=================> 2. Policy Gradient AGENT's world", world)
    self._exp_action = False
    self.cmiu = False
    self.onemore_ = False
    super().__init__(world, id, json_data) #!deepdive here
    return

  def reset(self):
    if self.cmiu and self.onemore_: print("=================>reset of pg_agents")
    super().reset()
    self._exp_action = False
    return

  def _check_action_space(self):
    if self.cmiu: print("=================>_check_action_space of pg_agents")
    action_space = self.get_action_space()
    return action_space == ActionSpace.Continuous

  def _load_params(self, json_data):
    if self.cmiu: print("=================>_load_params of pg_agents")
    super()._load_params(json_data)
    self.val_min, self.val_max = self._calc_val_bounds(self.discount)
    self.val_fail, self.val_succ = self._calc_term_vals(self.discount)
    return

  def _build_nets(self, json_data):
    if self.cmiu: print("=================>_build_nets of pg_agents")
    assert self.ACTOR_NET_KEY in json_data
    assert self.CRITIC_NET_KEY in json_data

    actor_net_name = json_data[self.ACTOR_NET_KEY]
    critic_net_name = json_data[self.CRITIC_NET_KEY]
    actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data
                                   ) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

    s_size = self.get_state_size()
    g_size = self.get_goal_size()
    a_size = self.get_action_size()

    print("[PG AGNT] INFO : ", actor_net_name, critic_net_name, actor_init_output_scale)
    print("[PG AGNT] SIZE INFO : ", s_size, g_size, a_size)

    # # setup input tensors
    # self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")  # observations
    # self.tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")  # target value s
    # self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")  # advantage
    # self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")  # target actions
    # self.g_tf = tf.placeholder(tf.float32,
    #                            shape=([None, g_size] if self.has_goal() else None),
    #                            name="g")  # goals
    self.s_pt = torch.zeros((1, s_size), dtype=torch.float32)
    self.a_pt = torch.zeros((1, a_size), dtype=torch.float32)
    self.tar_val_pt = torch.zeros((1, ), dtype=torch.float32)
    self.adv_pt = torch.zeros((1, ), dtype=torch.float32)
    self.g_pt = torch.zeros((1, g_size) if self.has_goal() else (1,), dtype=torch.float32)

    # with tf.variable_scope('main'):
    #   with tf.variable_scope('actor'):
    #     self.actor_tf = self._build_net_actor(actor_net_name, actor_init_output_scale)
    #   with tf.variable_scope('critic'):
    #     self.critic_tf = self._build_net_critic(critic_net_name)

    self.a_mean_pt = self._build_net_actor(actor_net_name, actor_init_output_scale)
    self.critic_pt = self._build_net_critic(critic_net_name)

    if (self.a_mean_pt != None):
      Logger.print2('Built actor net: ' + actor_net_name)

    if (self.critic_pt != None):
      Logger.print2('Built critic net: ' + critic_net_name)

    if self.netmap is None:
      self.netmap ={}
      
    Logger.print2('Actor net and critic net are added to the netmap!!')
    self.netmap['actor'] = self.a_mean_pt
    self.netmap['critic'] = self.critic_pt

    return

  def _build_normalizers(self):
    if self.cmiu: print("=================>_build_normalizers of pg_agents")
    super()._build_normalizers() 
    # with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
    #   with tf.variable_scope(self.RESOURCE_SCOPE):
    val_offset, val_scale = self._calc_val_offset_scale(self.discount)
    self.val_norm = TorchNormalizer(size = 1)
    self.val_norm.set_mean_std(-val_offset, 1.0 / val_scale)
    return

  def _init_normalizers(self):
    if self.cmiu: print("=================>_init_normalizers of pg_agents")
    super()._init_normalizers()
    # with self.sess.as_default(), self.graph.as_default():
    self.val_norm.update()
    return

  def _load_normalizers(self):
    if self.cmiu: print("=================>_load_normalizers of pg_agents")
    super()._load_normalizers()
    self.val_norm.load()
    return

  def _build_losses(self, json_data):
    if self.cmiu: print("=================>_build_losses of pg_agents")
    # actor_weight_decay = 0 if (
    #     self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
    # critic_weight_decay = 0 if (
    #     self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]

    # norm_val_diff = self.val_norm.normalize_tf(self.tar_val_tf) - self.val_norm.normalize_tf(
    #     self.critic_tf)
    # self.critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

    # if (critic_weight_decay != 0):
    #   self.critic_loss_tf += critic_weight_decay * self._weight_decay_loss('main/critic')

    # norm_a_mean_tf = self.a_norm.normalize_tf(self.actor_tf)
    # norm_a_diff = self.a_norm.normalize_tf(self.a_tf) - norm_a_mean_tf

    # self.actor_loss_tf = tf.reduce_sum(tf.square(norm_a_diff), axis=-1)
    # self.actor_loss_tf *= self.adv_tf
    # self.actor_loss_tf = 0.5 * tf.reduce_mean(self.actor_loss_tf)

    # norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
    # norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
    # a_bound_loss = TFUtil.calc_bound_loss(norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max)
    # a_bound_loss /= self.exp_params_curr.noise
    # self.actor_loss_tf += a_bound_loss

    # if (actor_weight_decay != 0):
    #   self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss('main/actor')

    return

  def _build_solvers(self, json_data):
    if self.cmiu: print("=================>_build_solvers of pg_agents")
    # actor_stepsize = 0.001 if (
    #     self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
    # actor_momentum = 0.9 if (
    #     self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
    # critic_stepsize = 0.01 if (
    #     self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
    # critic_momentum = 0.9 if (
    #     self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]

    # critic_vars = self._tf_vars('main/critic')
    # critic_opt = tf.train.MomentumOptimizer(learning_rate=critic_stepsize,
    #                                         momentum=critic_momentum)
    # self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
    # self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

    # actor_vars = self._tf_vars('main/actor')
    # actor_opt = tf.train.MomentumOptimizer(learning_rate=actor_stepsize, momentum=actor_momentum)
    # self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
    # self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

    return

  # def _build_net_actor_old(self, net_name, init_output_scale=None):

  #   # print("before : ", self.s_pt)
  #   # print("s_norm mean/std? : ", self.s_norm.mean_pt, self.s_norm.std_pt)
  #   norm_s_pt = self.s_norm.normalize_pt(self.s_pt)
  #   # print("normed : ", norm_s_pt) #make sense..
  #   input_tensors = [norm_s_pt]
  #   if (self.has_goal()):
  #     norm_g_pt = self.g_norm.normalize_pt(self.g_pt)
  #     input_tensors += [norm_g_pt]

  #   anet = NetBuilder.build_net(net_name, input_tensors)
  #   h = anet(input_tensors) #pass dummy?
  #   out_layer = nn.Linear(h.size(1), self.get_action_size())
  #   if init_output_scale is None: 
  #     init_output_scale = 0.1  
  #   nn.init.uniform_(out_layer.weight, -init_output_scale, init_output_scale)
  #   nn.init.constant_(out_layer.bias, 0)  # Initialize bias to zero
  #   norm_a_pt = out_layer(h)
  #   a_pt = self.a_norm.unnormalize_pt(norm_a_pt)
  #   print("a_pt: ", a_pt)

  #   return a_pt
  
  def _build_net_actor(self, net_name, init_output_scale=None):
    if self.cmiu: print("=================>_build_net_actor of pg_agents")    
    
    if init_output_scale is None: 
      init_output_scale = 0.1  
    
    norm_s_pt = self.s_norm.normalize_pt(self.s_pt)
    input_tensors = [norm_s_pt]
    if (self.has_goal()):
      norm_g_pt = self.g_norm.normalize_pt(self.g_pt)
      input_tensors += [norm_g_pt]   
 
    g_norm = self.g_norm if self.g_norm else None

    anet = pgactor.build_net(net_name, 
                             input_tensors, #only for sample..(To-Do : change!)
                             action_size= self.get_action_size(), 
                             state_norm= self.s_norm,
                             goal_norm=g_norm,
                             action_norm= self.a_norm,
                             init_output_scale = init_output_scale)
    print("anet : ", anet)

    a_pt = anet(norm_s_pt, self.g_pt)
    # norm_a_pt = self.a_norm.unnormalize_pt(a_pt)
    # anet.out_size = norm_a_pt.size()
    anet.out_size = a_pt.size() #To-Do: modify the style (ugly)
    print("Now out size of actor net == ", anet.out_size)
    return anet

  def _build_net_critic(self, net_name):
    if self.cmiu: print("=================>_build_net_critic of pg_agents")
    norm_s_pt = self.s_norm.normalize_pt(self.s_pt)
    input_tensors = [norm_s_pt]
    if self.has_goal():
      norm_g_pt = self.g_norm.normalize_pt(self.g_pt)
      input_tensors += [norm_g_pt]

    g_norm = self.g_norm if self.g_norm else None

    cnet = pgcritic.build_net(net_name, 
                             input_tensors,
                             state_norm= self.s_norm,
                             goal_norm=g_norm,
                             val_norm= self.val_norm,
                             )
    c_pt = cnet(norm_s_pt, self.g_pt) #pass dummy?
    cnet.out_size = c_pt.size() #To-Do: modify the style (ugly)
    print("Now out size of critic net == ", cnet.out_size)

    return cnet
  
  # def _build_net_critic_old(self, net_name):
  #   if self.cmiu: print("=================>_build_net_critic of pg_agents")
  #   norm_s_pt = self.s_norm.normalize_pt(self.s_pt)
  #   input_tensors = [norm_s_pt]
  #   if self.has_goal():
  #     norm_g_pt = self.g_norm.normalize_pt(self.g_pt)
  #     input_tensors += [norm_g_pt]

  #   cnet = NetBuilder.build_net(net_name, input_tensors)
  #   h = cnet(input_tensors) #pass dummy?
  #   out_layer = nn.Linear(h.size(1), 1)
  #   nn.init.xavier_uniform_(out_layer.weight)  # Initialize weights using xavier initialization
  #   nn.init.constant_(out_layer.bias, 0)  # Initialize bias to zero

  #   norm_val_pt = out_layer(h)
  #   val_pt = self.val_norm.unnormalize_pt(norm_val_pt.view(-1))

  #   return val_pt

  def _initialize_vars(self):
    if self.cmiu: print("=================>_initialize_vars of pg_agents")
    super()._initialize_vars()
    self._sync_solvers()
    return

  def _sync_solvers(self):
    if self.cmiu: print("=================>_sync_solvers of pg_agents")
    # self.actor_solver.sync() #없다 에러떠서 일단 커멘트해둠..
    # self.critic_solver.sync()
    return

  def _decide_action(self, s, g):
    if self.cmiu: print("=================>_decide_action of pg_agents")
    with self.sess.as_default(), self.graph.as_default():
      self._exp_action = False
      a = self._eval_actor(s, g)[0]
      logp = 0

      if self._enable_stoch_policy():
        # epsilon-greedy
        rand_action = MathUtil.flip_coin(self.exp_params_curr.rate)
        if rand_action:
          norm_exp_noise = np.random.randn(*a.shape)
          norm_exp_noise *= self.exp_params_curr.noise
          exp_noise = norm_exp_noise * self.a_norm.std
          a += exp_noise

          logp = self._calc_action_logp(norm_exp_noise)
          self._exp_action = True

    return a, logp

  def _enable_stoch_policy(self):
    if self.cmiu and self.onemore_: print("=================>_enable_stoch_policy of pg_agents")
    return self.enable_training and (self._mode == self.Mode.TRAIN or
                                     self._mode == self.Mode.TRAIN_END)

  def _eval_actor(self, s, g):
    if self.cmiu: print("=================>_eval_actor of pg_agents")
    # s = np.reshape(s, [-1, self.get_state_size()])
    # g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

    # feed = {self.s_tf: s, self.g_tf: g}

    # a = self.actor_tf.eval(feed)
    # return a

  # def _eval_critic_old(self, s, g):
  #   # if self.cmiu: print("=================>_eval_critic of pg_agents")
  #   with self.sess.as_default(), self.graph.as_default():
  #     s = np.reshape(s, [-1, self.get_state_size()])
  #     g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

  #     feed = {self.s_tf: s, self.g_tf: g}

  #     val = self.critic_tf.eval(feed)
  #   return val

  def _eval_critic(self, s_np, g_np):
    if self.cmiu: print("=================>_eval_critic of pg_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   s = np.reshape(s, [-1, self.get_state_size()])
    #   g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None
    self.s_pt = torch.from_numpy(s_np).to(dtype=torch.float32)
    if not self.has_goal():
      g_np = np.array([])
    self.g_pt = torch.from_numpy(g_np).to(dtype=torch.float32)
    s = torch.reshape(self.s_pt, [-1, self.get_state_size()])
    g = torch.reshape(self.g_pt, [-1, self.get_goal_size()]) if self.has_goal() else None
    with torch.no_grad():
      v_pt = self.critic_pt(s, g)
      v = v_pt.detach().numpy()

    return v
    #   feed = {self.s_tf: s, self.g_tf: g}

    #   val = self.critic_tf.eval(feed)
    # return val

  def _record_flags(self):
    if self.cmiu and self.onemore_: print("=================>_record_flags of pg_agents")
    flags = int(0)
    if (self._exp_action):
      # print("NEVER???"*11)
      flags = flags | self.EXP_ACTION_FLAG
    return flags

  def _train_step(self):
    # print("=================>_record_flags of pg_agents")
    if self.cmiu: print("Train Step of pg_agent!")
    super()._train_step()

    # critic_loss = self._update_critic()
    # actor_loss = self._update_actor()
    # critic_loss = MPIUtil.reduce_avg(critic_loss)
    # actor_loss = MPIUtil.reduce_avg(actor_loss)

    # critic_stepsize = self.critic_solver.get_stepsize()
    # actor_stepsize = self.actor_solver.get_stepsize()

    # self.logger.log_tabular('Critic_Loss', critic_loss)
    # self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
    # self.logger.log_tabular('Actor_Loss', actor_loss)
    # self.logger.log_tabular('Actor_Stepsize', actor_stepsize)

    return

  def _update_critic(self):
    if self.cmiu: print("=================>_update_critic of pg_agents")
    # idx = self.replay_buffer.sample(self._local_mini_batch_size)
    # s = self.replay_buffer.get('states', idx)
    # g = self.replay_buffer.get('goals', idx) if self.has_goal() else None

    # tar_V = self._calc_updated_vals(idx)
    # tar_V = np.clip(tar_V, self.val_min, self.val_max)

    # feed = {self.s_tf: s, self.g_tf: g, self.tar_val_tf: tar_V}

    # loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
    # self.critic_solver.update(grads)
    # return loss

  def _update_actor(self):
    if self.cmiu: print("=================>_update_actor of pg_agents")
    # key = self.EXP_ACTION_FLAG
    # idx = self.replay_buffer.sample_filtered(self._local_mini_batch_size, key)
    # has_goal = self.has_goal()

    # s = self.replay_buffer.get('states', idx)
    # g = self.replay_buffer.get('goals', idx) if has_goal else None
    # a = self.replay_buffer.get('actions', idx)

    # V_new = self._calc_updated_vals(idx)
    # V_old = self._eval_critic(s, g)
    # adv = V_new - V_old

    # feed = {self.s_tf: s, self.g_tf: g, self.a_tf: a, self.adv_tf: adv}

    # loss, grads = self.sess.run([self.actor_loss_tf, self.actor_grad_tf], feed)
    # self.actor_solver.update(grads)

    # return loss

  def _calc_updated_vals(self, idx):
    if self.cmiu: print("=================>_calc_updated_vals of pg_agents")
    r = self.replay_buffer.get('rewards', idx)

    if self.discount == 0:
      new_V = r
    else:
      next_idx = self.replay_buffer.get_next_idx(idx)
      s_next = self.replay_buffer.get('states', next_idx)
      g_next = self.replay_buffer.get('goals', next_idx) if self.has_goal() else None

      is_end = self.replay_buffer.is_path_end(idx)
      is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
      is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
      is_fail = np.logical_and(is_end, is_fail)
      is_succ = np.logical_and(is_end, is_succ)

      V_next = self._eval_critic(s_next, g_next)
      V_next[is_fail] = self.val_fail
      V_next[is_succ] = self.val_succ

      new_V = r + self.discount * V_next
    return new_V

  def _calc_action_logp(self, norm_action_deltas):
    if self.cmiu: print("=================>_calc_action_logp of pg_agents")
    # norm action delta are for the normalized actions (scaled by self.a_norm.std)
    stdev = self.exp_params_curr.noise
    assert stdev > 0

    a_size = self.get_action_size()
    logp = -0.5 / (stdev * stdev) * np.sum(np.square(norm_action_deltas), axis=-1)
    logp += -0.5 * a_size * np.log(2 * np.pi)
    logp += -a_size * np.log(stdev)
    return logp

  def _log_val(self, s, g):
    if self.cmiu: print("=================>_log_val of pg_agents")
    val = self._eval_critic(s, g)
    norm_val = self.val_norm.normalize(val)
    self.world.env.log_val(self.id, norm_val[0])
    return

  def _build_replay_buffer(self, buffer_size):
    if self.cmiu: print("=================>_build_replay_buffer of pg_agents")
    super()._build_replay_buffer(buffer_size)
    self.replay_buffer.add_filter_key(self.EXP_ACTION_FLAG)
    return
