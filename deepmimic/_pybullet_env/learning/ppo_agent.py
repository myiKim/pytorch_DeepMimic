import numpy as np
import copy as copy
# try:
#   print("tf 1.15 still take this ")
#   import tensorflow.compat.v1 as tf
# except Exception:
#   print("!!!!!EXCEPTION!!!!!!!!!!!!")
#   import tensorflow as tf
import torch
import torch.optim as optim

from _pybullet_env.learning.pg_agent import PGAgent
# from _pybullet_env.deep_mimic.learning.solvers.mpi_solver import MPISolver #tf안쓰면 큰 필요있나 싶다. 커멘트 계속 안켜도 될듯
# import _pybullet_env.deep_mimic.learning.tf_util as TFUtil
import _pybullet_env.learning.torch_util as PTUtil
import _pybullet_env.learning.rl_util as RLUtil
from _pybullet_utils.logger import Logger
import _pybullet_utils.mpi_util as MPIUtil
import _pybullet_utils.math_util as MathUtil
from _pybullet_env.env import Env
'''
Proximal Policy Optimization Agent
'''


class PPOAgent(PGAgent):
# class PPOAgent:
  NAME = "PPO"
  EPOCHS_KEY = "Epochs"
  BATCH_SIZE_KEY = "BatchSize"
  RATIO_CLIP_KEY = "RatioClip"
  NORM_ADV_CLIP_KEY = "NormAdvClip"
  TD_LAMBDA_KEY = "TDLambda"
  TAR_CLIP_FRAC = "TarClipFrac"
  ACTOR_STEPSIZE_DECAY = "ActorStepsizeDecay"
  OPTIMIZER_TYPE = 'SGD'

  def __init__(self, world, id, json_data):
    # print("######################################################"*20)
    print("=================> 1. Proximal Policy Optim. AGENT's world", world)
    super().__init__(world, id, json_data)
    return  

  def _load_params(self, json_data):
    super()._load_params(json_data)

    self.epochs = 1 if (self.EPOCHS_KEY not in json_data) else json_data[self.EPOCHS_KEY]
    self.batch_size = 1024 if (
        self.BATCH_SIZE_KEY not in json_data) else json_data[self.BATCH_SIZE_KEY]
    self.ratio_clip = 0.2 if (
        self.RATIO_CLIP_KEY not in json_data) else json_data[self.RATIO_CLIP_KEY]
    self.norm_adv_clip = 5 if (
        self.NORM_ADV_CLIP_KEY not in json_data) else json_data[self.NORM_ADV_CLIP_KEY]
    self.td_lambda = 0.95 if (
        self.TD_LAMBDA_KEY not in json_data) else json_data[self.TD_LAMBDA_KEY]
    self.tar_clip_frac = -1 if (
        self.TAR_CLIP_FRAC not in json_data) else json_data[self.TAR_CLIP_FRAC]
    self.actor_stepsize_decay = 0.5 if (
        self.ACTOR_STEPSIZE_DECAY not in json_data) else json_data[self.ACTOR_STEPSIZE_DECAY]

    num_procs = MPIUtil.get_num_procs()
    local_batch_size = int(self.batch_size / num_procs)
    min_replay_size = 2 * local_batch_size  # needed to prevent buffer overflow
    assert (self.replay_buffer_size > min_replay_size)

    self.replay_buffer_size = np.maximum(min_replay_size, self.replay_buffer_size)

    return

  def _build_nets(self, json_data):
    assert self.ACTOR_NET_KEY in json_data
    assert self.CRITIC_NET_KEY in json_data

    actor_net_name = json_data[self.ACTOR_NET_KEY]
    critic_net_name = json_data[self.CRITIC_NET_KEY]
    actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data
                                   ) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

    s_size = self.get_state_size()
    g_size = self.get_goal_size()
    a_size = self.get_action_size()

    print("INFO : ", actor_net_name, critic_net_name, actor_init_output_scale)
    print("SIZE INFO : ", s_size, g_size, a_size)

    self.netmap ={'actor':None, 'critic':None}

    # # setup input tensors

    # Create zero-initialized tensors
    self.s_pt = torch.zeros((1, s_size), dtype=torch.float32)
    self.a_pt = torch.zeros((1, a_size), dtype=torch.float32)
    self.tar_val_pt = torch.zeros((1, ), dtype=torch.float32)
    self.adv_pt = torch.zeros((1, ), dtype=torch.float32)
    self.g_pt = torch.zeros((1, g_size) if self.has_goal() else (1,), dtype=torch.float32)
    self.old_logp_pt = torch.zeros((1,), dtype=torch.float32)
    self.exp_mask_pt = torch.zeros((1,), dtype=torch.float32)
    # self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
    # self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
    # self.tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
    # self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
    # self.g_tf = tf.placeholder(tf.float32,
    #                            shape=([None, g_size] if self.has_goal() else None),
    #                            name="g")
    # self.old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")
    # self.exp_mask_tf = tf.placeholder(tf.float32, shape=[None], name="exp_mask")

    # with tf.variable_scope('main'):
    #   with tf.variable_scope('actor'):
    #     self.a_mean_tf = self._build_net_actor(actor_net_name, actor_init_output_scale)
    #   with tf.variable_scope('critic'):
    #     self.critic_tf = self._build_net_critic(critic_net_name)
    self.a_mean_pt = self._build_net_actor(actor_net_name, actor_init_output_scale)
    self.critic_pt = self._build_net_critic(critic_net_name)
    if (self.a_mean_pt != None):
      Logger.print2('Built actor net: ' + actor_net_name)

    if (self.critic_pt != None):
      Logger.print2('Built critic net: ' + critic_net_name)

    self.norm_a_std_pt = self.exp_params_curr.noise * torch.ones(a_size)
    # print("1: ", self.norm_a_std_pt, "b/c: ", self.norm_a_std_pt, "shape of those: ", self.norm_a_std_pt.size(), self.norm_a_std_pt.size())
    # norm_a_noise_pt = self.norm_a_std_pt * torch.randn(self.a_mean_pt.size())
    norm_a_noise_pt = self.norm_a_std_pt * torch.randn(self.a_mean_pt.out_size)
    # print("2: ", norm_a_noise_pt, "shape :", norm_a_noise_pt.shape)
    norm_a_noise_pt *= self.exp_mask_pt.view(-1, 1)
    # print("3: ", norm_a_noise_pt, "shape :", norm_a_noise_pt.shape)
    self.sample_a_pt = self.a_mean_pt(self.s_pt, self.g_pt) + norm_a_noise_pt * self.a_norm.std_pt
    # !!! -> net들어가는것이므로 샘플로만 써야함.
    # print("3.4: ", self.sample_a_pt, "shape :", self.sample_a_pt.shape)
    #not implementd yet for the below calc_logp_gaussian
    self.sample_a_logp_pt = PTUtil.calc_logp_gaussian(x_pt=norm_a_noise_pt, 
                                                      mean_pt=None, 
                                                      std_pt=self.norm_a_std_pt)
    # print(self.sample_a_logp_pt, "should not be None!") #토치유틸 들어가서, 임플리먼트 (9/16)
    #74.76457 이값이 eval_actor에서도 계속 나오는데, 토치 문법상 그럴수 밖에없는것으로 보이나, 
    # 어떻게 tf1.15코드는 그렇지 않은것인지 구현상 차이점 확인 필요 (~10.5)

    # self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)
    # norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(shape=tf.shape(self.a_mean_tf))
    # norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
    # self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
    # self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(x_tf=norm_a_noise_tf,
    #                                                   mean_tf=None,
    #                                                   std_tf=self.norm_a_std_tf)

    if self.netmap is None:
      Logger.print2('netmap is not created -> creating a new netmap object')
      self.netmap ={}

    Logger.print2('Actor net and critic net are added to the netmap!!')
    self.netmap['actor'] = self.a_mean_pt
    self.netmap['critic'] = self.critic_pt
    return 

  def _build_losses(self, json_data):
    actor_weight_decay = 0 if (
        self.ACTOR_WEIGHT_DECAY_KEY not in json_data) else json_data[self.ACTOR_WEIGHT_DECAY_KEY]
    critic_weight_decay = 0 if (
        self.CRITIC_WEIGHT_DECAY_KEY not in json_data) else json_data[self.CRITIC_WEIGHT_DECAY_KEY]

    #make it global
    self.critic_wdecay_value = critic_weight_decay
    self.actor_wdecay_value = actor_weight_decay
    # self.actor_wdecay_value = 0
    # norm_val_diff = self.val_norm.normalize_tf(self.tar_val_tf) - self.val_norm.normalize_tf(
    #     self.critic_tf)
    # norm_val_diff = self.val_norm.normalize_pt(self.tar_val_pt) - self.val_norm.normalize_pt(
    #     self.critic_pt) 
    # !!! -> (윗윗코드는 원본, 바로 윗 코드는 내가 트라이해본 토치버젼,,,but err) 
    # !!! -> self.tar_val_pt는 텐서고 self.critic_pt는 net 
    # !!! -> 이걸 보면, build loss단계랑 train쪽을 합칠 필요가 있어보임..
    # self.critic_loss_tf = 0.5 * tf.reduce_mean(tf.square(norm_val_diff))

    # if (critic_weight_decay != 0):
    #   self.critic_loss_tf += critic_weight_decay * self._weight_decay_loss('main/critic')

    # norm_tar_a_tf = self.a_norm.normalize_tf(self.a_tf)
    # self._norm_a_mean_tf = self.a_norm.normalize_tf(self.a_mean_tf)

    # self.logp_tf = TFUtil.calc_logp_gaussian(norm_tar_a_tf, self._norm_a_mean_tf,
    #                                          self.norm_a_std_tf)
    # ratio_tf = tf.exp(self.logp_tf - self.old_logp_tf)
    # actor_loss0 = self.adv_tf * ratio_tf
    # actor_loss1 = self.adv_tf * tf.clip_by_value(ratio_tf, 1.0 - self.ratio_clip,
    #                                              1 + self.ratio_clip)
    # self.actor_loss_tf = -tf.reduce_mean(tf.minimum(actor_loss0, actor_loss1))

    # norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
    # norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)
    # a_bound_loss = TFUtil.calc_bound_loss(self._norm_a_mean_tf, norm_a_bound_min, norm_a_bound_max)
    # self.actor_loss_tf += a_bound_loss

    # if (actor_weight_decay != 0):
    #   self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss('main/actor')

    # # for debugging
    # self.clip_frac_tf = tf.reduce_mean(
    #     tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip)))

    return

  def _build_solvers(self, json_data):
    actor_stepsize = 0.001 if (
        self.ACTOR_STEPSIZE_KEY not in json_data) else json_data[self.ACTOR_STEPSIZE_KEY]
    actor_momentum = 0.9 if (
        self.ACTOR_MOMENTUM_KEY not in json_data) else json_data[self.ACTOR_MOMENTUM_KEY]
    critic_stepsize = 0.01 if (
        self.CRITIC_STEPSIZE_KEY not in json_data) else json_data[self.CRITIC_STEPSIZE_KEY]
    critic_momentum = 0.9 if (
        self.CRITIC_MOMENTUM_KEY not in json_data) else json_data[self.CRITIC_MOMENTUM_KEY]
    
    #make them as a global variable
    self.optimizer_type = optimizer_type = 'sgd'
    self.critic_stepsize = critic_stepsize
    self.actor_stepsize = actor_stepsize
    self.critic_momentum = critic_momentum
    self.actor_momentum = actor_momentum

    # 실제 논문식으로 wdecay 구현할지, 아님 torch내장 방식으로 구현할지 고민 (디폴트로 0 넣어버리면 torch wdecay 꺼지는듯?)
    # Bias는 따로 빼? deepmimic에서 하려는 방식 (youtube : https://www.youtube.com/watch?v=hZE4Nja5zKA)
    self.apply_wdecay_manually = False
    if self.apply_wdecay_manually:
      self.torch_critic_wdecay = 0  
      self.torch_actor_wdecay = 0  
    else: 
      self.torch_critic_wdecay = self.critic_wdecay_value
      self.torch_actor_wdecay = self.actor_wdecay_value

    bias_included = False

    if not bias_included:

      if self.optimizer_type == 'sgd':
        self.critic_optimizer = optim.SGD(self._seperate_params_wdecay(self.critic_pt, self.torch_critic_wdecay), 
                                          lr=self.critic_stepsize, 
                                          momentum=self.critic_momentum)
        self.actor_optimizer = optim.SGD(self._seperate_params_wdecay(self.a_mean_pt, self.torch_actor_wdecay), 
                                        lr=self.actor_stepsize, 
                                        momentum=self.actor_momentum)
      else:
        self.critic_optimizer = optim.Adam(self._seperate_params_wdecay(self.critic_pt, self.torch_critic_wdecay), 
                                          lr=self.critic_stepsize)
        self.actor_optimizer = optim.Adam(self._seperate_params_wdecay(self.a_mean_pt, self.torch_actor_wdecay), 
                                          lr=self.actor_stepsize)
    
    else:
      if self.optimizer_type == 'sgd':
        self.critic_optimizer = optim.SGD(self.critic_pt.parameters(), lr=self.critic_stepsize, momentum=self.critic_momentum, weight_decay=self.torch_critic_wdecay)
        self.actor_optimizer = optim.SGD(self.a_mean_pt.parameters(), lr=self.actor_stepsize, momentum=self.actor_momentum, weight_decay=self.torch_actor_wdecay)
      else:
        self.critic_optimizer = optim.Adam(self.critic_pt.parameters(), lr=self.critic_stepsize, weight_decay=self.torch_critic_wdecay)
        self.actor_optimizer = optim.Adam(self.a_mean_pt.parameters(), lr=self.actor_stepsize, weight_decay=self.torch_actor_wdecay)

    # critic_vars = self._tf_vars('main/critic')
    # critic_opt = tf.train.MomentumOptimizer(learning_rate=critic_stepsize,
    #                                         momentum=critic_momentum)
    # self.critic_grad_tf = tf.gradients(self.critic_loss_tf, critic_vars)
    # self.critic_solver = MPISolver(self.sess, critic_opt, critic_vars)

    # self._actor_stepsize_tf = tf.get_variable(dtype=tf.float32,
    #                                           name='actor_stepsize',
    #                                           initializer=actor_stepsize,
    #                                           trainable=False)
    # self._actor_stepsize_ph = tf.get_variable(dtype=tf.float32, name='actor_stepsize_ph', shape=[])
    # self._actor_stepsize_update_op = self._actor_stepsize_tf.assign(self._actor_stepsize_ph)

    # actor_vars = self._tf_vars('main/actor')
    # actor_opt = tf.train.MomentumOptimizer(learning_rate=self._actor_stepsize_tf,
    #                                        momentum=actor_momentum)
    # self.actor_grad_tf = tf.gradients(self.actor_loss_tf, actor_vars)
    # self.actor_solver = MPISolver(self.sess, actor_opt, actor_vars)

    return

  def _decide_action(self, s, g):
    # with self.sess.as_default(), self.graph.as_default():
    self._exp_action = self._enable_stoch_policy() and MathUtil.flip_coin(
        self.exp_params_curr.rate) #이거 끄는게 이유였음..train안도는..
    #   #print("_decide_action._exp_action=",self._exp_action)
    #   a, logp = self._eval_actor(s, g, self._exp_action)
    a, logp = self._eval_actor(s, g, self._exp_action)
    # print("check what s is " , type(s)) #numpy.ndarray    
    # print("/ : ", a.shape, a[0].shape) #/ :  (1, 36) (36,)
    # print("/ : ", a[0], logp[0])
    return a[0], logp[0]
    #;,; for default running..

  def _eval_actor(self, s_np, g_np, enable_exp):
    # print("enable_exp : ", enable_exp)
    # s = np.reshape(s, [-1, self.get_state_size()])
    # g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None

    self.s_pt = torch.from_numpy(s_np).to(dtype=torch.float32)
    # print("g_np : ", g_np, type(g_np))
    self.g_pt = torch.from_numpy(g_np).to(dtype=torch.float32)
    self.exp_mask_pt = torch.tensor([1 if enable_exp else 0])
    # print("self.exp_mask_pt: ", self.exp_mask_pt) 

    # feed = {self.s_tf: s, self.g_tf: g, self.exp_mask_tf: np.array([1 if enable_exp else 0])}
    # a, logp = self.sess.run([self.sample_a_tf, self.sample_a_logp_tf], feed_dict=feed)
    s = torch.reshape(self.s_pt, [-1, self.get_state_size()])
    g = torch.reshape(self.g_pt, [-1, self.get_goal_size()]) if self.has_goal() else None

    # norm_a_std_pt = self.exp_params_curr.noise * torch.ones(self.a_mean_pt.out_size)
    # print("1: ", self.norm_a_std_pt, "b/c: ", self.norm_a_std_pt, "shape of those: ", self.norm_a_std_pt.size(), self.norm_a_std_pt.size())
    # norm_a_noise_pt = self.norm_a_std_pt * torch.randn(self.a_mean_pt.size())
    norm_a_noise_pt = self.norm_a_std_pt * torch.randn(self.a_mean_pt.out_size)
    norm_a_noise_pt *= self.exp_mask_pt.view(-1, 1)

    
    with torch.no_grad():
      a_pt = self.a_mean_pt(s, g) + norm_a_noise_pt * self.a_norm.std_pt
      a_logp_pt = PTUtil.calc_logp_gaussian(
        x_pt=norm_a_noise_pt, 
        mean_pt=None, 
        std_pt=self.norm_a_std_pt)
      
      a = a_pt.detach().numpy()
      a_logp = a_logp_pt.detach().numpy()

    return a, a_logp 
  #문제느 log_p가 계속 같은 값이 나옴. (path넣을때 path.check_vals()에서 쓰는거 말곤 쓰는데 딱히 없어보이는데)
  #계속 돌리니 아니네! tf1.15와 같음을 확인!

  def _train_step(self):
    # print("Train Step of ppo_agent!") #rl_agent에서 바로 여기로 넘어오는듯?
    adv_eps = 1e-5

    start_idx = self.replay_buffer.buffer_tail
    end_idx = self.replay_buffer.buffer_head
    # print("start_idx: ", start_idx, "and end_idx: ", end_idx)
    assert (start_idx == 0)
    assert (self.replay_buffer.get_current_size() <= self.replay_buffer.buffer_size
           )  # must avoid overflow
    assert (start_idx < end_idx)

    idx = np.array(list(range(start_idx, end_idx)))
    end_mask = self.replay_buffer.is_path_end(idx)
    end_mask = np.logical_not(end_mask)

    vals = self._compute_batch_vals(start_idx, end_idx)
    # print("vals shape: ",  vals.shape) #(4328,)
    new_vals = self._compute_batch_new_vals(start_idx, end_idx, vals)
    # print("and new_vals: ", new_vals) #(4328,)

    valid_idx = idx[end_mask]
    exp_idx = self.replay_buffer.get_idx_filtered(self.EXP_ACTION_FLAG).copy()
    num_valid_idx = valid_idx.shape[0]
    num_exp_idx = exp_idx.shape[0]
    exp_idx = np.column_stack([exp_idx, np.array(list(range(0, num_exp_idx)), dtype=np.int32)])

    local_sample_count = valid_idx.size
    global_sample_count = int(MPIUtil.reduce_sum(local_sample_count))
    mini_batches = int(np.ceil(global_sample_count / self.mini_batch_size))

    adv = new_vals[exp_idx[:, 0]] - vals[exp_idx[:, 0]]
    new_vals = np.clip(new_vals, self.val_min, self.val_max)

    adv_mean = np.mean(adv)
    adv_std = np.std(adv)
    adv = (adv - adv_mean) / (adv_std + adv_eps)
    adv = np.clip(adv, -self.norm_adv_clip, self.norm_adv_clip)
    # print("adv_mean / std: ", adv_mean, adv_std)
    # print("adv : ", adv)


    critic_loss = 0
    actor_loss = 0
    actor_clip_frac = 0

    for e in range(self.epochs):
      np.random.shuffle(valid_idx)
      np.random.shuffle(exp_idx)

      for b in range(mini_batches):
        batch_idx_beg = b * self._local_mini_batch_size
        batch_idx_end = batch_idx_beg + self._local_mini_batch_size

        critic_batch = np.array(range(batch_idx_beg, batch_idx_end), dtype=np.int32)
        actor_batch = critic_batch.copy()
        critic_batch = np.mod(critic_batch, num_valid_idx)
        actor_batch = np.mod(actor_batch, num_exp_idx)
        shuffle_actor = (actor_batch[-1] < actor_batch[0]) or (actor_batch[-1] == num_exp_idx - 1)
        
        checker = lambda variable : (variable, type(variable), variable.shape)
        rpc_bool = recycle_possible_check = False
        if rpc_bool: print("valid_idx: ", checker(valid_idx)) #(3676,)
        critic_batch = valid_idx[critic_batch]
        if rpc_bool: print("critic_batch: ", checker(critic_batch)) #(256,)
        if rpc_bool: print("exp_idx: ", checker(exp_idx)) #(3676, 2)
        actor_batch = exp_idx[actor_batch]
        if rpc_bool: print("actor_batch: ", checker(actor_batch)) #(256, 2)
        critic_batch_vals = new_vals[critic_batch]
        if rpc_bool: print("critic_batch_vals: ", checker(critic_batch_vals)) #(256,)
        actor_batch_adv = adv[actor_batch[:, 1]]
        if rpc_bool: print("actor_batch_adv: ", checker(actor_batch_adv)) #(256,)

        trainrpcchk = False
        critic_s = self.replay_buffer.get('states', critic_batch) #(256, 197)
        if trainrpcchk: print("critic_s: ", checker(critic_s))
        critic_g = self.replay_buffer.get('goals', critic_batch) if self.has_goal() else None
        if trainrpcchk and self.has_goal(): print("critic_g: ", checker(critic_g)) 
        curr_critic_loss = self._update_critic(critic_s, critic_g, critic_batch_vals)

        actor_s = self.replay_buffer.get("states", actor_batch[:, 0]) #(256, 197)
        if trainrpcchk: print("actor_s: ", checker(actor_s))
        actor_g = self.replay_buffer.get("goals", actor_batch[:, 0]) if self.has_goal() else None
        if trainrpcchk and self.has_goal(): print("actor_g: ", checker(actor_g))
        actor_a = self.replay_buffer.get("actions", actor_batch[:, 0]) #(256, 36)
        if trainrpcchk: print("actor_a: ", checker(actor_a))
        actor_logp = self.replay_buffer.get("logps", actor_batch[:, 0])
        curr_actor_loss, curr_actor_clip_frac = self._update_actor(actor_s, actor_g, actor_a,
                                                                   actor_logp, actor_batch_adv)

        critic_loss += curr_critic_loss
        actor_loss += np.abs(curr_actor_loss)
        actor_clip_frac += curr_actor_clip_frac

        if (shuffle_actor):
          np.random.shuffle(exp_idx)

    total_batches = mini_batches * self.epochs
    # print("mini_batches : ", mini_batches, "self.epochs: ", self.epochs) #mini_batches :  15 self.epochs:  1
    # print("critic loss (1) :", critic_loss)
    critic_loss /= total_batches
    actor_loss /= total_batches
    actor_clip_frac /= total_batches
    # print("critic loss (2) :", critic_loss)
    critic_loss = MPIUtil.reduce_avg(critic_loss)
    actor_loss = MPIUtil.reduce_avg(actor_loss)
    actor_clip_frac = MPIUtil.reduce_avg(actor_clip_frac)
    print("critic loss (3) :", critic_loss)
    # critic_stepsize = self.critic_solver.get_stepsize()
    critic_stepsize = self.critic_stepsize
    actor_stepsize = self.update_actor_stepsize(actor_clip_frac)

    self.logger.log_tabular('Critic_Loss', critic_loss)
    self.logger.log_tabular('Critic_Stepsize', critic_stepsize)
    self.logger.log_tabular('Actor_Loss', actor_loss)
    self.logger.log_tabular('Actor_Stepsize', actor_stepsize)
    self.logger.log_tabular('Clip_Frac', actor_clip_frac)
    self.logger.log_tabular('Adv_Mean', adv_mean)
    self.logger.log_tabular('Adv_Std', adv_std)

    self.replay_buffer.clear()

    return

  def _get_iters_per_update(self):
    return 1

  def _valid_train_step(self):

    samples = self.replay_buffer.get_current_size()
    # print("samples : ", samples)
    # print(self.EXP_ACTION_FLAG, ".d.d.d.d..d."*200) #True as tf1.15 ver
    exp_samples = self.replay_buffer.count_filtered(self.EXP_ACTION_FLAG)
    # print("chekk: ", self.replay_buffer._sample_buffers[self.EXP_ACTION_FLAG].count)
    # print("exp_samples : ", exp_samples)
    global_sample_count = int(MPIUtil.reduce_sum(samples))
    # print("global_sample_count : ", global_sample_count)
    global_exp_min = int(MPIUtil.reduce_min(exp_samples))
    # print("global_exp_min : ", global_exp_min)
    # print("batch size? ", self.batch_size)
    # print("Two booleans check : ", (global_sample_count > self.batch_size) ,"// ", (global_exp_min > 0))
    return (global_sample_count > self.batch_size) and (global_exp_min > 0)


  def _compute_batch_vals(self, start_idx, end_idx):
    states = self.replay_buffer.get_all("states")[start_idx:end_idx]
    goals = self.replay_buffer.get_all("goals")[start_idx:end_idx] if self.has_goal() else None
    # print("states: ", type(states), "goals: ", type(goals)) #states:  <class 'numpy.ndarray'> goals:  <class 'NoneType'>
    # print("states + its shp: ", states, states.shape, "goals: ", type(goals)) #(4119, 197)
    idx = np.array(list(range(start_idx, end_idx)))
    is_end = self.replay_buffer.is_path_end(idx)
    is_fail = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Fail)
    is_succ = self.replay_buffer.check_terminal_flag(idx, Env.Terminate.Succ)
    is_fail = np.logical_and(is_end, is_fail)
    is_succ = np.logical_and(is_end, is_succ)
    vals = self._eval_critic(states, goals)
    # print("vals_shape :: ", vals.shape)
    # print("vals : ", vals, vals.shape) #(4119, 36)
    vals[is_fail] = self.val_fail
    vals[is_succ] = self.val_succ

    return vals
    # return None


  def _compute_batch_new_vals(self, start_idx, end_idx, val_buffer): #vals in train_step coming into val_buffer
    rewards = self.replay_buffer.get_all("rewards")[start_idx:end_idx]
    # print("Reward [] : ", rewards, rewards.shape) #(4105,) when states.shape ==(4105,197)

    if self.discount == 0:
      new_vals = rewards.copy()
    else:
      new_vals = np.zeros_like(val_buffer)

      curr_idx = start_idx
      while curr_idx < end_idx:
        idx0 = curr_idx - start_idx
        idx1 = self.replay_buffer.get_path_end(curr_idx) - start_idx
        r = rewards[idx0:idx1]
        v = val_buffer[idx0:(idx1 + 1)]

        new_vals[idx0:idx1] = RLUtil.compute_return(r, self.discount, self.td_lambda, v)
        curr_idx = idx1 + start_idx + 1

    return new_vals
  
  def _update_critic(self, s_np, g_np, tar_vals):

    #1. Prepare network input
    #1-1. updating tensor variables
    self.s_pt = torch.from_numpy(s_np).to(dtype=torch.float32)
    if not self.has_goal():
      g_np = np.array([])
    self.g_pt = torch.from_numpy(g_np).to(dtype=torch.float32)
    self.tar_val_pt = torch.tensor(tar_vals)

    #1-2. preprocess(reshape) data
    s = torch.reshape(self.s_pt, [-1, self.get_state_size()])
    g = torch.reshape(self.g_pt, [-1, self.get_goal_size()]) if self.has_goal() else None
    
    #1-3 pass through the network
    value_net_out = self.critic_pt(s,g)

    #2. do some loss calculations
    norm_val_diff = self.val_norm.normalize_pt(self.tar_val_pt) - self.val_norm.normalize_pt(value_net_out) 
    critic_loss = 0.5 * torch.mean(torch.square(norm_val_diff))

    critic_weight_decay = self.critic_wdecay_value if self.critic_wdecay_value else 0
    if (critic_weight_decay != 0):
      # old way
      # self.critic_loss_pt += critic_weight_decay * self._weight_decay_loss('main/critic')

      #my own way (old way와 맞는지 체크 예정)
      self.do_weight_decay = True

    # 3. Backpropagate and update weights
    self.critic_optimizer.zero_grad()  # Clear gradients
    critic_loss.backward()  # Backpropagate the loss
    self.critic_optimizer.step()  # Update the critic network's weights using the optimizer (SGD in this case)

    return critic_loss.detach().numpy()  # Return the loss as a scalar


  # def _update_critic_old(self, s, g, tar_vals):
    # feed = {self.s_tf: s, self.g_tf: g, self.tar_val_tf: tar_vals}
    # self.tar_val_pt = torch.from_numpy(tar_vals) #이렇게 해야 loss에 더미값이 안들어가지 않을까..
    
    # self.tar_val_pt = torch.tensor(tar_vals)
    # loss, grads = self.sess.run([self.critic_loss_tf, self.critic_grad_tf], feed)
    # self.critic_solver.update(grads)
    # return loss

  
  # def _update_actor_old(self, s, g, a, logp, adv):
    # feed = {self.s_tf: s, self.g_tf: g, self.a_tf: a, self.adv_tf: adv, self.old_logp_tf: logp}

    # loss, grads, clip_frac = self.sess.run(
    #     [self.actor_loss_tf, self.actor_grad_tf, self.clip_frac_tf], feed)
    # self.actor_solver.update(grads)

    # return loss, clip_frac
  
  def _update_actor(self, s_np, g_np, a_np, logp, adv_np):

    # print("==== > s: ", s_np)

    #1. Prepare network input
    #1-1. updating tensor variables
    self.s_pt = torch.tensor(s_np).to(dtype=torch.float32)
    if not self.has_goal():
      g_np = np.array([])
    self.g_pt = torch.tensor(g_np).to(dtype=torch.float32)
    self.a_pt = torch.tensor(a_np).to(dtype=torch.float32)   
    self.adv_pt = torch.tensor(adv_np).to(dtype=torch.float32) 
    self.old_logp_pt = torch.tensor(logp).to(dtype=torch.float32) 

    #1-2. preprocess(reshape) data
    s = torch.reshape(self.s_pt, [-1, self.get_state_size()])
    g = torch.reshape(self.g_pt, [-1, self.get_goal_size()]) if self.has_goal() else None
    

    #1-3 pass through the network
    actor_net_out = self.a_mean_pt(s,g)    

    #2. do some loss calculations
    self._norm_a_mean_pt = self.a_norm.normalize_pt(actor_net_out)
    # print("self._norm_a_mean_pt : ",self._norm_a_mean_pt )
    norm_tar_a_pt = self.a_norm.normalize_pt(self.a_pt)
    
    self.logp_pt =  PTUtil.calc_logp_gaussian(x_pt=norm_tar_a_pt, 
                                              mean_pt=self._norm_a_mean_pt, 
                                              std_pt=self.norm_a_std_pt)
    ratio_pt = torch.exp(self.logp_pt - self.old_logp_pt)
    surrogate1 = self.adv_pt * ratio_pt
    surrogate2 = self.adv_pt * torch.clamp(ratio_pt, 1.0 - self.ratio_clip, 1.0 + self.ratio_clip)
    actor_loss_pt = -1 * torch.mean(torch.min(surrogate1, surrogate2))
    # print("self.a_bound_min : ", self.a_bound_min)
    norm_a_bound_min = self.a_norm.normalize(self.a_bound_min)
    norm_a_bound_max = self.a_norm.normalize(self.a_bound_max)

    a_bound_loss = PTUtil.calc_bound_loss(actor_net_out, torch.tensor(norm_a_bound_min), torch.tensor(norm_a_bound_max))
    actor_loss_pt += a_bound_loss

    # if (actor_weight_decay != 0):
    #   self.actor_loss_tf += actor_weight_decay * self._weight_decay_loss('main/actor')

    # 2.5. for debugging (have to convert to torch) (Chat gpt do the below)
    # self.clip_frac_tf = tf.reduce_mean(
    #     tf.to_float(tf.greater(tf.abs(ratio_tf - 1.0), self.ratio_clip)))
    clip_frac_pt = torch.mean((torch.abs(ratio_pt - 1.0) > self.ratio_clip).to(torch.float32))


    # 3. Backpropagate and update weights (Chat gpt do the below)
    self.actor_optimizer.zero_grad()  # Clear gradients
    actor_loss_pt.backward()  # Backpropagate the loss
    self.actor_optimizer.step()  # Update the actor network's weights using the optimizer (SGD in this case)

    return actor_loss_pt.detach().numpy(), clip_frac_pt.detach().numpy()


  def update_actor_stepsize(self, clip_frac):
    clip_tol = 1.5
    step_scale = 2
    max_stepsize = 1e-2
    min_stepsize = 1e-8
    warmup_iters = 5

    # actor_stepsize = self.actor_solver.get_stepsize()
    actor_stepsize = self.actor_stepsize
    if (self.tar_clip_frac >= 0 and self.iter > warmup_iters):
      min_clip = self.tar_clip_frac / clip_tol
      max_clip = self.tar_clip_frac * clip_tol
      under_tol = clip_frac < min_clip
      over_tol = clip_frac > max_clip

      if (over_tol or under_tol):
        if (over_tol):
          actor_stepsize *= self.actor_stepsize_decay
        else:
          actor_stepsize /= self.actor_stepsize_decay

        actor_stepsize = np.clip(actor_stepsize, min_stepsize, max_stepsize)
        self.set_actor_stepsize(actor_stepsize) 

    return actor_stepsize


  def set_actor_stepsize(self, stepsize):
    # feed = {
    #     self._actor_stepsize_ph: stepsize,
    # }
    # self.sess.run(self._actor_stepsize_update_op, feed)
    self.actor_stepsize =stepsize
    if self.optimizer_type == 'sgd':     
      self.actor_optimizer = optim.SGD(self.a_mean_pt.parameters(), lr=self.actor_stepsize, momentum=self.actor_momentum, weight_decay=self.torch_actor_wdecay)
    else:      
      self.actor_optimizer = optim.Adam(self.a_mean_pt.parameters(), lr=self.actor_stepsize, weight_decay=self.torch_actor_wdecay)
    
    return
  
  
