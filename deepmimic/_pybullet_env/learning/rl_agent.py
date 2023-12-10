import numpy as np
import copy
import os
import time
import sys
from abc import abstractmethod
import abc
if sys.version_info >= (3, 4):
  ABC = abc.ABC
else:
  ABC = abc.ABCMeta('ABC', (), {})

from enum import Enum

from _pybullet_env.learning.path import *
from _pybullet_env.learning.exp_params import ExpParams
from _pybullet_env.learning.normalizer import Normalizer
from _pybullet_env.learning.replay_buffer import ReplayBuffer
from _pybullet_utils.logger import Logger
import _pybullet_utils.mpi_util as MPIUtil
import _pybullet_utils.math_util as MathUtil


class RLAgent(ABC):

  class Mode(Enum):
    TRAIN = 0
    TEST = 1
    TRAIN_END = 2

  NAME = "None"

  UPDATE_PERIOD_KEY = "UpdatePeriod"
  ITERS_PER_UPDATE = "ItersPerUpdate"
  DISCOUNT_KEY = "Discount"
  MINI_BATCH_SIZE_KEY = "MiniBatchSize"
  REPLAY_BUFFER_SIZE_KEY = "ReplayBufferSize"
  INIT_SAMPLES_KEY = "InitSamples"
  NORMALIZER_SAMPLES_KEY = "NormalizerSamples"

  OUTPUT_ITERS_KEY = "OutputIters"
  INT_OUTPUT_ITERS_KEY = "IntOutputIters"
  TEST_EPISODES_KEY = "TestEpisodes"

  EXP_ANNEAL_SAMPLES_KEY = "ExpAnnealSamples"
  EXP_PARAM_BEG_KEY = "ExpParamsBeg"
  EXP_PARAM_END_KEY = "ExpParamsEnd"

  def __init__(self, world, id, json_data):
    print("=================> 4. Reinforcement Learning AGENT's world", world)
    
    self.world = world
    print("Myi's exp: self.world.agents", self.world.agents) 
    #하나가 들어가있네?
    self.id = id
    self.logger = Logger()
    self._mode = self.Mode.TRAIN
    self.cmiu_rl =  False

    assert self._check_action_space(), \
        Logger.print2("Invalid action space, got {:s}".format(str(self.get_action_space())))

    self._enable_training = False
    self.path = Path()
    self.iter = int(0)
    self.start_time = time.time()
    self._update_counter = 0

    self.update_period = 1.0  # simulated time (seconds) before each training update
    self.iters_per_update = int(1)
    self.discount = 0.95
    self.mini_batch_size = int(32)
    self.replay_buffer_size = int(50000)
    self.init_samples = int(1000)
    self.normalizer_samples = np.inf
    self._local_mini_batch_size = self.mini_batch_size  # batch size for each work for multiprocessing
    self._need_normalizer_update = True
    self._total_sample_count = 0

    self._output_dir = ""
    self._int_output_dir = ""
    self.output_iters = 100
    self.int_output_iters = 100

    self.train_return = 0.0
    self.test_episodes = int(0)
    self.test_episode_count = int(0)
    self.test_return = 0.0
    self.avg_test_return = 0.0

    self.exp_anneal_samples = 320000
    self.exp_params_beg = ExpParams()
    self.exp_params_end = ExpParams()
    self.exp_params_curr = ExpParams()

    

    self._load_params(json_data)
    # print("!")
    self._build_replay_buffer(self.replay_buffer_size)
    # print("!"*2)
    self._build_normalizers() #!deepdive here
    # print("!"*3)
    self._build_bounds()
    # print("!"*4)
    self.reset() #path clear하는 역할

    return

  def __str__(self):
    if self.cmiu_rl: print("=================>__str__ of rl_agents")
    action_space_str = str(self.get_action_space())
    info_str = ""
    info_str += '"ID": {:d},\n "set_output_dirType": "{:s}",\n "ActionSpace": "{:s}",\n "StateDim": {:d},\n "GoalDim": {:d},\n "ActionDim": {:d}'.format(
        self.id, self.NAME, action_space_str[action_space_str.rfind('.') + 1:],
        self.get_state_size(), self.get_goal_size(), self.get_action_size())
    return "{\n" + info_str + "\n}"

  def get_output_dir(self):
    if self.cmiu_rl: print("=================>get_output_dir of rl_agents")
    return self._output_dir

  def set_output_dir(self, out_dir):
    if self.cmiu_rl: print("=================>set_output_dir of rl_agents")
    self._output_dir = out_dir
    if (self._output_dir != ""):
      self.logger.configure_output_file(out_dir + "/agent" + str(self.id) + "_log.txt")
    return

  output_dir = property(get_output_dir, set_output_dir)

  def get_int_output_dir(self):
    if self.cmiu_rl: print("=================>get_int_output_dir of rl_agents")
    return self._int_output_dir

  def set_int_output_dir(self, out_dir):
    if self.cmiu_rl: print("=================>set_int_output_dir of rl_agents")
    self._int_output_dir = out_dir
    return

  int_output_dir = property(get_int_output_dir, set_int_output_dir)

  def reset(self):
    if self.cmiu_rl: print("=================>reset of rl_agents")
    self.path.clear()
    return

  def update(self, timestep):
    if self.cmiu_rl: print("=================>update of rl_agents")
    # print("timestep // 2: ", timestep) #rl world에서 agent.update하면 이거 타는듯.
    # print("need new action: ", self.need_new_action()) #인쇄 찍으면 에러가 남..
    if self.need_new_action():
      # print("Yest")
      # print("need new! update_new_action!!!")
      self._update_new_action()
    # else: #예스노 번갈아가며
      # print("No")

    if (self._mode == self.Mode.TRAIN and self.enable_training):
      self._update_counter += timestep
      # print("update counter : ", self._update_counter, "update period: ", self.update_period)
      # log followed :
      # ... 생략
      # / S:  (197,) shape
      # A:  [0.25775856, -0.04327505, -0.06654227, 0.20791441, -0.1950356, 0.00437055, 0.0224415, 0.21422549, 0.55615604, -0.11404589, 0.07292482, 0.2549699, -0.8531083, 0.6412591, 0.02251127, 0.03166192, 0.18994713, -0.1431605, 0.06846526, 0.03927562, 0.32796186, 0.9637947, -0.15190774, 0.02487963, 0.09093633, 0.18229035, -0.587962, 0.16542739, -0.08597768, 0.01767025, 0.28034002, 0.6529777, -0.0049221, 
      # 0.01850244, 0.1911766, 0.9632412]
      # update counter :  0.9666666666666422 update period:  1.0
      # update counter :  0.9708333333333089 update period:  1.0
      # update counter :  0.9749999999999756 update period:  1.0
      # update counter :  0.9791666666666422 update period:  1.0
      # update counter :  0.9833333333333089 update period:  1.0
      # update counter :  0.9874999999999755 update period:  1.0
      # update counter :  0.9916666666666422 update period:  1.0
      # update counter :  0.9958333333333088 update period:  1.0
      # update counter :  0.9999999999999755 update period:  1.0
      # / S:  (197,) shape
      # A:  [0.25775856, -0.04327505, -0.06654227, 0.20791441, -0.1950356, 0.00437055, 0.0224415, 0.21422549, 0.55615604, -0.11404589, 0.07292482, 0.2549699, -0.8531083, 0.6412591, 0.02251127, 0.03166192, 0.18994713, -0.1431605, 0.06846526, 0.03927562, 0.32796186, 0.9637947, -0.15190774, 0.02487963, 0.09093633, 0.18229035, -0.587962, 0.16542739, -0.08597768, 0.01767025, 0.28034002, 0.6529777, -0.0049221, 
      # 0.01850244, 0.1911766, 0.9632412]

      while self._update_counter >= self.update_period:
        # print("exceeded!")
        self._train() #torch_agent까지 갔다가 super()._train()으로 되돌아오는 모양새..
        self._update_exp_params()
        self.world.env.set_sample_count(self._total_sample_count)
        self._update_counter -= self.update_period

    return

  def end_episode(self): #rl_world._end_episode_agents() 발 
    if self.cmiu_rl: print("=================>end_episode of rl_agents")
    if (self.path.pathlength() > 0):
      self._end_path()

      if (self._mode == self.Mode.TRAIN or self._mode == self.Mode.TRAIN_END):
        # print("RL Agent - Train mode! ")
        if (self.enable_training and self.path.pathlength() > 0):
          # print("RL Agent - store path!")
          # print("When When When"*157)
          # print("self.path : ", self.path.actions, self.path.get_pathlen())
          # print("self.get_pathlen() : ", self.path.get_pathlen()) 
          self._store_path(self.path)
      elif (self._mode == self.Mode.TEST):
        # print("RL Agent - Test mode? ")
        self._update_test_return(self.path)
      else:
        assert False, Logger.print2("Unsupported RL agent mode" + str(self._mode))

      self._update_mode()
    return

  def has_goal(self):
    if self.cmiu_rl: print("=================>has_goal of rl_agents")
    return self.get_goal_size() > 0

  def predict_val(self):
    if self.cmiu_rl: print("=================>predict_val of rl_agents")
    return 0

  def get_enable_training(self):
    if self.cmiu_rl: print("=================>get_enable_training of rl_agents")
    return self._enable_training

  def set_enable_training(self, enable):
    if self.cmiu_rl: print("=================>set_enable_training of rl_agents")
    print("set_enable_training=", enable)
    self._enable_training = enable
    if (self._enable_training):
      self.reset()
    return

  enable_training = property(get_enable_training, set_enable_training)

  def enable_testing(self):
    if self.cmiu_rl: print("=================>enable_testing of rl_agents")
    return self.test_episodes > 0

  def get_name(self):
    if self.cmiu_rl: print("=================>get_name of rl_agents")
    return self.NAME
  
  ##Myi Added _create_modelmap
  @abstractmethod
  def _create_modelmap(self):
    if self.cmiu_rl: print("=================>_create_modelmap of rl_agents")
    pass

  @abstractmethod
  def save_model(self, out_path):
    if self.cmiu_rl: print("=================>save_model of rl_agents")
    pass

  @abstractmethod
  def load_model(self, in_path):
    if self.cmiu_rl: print("=================>load_model of rl_agents")
    pass

  @abstractmethod
  def _decide_action(self, s, g):
    if self.cmiu_rl: print("=================>_decide_action of rl_agents")
    pass

  @abstractmethod
  def _get_output_path(self):
    if self.cmiu_rl: print("=================>_get_output_path of rl_agents")
    pass

  @abstractmethod
  def _get_int_output_path(self):
    if self.cmiu_rl: print("=================>_get_int_output_path of rl_agents")
    pass

  @abstractmethod
  def _train_step(self):
    if self.cmiu_rl: print("=================>_train_step of rl_agents")
    pass

  @abstractmethod
  def _check_action_space(self):
    if self.cmiu_rl: print("=================>_check_action_space of rl_agents")
    pass

  def get_action_space(self):
    if self.cmiu_rl: print("=================>get_action_space of rl_agents")
    return self.world.env.get_action_space(self.id)

  def get_state_size(self):
    if self.cmiu_rl: print("=================>get_state_size of rl_agents")
    return self.world.env.get_state_size(self.id)

  def get_goal_size(self):
    if self.cmiu_rl: print("=================>get_goal_size of rl_agents")
    return self.world.env.get_goal_size(self.id)

  def get_action_size(self):
    if self.cmiu_rl: print("=================>get_action_size of rl_agents")
    return self.world.env.get_action_size(self.id)

  def get_num_actions(self):
    if self.cmiu_rl: print("=================>get_num_actions of rl_agents")
    return self.world.env.get_num_actions(self.id)

  def need_new_action(self):
    if self.cmiu_rl: print("=================>need_new_action of rl_agents")
    return self.world.env.need_new_action(self.id)

  def _build_normalizers(self):
    if self.cmiu_rl: print("=================>_build_normalizers of rl_agents")
    # print("build_normalizers!"*2900) #seems like no passing through here..
    self.s_norm = Normalizer(self.get_state_size(),
                             self.world.env.build_state_norm_groups(self.id))
    self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id),
                             1 / self.world.env.build_state_scale(self.id))

    self.g_norm = Normalizer(self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
    self.g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id),
                             1 / self.world.env.build_goal_scale(self.id))

    self.a_norm = Normalizer(self.world.env.get_action_size())
    self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id),
                             1 / self.world.env.build_action_scale(self.id))
    return

  def _build_bounds(self):
    if self.cmiu_rl: print("=================>_build_bounds of rl_agents")
    self.a_bound_min = self.world.env.build_action_bound_min(self.id)
    self.a_bound_max = self.world.env.build_action_bound_max(self.id)
    return

  def _load_params(self, json_data):
    if self.cmiu_rl: print("=================>_load_params of rl_agents")
    if (self.UPDATE_PERIOD_KEY in json_data):
      self.update_period = int(json_data[self.UPDATE_PERIOD_KEY])

    if (self.ITERS_PER_UPDATE in json_data):
      self.iters_per_update = int(json_data[self.ITERS_PER_UPDATE])

    if (self.DISCOUNT_KEY in json_data):
      self.discount = json_data[self.DISCOUNT_KEY]

    if (self.MINI_BATCH_SIZE_KEY in json_data):
      self.mini_batch_size = int(json_data[self.MINI_BATCH_SIZE_KEY])

    if (self.REPLAY_BUFFER_SIZE_KEY in json_data):
      self.replay_buffer_size = int(json_data[self.REPLAY_BUFFER_SIZE_KEY])

    if (self.INIT_SAMPLES_KEY in json_data):
      self.init_samples = int(json_data[self.INIT_SAMPLES_KEY])

    if (self.NORMALIZER_SAMPLES_KEY in json_data):
      self.normalizer_samples = int(json_data[self.NORMALIZER_SAMPLES_KEY])

    if (self.OUTPUT_ITERS_KEY in json_data):
      self.output_iters = json_data[self.OUTPUT_ITERS_KEY]

    if (self.INT_OUTPUT_ITERS_KEY in json_data):
      self.int_output_iters = json_data[self.INT_OUTPUT_ITERS_KEY]

    if (self.TEST_EPISODES_KEY in json_data):
      self.test_episodes = int(json_data[self.TEST_EPISODES_KEY])

    if (self.EXP_ANNEAL_SAMPLES_KEY in json_data):
      self.exp_anneal_samples = json_data[self.EXP_ANNEAL_SAMPLES_KEY]

    if (self.EXP_PARAM_BEG_KEY in json_data):
      self.exp_params_beg.load(json_data[self.EXP_PARAM_BEG_KEY])

    if (self.EXP_PARAM_END_KEY in json_data):
      self.exp_params_end.load(json_data[self.EXP_PARAM_END_KEY])

    num_procs = MPIUtil.get_num_procs()
    self._local_mini_batch_size = int(np.ceil(self.mini_batch_size / num_procs))
    self._local_mini_batch_size = np.maximum(self._local_mini_batch_size, 1)
    self.mini_batch_size = self._local_mini_batch_size * num_procs

    assert (self.exp_params_beg.noise == self.exp_params_end.noise)  # noise std should not change
    self.exp_params_curr = copy.deepcopy(self.exp_params_beg)
    self.exp_params_end.noise = self.exp_params_beg.noise

    self._need_normalizer_update = self.normalizer_samples > 0

    return

  def _record_state(self):
    if self.cmiu_rl: print("=================>_record_state of rl_agents")
    s = self.world.env.record_state(self.id)
    return s

  def _record_goal(self):
    if self.cmiu_rl: print("=================>_record_goal of rl_agents")
    g = self.world.env.record_goal(self.id)
    return g

  def _record_reward(self):
    if self.cmiu_rl: print("=================>_record_reward of rl_agents")
    r = self.world.env.calc_reward(self.id)
    return r

  def _apply_action(self, a):
    if self.cmiu_rl: print("=================>_apply_action of rl_agents")
    self.world.env.set_action(self.id, a)
    return

  def _record_flags(self):
    if self.cmiu_rl: print("=================>_record_flags of rl_agents")
    return int(0)

  def _is_first_step(self):
    if self.cmiu_rl: print("=================>_is_first_step of rl_agents")
    return len(self.path.states) == 0

  def _end_path(self):
    if self.cmiu_rl: print("=================>_end_path of rl_agents")
    s = self._record_state()
    g = self._record_goal()
    r = self._record_reward()

    self.path.rewards.append(r)
    self.path.states.append(s)
    self.path.goals.append(g)
    self.path.terminate = self.world.env.check_terminate(self.id)

    return

  def _update_new_action(self):
    if self.cmiu_rl: print("=================>_update_new_action of rl_agents")
    #print("_update_new_action!")
    s = self._record_state()
    # print("/ S: ", s.shape) #당연 계속 바뀜. <class 'numpy.ndarray'>
    # print("STATE SIZE :", self.get_state_size()) #197동일
    #np.savetxt("pb_record_state_s.csv", s, delimiter=",")
    g = self._record_goal()
    # print("/ g: ", g) #계속 None임.

    if not (self._is_first_step()):
      r = self._record_reward()
      self.path.rewards.append(r)

    a, logp = self._decide_action(s=s, g=g) #PPO_agent.py에 있음.
    # print("ACTION SIZE :", self.get_action_size()) #36동일
    # print("/ A: ", type(a), a.shape)
    assert len(np.shape(a)) == 1
    assert len(np.shape(logp)) <= 1

    flags = self._record_flags()
    # print("|--> flags: ", flags)
    self._apply_action(a)

    self.path.states.append(s)
    self.path.goals.append(g)
    self.path.actions.append(a)
    self.path.logps.append(logp)
    self.path.flags.append(flags)

    if self._enable_draw():
      self._log_val(s, g)

    return

  def _update_exp_params(self):
    if self.cmiu_rl: print("=================>_update_exp_params of rl_agents")
    lerp = float(self._total_sample_count) / self.exp_anneal_samples
    lerp = np.clip(lerp, 0.0, 1.0)
    self.exp_params_curr = self.exp_params_beg.lerp(self.exp_params_end, lerp)
    return

  def _update_test_return(self, path):
    if self.cmiu_rl: print("=================>_update_test_return of rl_agents")
    path_reward = path.calc_return()
    # print("path_reward of test: ", path_reward)
    self.test_return += path_reward
    self.test_episode_count += 1
    return

  def _update_mode(self):
    if self.cmiu_rl: print("=================>_update_mode of rl_agents")
    if (self._mode == self.Mode.TRAIN):
      self._update_mode_train()
    elif (self._mode == self.Mode.TRAIN_END):
      self._update_mode_train_end()
    elif (self._mode == self.Mode.TEST):
      self._update_mode_test()
    else:
      assert False, Logger.print2("Unsupported RL agent mode" + str(self._mode))
    return

  def _update_mode_train(self):
    if self.cmiu_rl: print("=================>_update_mode_train of rl_agents")
    return

  def _update_mode_train_end(self):
    if self.cmiu_rl: print("=================>_update_mode_train_end of rl_agents")
    self._init_mode_test()
    return

  def _update_mode_test(self):
    if self.cmiu_rl: print("=================>_update_mode_test of rl_agents")
    if (self.test_episode_count * MPIUtil.get_num_procs() >= self.test_episodes):
      global_return = MPIUtil.reduce_sum(self.test_return)
      global_count = MPIUtil.reduce_sum(self.test_episode_count)
      avg_return = global_return / global_count
      self.avg_test_return = avg_return

      if self.enable_training:
        self._init_mode_train()
    return

  def _init_mode_train(self):
    if self.cmiu_rl: print("=================>_init_mode_train of rl_agents")
    self._mode = self.Mode.TRAIN
    self.world.env.set_mode(self._mode)
    return

  def _init_mode_train_end(self):
    if self.cmiu_rl: print("=================>_init_mode_train_end of rl_agents")
    self._mode = self.Mode.TRAIN_END
    return

  def _init_mode_test(self):
    if self.cmiu_rl: print("=================>_init_mode_test of rl_agents")
    self._mode = self.Mode.TEST
    self.test_return = 0.0
    self.test_episode_count = 0
    self.world.env.set_mode(self._mode)
    return

  def _enable_output(self):
    if self.cmiu_rl: print("=================>_enable_output of rl_agents")
    return MPIUtil.is_root_proc() and self.output_dir != ""

  def _enable_int_output(self):
    if self.cmiu_rl: print("=================>_enable_int_output of rl_agents")
    return MPIUtil.is_root_proc() and self.int_output_dir != ""

  def _calc_val_bounds(self, discount):
    if self.cmiu_rl: print("=================>_calc_val_bounds of rl_agents")
    r_min = self.world.env.get_reward_min(self.id)
    r_max = self.world.env.get_reward_max(self.id)
    assert (r_min <= r_max)

    val_min = r_min / (1.0 - discount)
    val_max = r_max / (1.0 - discount)
    return val_min, val_max

  def _calc_val_offset_scale(self, discount):
    if self.cmiu_rl: print("=================>_calc_val_offset_scale of rl_agents")
    val_min, val_max = self._calc_val_bounds(discount)
    val_offset = 0
    val_scale = 1

    if (np.isfinite(val_min) and np.isfinite(val_max)):
      val_offset = -0.5 * (val_max + val_min)
      val_scale = 2 / (val_max - val_min)

    return val_offset, val_scale

  def _calc_term_vals(self, discount):
    if self.cmiu_rl: print("=================>_calc_term_vals of rl_agents")
    r_fail = self.world.env.get_reward_fail(self.id)
    r_succ = self.world.env.get_reward_succ(self.id)

    r_min = self.world.env.get_reward_min(self.id)
    r_max = self.world.env.get_reward_max(self.id)
    assert (r_fail <= r_max and r_fail >= r_min)
    assert (r_succ <= r_max and r_succ >= r_min)
    assert (not np.isinf(r_fail))
    assert (not np.isinf(r_succ))

    if (discount == 0):
      val_fail = 0
      val_succ = 0
    else:
      val_fail = r_fail / (1.0 - discount)
      val_succ = r_succ / (1.0 - discount)

    return val_fail, val_succ
  
  # def _check_status(self, models, optims): #(Myi) new
  #   # 모델의 state_dict 출력
  #   assert isinstance(models, dict)
  #   print("Model's state_dict:")
  #   for name, model in models.items():
  #     print("For model: ", name)
  #     for param_tensor in model.state_dict():
  #       print(param_tensor, "\t", model.state_dict()[param_tensor].size())
  #   assert isinstance(optims, dict)
  #   print("Optimizer's state_dict:")
  #   for optimizer in optims.items():
  #     for var_name in optimizer.state_dict():
  #       print(var_name, "\t", optimizer.state_dict()[var_name])

  #   return 


  def _update_iter(self, iter):
    if self.cmiu_rl: print("=================>_update_iter of rl_agents")
    
    # models = {"actor": self.a_mean_pt, "critic":self.critic_pt}
    # models = {"actor": self.a_mean_pt}
    # optims = {"actor_opt": self.actor_optimizer , "critic_opt": self.critic_optimizer}
    # self._check_status(models, optims)
    if (self._enable_output() and self.iter % self.output_iters == 0):
      output_path = self._get_output_path()
      output_dir = os.path.dirname(output_path)
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      # self.save_model(output_path)
      print("====>"*101, "1_update_iter!", iter)
      print("OUTPUT PATH : ", output_path)
      self.save_model(output_path.replace('.pth','_anet.pth'), 'actor')
      self.save_model(output_path.replace('.pth','_cnet.pth'), 'critic')

    if (self._enable_int_output() and self.iter % self.int_output_iters == 0):
      int_output_path = self._get_int_output_path()
      int_output_dir = os.path.dirname(int_output_path)
      if not os.path.exists(int_output_dir):
        os.makedirs(int_output_dir)
      # self.save_model(int_output_path)
      print("====>"*101, "2_update_iter!", iter)
      print("OUTPUT PATH : ", int_output_path)
      self.save_model(output_path.replace('.pth','_anet.pth'), 'actor')
      self.save_model(output_path.replace('.pth','_cnet.pth'), 'critic')

    self.iter = iter
    return

  def _enable_draw(self):
    if self.cmiu_rl: print("=================>_enable_draw of rl_agents")
    return self.world.env.enable_draw

  def _log_val(self, s, g):
    if self.cmiu_rl: print("=================>_log_val of rl_agents")
    pass

  def _build_replay_buffer(self, buffer_size):
    if self.cmiu_rl: print("=================>_build_replay_buffer of rl_agents")
    num_procs = MPIUtil.get_num_procs()
    buffer_size = int(buffer_size / num_procs)
    self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
    self.replay_buffer_initialized = False
    return

  def _store_path(self, path):
    if self.cmiu_rl: print("=================>_store_path of rl_agents")
    
    path_id = self.replay_buffer.store(path)
    # print("[rl agent] path stored!")
    valid_path = path_id != MathUtil.INVALID_IDX

    if valid_path:
      self.train_return = path.calc_return()

      if self._need_normalizer_update:
        self._record_normalizers(path)

    return path_id

  def _record_normalizers(self, path):
    if self.cmiu_rl: print("=================>_record_normalizers of rl_agents")
    states = np.array(path.states)
    self.s_norm.record(states)

    if self.has_goal():
      goals = np.array(path.goals)
      self.g_norm.record(goals)

    return

  def _update_normalizers(self):
    if self.cmiu_rl: print("=================>_update_normalizers of rl_agents")
    self.s_norm.update()

    if self.has_goal():
      self.g_norm.update()
    return

  def _train(self):
    if self.cmiu_rl: print("=================>_train of rl_agents")
    samples = self.replay_buffer.total_count
    # print("samples : ", samples)
    self._total_sample_count = int(MPIUtil.reduce_sum(samples))
    # print("self._total_sample_count : ", self._total_sample_count) #샘플스와 같음 (tf1.15버젼또한)
    end_training = False

    if (self.replay_buffer_initialized):
      # print("self.replay_buffer_initialized ", self.replay_buffer_initialized) #학습땐 트루
      # print("?? ", self._valid_train_step())
      # if (self._valid_train_step() or True):
      if (self._valid_train_step()): #PPOAgent의 _valid_train_step()을 타버림. 거기서 데이터가 안올라와서 못탐.
        # print("self._valid_train_step() : "*100, self._valid_train_step()) #한 패쓰 끝나고 트루되고 밑에 돈다,
        prev_iter = self.iter
        iters = self._get_iters_per_update()
        avg_train_return = MPIUtil.reduce_avg(self.train_return)

        for i in range(iters):
          curr_iter = self.iter
          wall_time = time.time() - self.start_time
          wall_time /= 60 * 60  # store time in hours

          has_goal = self.has_goal()
          s_mean = np.mean(self.s_norm.mean)
          s_std = np.mean(self.s_norm.std)
          g_mean = np.mean(self.g_norm.mean) if has_goal else 0
          g_std = np.mean(self.g_norm.std) if has_goal else 0

          self.logger.log_tabular("Iteration", self.iter)
          self.logger.log_tabular("Wall_Time", wall_time)
          self.logger.log_tabular("Samples", self._total_sample_count)
          self.logger.log_tabular("Train_Return", avg_train_return)
          self.logger.log_tabular("Test_Return", self.avg_test_return)
          self.logger.log_tabular("State_Mean", s_mean)
          self.logger.log_tabular("State_Std", s_std)
          self.logger.log_tabular("Goal_Mean", g_mean)
          self.logger.log_tabular("Goal_Std", g_std)
          self._log_exp_params()

          self._update_iter(self.iter + 1)
          self._train_step()

          Logger.print2("Agent " + str(self.id))
          self.logger.print_tabular()
          Logger.print2("")

          if (self._enable_output() and curr_iter % self.int_output_iters == 0):
            self.logger.dump_tabular()

        if (prev_iter // self.int_output_iters != self.iter // self.int_output_iters):
          end_training = self.enable_testing()

    else:

      Logger.print2("Agent " + str(self.id))
      Logger.print2("Samples: " + str(self._total_sample_count))
      Logger.print2("")

      if (self._total_sample_count >= self.init_samples):
        self.replay_buffer_initialized = True
        end_training = self.enable_testing()

    if self._need_normalizer_update:
      self._update_normalizers()
      self._need_normalizer_update = self.normalizer_samples > self._total_sample_count

    if end_training:
      self._init_mode_train_end()

    return

  def _get_iters_per_update(self):
    if self.cmiu_rl: print("=================>_get_iters_per_update of rl_agents")
    return MPIUtil.get_num_procs() * self.iters_per_update

  def _valid_train_step(self):
    if self.cmiu_rl: print("=================>_valid_train_step of rl_agents")
    return True

  def _log_exp_params(self):
    if self.cmiu_rl: print("=================>_log_exp_params of rl_agents")
    self.logger.log_tabular("Exp_Rate", self.exp_params_curr.rate)
    self.logger.log_tabular("Exp_Noise", self.exp_params_curr.noise)
    self.logger.log_tabular("Exp_Temp", self.exp_params_curr.temp)
    return