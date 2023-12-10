import numpy as np
# try:
#   import tensorflow.compat.v1 as tf
# except Exception:
#   import tensorflow as tf
import torch
from abc import abstractmethod

from _pybullet_env.learning.rl_agent import RLAgent
from _pybullet_utils.logger import Logger
# from _pybullet_env.deep_mimic.learning.tf_normalizer import TFNormalizer
from _pybullet_env.learning.torch_normalizer import TorchNormalizer

class TorchAgent(RLAgent):
  RESOURCE_SCOPE = 'resource'
  SOLVER_SCOPE = 'solvers'

  def __init__(self, world, id, json_data):
    print("=================> 3. Torch AGENT's world", world)
    self.tf_scope = 'agent'
    # self.graph = tf.Graph()
    # self.sess = tf.Session(graph=self.graph)
    self.cmiutf = self._myi_check_the_method_is_used = False

    super().__init__(world, id, json_data) #!deepdive here
    self._build_graph(json_data)
    
    
    self._init_normalizers()
    return

  def __del__(self):
    if self.cmiutf: print("=================>__del__ of tf_agents")
    # self.sess.close()
    return

  def save_model_old(self, out_path, models):
    # if self.cmiutf: print("=================>save_model of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   try:
    #     save_path = self.saver.save(self.sess, out_path, write_meta_graph=False, write_state=False)
    #     Logger.print2('Model saved to: ' + save_path)
    #   except:
    #     Logger.print2("Failed to save model to: " + save_path)
    for name, model in models.items():
      Logger.print2('Model == '+name)
      Logger.print2('Model saved to: ' + out_path)
      torch.save(model, out_path)
    return
  
  def save_model(self, out_path, net_name = 'actor'):
    if self.cmiutf: print("=================>save_model of tf_agents")
    # print("OUT PATH !!!"*121 , out_path*12)

    target_model = self.netmap[net_name]
    torch.save(target_model.state_dict(), out_path)

    return

  def load_model_old(self, in_path):
    # with self.sess.as_default(), self.graph.as_default():
    #   self.saver.restore(self.sess, in_path)
    #   self._load_normalizers()
    #   Logger.print2('Model loaded from: ' + in_path)
    return
  
  def load_model(self, in_path, net_name= 'actor'):
    if self.cmiutf: print("=================>load_model of tf_agents")
    model = self.netmap[net_name]
    model.load_state_dict(torch.load(in_path))
    #check!
    # for param_tensor in model.state_dict():
    #   print("[][][][][][][]" , "PARAM: ", param_tensor)

    self._load_normalizers()
    Logger.print2('Model loaded from: ' + in_path)
    return 
  
  def _create_modelmap(self):
    if self.netmap is not None: return #only when it is None
    self.netmap = {"actor": None, "critic":None}
    return

  def _get_output_path(self):
    if self.cmiutf: print("=================>_get_output_path of tf_agents")
    assert (self.output_dir != '')
    file_path = self.output_dir + '/agent' + str(self.id) + '_model.pth'
    return file_path

  def _get_int_output_path(self):
    if self.cmiutf: print("=================>_get_int_output_path of tf_agents")
    assert (self.int_output_dir != '')
    file_path = self.int_output_dir + (
        '/agent{:d}_models/agent{:d}_int_model_{:010d}.ckpt').format(self.id, self.id, self.iter)
    return file_path

  def _build_graph(self, json_data):
    if self.cmiutf: print("=================>_build_graph of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   with tf.variable_scope(self.tf_scope):
    self._build_nets(json_data)

    # with tf.variable_scope(self.SOLVER_SCOPE):
    self._build_losses(json_data)
    self._build_solvers(json_data)

    self._initialize_vars()
    self._build_saver()
    return

  def _init_normalizers(self):
    if self.cmiutf: print("=================>_init_normalizers of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   # update normalizers to sync the tensorflow tensors
    self.s_norm.update()
    self.g_norm.update()
    self.a_norm.update()
    return

  @abstractmethod
  def _build_nets(self, json_data):
    if self.cmiutf: print("=================>_build_nets of tf_agents")
    pass

  @abstractmethod
  def _build_losses(self, json_data):
    if self.cmiutf: print("=================>_build_losses of tf_agents")
    pass

  @abstractmethod
  def _build_solvers(self, json_data):
    if self.cmiutf: print("=================>_build_solvers of tf_agents")
    pass

  def _tf_vars(self, scope=''): #나중에 다끝나고 weight decay랑 같이 이해하며 구현 예정 (10월 중순전)
    if self.cmiutf: print("=================>_tf_vars of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.tf_scope + '/' + scope)
    #   assert len(res) > 0
    # return res
    res = []

    return res

  def _build_normalizers(self):
    print("=================>_build_normalizers of torch_agents") #torch normalizer를 구현 (23/09/03)
    # print("self.get_state_size() : ", self.get_state_size()) #197 #밑엔 [-1,0,0, ... ]
    # print("self.world.env.build_state_norm_groups(self.id): ", self.world.env.build_state_norm_groups(self.id)) #same as tf1.15 ver.
    self.s_norm = TorchNormalizer(self.get_state_size(), self.world.env.build_state_norm_groups(self.id))
    state_offset = -self.world.env.build_state_offset(self.id)
    print("state_offset=", state_offset)
    state_scale = 1 / self.world.env.build_state_scale(self.id)
    print("state_scale=", state_scale)
    # print("mean??? -self.world.env.build_state_offset(self.id)", -self.world.env.build_state_offset(self.id))
    self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id), #!deepdive here
                                 1 / self.world.env.build_state_scale(self.id))
    
    self.g_norm = TorchNormalizer(self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
    self.g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id),  1 / self.world.env.build_goal_scale(self.id))
    self.a_norm = TorchNormalizer(self.get_action_size())
    self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id), 1 / self.world.env.build_action_scale(self.id))
    # with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
    #   with tf.variable_scope(self.RESOURCE_SCOPE):
    #     self.s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(),
    #                                self.world.env.build_state_norm_groups(self.id))
    #     state_offset = -self.world.env.build_state_offset(self.id)
    #     print("state_offset=", state_offset)
    #     state_scale = 1 / self.world.env.build_state_scale(self.id)
    #     print("state_scale=", state_scale)
    #     self.s_norm.set_mean_std(-self.world.env.build_state_offset(self.id), #!deepdive here
    #                              1 / self.world.env.build_state_scale(self.id))

    #     self.g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(),
    #                                self.world.env.build_goal_norm_groups(self.id))
    #     self.g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id),
    #                              1 / self.world.env.build_goal_scale(self.id))

    #     self.a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
    #     self.a_norm.set_mean_std(-self.world.env.build_action_offset(self.id),
    #                              1 / self.world.env.build_action_scale(self.id))
    return

  def _load_normalizers(self):
    if self.cmiutf: print("=================>_load_normalizers of tf_agents")
    self.s_norm.load()
    self.g_norm.load()
    self.a_norm.load()
    return

  def _update_normalizers(self):
    if self.cmiutf: print("=================>_update_normalizers of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    super()._update_normalizers()
    return

  def _initialize_vars(self):
    if self.cmiutf: print("=================>_initialize_vars of tf_agents")
    # self.sess.run(tf.global_variables_initializer())
    return

  def _build_saver(self):
    if self.cmiutf: print("=================>_build_saver of tf_agents")
    # vars = self._get_saver_vars()
    # self.saver = tf.train.Saver(vars, max_to_keep=0)
    return

  def _get_saver_vars(self):
    vars = None
    if self.cmiutf: print("=================>_get_saver_vars of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    #   vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.tf_scope)
    #   vars = [v for v in vars if '/' + self.SOLVER_SCOPE + '/' not in v.name]
      #vars = [v for v in vars if '/target/' not in v.name]
    #   assert len(vars) > 0
    return vars

  def _weight_decay_loss(self, scope): #나중에 다끝나고 weight decay이해하며 구현 예정 (10월 중순전)
    # To-Do 이방식 말고 weight decay할수있는 방식이 있는지도 알아봐야함
    # Update (23/10/02 ) 굳이 하려면 이 같은 방식으로 할수 있으나, torch.optim에 이미 weight decay가 있어서
    # 이 방식 보다는 param을 나눠서 optim에 주는 방법도 괜찮아 보임 (_weight_decay_loss_no_bias)
    if self.cmiutf: print("=================>_weight_decay_loss of tf_agents")
    # vars = self._tf_vars(scope)
    # vars_no_bias = [v for v in vars if 'bias' not in v.name]
    # loss = tf.add_n([tf.nn.l2_loss(v) for v in vars_no_bias])
    # return loss
  
  def _seperate_params_wdecay(self, net, l2val):
    # Referece : https://www.youtube.com/watch?v=hZE4Nja5zKA
    decay, no_decay = [] , []
    loss = 0 
    for name, param in net.named_parameters():
      if not param.requires_grad: continue        

      if name.endswith(".bias"):
        no_decay.append(param)
      else:
        decay.append(param)     
    
    return [{'params': no_decay, 'weight_decay':0}, {'params': decay, 'weight_decay':l2val}]
      




  def _train(self):
    if self.cmiutf: print("=================>_train of tf_agents")
    # with self.sess.as_default(), self.graph.as_default():
    super()._train()
    return