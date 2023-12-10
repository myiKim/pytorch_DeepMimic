import numpy as np
import _pybullet_env.learning.agent_builder as AgentBuilder
# if False: import _pybullet_env.learning.tf_util as TFUtil   #gpu끄는것 밖에 없어서 딱히 필요있는지.. 
import os    
from _pybullet_env.learning.rl_agent import RLAgent
from _pybullet_utils.logger import Logger
import _pybullet_data


class RLWorld(object):

  def __init__(self, env, arg_parser, model_infodict = None, torchtest=False):
    # if False: TFUtil.disable_gpu()

    self.env = env
    self.arg_parser = arg_parser
    self._enable_training = True
    self.train_agents = []
    if torchtest and model_infodict:
      self.take_model_path(model_infodict)
    else:
      self.default_model_infodict = {'model':None, 'anet':None, 'cnet':None}
      self.take_model_path(self.default_model_infodict)
    
    
    self.parse_args(arg_parser)

    self.build_agents()

    return

  def get_enable_training(self):
    return self._enable_training

  def set_enable_training(self, enable):
    self._enable_training = enable
    for i in range(len(self.agents)):
      curr_agent = self.agents[i]
      if curr_agent is not None:
        enable_curr_train = self.train_agents[i] if (len(self.train_agents) > 0) else True
        curr_agent.enable_training = self.enable_training and enable_curr_train

    if (self._enable_training):
      self.env.set_mode(RLAgent.Mode.TRAIN)
    else:
      self.env.set_mode(RLAgent.Mode.TEST)

    return

  enable_training = property(get_enable_training, set_enable_training)

  def parse_args(self, arg_parser):
    self.train_agents = self.arg_parser.parse_bools('train_agents')
    num_agents = self.env.get_num_agents()
    assert (len(self.train_agents) == num_agents or len(self.train_agents) == 0)

    return

  def shutdown(self):
    self.env.shutdown()
    return
  
  def take_model_path(self, model_infodict):
    self.modelpath, self.cnetpath, self.anetpath = (None,)*3
    
    if 'model' in model_infodict:
      self.modelpath = model_infodict['model']
    if 'cnet' in model_infodict:
      self.cnetpath = model_infodict['cnet']
    if 'anet' in model_infodict:
      self.anetpath = model_infodict['anet']
    
    self.set_model_path(self.modelpath, self.cnetpath, self.anetpath)

  
  def set_model_path(self, outpath = None, critic_model_path = None, actor_model_path = None):
    # curr_agent.load_model(_pybullet_data.getDataPath() + "/" + curr_model_file)
    if outpath is None:
      Logger.print2("default model path is set!")
      self.model_output = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_newModel9' #임시로
    else:
      Logger.print2("given model path is set!")
      self.model_output = outpath

    # MANUAL_OUT_PATH = 'D:\ReinforcementLearning\_modified_bullet_main\output' #임시로
    if critic_model_path is None:
      self.critic_path =  os.path.join(self.model_output, 'agent0_model_cnet.pth')
    else:
      self.critic_path =  os.path.join(self.model_output, critic_model_path)

    if actor_model_path is None:
      self.actor_path = os.path.join(self.model_output, 'agent0_model_anet.pth')
    else:
      self.actor_path = os.path.join(self.model_output, actor_model_path)

  def build_agents(self):
    num_agents = self.env.get_num_agents()
    print("num_agents=", num_agents)
    self.agents = []

    Logger.print2('')
    Logger.print2('Num Agents: {:d}'.format(num_agents))

    agent_files = self.arg_parser.parse_strings('agent_files')
    print("len(agent_files)=", len(agent_files))
    assert (len(agent_files) == num_agents or len(agent_files) == 0)

    model_files = self.arg_parser.parse_strings('model_files')
    assert (len(model_files) == num_agents or len(model_files) == 0)

    output_path = self.arg_parser.parse_string('output_path')
    int_output_path = self.arg_parser.parse_string('int_output_path')

    for i in range(num_agents):
      print("agent_i : ", i)
      curr_file = agent_files[i]
      print("curr_file : ", curr_file) # data/agents/ct_agent_humanoid_ppo.txt 
      curr_agent = self._build_agent(i, curr_file) #!deepdive here
      # print(curr_agent is  None , "221121323"*102) #None아니고

      if curr_agent is not None:
        curr_agent.output_dir = output_path
        curr_agent.int_output_dir = int_output_path
        Logger.print2(str(curr_agent))
        # print("--------------"*121, "i th MODEL FILE? when BuildAgent applied?")
        if len(model_files) >0:  print(model_files[i]) #zero so pass the below..
        if (len(model_files) > 0):
          
          curr_model_file = model_files[i]
          if curr_model_file != 'none':
            
            curr_agent.load_model(self.critic_path, 'critic')
            curr_agent.load_model(self.actor_path, 'actor')

      self.agents.append(curr_agent)
      print(curr_agent, "is get inside self.agents")
      Logger.print2('')

    self.set_enable_training(self.enable_training)

    return

  def update(self, timestep):
    #print("world update!\n")
    self._update_agents(timestep)
    self._update_env(timestep)
    return

  def reset(self):
    self._reset_agents()
    self._reset_env()
    return

  def end_episode(self):
    self._end_episode_agents()
    return

  def _update_env(self, timestep):
    self.env.update(timestep)
    return

  def _update_agents(self, timestep):
    #print("len(agents)=",len(self.agents))
    for agent in self.agents:
      if (agent is not None):
        # print("timestep // 1: ", timestep) 
        agent.update(timestep) #RLagent.update로 가는듯. 타임스텝이 계속 동일함.
    return

  def _reset_env(self):
    self.env.reset()
    return

  def _reset_agents(self):
    for agent in self.agents:
      if (agent != None):
        agent.reset()
    return

  def _end_episode_agents(self):
    for agent in self.agents:
      if (agent != None):
        agent.end_episode()
    return

  def _build_agent(self, id, agent_file):
    Logger.print2('Agent {:d}: {}'.format(id, agent_file))
    if (agent_file == 'none'):
      agent = None
    else:
      agent = AgentBuilder.build_agent(self, id, agent_file) #!deepdive here
      assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)

    return agent
