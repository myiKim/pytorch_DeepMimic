import time
import os
import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)
# print("parentdir=", parentdir)
import json
from _pybullet_env.learning.rl_world import RLWorld
from _pybullet_env.learning.ppo_agent import PPOAgent
from collections import defaultdict
import _pybullet_data
from _pybullet_utils.arg_parser import ArgParser
from _pybullet_utils.logger import Logger
from _pybullet_env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
import sys
import random

update_timestep = 1. / 240.
animating = True
step = False
total_reward = 0
steps = 0

def update_world(world, time_elapsed):
  timeStep = update_timestep
  world.update(timeStep)
  reward = world.env.calc_reward(agent_id=0)
  global total_reward
  total_reward += reward
  global steps
  steps+=1
  
  # print("[INTERMED - print turned on by Myungin ] reward=",reward)
  # print("[INTERMED - print turned on by Myungin ] steps=",steps)
  end_episode = world.env.is_episode_end()
  if (end_episode or steps>= 1000):
    # print("[*]"*80," total_reward=",total_reward)
    # print("total_reward=",total_reward)
    total_reward=0
    # print("TOTAL STEPS = ", steps) #걱정.. 이게 tf1env꺼 보다 좀 길어서 리워드도 더 나오는편..
    steps = 0
    world.end_episode()
    world.reset()
  return


def build_arg_parser(args, train_needed=False):
  arg_parser = ArgParser()
  arg_parser.load_args(args)

  arg_file = arg_parser.parse_string('arg_file', '')
  if arg_file == '':
    # arg_file = "run_humanoid3d_backflip_args.txt"
    arg_file = "run_humanoid3d_walk_args.txt"
    # train_needed=False
    if train_needed: 
      # arg_file = "train_humanoid3d_dance_a_args.txt"
      # arg_file ="train_humanoid3d_walk_args.txt"
      arg_file ="train_humanoid3d_walk_args.txt"
      print("arg_file == walk")

  if (arg_file != ''):
    path = _pybullet_data.getDataPath() + "/args/" + arg_file
    succ = arg_parser.load_file(path)
    Logger.print2("Successfully load file!")
    Logger.print2(arg_file)
    assert succ, Logger.print2('Failed to load args from: ' + arg_file)
  return arg_parser



args = sys.argv[1:]


def build_world(args, enable_draw, train=False):
  arg_parser = build_arg_parser(args, train)
  print("enable_draw=", enable_draw)
  env = PyBulletDeepMimicEnv(arg_parser, enable_draw)
  print(";; env: ", env)
  model_info_dict = defaultdict(str)
  model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_newModel10'
  # model_info_dict['model'] =  'D:\ReinforcementLearning\_modified_bullet_main\ModifiedBulletDeepMimic\deepmimic\output\storage\800ish'
  # model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_barely_notrained'
  # model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_newstart\mm01'
  model_info_dict['cnet'] = 'agent0_model_cnet.pth'
  model_info_dict['anet'] = 'agent0_model_anet.pth'
  world = RLWorld(env, arg_parser, model_infodict=model_info_dict, torchtest=True)
  #world.env.set_playback_speed(playback_speed)

  motion_file = arg_parser.parse_string("motion_file")
  print("motion_file=", motion_file)
  bodies = arg_parser.parse_ints("fall_contact_bodies")
  print("bodies=", bodies)
  int_output_path = arg_parser.parse_string("int_output_path")
  print("int_output_path=", int_output_path)
  agent_files = _pybullet_data.getDataPath() + "/" + arg_parser.parse_string("agent_files")

  AGENT_TYPE_KEY = "AgentType"

  print("agent_file=", agent_files)
  with open(agent_files) as data_file:
    json_data = json.load(data_file)
    print("json_data=", json_data)
    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]
    print("agent_type=", agent_type)
    # print("What is ID!!!!!!!!!"*123, id)
    # print("PPO Agent is called here 222222")
    agent = PPOAgent(world, id, json_data) #<built-in function id>
    # print("PPO Agent is called here 222222 end")
    # print("what is world? ", world)

    agent.set_enable_training(False)
    world.reset()
  return world



if __name__ == '__main__':


  world = build_world(args, True)
  print("What is args? ", args)
  while (world.env._pybullet_client.isConnected()):
    # print("animating? ", animating) #와일 돌면서 계속 트루

    timeStep = update_timestep
    time.sleep(timeStep)
    keys = world.env.getKeyboardEvents()

    if world.env.isKeyTriggered(keys, ' '):
      print("this?") #안찍힘.
      animating = not animating
    if world.env.isKeyTriggered(keys, 'i'):
      step = True
    if (animating or step):
      # print("!!") #여러번 들어온다.
      update_world(world, timeStep)
      step = False