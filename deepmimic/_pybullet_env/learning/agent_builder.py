import json
import numpy as np
from _pybullet_env.learning.ppo_agent import PPOAgent
import _pybullet_data

AGENT_TYPE_KEY = "AgentType"


def build_agent(world, id, file):
  agent = None
  with open(_pybullet_data.getDataPath() + "/" + file) as data_file:
    json_data = json.load(data_file)

    assert AGENT_TYPE_KEY in json_data
    agent_type = json_data[AGENT_TYPE_KEY]

    if (agent_type == PPOAgent.NAME):
      print("PPO Agent is called here 111111")
      agent = PPOAgent(world, id, json_data) #!deepdive here
      print("PPO Agent is called here 111111 end")
    else:
      assert False, 'Unsupported agent type: ' + agent_type
#   print("Then, agent is what? "*12, agent)
  return agent
