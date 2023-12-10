import numpy as np
import sys
import os
import inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0, parentdir)
# print("parentdir=", parentdir)

from _pybullet_env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
from _pybullet_env.learning.rl_world import RLWorld
from _pybullet_utils.logger import Logger
from testrl import update_world, update_timestep, build_world
# import _pybullet_utils.mpi_util as MPIUtil

args = []
world = None
# print([1,0,0,1]&[0,1,1,1])

def run():
  global update_timestep
  global world
  done = False
  while not (done):
    update_world(world, update_timestep)
    # test = False
    # if test:
    #   if world.agents[0].replay_buffer.buffers:
    #     ss = world.agents[0].replay_buffer.buffers
    #     if 'flags' not in ss: 
    #       print("wow")
    #       continue
    #     else:
    #       if sum(ss['flags']) ==0: print("zero!") #tf1.15와 다르게 계속 zero만 나옴..
    #       else: print("------>>>>>>", ss['flags'])

  return


def shutdown():
  global world

  Logger.print2('Shutting down...')
  world.shutdown()
  return


def main():
  global args
  global world

  # Command line arguments
  args = sys.argv[1:]
  enable_draw = False
  print("args : ", args)
  world = build_world(args, enable_draw, train=True)
  print("======>")

  run()
  shutdown()

  return


if __name__ == '__main__':
  main()