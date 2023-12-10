
import os, inspect,sys
import json, time
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("curr dir : ", currentdir)
parentdir = os.path.dirname(currentdir)
print("parent dir : ", parentdir)
os.sys.path.insert(0, parentdir)
from policy_tester_env import PolicyTesterEnv
from _pybullet_env.learning.rl_world import RLWorld
from _pybullet_utils.arg_parser import ArgParser
from _pybullet_utils.logger import Logger
import _pybullet_data
import pybullet as pb

# args = sys.argv[1:]

class TestRunner():
    def __init__(self,
                 arg_file = "run_humanoid3d_walk_args.txt" ,
                 given_model = None
        ):
        self.pbcon = pb.connect(pb.DIRECT)
        self.env = PolicyTesterEnv(pybullet_client = self.pbcon)
        self.mini_config = dict()
        self.create_mini_config()
        
        self.arg_file= arg_file
        self.arg_path =_pybullet_data.getDataPath() + "/args/" + self.arg_file

        self.arg_parser = self.build_arg_parser(sys.argv[1:])
        self.rlworld = RLWorld(env=self.env, arg_parser=self.arg_parser, model_infodict=given_model, torchtest=True)
        self.motion_file = self.arg_parser.parse_string("motion_file")
        bodies = self.arg_parser.parse_ints("fall_contact_bodies")
        print("bodies=", bodies)
        agent_files = _pybullet_data.getDataPath() + "/" + self.arg_parser.parse_string("agent_files")

        # print("pah?"*129, os.path.exists(agent_files))
        AGENT_TYPE_KEY = "AgentType"
        with open(agent_files) as data_file:
            json_data = json.load(data_file)
            print("json_data=", json_data)
            assert AGENT_TYPE_KEY in json_data
            agent_type = json_data[AGENT_TYPE_KEY]
            print("agent_type=", agent_type)

            self.rlworld.reset()
            


    def build_arg_parser(self, args):
        arg_parser = ArgParser()
        arg_parser.load_args(args)
        arg_path = arg_parser.parse_string('arg_file', '')
        if arg_path == '':
            arg_path = self.arg_path
            Logger.print2("Use default one..")

        succ = arg_parser.load_file(arg_path)
        assert succ, Logger.print2('Failed to load args from: ' + arg_file)

        return arg_parser
    
    def create_mini_config(self):
        
        self.mini_config["update_timestep"] = 1. / 240.
        self.mini_config["animating"] = True
        self.mini_config["step"] = step = False
        self.mini_config["total_reward"]  = 0
        self.mini_config["steps"] = 0    

    def update_world(self):
        timeStep = self.mini_config["update_timestep"]
        self.rlworld.update(timeStep)
        reward = self.rlworld.env.calc_reward(agent_id=0)

        self.mini_config["total_reward"] += reward

        self.mini_config["steps"] +=1
        
        # print("[INTERMED - print turned on by Myungin ] reward=",reward)
        # print("[INTERMED - print turned on by Myungin ] steps=",steps)
        end_episode = self.rlworld.env.is_episode_end()
        # self.rlworld.reset()
        if (end_episode or self.mini_config["steps"] >= 1000):
            # print("[*]"*80," total_reward=",total_reward)
            # print("total_reward=",total_reward)
            print(self.mini_config["steps"])
            self.mini_config["total_reward"]=0
            # print("TOTAL STEPS = ", steps) #걱정.. 이게 tf1env꺼 보다 좀 길어서 리워드도 더 나오는편..
            self.mini_config["steps"] = 0
            self.rlworld.end_episode()
            self.rlworld.reset()
        return    

    
    def runTest(self):
        step = False

        while (self.rlworld.env._pybullet_client.isConnected()):
            timeStep = self.mini_config["update_timestep"]
            time.sleep(timeStep*20)
            keys = self.rlworld.env.getKeyboardEvents()
            if self.rlworld.env.isKeyTriggered(keys, 'i'):
                step = True
            if(self.mini_config["animating"] or step):
                self.update_world()
                step =False


        return
        
        

if __name__ == '__main__':

    from collections import defaultdict
    
    arg_file = "run_humanoid3d_walk_args.txt"
    model_info_dict = defaultdict(str)
    # model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_newModel5'
    # model_info_dict['model'] =  'D:\ReinforcementLearning\_modified_bullet_main\ModifiedBulletDeepMimic\deepmimic\output\storage\800ish'
    # model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_barely_notrained'
    model_info_dict['model'] = 'D:\ReinforcementLearning\_modified_bullet_main\output\preserve\_newstart\mm01'
    model_info_dict['cnet'] = 'agent0_model_cnet.pth'
    model_info_dict['anet'] = 'agent0_model_anet.pth'
    trun = TestRunner(arg_file, model_info_dict)
    trun.runTest()