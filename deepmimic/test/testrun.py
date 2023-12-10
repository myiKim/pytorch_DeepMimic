import gym 
import envs 

def initialize_env():
    global env
    # env = gym.make("heygogo-v0")
    # help(gym.make)
    env = gym.make("deep_mimic_env/hmyifEnv-v0")
    SEED =77777
    env.action_space.seed(SEED)

    return env.reset()

def drive_main(range_):

    for tt in range(range_):
        observation, reward, terminated, info = env.step(env.action_space.sample())
        if tt % 100 ==0:
            print("At : ", tt, "we observe : ", observation, "with reward : ", reward)
        
        if terminated: 
            env.reset()
    return {"finished at : ", tt}
        
def main():
    print(initialize_env())
    return drive_main(1000)
if __name__ == '__main__':
    print(main())
    