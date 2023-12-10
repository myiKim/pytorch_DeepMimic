import gym

def list_registered_environments():
    print(dir(gym.envs.registration.registry))
    all_envs = [env_spec for env_spec in gym.envs.registration.registry]
    return all_envs

if __name__ == "__main__":
    registered_envs = list_registered_environments()
    print("List of registered environments:")
    print(registered_envs)
