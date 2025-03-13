import gym
from gym import Wrapper
import time

class SwitchEnvWrapper(Wrapper):
    def __init__(self, env, env_old):
        super().__init__(env)
        env.unwrapped.unrealcv = env_old.unwrapped.unrealcv
        env.unwrapped.ue_binary = env_old.unwrapped.ue_binary
        env.unwrapped.unrealcv.set_map(env.unwrapped.env_name)
        print('Switching to environment:', env.unwrapped.env_name)
        env.unwrapped.launched = True
        env.unwrapped.init_agents()
        env.unwrapped.init_objects()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)  # take a step in the wrapped environment
        return obs, reward, terminated, truncated, info  # return the same results as the wrapped environment

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info  # return the same results as the wrapped environment
