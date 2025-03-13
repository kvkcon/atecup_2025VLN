import os
import glob
from gym.envs.registration import register
import logging
from gym_rescue.envs.utils.misc import load_env_setting
logger = logging.getLogger(__name__)
use_docker = False  # True: use nvidia docker   False: do not use nvidia-docker


Tasks = ['Rescue']
Observations = ['Color', 'Depth', 'Rgbd', 'Mask', 'Pose','MaskDepth','ColorMask']
Actions = ['Discrete', 'Continuous', 'Mixed']

# Task-oriented envs
UnrealEnv = os.environ.get('UnrealEnv')
setting_files = glob.glob(
    os.path.join(UnrealEnv , 'settings', '*.json')
)
for setting_file in setting_files:
    
    env = os.path.split(setting_file)[-1].split('.')[0]
    for task in Tasks:
        name = f'Unreal{task}-{env}'
        register(
            id=name,
            entry_point=f'gym_rescue.envs:{task}',
            kwargs={'env_file': setting_file,},
            max_episode_steps=10000
        )

