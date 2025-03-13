import argparse
import gym
import cv2
import time
from gym_rescue.envs.wrappers import time_dilation, early_done, monitor, population, configUE,task_cue
from gym_rescue.envs.utils.keyboard_util import get_key_action,on_press,on_release
from pynput import keyboard
import os
import json


#start keyboard listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', type=int, default=1, help='number of population')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', type=int, default=-1, help='time_dilation to keep fps in simulator')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', help='use nav agent to control the agents')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-k", '--keyboard', dest='keyboard', action='store_true', help='use keyboard to control the agents')
    parser.add_argument("-o", '--offscreen', dest='offscreen', action='store_true', help='offscreen rendering')
    parser.add_argument("-q", '--quality', dest='quality', type=int, default=3, help='render quality')
    parser.add_argument("-u", '--use_lumen', dest='use_lumen', action='store_true', help='use lumen')
    parser.add_argument("-v", '--record_video', dest='record_video', action='store_true', help='record task cues and observation into video')

    UnrealEnv = os.environ.get('UnrealEnv')
    TEST_JSONL = os.path.join(UnrealEnv, 'test_L1.jsonl')
    with open(TEST_JSONL, "r") as fp:
        try:
            point_id = 0
            for line in fp.readlines():
                content = json.loads(line)
                env_id = content.get("env_id")
                timeout = content.get("timeout")

                InterruptedException = False
                args = parser.parse_args()
                resolution = (320, 320)
                if args.record_video:
                    out = cv2.VideoWriter('output_{}.mp4'.format(point_id), cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                                          (int(2.5 * resolution[1]), resolution[1]))
                else:
                    out = None
                # create the environment
                env = gym.make(env_id, action_type='Mixed', observation_type='Color')  # available observation_type : ['Color', 'Depth', 'Rgbd', 'Mask', 'Pose','MaskDepth','ColorMask']
                # set the rendering mode, resolution and render quality, gpu_id
                env = configUE.ConfigUEWrapper(env, offscreen=args.offscreen, resolution=resolution, use_lumen=args.use_lumen, render_quality=args.quality)

                if int(args.time_dilation) > 0:  # -1 means no time_dilation, the number means the expected FPS in simulator
                    env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
                if int(timeout) > 0:  # -1 means no early_done, 180 mean 180 seconds
                    env = early_done.EarlyDoneWrapper(env, timeout)
                if args.monitor: # if dynamic_top_down is True, the top-down view will follow the agent
                    env = monitor.DisplayWrapper(env, dynamic_top_down=False, fix_camera=True,get_bbox=True)
                env = task_cue.TaskCueWrapper(env=env, test_point=content)

                agent = RandomAgent(env.action_space)
                rewards = 0
                done = False
                Total_rewards = 0
                obs,info = env.reset() # environment reset
                count_step = 0
                t0=time.time()
                while True:
                    try:
                        # print(info['reference_text'])  # print the text hints to terminal
                        if args.keyboard:
                            action = get_key_action() # get the action from keyboard
                        else:
                            action = agent.act(obs) # get the action from random agent
                        obs, reward, termination, truncation, info= env.step([action]) #environment update
                        rewards+=reward
                        if reward != 0:
                            print('Reward:', reward)
                        if args.render or args.record_video: # show the visual cues & text cues and first-person view observation
                            frame, out=env.render(mode='ref+obs', show=args.render,save=out)
                        count_step+=1

                        if termination:
                            fps = count_step / (time.time() - t0)
                            if out is not None:
                                out.release()
                            print('Success')
                            print('reward:',rewards)
                            print('Fps:' + str(fps))
                            break
                        if truncation or action[2]==4:
                            fps = count_step / (time.time() - t0)
                            if out is not None:
                                out.release()
                            print('Failed')
                            print('reward:',rewards)
                            print('Fps:' + str(fps))
                            break
                    except KeyboardInterrupt:
                        print('\nReceived CTRL+C. Cleaning up...')
                        if out is not None:
                            out.release()
                        InterruptedException = True
                        break
                    except Exception as e:
                        if out is not None:
                            out.release()
                        print(e)
                        InterruptedException = True
                        break
                point_id+=1
                if InterruptedException:
                    if out is not None:
                        out.release()
                    env.close()
                    break
                env.close()
        except Exception as e:
            print(e)
            if out is not None:
                out.release()
            env.close()
            exit(0)