"""
    Running Simple Policy in Fetch Enviroment
    to Generate Expert Policy and Saving Data
    Include Image, Feedback and Command
"""

import os
import gym
from gym.envs.robotics import FetchPickAndPlaceEnv
import numpy as np
from matplotlib import pyplot as plt
try:
    import fetch_remote.utils as utils
    from fetch_remote.utils.data_save import DataSaver
    from fetch_remote.utils.finite_state_machine import FSM
except ImportError:
    import utils
    from utils.data_save import DataSaver
    from utils.finite_state_machine import FSM


GRIPPER_STATE = 1
LIMIT_Z = .415
SCALE_SPEED = 4.0
# desired image size
IMG_SIZE = 256
GYM_PATH = gym.__path__[0]
XML_DIR = os.path.join(GYM_PATH, 'envs/robotics/assets/fetch/myenvs')


args = utils.get_args()
utils.set_env_variable(args.display)
env_xmls = utils.get_all_xml(XML_DIR)

for noenv, env_name in enumerate(env_xmls):
    env = FetchPickAndPlaceEnv(xml_file=env_name)

    data_save_path = os.path.join(args.dir, 'object_{}'.format(noenv))
    saver    = DataSaver(data_save_path)
    tar_info = DataSaver(os.path.join(data_save_path, 'target'), info=True)

    for i in range(args.start, args.end):
        obs = env.reset(rand_text=args.random, rand_shadow=args.random)
        g = GRIPPER_STATE
        # save object and goal pos
        tar_info.append(trajectory=np.append(obs['achieved_goal'], obs['desired_goal']))

        goal = obs['achieved_goal'].copy()
        goal[-1] = goal[-1] + .1
        simple_policy = FSM(np.append(obs['eeinfo'][0], g), obs['achieved_goal'], goal, LIMIT_Z)
        total_reward = 0

        a = np.array([0., 0., 0., 1.])
        while not simple_policy.done:
            if args.display:
                env.render()
            else:
                rgb_obs = env.sim.render(width=IMG_SIZE, height=IMG_SIZE, camera_name="external_camera_0", depth=False,
                    mode='offscreen', device_id=-1)
                # appending image to saver
                saver.append(image=rgb_obs)
    
            # appending current feedback: ee pos (x, y, z), all of robot joints angle and gripper state
            trajectory = np.append(obs['eeinfo'][0], obs['weneed'])
            trajectory = np.append(trajectory, obs['gripper_dense'])
            # trajectory = np.append(trajectory, a)

            x, y, z, g = simple_policy.execute()
            
            a = np.array([x, y, z, g])
            # appending control command: delta ee pos (x, y, z), gripper state
            trajectory = np.append(trajectory, a)

            # scale up action
            a = a * SCALE_SPEED 
            obs, r, done, info = env.step(a)
            # update robot state
            simple_policy.robot_state = np.append(obs['eeinfo'][0], g)
            total_reward += r

            # appending auxiliary: object and gripper pos
            trajectory = np.append(trajectory, obs['achieved_goal'])
            trajectory = np.append(trajectory, obs['eeinfo'][0])

            # appending trajectory to saver
            saver.append(trajectory=trajectory)
                
            if info['is_success'] or done: 
                break

        # plt.imshow(rgb_obs)
        # plt.show(block=False)
        # plt.pause(0.001)
        print(i, "total reward %0.2f" % total_reward)

        if args.save:
            saver.save(i)
            tar_info.save(i)
    
    # for run only one env
    break
